#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : section_summary_service.py
@Function:
    Section 摘要抽取 Service（业务编排层，不含 Kafka / DB 读依赖）。

    对应 text_splitter_service.py 的角色：编排 extract 层组件，不写具体算法。
    核心抽取逻辑（树构建、文本组装、LLM 摘要、rollup）已下沉到
    src/index/common_file_extract/extract/，本类只做：
    - 配置加载（components.json 的 section_summary 段）
    - LLM 客户端懒加载
    - 主编排：payload → 上下文 → 建树 → 并行叶子摘要 → 分层 rollup → 汇总结果

    设计原则：
    - **不读数据库**：split 阶段的 section/chunk 写库走 db_write.* 异步 Consumer，
      与 split.end 的消费无顺序保证；故 SectionSummary 的输入必须完全来自
      SplitEndMessage 消息体本身，避免「读库时数据尚未落盘」的竞态。
    - 与 TextSplitterService 一致：Service 不触碰 Kafka，仅返回结果 DTO。
    - 单 section 失败不阻断整文档：extract 层负责跳过，Service 层负责汇总。
@Modify History:
    2026/07/02 - 初版（从 DB 读改为从消息 payload 读，消除写库竞态）
    2026/07/08 - 重构：核心抽取逻辑下沉到 src/index/common_file_extract/extract/，
                 本类仅保留编排，与 parser/splitter 的分层模式对齐

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from src.index.common_file_extract.extract import (
    SectionNode,
    SectionSummarizer,
    build_section_tree,
    build_sections_from_payload,
    post_order,
)
from src.types.models.section_summary_result import SectionSummaryResult
from src.utils.component_config_manager import get_component_config_manager


class SectionSummaryService:
    """Section 摘要抽取服务（编排层，纯内存计算 + LLM 调用，无 DB 读）。"""

    COMPONENT_NAME = "section_summary"

    def __init__(self) -> None:
        self._config_manager = get_component_config_manager()
        self._llm_client = None  # 懒加载
        self._summarizer: Optional[SectionSummarizer] = None  # 懒加载
        self._max_concurrency: int = int(
            self._config_manager.get_component_config(self.COMPONENT_NAME).get(
                "max_concurrency", 4
            )
        )
        self._max_retries: int = int(
            self._config_manager.get_component_config(self.COMPONENT_NAME).get(
                "max_retries", 2
            )
        )

    # ========== 依赖懒加载 ==========

    def _get_llm_client(self):
        """懒加载 LLM 客户端（基于 components.json 的 section_summary 配置）。"""
        if self._llm_client is None:
            self._llm_client = (
                self._config_manager.get_llm_client_for_component(self.COMPONENT_NAME)
            )
            logger.info(
                f"SectionSummaryService LLM 客户端已创建: "
                f"model={self._llm_client.model_name}"
            )
        return self._llm_client

    def _get_summarizer(self) -> SectionSummarizer:
        """懒加载 SectionSummarizer（注入 LLM 客户端与重试配置）。"""
        if self._summarizer is None:
            self._summarizer = SectionSummarizer(
                llm_client=self._get_llm_client(),
                max_retries=self._max_retries,
            )
        return self._summarizer

    # ========== 主编排 ==========

    async def summarize_document_sections(
        self,
        document_id: str,
        sections_data: List[Dict[str, Any]],
        chunks_data: List[Dict[str, Any]],
        language: str = "unknown",
        knowledge_base_id: Optional[str] = None,
        knowledge_base_name: Optional[str] = None,
        parse_tool: Optional[str] = None,
    ) -> SectionSummaryResult:
        """
        编排整个文档的 section 摘要抽取（含建树 + 叶子 LLM + 父节点 rollup）。

        输入完全来自 SplitEndMessage 消息体（sections + chunks），不读数据库，
        消除「split 写库异步 → section_summary 读库竞态」问题。

        流程：
        1. build_sections_from_payload：flat sections 列表
        2. build_section_tree：建树（策略按 parse_tool 分派：
           markdown/html → heading_level（标题层级即真值），其余 → 标题编号推断）
        3. 叶子 section（有 chunk 的结构叶子）→ 并行 LLM 摘要
        4. 父节点后序遍历 rollup：LLM 合成子摘要为父摘要；LLM 失败降级为纯拼接
        5. 汇总所有 items 返回

        Args:
            document_id: 文档 ID
            sections_data: SplitEndMessage.sections
            chunks_data: SplitEndMessage.chunks
            language: 文档语言（来自 SplitEndMessage.language）
            knowledge_base_id / knowledge_base_name: 知识库归属（透传到结果）

        Returns:
            SectionSummaryResult（items 含叶子 + 父节点摘要，is_leaf 字段区分）
        """
        sections = build_sections_from_payload(sections_data, chunks_data)

        if not sections:
            logger.warning(
                f"SectionSummary: 消息 payload 无 section，返回空结果: "
                f"document_id={document_id}"
            )
            return SectionSummaryResult(
                document_id=document_id,
                items=[],
                llm_model="",
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
            )

        # Step 1: 建树（markdown/html/excel 的标题层级即真值，用 heading_level；
        # MinerU 系标题 text_level 扁平，走编号推断）
        strategy = "heading_level" if parse_tool in ("markdown", "html", "excel") else "numbering"
        roots = self._build_tree(sections, strategy=strategy)
        traversal = post_order(roots)

        # Step 2: 并行 LLM 摘要（凡有自身 chunk 的节点：结构叶子 + 混合节点）
        # 混合节点（有 chunk 也有子节点）先跑自己 chunk 的 LLM 摘要，
        # Step 3 rollup 时把这段作为「本节引言」并入父级 rollup 输入。
        semaphore = asyncio.Semaphore(self._max_concurrency)
        summarizer = self._get_summarizer()

        llm_nodes = [n for n in traversal if n.should_llm_summarize()]
        # 记录混合节点的 own LLM summary（rollup 前把 node.summary_item 清掉，
        # 由 rollup 结果覆盖，保证一个 section_id 只输出一个 SectionSummaryItem）
        own_llm_summary_map: Dict[str, str] = {}

        llm_tasks = [
            summarizer.summarize_leaf(
                n.section,
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                language=language,
                semaphore=semaphore,
            )
            for n in llm_nodes
        ]
        llm_results = await asyncio.gather(*llm_tasks, return_exceptions=False)

        for node, item in zip(llm_nodes, llm_results):
            if item is None:
                continue
            item.parent_section_id = (
                node.parent.section_id if node.parent else None
            )
            item.is_leaf = node.is_structural_leaf()

            if node.should_rollup():
                # 混合节点：暂存 own LLM summary，rollup 会覆盖 summary_item
                own_llm_summary_map[node.section_id] = item.summary_text
            else:
                # 纯叶子：LLM 结果就是最终 summary_item
                node.summary_item = item

        # Step 3: 后序遍历，父节点 rollup（同一层无依赖，可并行）
        # 按 inferred_level 从大到小分层（不含纯叶子层），逐层并发
        rollup_nodes = [n for n in traversal if n.should_rollup()]
        rollup_nodes_by_level: Dict[int, List[SectionNode]] = {}
        for n in rollup_nodes:
            rollup_nodes_by_level.setdefault(n.inferred_level, []).append(n)

        for level in sorted(rollup_nodes_by_level.keys(), reverse=True):
            level_nodes = rollup_nodes_by_level[level]
            rollup_tasks = [
                summarizer.rollup_parent(
                    node=n,
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    knowledge_base_name=knowledge_base_name,
                    language=language,
                    semaphore=semaphore,
                    own_llm_summary=own_llm_summary_map.get(n.section_id),
                )
                for n in level_nodes
            ]
            rollup_results = await asyncio.gather(
                *rollup_tasks, return_exceptions=False
            )
            for n, item in zip(level_nodes, rollup_results):
                if item is not None:
                    n.summary_item = item

        # Step 4: 收集所有 items（按后序：叶子 → 父）
        items = [n.summary_item for n in traversal if n.summary_item is not None]

        # 汇总 token 用量（LLMResponse.usage 在 client 内部已记录日志，这里做粗略聚合）
        token_usage: Dict[str, int] = {"input": 0, "output": 0, "total": 0}
        llm_model = self._get_llm_client().model_name

        pure_leaf_nodes = [n for n in llm_nodes if not n.should_rollup()]
        mixed_nodes = [n for n in llm_nodes if n.should_rollup()]
        leaf_success = sum(1 for n in pure_leaf_nodes if n.summary_item is not None)
        rollup_success = sum(
            1 for n in rollup_nodes if n.summary_item is not None
        )
        logger.info(
            f"SectionSummary: 文档抽取完成 document_id={document_id}, "
            f"total_sections={len(sections)}, "
            f"leaf_llm={leaf_success}/{len(pure_leaf_nodes)}, "
            f"mixed_nodes={len(mixed_nodes)}, "
            f"rollup={rollup_success}/{len(rollup_nodes)}, "
            f"total_items={len(items)}"
        )

        return SectionSummaryResult(
            document_id=document_id,
            items=items,
            llm_model=llm_model,
            token_usage=token_usage,
            knowledge_base_id=knowledge_base_id,
            knowledge_base_name=knowledge_base_name,
        )

    # ========== 内部辅助 ==========

    @staticmethod
    def _build_tree(sections, strategy: str = "numbering"):
        """建树的薄封装，便于在 Service 层打日志或扩展时统一入口。"""
        return build_section_tree(sections, strategy=strategy)
