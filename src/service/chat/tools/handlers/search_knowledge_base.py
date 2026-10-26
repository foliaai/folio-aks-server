#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""search_knowledge_base 工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

from src.client.llm.types import ToolSchema
from src.prompts.chat.retrieval_hints import SEMANTIC_RECALL_LITERAL_HINT
from src.retrieve.pipeline.types import DirectAnswer
from src.retrieve.types.result import ChunkItem
from src.service.chat.chunk_alias_map import NavAliasMap
from src.service.chat.tools.base import ToolDefinition
from src.service.chat.tools.helpers import format_chunks_for_llm
from src.service.chat.tools.runtime import get_current_tool_call_id

if TYPE_CHECKING:
    from src.service.chat.tools.kit import KnowledgeNavToolKit

NAME = "search_knowledge_base"

SCHEMA: ToolSchema = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "在知识库中做**语义相关**片段检索（路由规划 + 多路召回 + 融合 + 精排），"
            "返回与 query 最相关的 Top-K 段落，适合概念探索与开放式问题。"
            "若 qa_dense 高置信命中，会先置顶一条原子问答及其依据原文，再跟精排 Top-K；"
            "**不会**因此跳过其它路的对齐/融合/精排。"
            "**不保证**某术语在全文中的字面全部命中；若需穷举某词的全部出现或确认精确数值/配置，"
            "再用 `grep_chunks` 做字面全扫，并用 `read_chunks` 取全文。"
            "返回的每条 chunk 正文为预览（默认 200 字）；preview 不完整时用 `read_chunks`，"
            "**不要**用 `context_window`（只取邻居、不会让当前 chunk 变全文）。"
            "可用不同 query 多次调用；`chunk_type` 可过滤 text/image/table/code_block。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "检索查询文本，描述需要查找的信息",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 30,
                },
                "chunk_type": {
                    "type": "string",
                    "description": "过滤 chunk 类型：text=文本, image=图片, table=表格, code_block=代码块。不指定则返回所有类型",
                    "enum": ["text", "image", "table", "code_block"],
                },
            },
            "required": ["query_text"],
        },
    },
}


def _chunks_brief(chunks: List[ChunkItem]) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "score": chunk.score,
            "preview": (chunk.text or "")[:200],
        }
        for chunk in chunks
    ]


def format_pinned_search_for_llm(
    pinned: Optional[DirectAnswer],
    evidence: List[ChunkItem],
    items: List[ChunkItem],
    *,
    alias_map: Optional[NavAliasMap] = None,
) -> str:
    """置顶 QA + 依据原文 + 精排 Top-K，给 LLM 看。"""
    if pinned is None:
        return format_chunks_for_llm(
            items,
            alias_map=alias_map,
            append_semantic_literal_hint=True,
        )

    source_labels: List[str] = []
    for cid in pinned.source_chunk_ids:
        if not cid:
            continue
        source_labels.append(
            alias_map.alias_for(cid) if alias_map else cid,
        )
    source_text = ", ".join(source_labels) or "（无）"

    parts: List[str] = [
        "【高置信原子问答】"
        f"（qa_dense 相似度 {pinned.score:.4f}，已置顶；"
        "其它路仍走对齐/融合/精排，本条未短路）\n"
        f"Q: {pinned.question}\n"
        f"A: {pinned.answer}\n"
        f"依据 chunk: {source_text}（正文见下方「依据原文」，不经精排）",
    ]
    if evidence:
        parts.append(
            "【依据原文】\n"
            + format_chunks_for_llm(
                evidence,
                alias_map=alias_map,
                append_semantic_literal_hint=False,
            ),
        )
    elif pinned.source_chunk_ids:
        parts.append(
            "（依据 chunk 正文未能从库中补全，可用 read_chunks 按上方 id 取全文）",
        )
    else:
        parts.append("（该 QA 未标注依据 chunk）")

    if items:
        parts.append(
            "【其它相关片段】\n"
            + format_chunks_for_llm(
                items,
                alias_map=alias_map,
                append_semantic_literal_hint=True,
            ),
        )
    else:
        parts.append(SEMANTIC_RECALL_LITERAL_HINT)

    return "\n\n".join(parts)


async def handle(
    kit: KnowledgeNavToolKit,
    query_text: str,
    top_k: int = 10,
    chunk_type: Optional[str] = None,
) -> str:
    if kit.retrieve_service is None:
        return "search_knowledge_base: 检索服务不可用。"

    from src.retrieve.pipeline.types import RetrieveRequest
    from src.service.knowledge.kb_access import build_kb_scoped_filter
    from src.service.knowledge.tombstone import exclude_deleted

    # 可见性分流：共享库（team/public）向量按 kb 隔离，不能带 user_id 过滤
    # （否则只能检索到自己上传的部分）；会话绑定的库已不可访问时软降级。
    # 多 kb 会话按全量列表过滤（knowledge_base_id in [...]），不再只取第一个
    try:
        filters = build_kb_scoped_filter(
            kit.user_id,
            None,
            knowledge_base_ids=list(kit.knowledge_base_ids or []),
            chunk_type=chunk_type,
        )
    except Exception:
        return "search_knowledge_base: 当前会话绑定的知识库已不可访问。"
    if kit.scope_document_ids:
        filters.document_ids = list(kit.scope_document_ids)
    exclude_deleted(filters)

    request = RetrieveRequest(
        query_text=query_text,
        filters=filters,
        top_k=top_k,
        conversation_context=None,
        enable_route_planner=getattr(kit, "enable_route_planner", False),
    )

    async def on_progress(stage: str) -> None:
        await kit.emit_progress(stage, channel="retrieval")

    try:
        response = await kit.retrieve_service.retrieve(
            request,
            on_progress=on_progress,
        )

        pinned = response.direct_answer
        evidence = list(response.pinned_evidence or [])
        items = list(response.items or [])
        visible = evidence + items

        kit.supplemented.extend(visible)
        kit.note_result_count(len(visible))
        if response.planner_model:
            kit.note_execution_model(response.planner_model)

        tc_id = get_current_tool_call_id()
        if tc_id:
            params: Dict[str, Any] = {
                "query_text": query_text,
                "top_k": top_k,
            }
            if chunk_type:
                params["chunk_type"] = chunk_type
            if pinned is not None:
                params["direct_answer"] = pinned.model_dump(exclude_none=True)
                params["qa_pinned"] = True
            if response.route_plan:
                params["route_plan"] = response.route_plan.model_dump(
                    exclude_none=True,
                )
            recall_stats_dict = (
                response.recall_stats.model_dump(exclude_none=True)
                if response.recall_stats is not None
                else None
            )
            kit.search_results[tc_id] = (
                _chunks_brief(visible),
                params,
                recall_stats_dict,
            )

        if pinned is not None:
            logger.debug(
                f"search_knowledge_base({query_text!r}) → QA 置顶 "
                f"qa_id={pinned.qa_id} score={pinned.score:.4f} "
                f"evidence={len(evidence)} items={len(items)}"
            )
        else:
            logger.debug(
                f"search_knowledge_base({query_text!r}) → {len(items)} chunks"
            )

        return format_pinned_search_for_llm(
            pinned,
            evidence,
            items,
            alias_map=kit.alias_map,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"search_knowledge_base 执行异常: {e}")
        return f"检索失败: {e}"


DEFINITION = ToolDefinition(name=NAME, schema=SCHEMA, handler=handle)
