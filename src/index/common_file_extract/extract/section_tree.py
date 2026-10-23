#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : section_tree.py
@Function:
    Section 树构建与遍历（纯算法，无 LLM / 无 DB 依赖）。

    对应 splitter/text_splitter.py 的角色：核心算法组件。

    背景：MinerU 把所有标题都标成同一 text_level（扁平），无法反映真实层级。
    但标题文本本身通常带编号（"1"、"1.1"、"2.6.1"），层级信息藏在编号里。
    本模块解析编号、建树，供 section_summarizer 做自底向上 rollup。

    - build_section_tree：从扁平 sections 列表构造 section 树
      （编号推层级 + 父子关系；无编号 section 一律挂根）
    - post_order：后序遍历（叶子在前，父节点在后）
    - collect_descendant_chunk_ids：递归合并节点及所有后代的 chunk_id_list
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import Dict, List, Optional

from loguru import logger

from src.index.common_file_extract.extract.models import SectionNode, SectionWithChunks
from src.utils.section_numbering import parse_numbering


def _build_tree_by_heading_level(
    sections: List[SectionWithChunks],
) -> List[SectionNode]:
    """heading_level 策略：level 栈建树（Markdown 链路，level 即真层级）。"""
    roots: List[SectionNode] = []
    # 栈元素：(level, node)；栈顶是当前最近的开阔祖先
    stack: List[tuple] = []

    for sec in sections:
        level = sec.level if (sec.level or 0) > 0 else 1
        node = SectionNode(section=sec, numbering=None)

        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            parent_node = stack[-1][1]
            node.parent = parent_node
            node.inferred_level = parent_node.inferred_level + 1
            parent_node.children.append(node)
        else:
            node.inferred_level = 1
            roots.append(node)

        stack.append((level, node))

    logger.debug(
        f"SectionSummary: heading_level 树构建完成 total={len(sections)}, "
        f"roots={len(roots)}"
    )
    return roots


def build_section_tree(
    sections: List[SectionWithChunks],
    strategy: str = "numbering",
) -> List[SectionNode]:
    """
    从扁平 section 列表构造 section 树。

    建树策略按解析来源分派（parse_tool）：

    strategy="numbering"（默认，MinerU 系 pdf/word/ppt）：
    不依赖 MinerU 的 text_level（扁平、无区分度），从标题编号推层级：
    1. 对每个 section 跑 parse_numbering(title) → NumberingInfo（或 None）
    2. 有编号的 section：
       - 顺着 numbering.parent() 逐级往上找已注册的祖先编号
       - 找到 → 挂到该祖先的 children；找不到 → 挂根
    3. 无编号的 section（如「Introduction / Discussion / References」）：
       - 一律挂根（作为独立顶级 section）
       - 理由：无编号 section 在学术/技术文档中多为顶级 section（Abstract,
         Introduction, Discussion, Conclusion, References, Acknowledgments 等）；
         若挂"最近编号祖先"会误把 Discussion 挂到 3.1 下这种典型误判
    4. inferred_level：有编号 = 编号段数（or 匹配的父级 level+1）；
       无编号 = 1（顶级）

    strategy="heading_level"（Markdown 解析链路）：
    md 标题层级（#~###### → level 1~6）是真值，编号反而冗余甚至写错。
    维护 level 栈：level 更深 → 挂栈顶；不深 → 弹栈到最近更浅的祖先，
    挂根为止。容忍跳级（# 直接接 ####），编号不参与建树。

    Returns:
        根节点列表（顶级 section，按文档顺序）
    """
    if strategy == "heading_level":
        return _build_tree_by_heading_level(sections)

    roots: List[SectionNode] = []
    # 已注册编号 key → 节点，用于 O(1) 找父
    key_to_node: Dict[str, SectionNode] = {}

    for sec in sections:
        info = parse_numbering(sec.title)
        node = SectionNode(section=sec, numbering=info)

        if info is not None:
            # 有编号：沿父编号链找已注册祖先
            parent_node: Optional[SectionNode] = None
            parent_info = info.parent()
            while parent_info is not None:
                p = key_to_node.get(parent_info.key())
                if p is not None:
                    parent_node = p
                    break
                parent_info = parent_info.parent()

            if parent_node is None:
                node.inferred_level = 1
                roots.append(node)
            else:
                node.parent = parent_node
                node.inferred_level = parent_node.inferred_level + 1
                parent_node.children.append(node)

            key_to_node[info.key()] = node
        else:
            # 无编号 section 一律挂根（作为顶级 section，避免误挂到最近编号祖先下）
            node.inferred_level = 1
            roots.append(node)

    logger.debug(
        f"SectionSummary: section 树构建完成 total={len(sections)}, "
        f"roots={len(roots)}, numbered={len(key_to_node)}, "
        f"unnumbered={len(sections) - len(key_to_node)}"
    )
    return roots


def post_order(roots: List[SectionNode]) -> List[SectionNode]:
    """对 section 树做后序遍历（叶子在前，父节点在后）。"""
    result: List[SectionNode] = []

    def visit(node: SectionNode) -> None:
        for child in node.children:
            visit(child)
        result.append(node)

    for root in roots:
        visit(root)
    return result


def collect_descendant_chunk_ids(node: SectionNode) -> List[str]:
    """
    递归合并该节点及所有后代的 chunk_id_list（去重，保序）。

    父 section 用于溯源：Milvus 命中父 summary 后可拿到子树下所有 chunk。
    """
    seen: set = set()
    result: List[str] = []

    def visit(n: SectionNode) -> None:
        for cid in n.section.chunk_id_list:
            if cid and cid not in seen:
                seen.add(cid)
                result.append(cid)
        for child in n.children:
            visit(child)

    visit(node)
    return result
