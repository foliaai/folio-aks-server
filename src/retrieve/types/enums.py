#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : enums.py
@Author  : caixiongjiang
@Date    : 2026/02/28
@Function: 
    检索模块枚举定义
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from enum import Enum


class SearchMode(str, Enum):
    """检索模式"""
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    STRUCTURED = "structured"
    GRAPH = "graph"


class GranularityLevel(str, Enum):
    """检索粒度层级"""
    DOCUMENT = "document"
    SECTION = "section"
    CHUNK = "chunk"
    ELEMENT = "element"
    LINK = "link"


class ElementType(str, Enum):
    """文档元素类型"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    EQUATION = "equation"
    CODE_BLOCK = "code_block"


class SortOrder(str, Enum):
    """排序方向"""
    RELEVANCE = "relevance"
    CHRONOLOGICAL = "chronological"
    POSITION = "position"


class MatchMode(str, Enum):
    """字面匹配模式"""
    EXACT = "exact"
    PREFIX = "prefix"
    REGEX = "regex"
    FUZZY = "fuzzy"


class TraverseDirection(str, Enum):
    """导航遍历方向"""
    PREV = "prev"
    NEXT = "next"
    BOTH = "both"
    UP = "up"
    DOWN = "down"


class ConsistencyLevel(str, Enum):
    """Milvus 一致性级别

    - STRONG:     强一致性，读取能看到所有已提交的写入。延迟最高但数据最准确。
    - BOUNDED:    有界一致性，读取可能滞后一个可配置的时间窗口。
    - SESSION:    会话一致性，同一会话内读取能看到该会话的所有写入。
    - EVENTUALLY: 最终一致性，延迟最低但可能读到旧数据。适合对实时性要求不高的场景。
    """
    STRONG = "Strong"
    BOUNDED = "Bounded"
    SESSION = "Session"
    EVENTUALLY = "Eventually"


class SemanticTarget(str, Enum):
    """语义检索目标 Collection 类型

    与 Milvus 中的 Collection 一一对应：
    - CHUNK           → chunk_store
    - SECTION         → section_store
    - ENHANCED        → enhanced_chunk_store
    - ATOMIC_QA       → atomic_qa_store
    - FILE_SUMMARY    → file_summary_store
    - SECTION_SUMMARY → section_summary_store
    """
    CHUNK = "chunk"
    SECTION = "section"
    ENHANCED = "enhanced"
    ATOMIC_QA = "atomic_qa"
    FILE_SUMMARY = "file_summary"
    SECTION_SUMMARY = "section_summary"
