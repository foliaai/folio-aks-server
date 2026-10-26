#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : query.py
@Author  : caixiongjiang
@Date    : 2026/02/28
@Function: 
    检索查询参数模型定义
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from src.retrieve.types.enums import (
    ConsistencyLevel, ElementType, GranularityLevel, MatchMode,
    SemanticTarget, TraverseDirection,
)


class MetadataFilter(BaseModel):
    """通用元数据过滤条件

    在各类检索中用于缩小查询范围。
    所有字段均为可选，多个字段同时指定时取 AND 交集。
    """
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    exclude_document_ids: Optional[List[str]] = None
    section_ids: Optional[List[str]] = None
    knowledge_base_id: Optional[str] = None
    knowledge_base_ids: Optional[List[str]] = None
    knowledge_base_name: Optional[str] = None
    label_id: Optional[str] = None
    source_type: Optional[str] = None
    chunk_type: Optional[str] = None
    date_range: Optional[Tuple[str, str]] = None

    def to_milvus_filter_expr(self) -> Optional[str]:
        """将过滤条件转化为 Milvus filter expression 字符串

        Returns:
            Milvus 兼容的过滤表达式，无条件时返回 None
        """
        parts: List[str] = []
        if self.user_id:
            parts.append(f"user_id == '{self.user_id}'")
        if self.document_id:
            parts.append(f"document_id == '{self.document_id}'")
        if self.document_ids:
            ids_str = ", ".join(f"'{d}'" for d in self.document_ids)
            parts.append(f"document_id in [{ids_str}]")
        if self.exclude_document_ids:
            ids_str = ", ".join(f"'{d}'" for d in self.exclude_document_ids)
            parts.append(f"document_id not in [{ids_str}]")
        if self.knowledge_base_ids:
            ids_str = ", ".join(f"'{k}'" for k in self.knowledge_base_ids)
            parts.append(f"knowledge_base_id in [{ids_str}]")
        elif self.knowledge_base_id:
            parts.append(f"knowledge_base_id == '{self.knowledge_base_id}'")
        if self.label_id:
            parts.append(f"label_id == '{self.label_id}'")
        if self.chunk_type:
            parts.append(f"type == '{self.chunk_type}'")
        return " and ".join(parts) if parts else None


class SemanticQuery(BaseModel):
    """语义向量检索查询参数

    Attributes:
        query_text: 自然语言查询文本（与 query_vector 二选一）
        query_vector: 预计算的查询向量（与 query_text 二选一）
        target: 目标 Collection 类型
        top_k: 返回结果数量上限
        filters: 元数据过滤条件
        return_content: 是否同时从 MongoDB 获取全文内容（默认仅返回 ID + Score）
        consistency_level: Milvus 一致性级别（None 时使用 Collection 默认级别）
    """
    target: SemanticTarget
    top_k: int = 10
    query_text: Optional[str] = None
    query_vector: Optional[List[float]] = None
    filters: MetadataFilter = Field(default_factory=MetadataFilter)
    return_content: bool = False
    consistency_level: Optional[ConsistencyLevel] = None
    score_threshold: Optional[float] = Field(
        default=None,
        description="召回层分数阈值 (COSINE: 0~1), 低于此值的结果在召回阶段即被过滤; None 表示不过滤",
    )

    @model_validator(mode="after")
    def _check_query_input(self) -> "SemanticQuery":
        if not self.query_text and not self.query_vector:
            raise ValueError("query_text 和 query_vector 必须至少提供一个")
        return self


class LexicalQuery(BaseModel):
    """字面检索统一查询参数

    Attributes:
        query_text: 自然语言查询文本（BM25Search 使用）
        keywords: 关键词列表（ExactMatch 使用）
        match_mode: 字面匹配模式（ExactMatch 使用）
        bool_expression: 布尔表达式字符串（BooleanSearch 使用）
        top_k: 返回结果数量上限
        target_granularity: 目标检索粒度
        filters: 元数据过滤条件
        score_threshold: 召回层分数阈值 (IP score), 低于此值的结果被过滤
    """
    top_k: int = 10
    query_text: Optional[str] = None
    keywords: Optional[List[str]] = None
    match_mode: MatchMode = MatchMode.EXACT
    bool_expression: Optional[str] = None
    target_granularity: GranularityLevel = GranularityLevel.CHUNK
    filters: MetadataFilter = Field(default_factory=MetadataFilter)
    score_threshold: Optional[float] = Field(
        default=None,
        description="召回层分数阈值 (IP score), 低于此值的结果在召回阶段即被过滤; None 表示不过滤",
    )


class NavigationQuery(BaseModel):
    """结构化导航查询参数

    所有导航能力共用此查询参数模型。各能力按需使用相关字段，
    不相关的字段保持默认值即可。

    Attributes:
        anchor_id: 锚点 ID（chunk_id / section_id / document_id / element_id）
        anchor_type: 锚点的粒度类型
        direction: 遍历方向（ContextWindow 使用 PREV/NEXT/BOTH）
        target_granularity: 目标粒度（DrillDown / RollUp 使用）
        window_size: 上下文窗口大小（ContextWindow 使用，表示单方向 Chunk 数量）
        max_depth: 最大展开深度（Skeleton 使用）
        element_type_filter: 元素类型过滤（DrillDown 到 Element 粒度时使用）
        include_content: 是否从 MongoDB 获取全文内容
        filters: 元数据过滤条件
    """
    anchor_id: str
    anchor_type: GranularityLevel
    direction: TraverseDirection = TraverseDirection.BOTH
    target_granularity: Optional[GranularityLevel] = None
    window_size: int = 3
    max_depth: int = 3
    element_type_filter: Optional[ElementType] = None
    include_content: bool = True
    filters: MetadataFilter = Field(default_factory=MetadataFilter)
