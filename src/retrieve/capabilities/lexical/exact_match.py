#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : exact_match.py
@Author  : caixiongjiang
@Date    : 2026/03/02
@Function: 
    精确 / 前缀 / 正则 字面匹配原子能力
    
    核心流程:
      keywords + match_mode → MongoDB $regex 查询
      → ChunkData 结果 → RetrieveResult[ChunkItem]
    
    依赖:
      - MongoDB ChunkDataRepository / ChunkData 模型
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
import re
from typing import Any, Dict, List, Optional

from src.db.mongodb.models.chunk_data import ChunkData
from src.retrieve.capabilities.base import BaseCapability, CapabilityDescriptor
from src.types.utils.chunk_search_text import resolve_chunk_display_text
from src.retrieve.capabilities.lexical._filter_helper import (
    filter_has_chunk_scope,
    resolve_chunk_ids_from_filters,
)
from src.retrieve.types.enums import MatchMode
from src.retrieve.types.query import LexicalQuery, MetadataFilter
from src.retrieve.types.result import ChunkItem, RetrieveResult


class ExactMatch(BaseCapability):
    """精确 / 前缀 / 正则 字面匹配

    根据 MatchMode 构建 MongoDB $regex 查询，对 chunk_data 的 text 字段
    进行字面级别匹配。适用于专有名词、型号代码、公式符号等精确查找。

    支持模式:
      - EXACT:  完全匹配关键词
      - PREFIX: 前缀匹配
      - REGEX:  用户自定义正则表达式
      - FUZZY:  大小写不敏感的包含匹配

    对应 MongoDB Collection: chunk_data
    """

    async def _do_execute(self, **kwargs: Any) -> RetrieveResult:
        query: LexicalQuery = kwargs["query"]

        if not query.keywords:
            raise ValueError("ExactMatch 需要 keywords 参数")

        mongo_query = self._build_mongo_query(query.keywords, query.match_mode)

        # 透传 MetadataFilter：先在 MySQL 解析 chunk_id 集合，再注入 _id $in 条件
        # （2026-10-26 起解析失败 fail-closed 返回空集，不再回退无限制查询）
        if filter_has_chunk_scope(query.filters):
            allowed_ids = resolve_chunk_ids_from_filters(query.filters)
            if allowed_ids is None:
                # 理论不可达（前置 filter_has_chunk_scope 已保证有裁剪字段），
                # 防御性 fail-closed
                return RetrieveResult(items=[], total_count=0)
            if not allowed_ids:
                return RetrieveResult(items=[], total_count=0)
            mongo_query["_id"] = {"$in": allowed_ids}

        results = await ChunkData.find(
            mongo_query,
        ).limit(query.top_k).to_list()

        items = self._build_result_items(results)

        return RetrieveResult(
            items=items,
            total_count=len(items),
        )

    @staticmethod
    def _build_regex_pattern(keyword: str, mode: MatchMode) -> str:
        """根据匹配模式构建正则表达式"""
        escaped = re.escape(keyword)
        if mode == MatchMode.EXACT:
            return f"^{escaped}$"
        elif mode == MatchMode.PREFIX:
            return f"^{escaped}"
        elif mode == MatchMode.REGEX:
            return keyword
        else:
            return escaped

    def _build_mongo_query(
        self,
        keywords: List[str],
        mode: MatchMode,
    ) -> Dict[str, Any]:
        """构建 MongoDB 查询条件

        多个关键词之间取 OR 关系。
        """
        conditions: List[Dict[str, Any]] = []
        for keyword in keywords:
            pattern = self._build_regex_pattern(keyword, mode)
            conditions.append({
                "$or": [
                    {"search_text": {"$regex": pattern, "$options": "i"}},
                    {"text_meta.image_caption": {"$regex": pattern, "$options": "i"}},
                    {"text_meta.table_caption": {"$regex": pattern, "$options": "i"}},
                    {"text_meta.text": {"$regex": pattern, "$options": "i"}},
                ],
            })

        query: Dict[str, Any] = {"deleted": 0}
        if len(conditions) == 1:
            query.update(conditions[0])
        else:
            query["$or"] = conditions

        return query

    @staticmethod
    def _build_result_items(docs: List[ChunkData]) -> List[ChunkItem]:
        items: List[ChunkItem] = []
        for doc in docs:
            meta: Dict[str, Any] = {"chunk_type": doc.chunk_type}
            text_meta = doc.text_meta or {}
            if text_meta.get("image_caption"):
                meta["image_caption"] = text_meta["image_caption"]
            if text_meta.get("image_footnote"):
                meta["image_footnote"] = text_meta["image_footnote"]
            if text_meta.get("table_caption"):
                meta["table_caption"] = text_meta["table_caption"]
            if text_meta.get("table_footnote"):
                meta["table_footnote"] = text_meta["table_footnote"]
            items.append(ChunkItem(
                chunk_id=str(doc.id),
                score=1.0,
                text=resolve_chunk_display_text(doc),
                metadata=meta,
            ))
        return items

    def describe(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            name="exact_match",
            display_name="精确字面匹配",
            description=(
                "对 MongoDB chunk_data 进行精确 / 前缀 / 正则字面匹配。"
                "适用于专有名词、型号代码、公式符号等需要字符级精确查找的场景。"
            ),
            input_schema={
                "keywords": "List[str] - 关键词列表（多个取 OR）",
                "match_mode": "MatchMode - 匹配模式（EXACT/PREFIX/REGEX/FUZZY）",
                "top_k": "int - 返回数量上限，默认 10",
                "filters": "MetadataFilter - 元数据过滤条件（可选）",
            },
            output_type="RetrieveResult[ChunkItem]",
        )
