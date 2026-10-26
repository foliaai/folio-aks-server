#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : tombstone.py
@Author  : caixiongjiang
@Date    : 2026/09/01
@Function:
    待清理文件的墓碑查询，供检索侧排除已删除内容
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import time
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.db.mysql.connection.factory import get_mysql_manager
from src.db.mysql.repositories.business.workspace_file_system_repo import (
    workspace_file_system_repo,
)
from src.retrieve.types.query import MetadataFilter

# 墓碑存活时间以秒计（清理 worker 跑完就物理删行），缓存只为压掉同一轮召回里的重复查询
_CACHE_TTL_SECONDS = 5.0

# 共享库（team/public）维度的缓存键标记；真实 user_id 不会是这种形态，键空间不冲突
_KB_SCOPE = "__kb__"

_cache: Dict[Tuple[str, Optional[str]], Tuple[float, List[str]]] = {}


def get_tombstoned_document_ids(
    user_id: str,
    knowledge_base_id: Optional[str] = None,
) -> List[str]:
    """获取已删除但尚未清理完的 document_id，用于在检索中排除

    删除是「同步写墓碑 + 异步清理」的：向量在 worker 跑完之前仍留在 Milvus 里，
    而且 Milvus 搜索走默认的 Bounded 一致性，即便已经删除并 flush，
    短窗口内依然可能召回。所以排除不能依赖清理速度，必须显式过滤。

    查询失败时返回空列表：此时排除失效，宁可短暂多召回，也不要让检索整体挂掉。

    Args:
        user_id: 用户ID
        knowledge_base_id: 可选，按知识库缩小范围

    Returns:
        document_id 列表，通常为空或个位数
    """
    if not user_id:
        return []

    key = (user_id, knowledge_base_id)
    now = time.monotonic()

    cached = _cache.get(key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        manager = get_mysql_manager()
        with manager.get_session() as session:
            document_ids = workspace_file_system_repo.get_tombstoned_document_ids(
                session, user_id, knowledge_base_id
            )
    except Exception as e:
        logger.warning(f"查询待清理 document_id 失败，本轮不排除: {e}")
        return []

    _cache[key] = (now, document_ids)
    return document_ids


def get_kb_tombstoned_document_ids(
    knowledge_base_id: str,
) -> List[str]:
    """共享库（team/public）维度的墓碑：任一成员删除即对全员生效。

    共享库检索的 MetadataFilter 不带 user_id（向量按 kb 隔离），
    排除也必须按 kb 聚合，否则成员 A 删除的文件仍会召回给成员 B。
    """
    if not knowledge_base_id:
        return []

    key = (_KB_SCOPE, knowledge_base_id)
    now = time.monotonic()

    cached = _cache.get(key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        manager = get_mysql_manager()
        with manager.get_session() as session:
            document_ids = (
                workspace_file_system_repo.get_tombstoned_document_ids_by_kb(
                    session, knowledge_base_id
                )
            )
    except Exception as e:
        logger.warning(f"查询共享库待清理 document_id 失败，本轮不排除: {e}")
        return []

    _cache[key] = (now, document_ids)
    return document_ids


def invalidate(user_id: str) -> None:
    """删除受理后立刻失效缓存，避免刚删的文件在 TTL 内仍被召回"""
    for key in [k for k in _cache if k[0] == user_id]:
        _cache.pop(key, None)


def invalidate_kb(knowledge_base_id: str) -> None:
    """共享库维度的缓存失效（删除对全员生效，须立即清）"""
    for key in [
        k for k in _cache
        if k[0] == _KB_SCOPE and k[1] == knowledge_base_id
    ]:
        _cache.pop(key, None)


def exclude_deleted(filters: MetadataFilter) -> MetadataFilter:
    """就地把待清理的文件从检索范围里剔除

    所有构造 ``MetadataFilter`` 去做召回的地方都应该调一次，
    否则已删文件的向量在清理完成前仍会被召回到回答里。

    口径：filters 带 user_id（私有库/未指定库）→ 按 (user, kb) 排除；
    filters 只带 kb（共享库分流后的形态）→ 按 kb 聚合排除（全员生效）；
    filters 带多 kb（会话绑定多库）→ 逐 kb 聚合后并集。

    Returns:
        传入的 filters 本身，便于链式书写
    """
    kb_ids = list(filters.knowledge_base_ids or [])
    if len(kb_ids) > 1:
        excluded: List[str] = []
        seen: set = set()
        for kb_id in kb_ids:
            for doc_id in get_kb_tombstoned_document_ids(kb_id):
                if doc_id not in seen:
                    seen.add(doc_id)
                    excluded.append(doc_id)
    else:
        kb_id = filters.knowledge_base_id or (kb_ids[0] if kb_ids else None)
        if kb_id and not filters.user_id:
            excluded = get_kb_tombstoned_document_ids(kb_id)
        else:
            excluded = get_tombstoned_document_ids(
                filters.user_id or "", kb_id
            )
    if excluded:
        filters.exclude_document_ids = excluded
    return filters
