#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : kb_access.py
@Author  : caixiongjiang
@Date    : 2026/10/18
@Function:
    知识库访问权限的唯一判定入口

    设计文档：docs/特殊功能设计/团队知识库与公共知识库设计.md §5.1
    判权矩阵（唯一事实源，所有端点过它，不允许各自写条件）：

    | visibility | can_read    | can_write    | can_manage                       |
    |----------- |-------------|--------------|----------------------------------|
    | private    | owner       | owner        | owner                            |
    | team       | team member | team member  | team owner/admin / KB创建者 / 平台 admin |
    | public     | 任何登录用户 | KB创建者/admin | 同 can_write                     |

    错误口径：KB 不存在 → 404；无权限 → 私库 404（不泄露存在性）、
    共享库（team/public）403（其存在性本就通过列表可见）。
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.db.mysql.models.business.knowledge_base import KnowledgeBase
from src.db.mysql.repositories.business.knowledge_base_repo import (
    knowledge_base_repo,
)
from src.db.mysql.repositories.business.team_member_repo import team_member_repo

# 可见性枚举（knowledge_base.visibility）
VISIBILITY_PRIVATE = "private"
VISIBILITY_TEAM = "team"
VISIBILITY_PUBLIC = "public"

_LEVEL_LABELS = {"read": "读取", "write": "写入", "manage": "管理"}


def is_platform_admin(user_id: str) -> bool:
    """平台管理员白名单判定。

    与 auth_core.require_admin 的白名单分支同源（FOLIO_AUTH_ADMIN_USER_IDS），
    不含 role claim 判定——role 需解析 token（get_current_user），
    由需要 admin 语义的端点在依赖层取到后经 is_admin 传入。
    """
    try:
        from auth_core.verifier import get_verifier

        return user_id in get_verifier().settings.admin_user_ids
    except Exception:
        return False


def can_create_public_kb(user_id: str) -> bool:
    """公共知识库创建权。

    PUBLIC_KB_ADMIN_ONLY=true 时仅平台 admin 可建（prd 公网环境建议 true，
    防止任意用户向全体注册用户广播内容）；false（默认，JP/dev）不限制。
    """
    try:
        from src.utils.env_manager import get_env_manager

        restricted = (
            get_env_manager().get("PUBLIC_KB_ADMIN_ONLY") or ""
        ).strip().lower() == "true"
    except Exception:
        restricted = False
    if not restricted:
        return True
    return is_platform_admin(user_id)


@dataclass
class KBAccess:
    """单个用户对单个知识库的访问能力"""

    can_read: bool = False
    can_write: bool = False
    can_manage: bool = False


def resolve_kb_access(
    session: Session,
    user_id: str,
    kb: KnowledgeBase,
    *,
    is_admin: bool = False,
) -> KBAccess:
    """纯判定函数（无 HTTP 语义），按可见性分流。

    异常数据兜底：visibility='team' 但 team_id 缺失时按 private 处理
    （最保守：仅 owner 可访问）。
    """
    visibility = kb.visibility or VISIBILITY_PRIVATE

    if visibility == VISIBILITY_TEAM and kb.team_id:
        role = team_member_repo.get_member_role(session, kb.team_id, user_id)
        if role is not None:
            manage = (
                role in ("owner", "admin")
                or kb.user_id == user_id  # KB 创建者
                or is_admin
            )
            return KBAccess(can_read=True, can_write=True, can_manage=manage)
        if is_admin:
            return KBAccess(can_read=True, can_write=True, can_manage=True)
        return KBAccess()

    if visibility == VISIBILITY_PUBLIC:
        write = kb.user_id == user_id or is_admin
        return KBAccess(can_read=True, can_write=write, can_manage=write)

    # private（含 team 可见性缺 team_id 的异常数据）
    if kb.user_id == user_id:
        return KBAccess(can_read=True, can_write=True, can_manage=True)
    return KBAccess()


def has_read_any_kb(
    session: Session,
    user_id: str,
    knowledge_base_ids,
    *,
    is_admin: bool = False,
) -> bool:
    """对一组知识库逐一判定，任一可读即 True（用于 document_id 反查多库场景）"""
    for kb_id in knowledge_base_ids:
        kb = knowledge_base_repo.get_by_id(session, kb_id)
        if kb is not None and resolve_kb_access(
            session, user_id, kb, is_admin=is_admin
        ).can_read:
            return True
    return False


def build_kb_scoped_filter(
    user_id: str,
    knowledge_base_id: Optional[str],
    *,
    knowledge_base_ids=None,
    document_id: Optional[str] = None,
    document_ids=None,
    chunk_type: Optional[str] = None,
):
    """构造检索过滤条件（判权 + 可见性分流，设计文档 §5.2.4）

    - 未指定 kb → user_id 过滤（现状语义：我的库全集）
    - private kb → user_id + kb（纵深防御保留）
    - team/public kb → 仅 kb（共享库向量按 kb 隔离，带 user_id 必然 0 命中）
    - 多 kb（``knowledge_base_ids`` 非空，会话绑定多库场景）→ 逐库判权后
      ``knowledge_base_id in [...]``；全部为私库时额外保留 user_id 纵深防御

    墓碑排除（exclude_deleted）由调用方按返回的 filter 形态自行套用。

    Raises:
        HTTPException: 404/403 目标知识库不可读
    """
    from src.db.mysql.connection.factory import get_mysql_manager
    from src.retrieve.types.query import MetadataFilter

    kwargs: dict = {}
    if knowledge_base_ids:
        kb_list = [k for k in dict.fromkeys(knowledge_base_ids) if k]
        manager = get_mysql_manager()
        with manager.get_session() as session:
            kbs = [
                require_kb_access(session, user_id, k, level="read")
                for k in kb_list
            ]
            all_private = all(
                (kb.visibility or VISIBILITY_PRIVATE) == VISIBILITY_PRIVATE
                for kb in kbs
            )
        if all_private:
            kwargs["user_id"] = user_id
        filters = MetadataFilter(knowledge_base_ids=kb_list, **kwargs)
    elif knowledge_base_id:
        manager = get_mysql_manager()
        with manager.get_session() as session:
            kb = require_kb_access(
                session, user_id, knowledge_base_id, level="read"
            )
            shared = (kb.visibility or VISIBILITY_PRIVATE) in (
                VISIBILITY_TEAM,
                VISIBILITY_PUBLIC,
            )
        if not shared:
            kwargs["user_id"] = user_id
        kwargs["knowledge_base_id"] = knowledge_base_id
        filters = MetadataFilter(**kwargs)
    else:
        kwargs["user_id"] = user_id
        filters = MetadataFilter(**kwargs)

    if document_id:
        filters.document_id = document_id
    if document_ids:
        filters.document_ids = list(document_ids)
    if chunk_type:
        filters.chunk_type = chunk_type
    return filters


def require_kb_access(
    session: Session,
    user_id: str,
    knowledge_base_id: str,
    level: str = "read",
    *,
    is_admin: bool = False,
) -> KnowledgeBase:
    """取知识库并按级别校验权限，失败抛 HTTPException；成功返回 KB 对象。

    Args:
        level: read / write / manage（映射 KBAccess.can_read/...）
    """
    kb = knowledge_base_repo.get_by_id(session, knowledge_base_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="知识库不存在")

    access = resolve_kb_access(session, user_id, kb, is_admin=is_admin)
    if not getattr(access, f"can_{level}", False):
        if (kb.visibility or VISIBILITY_PRIVATE) == VISIBILITY_PRIVATE:
            # 私库越权不泄露存在性
            raise HTTPException(
                status_code=404, detail="知识库不存在或无权限"
            )
        raise HTTPException(
            status_code=403,
            detail=f"无权{_LEVEL_LABELS.get(level, level)}该知识库",
        )
    return kb
