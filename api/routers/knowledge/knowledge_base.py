#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : knowledge_base.py
@Author  : caixiongjiang
@Date    : 2026/02/19
@Function:
    知识库管理路由
    提供知识库相关的 API 端点：
      POST   /create               - 创建知识库（private/team/public）
      GET    /list                  - 可访问知识库列表（个人+团队+公共 三路 union）
      GET    /children              - 获取子知识库列表
      GET    /{knowledge_base_id}   - 获取知识库详情
      PUT    /{knowledge_base_id}   - 更新知识库（管理权）
      DELETE /{knowledge_base_id}   - 删除知识库（物理删除，需无文件）

    可见性与判权（设计文档：docs/特殊功能设计/团队知识库与公共知识库设计.md）：
    - visibility ∈ {private(默认), team, public}；team 需绑定 team_id
    - 全部端点经 KBAccess 统一判权（require_kb_access），
      私库越权 404（不泄露存在性）、共享库越权 403
    - 公共库创建权受 PUBLIC_KB_ADMIN_ONLY 开关控制
@Modify History:
    2026/10/18 - 引入 visibility/team_id 与 KBAccess 判权；list 改三路 union

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from api.dependencies.auth import get_current_user_id
from api.dependencies.database import get_db_session
from api.schemas.common import ApiResponse
from api.schemas.knowledge.knowledge_base import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseDeleteResponse,
    KnowledgeBaseInfo,
    KnowledgeBaseListPermissions,
    KnowledgeBaseListResponse,
    KnowledgeBaseUpdateRequest,
)
from src.db.mysql.models.business.knowledge_base import KnowledgeBase
from src.db.mysql.models.business.team import Team
from src.db.mysql.repositories.business.knowledge_base_repo import (
    knowledge_base_repo,
)
from src.db.mysql.repositories.business.team_member_repo import team_member_repo
from src.service.knowledge.kb_access import (
    can_create_public_kb,
    require_kb_access,
)

router = APIRouter(tags=["KnowledgeBase"])

_VALID_VISIBILITIES = ("private", "team", "public")


def _team_name_map(
    session: Session, kbs: list[KnowledgeBase]
) -> dict[str, str]:
    """批量取 team_id → team_name（列表展示用，避免 N+1 由前端缓存取名）"""
    team_ids = {kb.team_id for kb in kbs if kb.team_id}
    if not team_ids:
        return {}
    teams = session.query(Team).filter(Team.team_id.in_(team_ids)).all()
    return {t.team_id: t.name for t in teams}


def _to_kb_info(kb: KnowledgeBase, team_names: Optional[dict] = None) -> KnowledgeBaseInfo:
    team_names = team_names or {}
    return KnowledgeBaseInfo(
        knowledge_base_id=kb.knowledge_base_id,
        knowledge_base_name=kb.knowledge_base_name,
        parent_knowledge_base_id=kb.parent_knowledge_base_id,
        knowledge_type=kb.knowledge_type,
        description=kb.description,
        visibility=kb.visibility or "private",
        team_id=kb.team_id,
        team_name=team_names.get(kb.team_id) if kb.team_id else None,
        owner_user_id=kb.user_id,
    )


# ==================== 创建 ====================


@router.post(
    "/create",
    response_model=ApiResponse[KnowledgeBaseInfo],
    summary="创建知识库",
    description=(
        "创建一个新的知识库。同一用户下知识库名称不能重复。"
        "visibility=team 时须传 team_id 且创建者为该团队成员；"
        "visibility=public 的创建权受 PUBLIC_KB_ADMIN_ONLY 开关控制。"
    ),
)
async def create_knowledge_base(
    request: KnowledgeBaseCreateRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[KnowledgeBaseInfo]:
    visibility = request.visibility or "private"
    if visibility not in _VALID_VISIBILITIES:
        raise HTTPException(
            status_code=400,
            detail=f"无效的 visibility: {visibility}，可选 {_VALID_VISIBILITIES}",
        )

    team_id: Optional[str] = None
    if visibility == "team":
        if not request.team_id:
            raise HTTPException(
                status_code=400, detail="团队知识库必须指定 team_id"
            )
        # 创建者须为该团队成员（团队不存在与无权限同口径 404，不泄露存在性）
        role = team_member_repo.get_member_role(
            session, request.team_id, user_id
        )
        if role is None:
            raise HTTPException(status_code=404, detail="团队不存在或无权限")
        team_id = request.team_id
    elif visibility == "public":
        if not can_create_public_kb(user_id):
            raise HTTPException(
                status_code=403, detail="当前环境仅平台管理员可创建公共知识库"
            )

    if knowledge_base_repo.name_exists(
        session, user_id, request.knowledge_base_name
    ):
        raise HTTPException(
            status_code=409,
            detail=f"知识库名称已存在: {request.knowledge_base_name}",
        )

    if request.parent_knowledge_base_id:
        # 父库须对创建者可写（共享库子树由团队成员共同构建）
        require_kb_access(
            session,
            user_id,
            request.parent_knowledge_base_id,
            level="write",
        )

    kb_id = f"kb-{uuid.uuid4()}"
    kb = knowledge_base_repo.create(
        session,
        knowledge_base_id=kb_id,
        user_id=user_id,
        knowledge_base_name=request.knowledge_base_name,
        parent_knowledge_base_id=request.parent_knowledge_base_id,
        knowledge_type=request.knowledge_type,
        description=request.description,
        visibility=visibility,
        team_id=team_id,
        creator=user_id,
    )
    if not kb:
        raise HTTPException(status_code=500, detail="创建知识库失败")

    logger.info(
        f"创建知识库: user_id={user_id}, kb_id={kb_id}, "
        f"name={request.knowledge_base_name}, visibility={visibility}, "
        f"team_id={team_id or '-'}"
    )
    return ApiResponse.success(
        data=_to_kb_info(kb), message="知识库创建成功"
    )


# ==================== 查询 ====================


@router.get(
    "/list",
    response_model=ApiResponse[KnowledgeBaseListResponse],
    summary="获取可访问的知识库列表",
    description=(
        "返回当前用户可读的全量知识库（按创建时间倒序）："
        "自己的 private + 所在团队的 team + 全部 public。"
        "响应附 permissions.can_create_public 供前端显隐创建入口。"
    ),
)
async def list_knowledge_bases(
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[KnowledgeBaseListResponse]:
    team_ids = team_member_repo.list_team_ids_by_user(session, user_id)
    kbs = knowledge_base_repo.get_accessible(session, user_id, team_ids)
    team_names = _team_name_map(session, kbs)
    items = [_to_kb_info(kb, team_names) for kb in kbs]
    return ApiResponse.success(
        data=KnowledgeBaseListResponse(
            knowledge_bases=items,
            total=len(items),
            permissions=KnowledgeBaseListPermissions(
                can_create_public=can_create_public_kb(user_id)
            ),
        )
    )


@router.get(
    "/children",
    response_model=ApiResponse[KnowledgeBaseListResponse],
    summary="获取子知识库列表",
    description=(
        "获取指定父知识库下的直接子知识库。"
        "parent_knowledge_base_id 不传则返回顶级知识库（同 /list 的可访问范围）。"
        "父库须对当前用户可读；子树按 kb 维度返回（含其他成员创建的子库）。"
    ),
)
async def list_children(
    parent_knowledge_base_id: Optional[str] = Query(
        default=None, description="父知识库ID（不传则查询顶级）"
    ),
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[KnowledgeBaseListResponse]:
    if parent_knowledge_base_id is None:
        team_ids = team_member_repo.list_team_ids_by_user(session, user_id)
        kbs = knowledge_base_repo.get_accessible(session, user_id, team_ids)
        kbs = [kb for kb in kbs if kb.parent_knowledge_base_id is None]
    else:
        # 判权：父库可读即可浏览其子树
        require_kb_access(
            session, user_id, parent_knowledge_base_id, level="read"
        )
        kbs = knowledge_base_repo.get_children_by_parent(
            session, parent_knowledge_base_id
        )
    team_names = _team_name_map(session, kbs)
    items = [_to_kb_info(kb, team_names) for kb in kbs]
    return ApiResponse.success(
        data=KnowledgeBaseListResponse(
            knowledge_bases=items,
            total=len(items),
            permissions=KnowledgeBaseListPermissions(
                can_create_public=can_create_public_kb(user_id)
            ),
        )
    )


@router.get(
    "/{knowledge_base_id}",
    response_model=ApiResponse[KnowledgeBaseInfo],
    summary="获取知识库详情",
    description="根据知识库ID获取详细信息（须可读）。",
)
async def get_knowledge_base(
    knowledge_base_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[KnowledgeBaseInfo]:
    kb = require_kb_access(session, user_id, knowledge_base_id, level="read")
    return ApiResponse.success(data=_to_kb_info(kb))


# ==================== 更新 ====================


@router.put(
    "/{knowledge_base_id}",
    response_model=ApiResponse[KnowledgeBaseInfo],
    summary="更新知识库（管理权）",
    description=(
        "更新知识库的名称或描述。需要管理权：private=owner、"
        "team=团队 owner / KB 创建者 / 平台 admin、public=创建者 / 平台 admin。"
        "名称不能与归属用户下的其他知识库重复。"
    ),
)
async def update_knowledge_base(
    knowledge_base_id: str,
    request: KnowledgeBaseUpdateRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[KnowledgeBaseInfo]:
    kb = require_kb_access(
        session, user_id, knowledge_base_id, level="manage"
    )

    if request.knowledge_base_name and request.knowledge_base_name != kb.knowledge_base_name:
        # 重名范围固定为归属用户（共享库改名也不与创建者的其他库重名）
        if knowledge_base_repo.name_exists(
            session, kb.user_id, request.knowledge_base_name,
            exclude_id=knowledge_base_id,
        ):
            raise HTTPException(
                status_code=409,
                detail=f"知识库名称已存在: {request.knowledge_base_name}",
            )

    update_fields: dict = {}
    if request.knowledge_base_name is not None:
        update_fields["knowledge_base_name"] = request.knowledge_base_name
    if request.description is not None:
        update_fields["description"] = request.description

    if not update_fields:
        return ApiResponse.success(data=_to_kb_info(kb))

    updated = knowledge_base_repo.update(
        session, knowledge_base_id, updater=user_id, **update_fields
    )
    if not updated:
        raise HTTPException(status_code=500, detail="更新失败")

    return ApiResponse.success(
        data=_to_kb_info(updated), message="知识库更新成功"
    )


# ==================== 删除 ====================


@router.delete(
    "/{knowledge_base_id}",
    response_model=ApiResponse[KnowledgeBaseDeleteResponse],
    summary="删除知识库（级联删除子知识库，管理权）",
    description=(
        "物理删除知识库及其所有子知识库，同时清理其下的所有空文件夹。"
        "子树按 kb 维度级联（含团队成员创建的子库与文件夹）。"
        "前置条件：该知识库及所有子知识库下不存在任何文件（含正在清理中的）。"
        "如有文件，请先删除所有文件后重试。"
    ),
)
async def delete_knowledge_base(
    knowledge_base_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[KnowledgeBaseDeleteResponse]:
    kb = require_kb_access(
        session, user_id, knowledge_base_id, level="manage"
    )

    descendant_ids = knowledge_base_repo.get_all_descendants_by_kb(
        session, knowledge_base_id
    )
    all_kb_ids = [knowledge_base_id] + descendant_ids

    file_count = knowledge_base_repo.check_kb_tree_has_files(
        session, all_kb_ids
    )
    if file_count < 0:
        raise HTTPException(status_code=500, detail="检查文件数量失败")
    if file_count > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                f"知识库树下仍有 {file_count} 个文件（含正在清理中的），"
                "请先删除所有文件后重试"
            ),
        )

    try:
        total_folder_count = 0
        for kb_id in reversed(all_kb_ids):
            total_folder_count += knowledge_base_repo.hard_delete_by_kb(
                session, kb_id
            )
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"删除知识库失败: {e}")
        raise HTTPException(status_code=500, detail="删除失败")

    logger.info(
        f"删除知识库: operator={user_id}, kb_id={knowledge_base_id}, "
        f"owner={kb.user_id}, name={kb.knowledge_base_name}, "
        f"kb_count={len(all_kb_ids)}, folders={total_folder_count}"
    )

    return ApiResponse.success(
        data=KnowledgeBaseDeleteResponse(
            knowledge_base_id=knowledge_base_id,
            deleted_folder_count=total_folder_count,
            deleted_kb_count=len(all_kb_ids),
        ),
        message=f"已删除 {len(all_kb_ids)} 个知识库，清理 {total_folder_count} 个文件夹",
    )
