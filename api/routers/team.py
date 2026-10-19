#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : team.py
@Author  : caixiongjiang
@Date    : 2026/10/18
@Function:
    团队管理路由（授权客体，挂 /api/team —— team 是共享授权对象而非
    知识概念，将来 agent/技能侧复用时不必挂在 /api/knowledge 下）

    设计文档：docs/特殊功能设计/团队知识库与公共知识库设计.md §5.2.2
      POST   /api/team                        - 创建团队（创建者即 owner）
      GET    /api/team/list                   - 我相关的团队（own + 加入）
      GET    /api/team/{team_id}              - 团队详情 + 成员列表
      PUT    /api/team/{team_id}              - 改名/描述（owner）
      DELETE /api/team/{team_id}              - 解散（owner，有库绑定 409）
      POST   /api/team/{team_id}/members      - 批量加成员（owner）
      DELETE /api/team/{team_id}/members/{uid} - 移除成员（owner，不可移 owner）

    JP 的"从部门导入"不是这里的专属端点：前端先调 auth 的部门成员 API
    拿 user_id 列表，随创建请求的 member_user_ids 一次带入（快照语义）。
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from api.dependencies.auth import get_current_user_id
from api.dependencies.database import get_db_session
from api.schemas.common import ApiResponse
from api.schemas.team import (
    MAX_TEAM_MEMBERS,
    TeamCreateRequest,
    TeamDetailResponse,
    TeamInfo,
    TeamListResponse,
    TeamMemberAddRequest,
    TeamMemberInfo,
    TeamUpdateRequest,
)
from src.db.mysql.models.business.team import Team
from src.db.mysql.repositories.business.knowledge_base_repo import (
    knowledge_base_repo,
)
from src.db.mysql.repositories.business.team_member_repo import team_member_repo
from src.db.mysql.repositories.business.team_repo import team_repo

router = APIRouter(prefix="/api/team", tags=["Team"])


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt else None


def _require_team_role(
    session: Session, team_id: str, user_id: str
) -> tuple[Team, str]:
    """取团队并返回当前用户角色；团队不存在或非成员一律 404（不泄露存在性）"""
    team = team_repo.get_by_id(session, team_id)
    role = (
        team_member_repo.get_member_role(session, team_id, user_id)
        if team
        else None
    )
    if team is None or role is None:
        raise HTTPException(status_code=404, detail="团队不存在或无权限")
    return team, role


# ==================== 创建 ====================


@router.post(
    "",
    response_model=ApiResponse[TeamInfo],
    summary="创建团队",
    description="创建团队，创建者即 owner。可随请求带入初始成员 user_id 列表（JP 从部门导入即走此参数）。",
)
async def create_team(
    request: TeamCreateRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[TeamInfo]:
    # 清洗初始成员：去重、去空、排除创建者本人（owner 档单独写入）
    member_ids = [
        uid for uid in dict.fromkeys(
            u.strip() for u in request.member_user_ids if u and u.strip()
        )
        if uid != user_id
    ]
    if 1 + len(member_ids) > MAX_TEAM_MEMBERS:
        raise HTTPException(
            status_code=400,
            detail=f"团队成员总数（含创建者）不能超过 {MAX_TEAM_MEMBERS}",
        )

    team_id = f"team-{uuid.uuid4()}"
    try:
        team = Team(
            team_id=team_id,
            name=request.name,
            description=request.description,
            creator=user_id,
        )
        session.add(team)
        team_member_repo.add_members(session, team_id, [user_id], role="owner")
        team_member_repo.add_members(session, team_id, member_ids)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"创建团队失败: {e}")
        raise HTTPException(status_code=500, detail="创建团队失败")

    logger.info(
        f"创建团队: team_id={team_id}, name={request.name}, "
        f"owner={user_id}, 初始成员={len(member_ids)}"
    )
    return ApiResponse.success(
        data=TeamInfo(
            team_id=team_id,
            name=team.name,
            description=team.description,
            my_role="owner",
            member_count=1 + len(member_ids),
            create_time=_iso(team.create_time),
        ),
        message="团队创建成功",
    )


# ==================== 查询 ====================


@router.get(
    "/list",
    response_model=ApiResponse[TeamListResponse],
    summary="我的团队列表",
    description="返回当前用户相关（创建或加入）的团队，附我在其中的角色与成员数。",
)
async def list_teams(
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[TeamListResponse]:
    team_ids = team_member_repo.list_team_ids_by_user(session, user_id)
    teams: list[TeamInfo] = []
    for team_id in team_ids:
        team = team_repo.get_by_id(session, team_id)
        if not team:
            continue
        role = team_member_repo.get_member_role(session, team_id, user_id)
        if role is None:
            continue
        teams.append(
            TeamInfo(
                team_id=team.team_id,
                name=team.name,
                description=team.description,
                my_role=role,
                member_count=team_member_repo.count_by_team(session, team_id),
                create_time=_iso(team.create_time),
            )
        )
    return ApiResponse.success(
        data=TeamListResponse(teams=teams, total=len(teams))
    )


@router.get(
    "/{team_id}",
    response_model=ApiResponse[TeamDetailResponse],
    summary="团队详情",
    description="返回团队信息与成员列表（owner 在前）。仅团队成员可见。",
)
async def get_team(
    team_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[TeamDetailResponse]:
    team, my_role = _require_team_role(session, team_id, user_id)
    members = [
        TeamMemberInfo(
            user_id=m.user_id,
            role=m.role,
            joined_time=_iso(m.joined_time),
        )
        for m in team_member_repo.list_by_team(session, team_id)
    ]
    return ApiResponse.success(
        data=TeamDetailResponse(
            team_id=team.team_id,
            name=team.name,
            description=team.description,
            my_role=my_role,
            members=members,
        )
    )


# ==================== 更新 ====================


@router.put(
    "/{team_id}",
    response_model=ApiResponse[TeamInfo],
    summary="更新团队（owner）",
    description="修改团队名称或描述。仅团队 owner 可操作。",
)
async def update_team(
    team_id: str,
    request: TeamUpdateRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[TeamInfo]:
    team, my_role = _require_team_role(session, team_id, user_id)
    if my_role != "owner":
        raise HTTPException(status_code=403, detail="仅团队 owner 可修改团队")

    if request.name is not None:
        team.name = request.name
    if request.description is not None:
        team.description = request.description
    try:
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"更新团队失败: {e}")
        raise HTTPException(status_code=500, detail="更新团队失败")

    return ApiResponse.success(
        data=TeamInfo(
            team_id=team.team_id,
            name=team.name,
            description=team.description,
            my_role=my_role,
            member_count=team_member_repo.count_by_team(session, team_id),
            create_time=_iso(team.create_time),
        ),
        message="团队更新成功",
    )


@router.delete(
    "/{team_id}",
    response_model=ApiResponse,
    summary="解散团队（owner）",
    description=(
        "解散团队并清空成员关系。仍有团队知识库绑定时返回 409，"
        "需先删除或转移这些知识库。"
    ),
)
async def delete_team(
    team_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse:
    team, my_role = _require_team_role(session, team_id, user_id)
    if my_role != "owner":
        raise HTTPException(status_code=403, detail="仅团队 owner 可解散团队")

    bound_kbs = knowledge_base_repo.count_active_by_team(session, team_id)
    if bound_kbs > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                f"仍有 {bound_kbs} 个团队知识库绑定在该团队上，"
                "请先删除或转移后再解散"
            ),
        )

    try:
        team_member_repo.remove_all_by_team(session, team_id)
        team.deleted = 1
        team.updater = user_id
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"解散团队失败: {e}")
        raise HTTPException(status_code=500, detail="解散团队失败")

    logger.info(f"解散团队: team_id={team_id}, operator={user_id}")
    return ApiResponse.success(message="团队已解散")


# ==================== 成员管理 ====================


@router.post(
    "/{team_id}/members",
    response_model=ApiResponse,
    summary="批量添加成员（owner）",
    description="向团队批量添加成员。已存在的 user_id 自动跳过。",
)
async def add_team_members(
    team_id: str,
    request: TeamMemberAddRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse:
    _, my_role = _require_team_role(session, team_id, user_id)
    if my_role != "owner":
        raise HTTPException(status_code=403, detail="仅团队 owner 可管理成员")

    new_ids = [u.strip() for u in request.user_ids if u and u.strip()]
    if not new_ids:
        raise HTTPException(status_code=400, detail="成员列表为空")

    current = team_member_repo.count_by_team(session, team_id)
    existing = {
        m.user_id for m in team_member_repo.list_by_team(session, team_id)
    }
    to_add = [uid for uid in dict.fromkeys(new_ids) if uid not in existing]
    if current + len(to_add) > MAX_TEAM_MEMBERS:
        raise HTTPException(
            status_code=400,
            detail=f"团队成员总数（含 owner）不能超过 {MAX_TEAM_MEMBERS}",
        )

    try:
        added = team_member_repo.add_members(session, team_id, to_add)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"添加团队成员失败: {e}")
        raise HTTPException(status_code=500, detail="添加成员失败")

    logger.info(
        f"添加团队成员: team_id={team_id}, operator={user_id}, "
        f"requested={len(new_ids)}, added={len(added)}"
    )
    return ApiResponse.success(
        message=f"成功添加 {len(added)} 名成员"
        + (f"（跳过 {len(new_ids) - len(added)} 名已存在成员）"
           if len(new_ids) > len(added) else "")
    )


@router.delete(
    "/{team_id}/members/{member_user_id}",
    response_model=ApiResponse,
    summary="移除成员（owner）",
    description="将成员移出团队。owner 不可被移除（团队解散走 DELETE /{team_id}）。",
)
async def remove_team_member(
    team_id: str,
    member_user_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse:
    _, my_role = _require_team_role(session, team_id, user_id)
    if my_role != "owner":
        raise HTTPException(status_code=403, detail="仅团队 owner 可管理成员")

    target_role = team_member_repo.get_member_role(
        session, team_id, member_user_id
    )
    if target_role is None:
        raise HTTPException(status_code=404, detail="该用户不是团队成员")
    if target_role == "owner":
        raise HTTPException(
            status_code=400, detail="不能移除团队 owner（解散请用 DELETE /{team_id}）"
        )

    try:
        team_member_repo.remove_member(session, team_id, member_user_id)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"移除团队成员失败: {e}")
        raise HTTPException(status_code=500, detail="移除成员失败")

    logger.info(
        f"移除团队成员: team_id={team_id}, member={member_user_id}, "
        f"operator={user_id}"
    )
    return ApiResponse.success(message="成员已移除")
