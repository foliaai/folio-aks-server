#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : oa.py
@Author  : caixiongjiang
@Date    : 2026/09/16
@Function: 
    OA 授权码登录路由（内部部署）
    端点：POST /api/auth/oa/login
    前端拿到 OA 回调 code 后调用本接口，由服务端完成：
      1. access_token 获取与缓存（Redis）
      2. code 换取员工身份（工号 / 姓名）
      3. 首次登录写入 user_profile（user_id = 工号）
      4. 签发本域 JWT 返回前端
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.orm import Session

from api.dependencies.database import get_db_session
from api.schemas.auth.oa import OALoginData, OALoginRequest, OAUserProfile
from api.schemas.common import ApiResponse
from src.auth import OAApiError, create_access_token, is_oa_auth_enabled, oa_client
from src.db.mysql.repositories.user.user_profile_repo import user_profile_repo

router = APIRouter(tags=["Auth"])

FRONTEND_REDIRECT_HINT = "请返回登录页重新发起登录"


def _extract_identity(user_info: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
    """
    从 OA UserInfo 中提取本系统用户标识与展示信息

    Returns:
        (user_id, name, extra)：user_id 优先取工号 EmployeeNo，缺失时降级为 EmployeeID
    """
    employee_no = str(user_info.get("EmployeeNo") or "").strip()
    employee_id = user_info.get("EmployeeID")
    name = str(user_info.get("Name") or "").strip()
    alias_name = str(user_info.get("AlisName") or "").strip()

    user_id = employee_no
    if not user_id and employee_id is not None:
        user_id = str(employee_id).strip()

    if not user_id:
        raise HTTPException(status_code=502, detail="OA 未返回有效的用户标识")

    extra = {
        "employee_no": employee_no or None,
        "employee_id": int(employee_id) if str(employee_id or "").isdigit() else None,
        "name": name or None,
        "alias_name": alias_name or None,
    }
    return user_id, name or alias_name, extra


@router.post(
    "/oa/login",
    response_model=ApiResponse[OALoginData],
    summary="OA 授权码登录",
    description=(
        "内部部署登录入口。接收 OA 网页授权回调返回的 code，"
        "服务端换取员工身份后签发本域 JWT。"
        "AppSecret 与 OA access_token 均不会下发前端。"
    ),
)
async def oa_login(
    body: OALoginRequest,
    session: Session = Depends(get_db_session),
) -> ApiResponse[OALoginData]:
    if not is_oa_auth_enabled():
        raise HTTPException(status_code=403, detail="当前部署未启用 OA 登录")

    code = (body.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail=f"缺少 OA 授权码，{FRONTEND_REDIRECT_HINT}")

    try:
        user_info = await oa_client.get_user_by_code(code)
    except OAApiError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:  # noqa: BLE001 - 兜底避免登录链路异常泄漏
        logger.exception(f"OA 登录异常: {e}")
        raise HTTPException(status_code=500, detail="OA 登录服务异常") from e

    user_id, display_name, extra = _extract_identity(user_info)

    # 首次登录建档；已存在的记录补全昵称，但尊重用户后续自定义
    profile = user_profile_repo.get_or_create(session, user_id, nickname=display_name or None)
    if display_name and profile is not None and not profile.nickname:
        user_profile_repo.update_profile(session, user_id=user_id, nickname=display_name)

    token, expires_in = create_access_token(user_id, display_name or None)

    logger.info(f"OA 登录成功: user_id={user_id}, name={display_name or '-'}")

    return ApiResponse.success(
        data=OALoginData(
            access_token=token,
            token_type="bearer",
            expires_in=expires_in,
            user=OAUserProfile(
                user_id=user_id,
                employee_no=extra["employee_no"],
                employee_id=extra["employee_id"],
                name=extra["name"],
                alias_name=extra["alias_name"],
            ),
        ),
        message="登录成功",
    )