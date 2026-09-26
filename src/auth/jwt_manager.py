#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : jwt_manager.py
@Author  : caixiongjiang
@Date    : 2026/09/16
@Function: 
    本域 JWT 签发与校验（OA 登录模式）
    OA 授权码换取的用户信息由后端校验后签发 JWT，前端仅持有该 JWT，
    后端各服务统一从 JWT.sub 取 user_id。
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import jwt
from loguru import logger

from src.auth.config import get_jwt_expires_seconds, get_jwt_secret

ALGORITHM = "HS256"
ISSUER = "aks-auth"


class AuthTokenError(Exception):
    """JWT 校验失败（过期 / 非法 / 缺少配置）"""


def create_access_token(user_id: str, name: Optional[str] = None) -> Tuple[str, int]:
    """
    为用户签发本域 JWT

    Args:
        user_id: 用户标识（OA 模式为工号 EmployeeNo）
        name: 用户姓名（仅用于展示，不参与鉴权）

    Returns:
        (token, expires_in)：token 字符串与有效期（秒）
    """
    if not user_id or not user_id.strip():
        raise ValueError("签发 JWT 需要有效的 user_id")

    expires_in = get_jwt_expires_seconds()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id.strip(),
        "iss": ISSUER,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }
    if name:
        payload["name"] = name

    token = jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)
    return token, expires_in


def decode_access_token(token: str) -> str:
    """
    校验本域 JWT 并返回 user_id

    Args:
        token: JWT 字符串

    Returns:
        JWT.sub，即用户标识

    Raises:
        AuthTokenError: token 缺失、过期或非法
    """
    if not token or not token.strip():
        raise AuthTokenError("缺少有效的登录凭证")

    try:
        payload = jwt.decode(
            token.strip(),
            get_jwt_secret(),
            algorithms=[ALGORITHM],
            issuer=ISSUER,
        )
    except jwt.ExpiredSignatureError as e:
        raise AuthTokenError("登录已过期，请重新登录") from e
    except jwt.InvalidTokenError as e:
        logger.warning(f"JWT 校验失败: {e}")
        raise AuthTokenError("无效的登录凭证") from e
    except RuntimeError as e:
        # get_jwt_secret 未配置
        logger.error(str(e))
        raise AuthTokenError("服务端未正确配置登录密钥") from e

    subject = payload.get("sub")
    if not subject or not str(subject).strip():
        raise AuthTokenError("登录凭证缺少用户标识")

    return str(subject).strip()