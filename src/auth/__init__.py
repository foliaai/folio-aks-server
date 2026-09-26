#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : __init__.py
@Author  : caixiongjiang
@Date    : 2026/09/16
@Function: 
    认证模块
    - config:       认证相关环境变量（AUTH_MODE / JWT / OA 开放平台）
    - jwt_manager:  本域 JWT 签发与校验（OA 模式）
    - oa_client:    企业 OA 开放平台客户端（access_token 缓存 + code 换用户）
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from src.auth.config import (
    get_auth_mode,
    get_jwt_expires_seconds,
    get_jwt_secret,
    get_oa_api_base,
    get_oa_app_key,
    get_oa_app_secret,
    is_oa_auth_enabled,
)
from src.auth.jwt_manager import AuthTokenError, create_access_token, decode_access_token
from src.auth.oa_client import OAApiError, oa_client

__all__ = [
    "get_auth_mode",
    "is_oa_auth_enabled",
    "get_jwt_secret",
    "get_jwt_expires_seconds",
    "get_oa_api_base",
    "get_oa_app_key",
    "get_oa_app_secret",
    "AuthTokenError",
    "create_access_token",
    "decode_access_token",
    "OAApiError",
    "oa_client",
]