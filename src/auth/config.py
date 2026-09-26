#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : config.py
@Author  : caixiongjiang
@Date    : 2026/09/16
@Function: 
    认证相关配置读取
    同一份代码通过 AUTH_MODE 区分两种部署的登录方式：
      - logto: 公网部署，前端对接自建 Logto（后端沿用 X-User-Id 透传）
      - oa:    内部部署，前端对接企业 OA SSO，后端签发自域 JWT
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import Literal
from src.utils.env_manager import get_env_manager

AuthMode = Literal["logto", "oa"]

AUTH_MODE_LOGTO: AuthMode = "logto"
AUTH_MODE_OA: AuthMode = "oa"

# JWT 默认有效期：7 天（与前端 cookie 存储时长对齐）
DEFAULT_JWT_EXPIRES_SECONDS = 7 * 24 * 3600

# OA 开放平台默认地址
DEFAULT_OA_API_BASE = "https://ehrwuji.jiepei.com/oaopenapi"


def get_auth_mode() -> str:
    """获取当前部署的登录模式（默认 logto，保持公网部署行为不变）"""
    return (get_env_manager().get("AUTH_MODE", AUTH_MODE_LOGTO) or "").strip().lower()


def is_oa_auth_enabled() -> bool:
    """是否为 OA 登录模式"""
    return get_auth_mode() == AUTH_MODE_OA


def get_jwt_secret() -> str:
    """
    获取本域 JWT 签名密钥（OA 模式必填）

    优先读 AUTH_JWT_SECRET，未配置时回退到系统已有的 JWT_SECRET_KEY。
    """
    env = get_env_manager()
    secret = (env.get("AUTH_JWT_SECRET", "") or "").strip()
    if not secret:
        secret = (env.get("JWT_SECRET_KEY", "") or "").strip()
    if not secret:
        raise RuntimeError(
            "OA 登录模式下必须配置 AUTH_JWT_SECRET（或复用 JWT_SECRET_KEY）"
        )
    return secret


def get_jwt_expires_seconds() -> int:
    """获取本域 JWT 有效期（秒）"""
    raw = get_env_manager().get("AUTH_JWT_EXPIRES_SECONDS", "")
    if raw and str(raw).strip().isdigit():
        value = int(str(raw).strip())
        if value > 0:
            return value
    return DEFAULT_JWT_EXPIRES_SECONDS


def get_oa_api_base() -> str:
    """获取 OA 开放平台 API 根地址"""
    base = (get_env_manager().get("OA_API_BASE", DEFAULT_OA_API_BASE) or "").strip()
    return (base or DEFAULT_OA_API_BASE).rstrip("/")


def get_oa_app_key() -> str:
    """获取 OA 应用 AppKey（认证中台创建应用后下发）"""
    return (get_env_manager().get("OA_APP_KEY", "") or "").strip()


def get_oa_app_secret() -> str:
    """获取 OA 应用 AppSecret"""
    return (get_env_manager().get("OA_APP_SECRET", "") or "").strip()