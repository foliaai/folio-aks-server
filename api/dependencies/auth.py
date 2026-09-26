#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : auth.py
@Author  : caixiongjiang
@Date    : 2026/01/21 10:00
@Function: 
    认证依赖模块（已切换到 folio-auth-core 统一验签）

    本文件从"自己实现验签"降级为薄转发层：四个同名依赖全部委托给
    folio-auth-core（RS256 + JWKS），业务路由零改动。

    验签规则（auth-core 统一实现）：
      - Authorization: Bearer <folio-auth RS256 JWT>（folio-auth-server 签发）
      - 过渡兼容①：旧 AKS HS256 token（issuer aks-auth，
        密钥取 AUTH_JWT_SECRET / JWT_SECRET_KEY），存量 token 到期自然淘汰
      - 过渡兼容②：无 Bearer 时透传 X-User-Id / 明文 query user_id
        （AUTH_HEADER_PASSTHROUGH 控制，默认 true；
        前端全面切换 /auth-api 登录后应设为 false 关闭裸透传）

    环境变量：
      AUTH_SERVER_URL         folio-auth-server 根地址（默认 http://localhost:8003）
      AUTH_HEADER_PASSTHROUGH 过渡透传开关（默认 true，切换完成后改 false）
      AUTH_ADMIN_USER_IDS     引导管理员白名单（AKS 自身暂无 admin 接口，预留）

@Modify History:
    2026/02/18 - 实现简化版用户认证（Header 提取 user_id）
    2026/09/16 - 新增 AUTH_MODE=oa 的本域 JWT 校验（HTTP / query / WebSocket 三通道）
    2026/09/26 - 验签切换 folio-auth-core（RS256 为主，HS256/X-User-Id 过渡兼容）
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from auth_core import AuthCoreSettings, configure_verifier
from auth_core.deps import (
    close_unauthorized,
    get_current_user_id,
    get_current_user_id_from_token,
    get_current_user_id_ws,
)
from src.utils.env_manager import get_env_manager

__all__ = [
    "get_current_user_id",
    "get_current_user_id_from_token",
    "get_current_user_id_ws",
    "close_unauthorized",
]


def _configure_auth_core() -> None:
    """按 AKS 环境变量装配 auth-core 校验器（import 时执行一次）"""
    env = get_env_manager()

    server_base = (env.get("AUTH_SERVER_URL", "") or "").strip().rstrip("/")
    if not server_base:
        server_base = "http://localhost:8003"

    # 旧 HS256 密钥：优先 AUTH_JWT_SECRET，回退 JWT_SECRET_KEY；都没有则不校验旧 token
    legacy_secret = (env.get("AUTH_JWT_SECRET", "") or "").strip()
    if not legacy_secret:
        legacy_secret = (env.get("JWT_SECRET_KEY", "") or "").strip()
    legacy_secret = legacy_secret or None

    passthrough_raw = (env.get("AUTH_HEADER_PASSTHROUGH", "") or "").strip().lower()
    header_passthrough = passthrough_raw not in {"false", "0", "no", "off"}

    admin_ids = tuple(
        item.strip()
        for item in (env.get("AUTH_ADMIN_USER_IDS", "") or "").split(",")
        if item.strip()
    )

    configure_verifier(
        AuthCoreSettings(
            jwks_url=f"{server_base}/api/auth/jwks",
            issuer="folio-auth",
            legacy_secret=legacy_secret,
            legacy_issuer="aks-auth",
            header_passthrough=header_passthrough,
            admin_user_ids=admin_ids,
        )
    )


_configure_auth_core()
