#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : auth.py
@Author  : caixiongjiang
@Date    : 2026/01/21 10:00
@Function:
    认证依赖模块（纯委托 folio-auth-core 统一验签）

    登录与签发完全由 folio-auth-server 承担（Logto / 企业OA 两种上游，
    由其部署环境决定），AKS 只消费它签发的 RS256 JWT：

      - Authorization: Bearer <folio-auth RS256 JWT>（HTTP 通道）
      - ?token=<JWT>（react-pdf / <img> 等无法带 header 的资源加载）
      - WebSocket 握手 query token / 子协议（见 auth_core.deps）

    本文件不再保留任何兼容层：旧 AKS 自签 HS256 token 与
    X-User-Id 明文透传均已下线，业务路由通过下方四个同名依赖
    零改动使用 auth-core 实现。

    环境变量：
      AUTH_SERVER_URL   folio-auth-server 根地址（默认 http://localhost:8003）

@Modify History:
    2026/02/18 - 实现简化版用户认证（Header 提取 user_id）
    2026/09/16 - 新增 AUTH_MODE=oa 的本域 JWT 校验（HTTP / query / WebSocket 三通道）
    2026/09/26 - 验签切换 folio-auth-core（RS256 为主，HS256/X-User-Id 过渡兼容）
    2026/09/27 - 移除旧认证实现与全部兼容层（旧登录路由 / src/auth 已删除）
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
    """按 AKS 环境变量装配 auth-core 校验器（import 时执行一次）

    只覆盖 JWKS 端点；issuer 用 auth-core 默认值（folio-auth），
    header_passthrough / legacy_secret 均保持默认关闭——本域只认
    folio-auth-server 签发的 RS256 JWT。
    """
    env = get_env_manager()
    server_base = (env.get("AUTH_SERVER_URL") or "").strip().rstrip("/")
    if not server_base:
        server_base = "http://localhost:8003"

    configure_verifier(
        AuthCoreSettings(
            jwks_url=f"{server_base}/api/auth/jwks",
        )
    )


_configure_auth_core()
