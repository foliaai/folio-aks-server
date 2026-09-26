#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : oa_client.py
@Author  : caixiongjiang
@Date    : 2026/09/16
@Function: 
    企业 OA 开放平台客户端
    - access_token：服务端缓存（Redis，降级内存），过期前自动刷新，失败重取
    - 用户信息：授权码 code 换取员工身份（EmployeeNo / EmployeeID / Name 等）
    安全约束：AppSecret 与 access_token 均只存在于服务端，不下发给前端。
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import time
from typing import Any, Dict, Optional, Tuple

import httpx
from loguru import logger

from src.auth.config import get_oa_api_base, get_oa_app_key, get_oa_app_secret

# access_token 缓存 key（每个应用的 token 相互独立）
_TOKEN_CACHE_KEY_TEMPLATE = "auth:oa:accesstoken:{app_key}"
# 提前刷新余量（秒），避免边界过期
_TOKEN_SAFETY_MARGIN = 300
_HTTP_TIMEOUT_SECONDS = 10.0


class OAApiError(Exception):
    """OA 开放平台调用失败"""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class OAClient:
    """企业 OA 开放平台客户端（进程内单例使用）"""

    def __init__(self) -> None:
        # 内存缓存兜底：{app_key: (token, expire_at_timestamp)}
        self._memory_cache: Dict[str, Tuple[str, float]] = {}

    # ==================== 配置 ====================

    def _ensure_configured(self) -> None:
        if not get_oa_app_key() or not get_oa_app_secret():
            raise OAApiError(
                "OA 登录未配置：缺少 OA_APP_KEY / OA_APP_SECRET",
                status_code=503,
            )

    def _cache_key(self) -> str:
        return _TOKEN_CACHE_KEY_TEMPLATE.format(app_key=get_oa_app_key())

    # ==================== access_token 缓存 ====================

    async def _read_cached_token(self) -> Optional[str]:
        """读取缓存的 access_token（Redis 优先，失败降级内存）"""
        key = self._cache_key()

        try:
            from src.db.redis.connection.factory import RedisManagerFactory

            manager = await RedisManagerFactory.get_manager()
            token = await manager.execute("GET", key)
            if token:
                return str(token)
        except Exception as e:  # noqa: BLE001 - Redis 不可用时降级
            logger.warning(f"读取 OA access_token 缓存失败，降级内存缓存: {e}")

        cached = self._memory_cache.get(key)
        if cached and cached[1] > time.time():
            return cached[0]

        return None

    async def _write_cached_token(self, token: str, ttl: int) -> None:
        """写入 access_token 缓存"""
        key = self._cache_key()
        ttl = max(ttl, 60)
        self._memory_cache[key] = (token, time.time() + ttl)

        try:
            from src.db.redis.connection.factory import RedisManagerFactory

            manager = await RedisManagerFactory.get_manager()
            await manager.execute("SET", key, token, ex=ttl)
        except Exception as e:  # noqa: BLE001 - Redis 不可用时降级
            logger.warning(f"写入 OA access_token 缓存失败，仅使用内存缓存: {e}")

    async def _fetch_access_token(self) -> Tuple[str, int]:
        """调用 OA 接口获取新的 access_token"""
        self._ensure_configured()

        url = f"{get_oa_api_base()}/accesstoken/get"
        body = {"AppKey": get_oa_app_key(), "AppSecret": get_oa_app_secret()}

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
                response = await client.post(url, json=body)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as e:
            logger.error(f"获取 OA access_token 请求失败: {e}")
            raise OAApiError("OA 认证服务暂不可用，请稍后重试") from e
        except ValueError as e:
            logger.error(f"解析 OA access_token 响应失败: {e}")
            raise OAApiError("OA 认证服务返回格式异常") from e

        base_resp = self._parse_base_resp(payload)
        token = str(payload.get("AccessToken") or "").strip()

        if base_resp.get("code") != 0 or not token:
            message = base_resp.get("message") or "OA 应用凭证校验失败"
            logger.error(f"获取 OA access_token 失败: code={base_resp.get('code')}, message={message}")
            raise OAApiError(f"OA 应用凭证校验失败: {message}")

        expires_in = int(payload.get("ExpiresIn") or 7200)
        ttl = max(expires_in - _TOKEN_SAFETY_MARGIN, 60)
        logger.info(f"已获取 OA access_token，缓存 {ttl}s")
        return token, ttl

    async def _get_access_token(self, force_refresh: bool = False) -> str:
        """获取 access_token（优先缓存，force_refresh 时强制重取）"""
        if not force_refresh:
            cached = await self._read_cached_token()
            if cached:
                return cached

        token, ttl = await self._fetch_access_token()
        await self._write_cached_token(token, ttl)
        return token

    # ==================== 业务接口 ====================

    async def get_user_by_code(self, code: str) -> Dict[str, Any]:
        """
        授权码 code 换取员工身份信息

        Args:
            code: OA 网页授权回调返回的 code（一次性，最大 512 字节）

        Returns:
            UserInfo 原始字典（EmployeeID / EmployeeNo / Name / AlisName 等）

        Raises:
            OAApiError: code 无效（401）或 OA 服务异常（502）
        """
        if not code or not code.strip():
            raise OAApiError("缺少 OA 授权码", status_code=401)

        # 首次调用；若 OA 提前使 token 失效（BaseResp 非 0），强制刷新后重试一次
        for attempt in range(2):
            token = await self._get_access_token(force_refresh=attempt > 0)
            payload = await self._post_user_get(token, code.strip())
            base_resp = self._parse_base_resp(payload)

            if base_resp.get("code") == 0:
                user_info = payload.get("UserInfo")
                if not isinstance(user_info, dict) or not user_info:
                    raise OAApiError("OA 未返回有效的用户信息", status_code=401)
                return user_info

            message = base_resp.get("message") or "未知错误"
            if attempt == 0:
                logger.warning(f"OA code 换用户失败，刷新 access_token 后重试: {message}")
                continue

            logger.error(
                f"OA code 换用户失败: code={base_resp.get('code')}, message={message}"
            )
            raise OAApiError(f"OA 登录校验失败: {message}", status_code=401)

        raise OAApiError("OA 登录校验失败，请重新登录", status_code=401)

    async def _post_user_get(self, access_token: str, code: str) -> Dict[str, Any]:
        """调用 /auth/user/get 将 code 换取用户信息"""
        url = f"{get_oa_api_base()}/auth/user/get"
        params = {"access_token": access_token}
        body = {"Code": code}

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
                response = await client.post(url, params=params, json=body)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"OA 获取用户信息请求失败: {e}")
            raise OAApiError("OA 用户服务暂不可用，请稍后重试") from e
        except ValueError as e:
            logger.error(f"解析 OA 用户信息响应失败: {e}")
            raise OAApiError("OA 用户服务返回格式异常") from e

    # ==================== 工具方法 ====================

    @staticmethod
    def _parse_base_resp(payload: Dict[str, Any]) -> Dict[str, Any]:
        """归一化 BaseResp（Code/Message）"""
        base_resp = payload.get("BaseResp")
        if not isinstance(base_resp, dict):
            return {"code": 0, "message": ""}

        code = base_resp.get("Code")
        try:
            code = int(code) if code is not None else 0
        except (TypeError, ValueError):
            code = -1

        return {"code": code, "message": str(base_resp.get("Message") or "")}


# 全局单例
oa_client = OAClient()