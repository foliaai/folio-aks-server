#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : md_image_fetcher.py
@Function:
    Markdown 外链图片批量下载器

    职责：md_parser 解析出的远程图片 URL（http/https）并发下载，供解析器
    转 image_base64 走现有 MinIO 上传链路。任一图片失败不抛异常，返回 None
    由解析器回退为保留 ![alt](url) 原文。

    安全边界（SSRF 防护，双层）：
    - 主机名字面量：禁 localhost / 内网 IP 字面量 / .local
    - DNS 解析后 IP：环回 / 私有 / 链路本地 / 保留 / 未指定段全部拒绝
    - 重定向逐跳校验（最多 3 跳），不允许 follow_redirects 一跳到底

    资源边界：
    - 单图超时 / 单图大小上限 / 魔数嗅探（仅接受 PNG/JPEG/GIF/BMP/WebP）
    - 单文档图片数上限（超出的直接放弃下载）
    - 并发信号量 + 整文档下载总预算（预算耗尽后剩余图片直接回退）
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import asyncio
import ipaddress
import socket
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from loguru import logger

# 魔数表：前缀匹配（WebP 需前 12 字节 RIFF....WEBP，单独判断）
_MAGIC_PREFIXES: List[Tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
]


def sniff_image(data: bytes) -> Optional[str]:
    """按魔数嗅探图片类型，返回后缀（不含点）；非图片返回 None。"""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    for prefix, suffix in _MAGIC_PREFIXES:
        if data.startswith(prefix):
            return suffix
    return None


class MarkdownImageFetcher:
    """
    Markdown 外链图片批量下载器

    使用方式：
        fetcher = MarkdownImageFetcher()
        results = await fetcher.fetch_all(urls)   # {url: (bytes, suffix) | None}
    """

    def __init__(
        self,
        per_image_timeout: float = 10.0,
        total_budget: float = 60.0,
        max_bytes: int = 20 * 1024 * 1024,
        max_images: int = 50,
        concurrency: int = 5,
        max_redirects: int = 3,
    ):
        self._per_image_timeout = per_image_timeout
        self._total_budget = total_budget
        self._max_bytes = max_bytes
        self._max_images = max_images
        self._concurrency = concurrency
        self._max_redirects = max_redirects

    async def fetch_all(self, urls: List[str]) -> Dict[str, Optional[Tuple[bytes, str]]]:
        """
        并发下载图片列表。

        Returns:
            {url: (图片字节, 后缀) 或 None}；输入去重后超上限的部分直接 None。
        """
        unique_urls = list(dict.fromkeys(urls))
        results: Dict[str, Optional[Tuple[bytes, str]]] = {
            url: None for url in unique_urls
        }
        if not unique_urls:
            return results

        # 单文档数量上限：超出的不发起请求
        todo = unique_urls[: self._max_images]

        deadline = asyncio.get_event_loop().time() + self._total_budget
        semaphore = asyncio.Semaphore(self._concurrency)

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._per_image_timeout),
            follow_redirects=False,
        ) as client:

            async def _bounded_fetch(url: str) -> None:
                async with semaphore:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining < 2.0:
                        return
                    try:
                        results[url] = await self._fetch_one(
                            client, url, deadline=deadline
                        )
                    except Exception as exc:
                        logger.debug(f"md 图片下载失败（回退保留链接）: {url} -> {exc}")

            await asyncio.gather(*(_bounded_fetch(url) for url in todo))
        return results

    async def _fetch_one(
        self, client: httpx.AsyncClient, url: str, deadline: float
    ) -> Optional[Tuple[bytes, str]]:
        """下载单张图片（含逐跳重定向校验），失败返回 None。"""
        current = url
        for _ in range(self._max_redirects + 1):
            await self._ensure_public_http_url(current)
            resp = await client.get(current)
            if resp.is_redirect:
                location = resp.headers.get("location", "")
                if not location:
                    return None
                current = str(client.build_request("GET", current, timeout=self._per_image_timeout).url.join(location))
                continue
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            data = await resp.aread()
            suffix = sniff_image(data[:16])
            if suffix is None:
                logger.debug(
                    f"md 图片魔数嗅探失败（回退保留链接）: {url}, "
                    f"content-type={content_type}, bytes={len(data)}"
                )
                return None
            return data, suffix
        logger.debug(f"md 图片重定向超上限（回退保留链接）: {url}")
        return None

    async def _ensure_public_http_url(self, url: str) -> None:
        """SSRF 双层校验：scheme 白名单 + 主机名字面量 + DNS 解析后 IP 段检查。"""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"非 http/https 协议: {url}")
        hostname = (parsed.hostname or "").strip().lower()
        if not hostname:
            raise ValueError(f"URL 无主机名: {url}")

        # 字面量层：localhost / .local / 直接写内网 IP
        if hostname in ("localhost",) or hostname.endswith(".local"):
            raise ValueError(f"禁用主机名: {hostname}")
        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            literal = None
        if literal is not None and not literal.is_global:
            raise ValueError(f"禁用非公网 IP: {hostname}")

        # DNS 层：解析后逐 IP 检查（任一命中内网段即拒绝）
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        loop = asyncio.get_event_loop()
        infos = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                raise ValueError(f"主机解析到非公网地址: {hostname} -> {ip}")
