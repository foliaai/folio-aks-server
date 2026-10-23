#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : md_parser.py
@Author  : caixiongjiang
@Date    : 2025/12/31 14:30
@Function:
    Markdown 解析器 - 本地 token 遍历，不依赖 MinerU

    策略：markdown-it-py 产出扁平有序 token 流，逐块映射为元素流契约
    （struct_content.root[].page_info[]，与 MinerU content_list 同构）。

    块映射：
    - heading（#~######）   → text + text_level=1..6（section 锚点，真层级）
    - 段落/列表/引用/HTML块 → text（保留 md 原文，不做渲染）
    - GFM 表格              → table（table_body 存管道原文，caption/footnote 留空）
    - fence/缩进代码块      → code_block（保留围栏原文；下游独立成 chunk、免清洗）
    - $$ 独立公式块         → equation（纯 LaTeX，剥 $$ 包裹；下游免清洗链路现成）
    - 独立段落图片          → image（http 外链批量下载转 base64 走 MinIO 现成链路；
                              data: URI 直接解码；下载失败/相对路径一律回退：
                              原文留在 text 元素，链接仍可被模型读到）
    - front-matter / hr     → 剥离 / 丢弃

    页与 bbox：md 无页无坐标，total_pages=1、page_idx=0、bbox 全空，
    引用回溯退化为文本匹配（与 JSON/CSV 内联解析先例一致）。

    parser_source="markdown" 顶层标记：_convert_to_parse_result 据此把 parse_tool
    设为 "markdown"，沿 ParseEndMessage → split → SplitEndMessage 透传，
    section_summary 建树阶段分派 heading_level 策略（MinerU 系仍走编号推断）。
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import base64
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.index.common_file_extract.parser.md_image_fetcher import (
    MarkdownImageFetcher,
    sniff_image,
)

try:
    from markdown_it import MarkdownIt
except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确指引
    raise ImportError(
        "Markdown 解析依赖 markdown-it-py 未安装，请执行: uv add markdown-it-py"
    ) from exc

# YAML front-matter：文档起始的 --- ... --- 块（文档头部元数据，剥离后参与切行）
_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*\r?\n?", re.DOTALL)

# 整段 $$ 公式块：strip 后整体被 $$ 包裹
_DISPLAY_MATH_RE = re.compile(r"\A\$\$([\s\S]+)\$\$\Z")

# data: URI 内嵌图片
_DATA_URI_RE = re.compile(r"\Adata:(image/[a-zA-Z0-9.+-]+);base64,(.+)\Z", re.DOTALL)


class MarkdownParser:
    """
    Markdown 解析器

    功能：
    - 编码降级链（UTF-8 → GBK → UTF-8/replace），坏字节只降级不拒绝
    - markdown-it-py（commonmark + GFM table）token 遍历，按块映射元素流
    - 独立段落图片批量并发下载，失败逐张回退保留原文

    职责：
    - 仅负责单文件解析，不访问数据库与对象存储（图片上传由下游完成）
    - 无 MinerU 依赖，解析结果与 MinerU 链路契约同构
    """

    def __init__(self, image_fetcher: Optional[MarkdownImageFetcher] = None):
        self._image_fetcher = image_fetcher or MarkdownImageFetcher()
        self._md = MarkdownIt("commonmark").enable("table")

    async def parse(self, file_path: Path, file_name: str) -> Dict[str, Any]:
        """
        解析 Markdown 文件

        Args:
            file_path: 文件路径
            file_name: 原始文件名（用于 file_type 推断）

        Returns:
            与 MinerU 链路同构的解析结果 dict
        """
        file_path = Path(file_path)
        src = self._read_text(file_path)

        # 剥离 front-matter（token map 行号基于剥后的 src，自洽）
        src = _FRONT_MATTER_RE.sub("", src, count=1)

        lines = src.split("\n")
        tokens = self._md.parse(src)

        elements: List[Dict[str, Any]] = []
        pending_images: List[Dict[str, Any]] = []

        def emit(element: Dict[str, Any]) -> None:
            element["element_index"] = len(elements)
            element["bbox"] = []
            elements.append(element)

        def slice_map(tok) -> str:
            start, end = tok.map or (0, 0)
            return "\n".join(lines[start:end]).strip("\n")

        i = 0
        total = len(tokens)
        while i < total:
            tok = tokens[i]

            if tok.type == "heading_open":
                # heading_open → inline(标题文本) → heading_close
                inline = tokens[i + 1] if i + 1 < total else None
                title = (inline.content if inline is not None else "") or ""
                emit({
                    "type": "text",
                    "text_level": int(tok.tag[1]),
                    "text": title.strip(),
                })
                i += 3
                continue

            if tok.type in ("fence", "code_block"):
                lang = (tok.info or "").strip() if tok.type == "fence" else ""
                raw = slice_map(tok)
                if raw:
                    emit({"type": "code_block", "text": raw, "code_language": lang})
                i += 1
                continue

            if tok.type in ("ordered_list_open", "bullet_list_open", "blockquote_open"):
                # 列表/引用整块一个 text 元素（保留原文，嵌套结构不展开）
                raw = slice_map(tok)
                if raw:
                    emit({"type": "text", "text": raw})
                i = self._skip_container(tokens, i)
                continue

            if tok.type == "table_open":
                raw = slice_map(tok)
                if raw:
                    emit({
                        "type": "table",
                        "table_body": raw,
                        "table_caption": [],
                        "table_footnote": [],
                    })
                i = self._skip_container(tokens, i)
                continue

            if tok.type == "paragraph_open":
                inline = tokens[i + 1] if i + 1 < total else None
                raw = slice_map(tok)
                math_match = _DISPLAY_MATH_RE.match(raw.strip())

                if math_match and math_match.group(1).strip():
                    emit({
                        "type": "equation",
                        "text": math_match.group(1).strip(),
                        "text_format": "latex",
                    })
                elif inline is not None and self._is_standalone_image(inline):
                    placeholder = {
                        "type": "image",
                        "__alt__": self._image_alt(inline),
                        "__src__": self._image_src(inline),
                        "__fallback_raw__": raw,
                        # element_index / bbox 由落位阶段补齐
                    }
                    placeholder["element_index"] = len(elements)
                    placeholder["bbox"] = []
                    pending_images.append(placeholder)
                    elements.append(placeholder)
                elif raw:
                    emit({"type": "text", "text": raw})
                i += 3
                continue

            if tok.type == "html_block":
                raw = slice_map(tok)
                if raw:
                    emit({"type": "text", "text": raw})
                i += 1
                continue

            # hr / *_close / 孤儿 inline 等：丢弃
            i += 1

        # 待落位图片（占位 dict 原地替换，保序保 element_index）
        await self._resolve_images(pending_images)

        suffix = (Path(file_name).suffix.lstrip(".") or "md").lower()
        return {
            "status": "success",
            "total_pages": 1,
            "parser_source": "markdown",
            "file_type": suffix,
            "content": src,
            "struct_content": {"root": [{"page_idx": 0, "page_info": elements}]},
        }

    # ========== 内部方法 ==========

    def _read_text(self, file_path: Path) -> str:
        """编码降级链：UTF-8 → GBK → UTF-8/replace（坏字节只降级不拒绝）。"""
        data = file_path.read_bytes()
        for encoding in ("utf-8", "gbk"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        logger.warning(f"Markdown 编码降级到 replace: {file_path.name}")
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _skip_container(tokens, start: int) -> int:
        """跳过 open..close 容器体（含嵌套），返回 close 后的索引。"""
        depth = 1
        j = start + 1
        while j < len(tokens):
            t = tokens[j]
            if t.type.endswith("_open"):
                depth += 1
            elif t.type.endswith("_close"):
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return len(tokens)

    @staticmethod
    def _is_standalone_image(inline) -> bool:
        """段落是否只含一张图片（允许空白文本与软换行）。"""
        children = inline.children or []
        if not any(c.type == "image" for c in children):
            return False
        for child in children:
            if child.type in ("image", "softbreak"):
                continue
            if child.type == "text" and not child.content.strip():
                continue
            return False
        return True

    @staticmethod
    def _image_src(inline) -> str:
        for child in inline.children or []:
            if child.type == "image":
                return child.attrs.get("src", "") or ""
        return ""

    @staticmethod
    def _image_alt(inline) -> str:
        parts: List[str] = []
        for child in inline.children or []:
            if child.type == "image":
                for sub in child.children or []:
                    if sub.content:
                        parts.append(sub.content)
        return " ".join(parts).strip()

    async def _resolve_images(self, placeholders: List[Dict[str, Any]]) -> None:
        """占位图片落位：data URI 直接解码；http 外链批量下载；失败回退 text 原文。"""
        if not placeholders:
            return

        remote_urls: List[str] = []
        for ph in placeholders:
            src = ph.get("__src__", "")
            data_match = _DATA_URI_RE.match(src)
            if data_match:
                try:
                    image_bytes = base64.b64decode(data_match.group(2))
                    suffix = sniff_image(image_bytes[:16])
                    if suffix:
                        ph["__data__"] = (image_bytes, suffix)
                except Exception:
                    pass  # 解码失败走回退
            elif src.startswith(("http://", "https://")):
                remote_urls.append(src)

        fetched: Dict[str, Optional[Tuple[bytes, str]]] = {}
        if remote_urls:
            try:
                fetched = await self._image_fetcher.fetch_all(remote_urls)
            except Exception as exc:
                logger.warning(f"md 图片批量下载异常（全部回退保留链接）: {exc}")
                fetched = {url: None for url in remote_urls}

        for ph in placeholders:
            data = ph.pop("__data__", None)
            if data is None:
                data = fetched.get(ph.get("__src__", ""))
            alt = ph.pop("__alt__", "")
            raw = ph.pop("__fallback_raw__", "")
            src = ph.pop("__src__", "")
            index, bbox = ph["element_index"], ph["bbox"]

            ph.clear()
            if data is not None:
                image_bytes, suffix = data
                ph.update({
                    "type": "image",
                    "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                    "img_path": self._image_name(src, suffix),
                    "image_caption": [alt] if alt else [],
                    "image_footnote": [],
                })
            else:
                # 下载失败回退：原文（含链接）留在 text 元素
                ph.update({"type": "text", "text": raw})
            ph["element_index"] = index
            ph["bbox"] = bbox

    @staticmethod
    def _image_name(src: str, suffix: str) -> str:
        """图片对象名：优先 URL basename，退化到 md-image-{suffix}。"""
        name = ""
        if src:
            path = (src.split("?", 1)[0]).rstrip("/")
            name = path.rsplit("/", 1)[-1] if "/" in path else path
        if name and "." in name:
            return name
        return f"md-image.{suffix}"
