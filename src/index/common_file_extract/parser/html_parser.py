#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : html_parser.py
@Function:
    HTML 解析器 - BeautifulSoup(lxml) 容错遍历，不依赖 MinerU

    策略：bs4+lxml 解析为 DOM，按块遍历 body 映射为元素流契约
    （struct_content.root[].page_info[]，与 MinerU content_list 同构）。

    块映射：
    - h1~h6                  → text + text_level=1..6（section 锚点，真层级）
    - p / ul / ol / dl       → text（行内标签展平；列表缩进序列化）
    - 块级容器（div/section
      /article/figure 等）    → 递归下钻（布局标签不成元素）
    - table                  → table（table_body 存洗白后的 <table><tr><td>
                               受限 HTML，caption → table_caption；嵌套表格
                               行跳过、单元格内嵌表格拍平为文本）
    - pre                    → code_block（重组 ``` 围栏标准形态，与 md 契约一致，
                               语言从 class 的 language-xxx/lang-xxx 提取；
                               下游独立 chunk 免清洗、缩进保留）
    - img                    → image（data-src > data-original > src 取址；
                               http 外链批量下载走 md 同款 fetcher，
                               data: URI 直接解码，相对路径/失败回退保留
                               <img> 原文；figcaption > alt 填 caption）
    - script math/tex 与
      KaTeX annotation       → equation（display 块独立元素；行内公式替换为
                               $latex$ 文本留在段落流；MathML/mjx 不还原，剔除）
    - hr / 注释 / 处理指令    → 丢弃

    噪音剔除（遍历前 decompose）：script/style/noscript/template/iframe/
    object/embed/svg/math/mjx-container/link/meta、表单控件（input/select/
    textarea/button）、hidden 属性 / aria-hidden / 行内 display:none。
    header/nav/footer/aside 语义壳标签保守保留（只剥确定噪音）。

    编码链：BOM → meta charset 预扫（head 前 2KB）→ UTF-8 → GBK → replace。

    页与 bbox：HTML 无页无坐标，total_pages=1、page_idx=0、bbox 全空，
    引用回溯退化为文本匹配（与 md/JSON/CSV 内联解析先例一致）。

    parser_source="html" 顶层标记：_convert_to_parse_result 据此把 parse_tool
    设为 "html"，section_summary 建树阶段分派 heading_level 策略
    （h1-h6 层级即真值；MinerU 系仍走编号推断）。

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import base64
import html as html_mod
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from bs4 import BeautifulSoup
    from bs4.element import (
        CData,
        Comment,
        Declaration,
        Doctype,
        NavigableString,
        ProcessingInstruction,
        Tag,
    )
except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确指引
    raise ImportError(
        "HTML 解析依赖 beautifulsoup4/lxml 未安装，请执行: uv add beautifulsoup4 lxml"
    ) from exc

from src.index.common_file_extract.parser.md_image_fetcher import (
    MarkdownImageFetcher,
    sniff_image,
)

# data: URI 内嵌图片（单文件导出 HTML 的主力图片形态）
_DATA_URI_RE = re.compile(r"\Adata:(image/[a-zA-Z0-9.+-]+);base64,(.+)\Z", re.DOTALL)

# meta charset 预扫（HTML 文件自身声明编码，GBK/GB2312 中文页常见）
_CHARSET_RE = re.compile(rb"""charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.IGNORECASE)

# 代码块语言：class 里的 language-xxx / lang-xxx
_CODE_LANG_RE = re.compile(r"\b(?:language|lang)-([A-Za-z0-9_+#.-]+)")

# 行内 display:none（不做 CSS 解析，只认字面声明）
_DISPLAY_NONE_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)

# 噪音标签：遍历前整体剔除（math/tex 公式先于 script 剔除前提取）
_NOISE_TAGS = (
    "script", "style", "noscript", "template", "iframe",
    "object", "embed", "svg", "math", "mjx-container",
    "link", "meta", "canvas", "audio", "video",
    "input", "select", "textarea", "button",
)

# 块级容器：不成元素，递归下钻
_CONTAINER_TAGS = {
    "html", "body", "head", "div", "section", "article", "main",
    "header", "footer", "aside", "nav", "blockquote", "details",
    "summary", "dialog", "fieldset", "form", "center", "picture",
    "label",
}

# 行内标签：直接挂块级时并入散落文本（内裹块级结构的除外，见 _has_block_descendant）
_INLINE_TAGS = {
    "a", "b", "i", "em", "strong", "u", "s", "small", "big", "sub", "sup",
    "code", "kbd", "samp", "var", "mark", "abbr", "cite", "q", "time",
    "ruby", "rt", "rp", "wbr", "bdi", "bdo", "ins", "del", "span", "font",
}

# 块级特征标签：行内标签内出现即视为"行内标签裹块级"，退化为递归下钻
_BLOCKISH_TAGS = (
    "p", "div", "table", "ul", "ol", "dl", "pre", "blockquote", "figure",
    "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "img",
)

# KaTeX annotation 里的 latex 源码
_KATEX_TEX_SELECTOR = 'annotation[encoding="application/x-tex"]'

# 公式占位 div 的 class 标记（display 公式替换后由遍历识别）
_EQUATION_MARKER_CLASS = "__html_equation__"


def _squash(text: str) -> str:
    """压平水平空白（空格/制表符折叠为单空格），适用于单行文本与单元格。"""
    return " ".join(text.split())


def _clean_text(text: str) -> str:
    """按行压平水平空白、去空行；保留 <br> 产生的换行。"""
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


class HtmlParser:
    """
    HTML 解析器

    功能：
    - 编码链（BOM → meta charset → UTF-8 → GBK → replace），坏字节只降级不拒绝
    - bs4+lxml 容错解析（真实世界 HTML 大量未闭合/嵌套错乱，lxml 后端可吞）
    - 块遍历映射元素流；独立段落图片批量并发下载，失败逐张回退保留原文

    职责：
    - 仅负责单文件解析，不访问数据库与对象存储（图片上传由下游完成）
    - 无 MinerU 依赖，解析结果与 MinerU 链路契约同构
    """

    def __init__(self, image_fetcher: Optional[MarkdownImageFetcher] = None):
        self._image_fetcher = image_fetcher or MarkdownImageFetcher()

    async def parse(self, file_path: Path, file_name: str) -> Dict[str, Any]:
        """
        解析 HTML 文件

        Args:
            file_path: 文件路径
            file_name: 原始文件名（用于 file_type 推断）

        Returns:
            与 MinerU 链路同构的解析结果 dict
        """
        file_path = Path(file_path)
        src = self._read_text(file_path)

        soup = BeautifulSoup(src, "lxml")
        self._preprocess(soup)

        elements: List[Dict[str, Any]] = []
        pending_images: List[Dict[str, Any]] = []

        def emit(element: Dict[str, Any]) -> None:
            if not element.get("bbox"):
                element["bbox"] = []
            elements.append(element)
            element["element_index"] = len(elements) - 1

        root = soup.body or soup
        self._walk(root, emit, pending_images)

        # 待落位图片（占位 dict 原地替换，保序保 element_index）
        await self._resolve_images(pending_images)

        suffix = (Path(file_name).suffix.lstrip(".") or "html").lower()
        return {
            "status": "success",
            "total_pages": 1,
            "parser_source": "html",
            "file_type": suffix,
            "content": src,
            "struct_content": {"root": [{"page_idx": 0, "page_info": elements}]},
        }

    # ========== 编码 ==========

    def _read_text(self, file_path: Path) -> str:
        """编码链：BOM → meta charset 预扫 → UTF-8 → GBK → replace。"""
        data = file_path.read_bytes()

        # UTF-8 BOM（UTF-16/32 BOM 场景极少，交给 meta/降级链）
        if data.startswith(b"\xef\xbb\xbf"):
            return data[3:].decode("utf-8", errors="replace")

        # meta charset 预扫：声明只可能出现在文件头部，按 latin-1 安全扫前 2KB
        declared = None
        m = _CHARSET_RE.search(data[:2048])
        if m:
            declared = m.group(1).decode("ascii", errors="ignore").lower()
        candidates = []
        if declared and declared not in ("utf-8", "utf8"):
            candidates.append(declared)
        candidates.extend(["utf-8", "gbk", "gb18030"])
        for encoding in candidates:
            try:
                return data.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        logger.warning(f"HTML 编码降级到 replace: {file_path.name}")
        return data.decode("utf-8", errors="replace")

    # ========== 预处理 ==========

    def _preprocess(self, soup: BeautifulSoup) -> None:
        """
        遍历前的整树预处理（顺序敏感）：

        1. 公式提取（必须先于 script 剔除）：
           - script math/tex（MathJax 源码形态）：mode=display 替换为占位 div，
             行内替换为 $latex$ 文本留在段落流
           - KaTeX：katex-display 整块替换为占位 div，行内 katex 替换为 $latex$
           - MathML / mjx-container 不还原 latex，直接剔除（渲染残留是噪音）
        2. <br> → 换行文本（get_text 自然带出，行结构保留）
        3. 噪音剔除：script 系 / 表单控件 / 嵌入对象
        4. 隐藏元素剔除：hidden 属性 / aria-hidden / 行内 display:none
        """
        # --- 1. 公式 ---
        for tag in soup.find_all("script", type=re.compile(r"math/tex", re.I)):
            latex = (tag.string or tag.get_text() or "").strip()
            if not latex:
                tag.decompose()
                continue
            if "mode=display" in (tag.get("type") or "").lower():
                tag.replace_with(self._equation_marker(soup, latex))
            else:
                tag.replace_with(f"${latex}$")

        for display in soup.select(".katex-display"):
            ann = display.select_one(_KATEX_TEX_SELECTOR)
            latex = (ann.get_text() if ann else "").strip()
            if latex:
                display.replace_with(self._equation_marker(soup, latex))
            else:
                display.decompose()  # 无 latex 源码，渲染残留是字符碎片噪音

        for inline in soup.select(".katex"):
            ann = inline.select_one(_KATEX_TEX_SELECTOR)
            latex = (ann.get_text() if ann else "").strip()
            if latex:
                inline.replace_with(f"${latex}$")
            else:
                inline.decompose()

        # --- 2. <br> → 换行 ---
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # --- 3. 噪音标签 ---
        for name in _NOISE_TAGS:
            for tag in soup.find_all(name):
                tag.decompose()

        # --- 4. 隐藏元素（HTML 层面可见的三种声明） ---
        hidden_tags = soup.find_all(
            attrs={"hidden": True}
        ) + soup.find_all(attrs={"aria-hidden": "true"})
        for tag in hidden_tags:
            tag.decompose()
        for tag in soup.find_all(style=True):
            if _DISPLAY_NONE_RE.search(tag.get("style") or ""):
                tag.decompose()

    @staticmethod
    def _equation_marker(soup: BeautifulSoup, latex: str) -> Tag:
        """display 公式占位 div：替换原标签，遍历时映射为 equation 元素。"""
        marker = soup.new_tag("div", attrs={"class": _EQUATION_MARKER_CLASS})
        marker.string = latex
        return marker

    # ========== 块遍历 ==========

    def _walk(self, container: Tag, emit, pending_images: List[Dict[str, Any]]) -> None:
        """
        按文档顺序遍历容器子节点，块级内容映射为元素。

        容器间的散落文本（bad HTML 里不裹 p 的裸文本、行内标签直接挂块级）
        按出现位置聚合成 text 元素；遇到块级子节点先冲刷缓冲。
        """
        loose: List[str] = []

        def flush() -> None:
            if loose:
                text = _clean_text(" ".join(loose))
                if text:
                    emit({"type": "text", "text": text})
                loose.clear()

        for child in list(container.children):
            if isinstance(child, (Comment, Doctype, ProcessingInstruction, CData, Declaration)):
                continue
            if isinstance(child, NavigableString):
                if str(child).strip():
                    loose.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue

            name = (child.name or "").lower()

            # 公式占位（display 公式替换标记）
            if _EQUATION_MARKER_CLASS in (child.get("class") or []):
                flush()
                latex = child.get_text().strip()
                if latex:
                    emit({"type": "equation", "text": latex, "text_format": "latex"})
                continue

            # 标题：真层级锚点
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                flush()
                title = _clean_text(child.get_text())
                if title:
                    emit({"type": "text", "text_level": int(name[1]), "text": title})
                continue

            if name == "p":
                flush()
                self._handle_p(child, emit, pending_images)
                continue

            if name in ("ul", "ol"):
                flush()
                text = self._serialize_list(child)
                if text:
                    emit({"type": "text", "text": text})
                continue

            if name == "dl":
                flush()
                text = self._serialize_dl(child)
                if text:
                    emit({"type": "text", "text": text})
                continue

            if name == "table":
                flush()
                table_body, caption = self._serialize_table(child)
                if table_body:
                    emit({
                        "type": "table",
                        "table_body": table_body,
                        "table_caption": [caption] if caption else [],
                        "table_footnote": [],
                    })
                continue

            if name == "pre":
                flush()
                code = child.get_text().strip("\n")
                if code:
                    lang = self._code_language(child)
                    # text 存 ``` 围栏标准形态（与 md 契约一致）：
                    # 下游 create_code_block_chunk 从 text 重新解析围栏取语言与内容
                    emit({
                        "type": "code_block",
                        "text": f"```{lang}\n{code}\n```",
                        "code_language": lang,
                    })
                continue

            if name == "figure":
                flush()
                self._handle_figure(child, emit, pending_images)
                continue

            if name == "img":
                flush()
                self._emit_image(child, emit, pending_images, caption="")
                continue

            if name == "hr":
                flush()
                continue

            if name in _CONTAINER_TAGS:
                flush()
                self._walk(child, emit, pending_images)
                continue

            # 行内标签直接挂块级（bad HTML）：按散落文本处理；
            # 例外：行内标签里裹了块级结构（编辑器产物），退化为递归下钻
            if name in _INLINE_TAGS:
                if self._has_block_descendant(child):
                    flush()
                    self._walk(child, emit, pending_images)
                else:
                    text = child.get_text()
                    if text.strip():
                        loose.append(text)
                continue

            # 未知标签：保守当下钻（布局框架的自定义标签居多）
            flush()
            self._walk(child, emit, pending_images)

        flush()

    # ========== 块处理 ==========

    def _handle_p(self, p: Tag, emit, pending_images: List[Dict[str, Any]]) -> None:
        """
        段落：行内内容展平为 text；img 分流。

        - 段落只含图片（其余为空白）：每张 img 独立成 image 元素（alt 填 caption）
        - 段落混排文字与图片：先出 text（img 不贡献文本），图片按顺序跟在
          text 后出 image 元素（阅读顺序近似成立，占位符链路不受影响）
        """
        imgs = p.find_all("img")
        text_without_img = _clean_text(self._text_excluding(p, imgs))
        if not imgs:
            if text_without_img:
                emit({"type": "text", "text": text_without_img})
            return

        if text_without_img:
            emit({"type": "text", "text": text_without_img})
        for img in imgs:
            alt = _squash(img.get("alt") or "")
            self._emit_image(img, emit, pending_images, caption=alt)

    def _handle_figure(self, figure: Tag, emit, pending_images: List[Dict[str, Any]]) -> None:
        """
        figure：figcaption 是正文级图注（质量高于 alt）。

        有 img：捕获首个 figcaption 作图注 → 剔除全部 figcaption →
        img 逐张出 image 元素 → 其余子节点（如源码 figure 里混的 pre）
        正常递归。
        无 img：整块正常递归（figcaption 作为普通文本保留）。
        """
        imgs = figure.find_all("img")
        if not imgs:
            self._walk(figure, emit, pending_images)
            return

        caption_tag = figure.find("figcaption")
        caption = _squash(caption_tag.get_text()) if caption_tag else ""
        for fc in figure.find_all("figcaption"):
            fc.decompose()
        for img in imgs:
            self._emit_image(img, emit, pending_images, caption=caption)
            img.decompose()  # 已发射，防止后续 _walk 双重发射
        self._walk(figure, emit, pending_images)

    # ========== 图片 ==========

    def _emit_image(
        self,
        img: Tag,
        emit,
        pending_images: List[Dict[str, Any]],
        caption: str,
    ) -> None:
        """
        图片占位：src 取址优先级 data-src > data-original > src
        （懒加载页面的 src 常是占位图，真实地址在 data-* 属性）。
        解码/下载/回退延后到 _resolve_images 统一批处理。
        """
        src = ""
        for attr in ("data-src", "data-original", "src"):
            value = (img.get(attr) or "").strip()
            if value:
                src = value
                break
        alt = _squash(img.get("alt") or "")
        fallback = self._img_fallback_raw(src, alt)

        placeholder = {
            "type": "image",
            "__src__": src,
            "__caption__": caption or alt,
            "__fallback_raw__": fallback,
            "bbox": [],
        }
        pending_images.append(placeholder)
        emit(placeholder)  # element_index 由 emit 统一编号

    @staticmethod
    def _img_fallback_raw(src: str, alt: str) -> str:
        """回退原文：重建可读的 <img> 标签（链接保留在文本里，模型可读）。"""
        attrs = f'src="{html_mod.escape(src, quote=True)}"'
        if alt:
            attrs += f' alt="{html_mod.escape(alt, quote=True)}"'
        return f"<img {attrs}>"

    async def _resolve_images(self, placeholders: List[Dict[str, Any]]) -> None:
        """占位图片落位：data URI 直接解码；http 外链批量下载；失败回退 img 原文。"""
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
                logger.warning(f"html 图片批量下载异常（全部回退保留链接）: {exc}")
                fetched = {url: None for url in remote_urls}

        for ph in placeholders:
            data = ph.pop("__data__", None)
            if data is None:
                data = fetched.get(ph.get("__src__", ""))
            caption = ph.pop("__caption__", "")
            raw = ph.pop("__fallback_raw__", "")
            src = ph.pop("__src__", "")
            index, bbox = ph.get("element_index"), ph.get("bbox") or []

            ph.clear()
            if data is not None:
                image_bytes, suffix = data
                ph.update({
                    "type": "image",
                    "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                    "img_path": self._image_name(src, suffix),
                    "image_caption": [caption] if caption else [],
                    "image_footnote": [],
                })
            else:
                # 回退：重建的 <img> 原文留在 text 元素（链接可被模型读到）
                ph.update({"type": "text", "text": raw})
            ph["element_index"] = index
            ph["bbox"] = bbox

    @staticmethod
    def _image_name(src: str, suffix: str) -> str:
        """图片对象名：优先 URL basename（去查询串），退化到 html-image-{suffix}。"""
        name = ""
        if src:
            path = (src.split("?", 1)[0]).rstrip("/")
            name = path.rsplit("/", 1)[-1] if "/" in path else path
        if name and "." in name:
            return name
        return f"html-image.{suffix}"

    # ========== 列表 / 表格 / 代码 ==========

    def _serialize_list(self, el: Tag, depth: int = 0) -> str:
        """
        列表序列化：ul 行首 "- "、ol 按序号 "N. "，嵌套列表缩进两空格。
        嵌套列表先从 li 摘出（单次访问，树变异无副作用）再递归序列化。
        """
        ordered = el.name == "ol"
        lines: List[str] = []
        for idx, li in enumerate(el.find_all("li", recursive=False), start=1):
            nested = li.find_all(["ul", "ol"], recursive=False)
            for sub in nested:
                sub.extract()
            text = _squash(li.get_text("\n"))
            prefix = f"{idx}. " if ordered else "- "
            lines.append("  " * depth + prefix + text)
            for sub in nested:
                sub_text = self._serialize_list(sub, depth + 1)
                if sub_text:
                    lines.append(sub_text)
        return "\n".join(lines)

    @staticmethod
    def _serialize_dl(dl: Tag) -> str:
        """定义列表：dt 一行，dd 缩进一行。"""
        lines: List[str] = []
        for child in dl.children:
            if not isinstance(child, Tag):
                continue
            name = (child.name or "").lower()
            if name == "dt":
                text = _squash(child.get_text("\n"))
                if text:
                    lines.append(text)
            elif name == "dd":
                text = _squash(child.get_text("\n"))
                if text:
                    lines.append("  " + text)
        return "\n".join(lines)

    def _serialize_table(self, table: Tag) -> Tuple[str, str]:
        """
        表格序列化为洗白的受限 HTML（MinerU 风格 table_body）。

        - 只放行 table/tr/td/th + rowspan/colspan（数值校验），其余属性全剥
          （table_body 会被前端预览渲染，防属性注入）
        - 嵌套表格的行跳过（只收本表直系行）；单元格内嵌套拍平为文本
        - caption → table_caption
        """
        caption_tag = table.find("caption", recursive=False)
        caption = _squash(caption_tag.get_text()) if caption_tag else ""

        parts: List[str] = ["<table>"]
        for tr in table.find_all("tr"):
            # 跳过嵌套内层表格的行（find_all 全树搜索）
            if tr.find_parent("table") is not table:
                continue
            cells = tr.find_all(["td", "th"], recursive=False)
            if not cells:
                continue
            parts.append("<tr>")
            for cell in cells:
                tag_name = "th" if cell.name == "th" else "td"
                attrs = ""
                for attr in ("rowspan", "colspan"):
                    value = cell.get(attr)
                    if value and str(value).strip().isdigit():
                        attrs += f' {attr}="{int(value)}"'
                text = html_mod.escape(_squash(cell.get_text(" ")))
                parts.append(f"<{tag_name}{attrs}>{text}</{tag_name}>")
            parts.append("</tr>")
        parts.append("</table>")

        body = "".join(parts)
        has_row = "<tr>" in body
        return (body, caption) if has_row else ("", caption)

    @staticmethod
    def _has_block_descendant(tag: Tag) -> bool:
        """行内标签内是否裹了块级结构（决定散落文本 vs 递归下钻）。"""
        return tag.find(_BLOCKISH_TAGS) is not None

    @staticmethod
    def _code_language(pre: Tag) -> str:
        """代码语言：pre 自身或其内 code 标签 class 里的 language-xxx / lang-xxx。"""
        targets = [pre] + pre.find_all("code")
        for tag in targets:
            classes = tag.get("class") or []
            joined = " ".join(classes) if isinstance(classes, list) else str(classes)
            m = _CODE_LANG_RE.search(joined)
            if m:
                return m.group(1).lower()
        return ""

    @staticmethod
    def _text_excluding(el: Tag, excluded: List[Tag]) -> str:
        """取 el 的文本但排除指定子树（段落文字与 img 分流用）。"""
        for tag in excluded:
            tag.extract()  # 单次访问的子树，摘除无副作用
        return el.get_text()
