#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : excel_parser.py
@Author  : caixiongjiang
@Date    : 2026/10/27
@Function:
    Excel 解析器（.xlsx / .xls）→ 统一元素流

    结构约定：
    - 工作表 = Section（sheet 名发 text_level=1 标题元素，heading_level 建树）
    - 工作表内识别若干 TableRegion，每个区域一个 table 元素（HTML table_body）
    - 区域识别：原生表对象(ws.tables)优先 → 非空格子 4 邻接连通块 →
      垂直相邻同列范围且无新表特征时合并（表内空行不拆表）
    - 标题/备注行归属邻近区域（caption/footnote），不产生独立 text 元素
    - 图片从 zip 层直读（xl/media + drawing 锚点），按锚点归属区域，
      caption 注明 工作表｜区域｜单元格
    - page_idx = 工作表序号（仅计非空工作表），total_pages = 工作表数

    .xls 走 LibreOffice headless 转 .xlsx 后进同一条链路
    （转换模式同 OfficeToPdfParser：独立 profile 防并发锁、超时控制）。
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import asyncio
import base64
import html
import posixpath
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from loguru import logger

# 防御 openpyxl 虚胖 dimension（格式污染的空行列也算），超限截断并告警
_MAX_ROWS = 100_000
_MAX_COLS = 512
# 原生表/连通块之间的空隙 ≤ 该行数视为"紧邻"，参与归属与合并判定
_ADJACENT_GAP = 2
# zip 层图片支持的后缀（emf/wmf 为 Windows 矢量格式，跳过）
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}

_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_NS_XDR = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
_NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"


@dataclass
class _Region:
    """工作表内的一个表格区域（1-based 闭区间）"""

    min_row: int
    min_col: int
    max_row: int
    max_col: int
    caption: str = ""
    footnote: str = ""
    native: bool = False  # Excel 原生表对象（边界明确，不参与合并校正）
    cells: int = 0  # 区域内非空格子数（note 判定用）
    # 表体序列化范围（首行全宽合并标题可被提为 caption、尾行提为 footnote）
    body_min_row: Optional[int] = None
    body_max_row: Optional[int] = None

    def body_rows(self) -> range:
        top = self.body_min_row if self.body_min_row is not None else self.min_row
        bottom = self.body_max_row if self.body_max_row is not None else self.max_row
        return range(top, bottom + 1)

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1

    def contains(self, row0: int, col0: int) -> bool:
        """0-based 锚点是否落在区域内"""
        return (
            self.min_row - 1 <= row0 <= self.max_row - 1
            and self.min_col - 1 <= col0 <= self.max_col - 1
        )


@dataclass
class _SheetImage:
    """zip 层解析出的一张内嵌图片（锚点 0-based）"""

    row: int
    col: int
    name: str
    data: bytes = field(repr=False)


class ExcelParser:
    """Excel → 统一元素流解析器"""

    def __init__(self, soffice_path: str = "soffice", convert_timeout: int = 120):
        self.soffice_path = soffice_path
        self.convert_timeout = convert_timeout

    # ========== 入口 ==========

    async def parse(self, file_path: Path, file_name: str) -> Dict:
        from openpyxl import load_workbook
        from openpyxl.utils import range_boundaries

        suffix = file_path.suffix.lower()
        xls_temp_dir: Optional[Path] = None
        try:
            if suffix == ".xls":
                xlsx_path = await self._convert_xls_to_xlsx(file_path)
                xls_temp_dir = xlsx_path.parent
            else:
                xlsx_path = file_path

            # data_only 取计算值；公式缓存缺失时懒加载公式版回退
            wb_data = load_workbook(xlsx_path, data_only=True)
            wb_formula = None

            def formula_ws(title: str):
                nonlocal wb_formula
                if wb_formula is None:
                    wb_formula = load_workbook(xlsx_path, data_only=False)
                return wb_formula[title]

            images_by_sheet = self._extract_images(xlsx_path)

            pages: List[Dict] = []
            full_text_parts: List[str] = []

            for ws in wb_data.worksheets:
                if ws.sheet_state != "visible":
                    continue
                grid, merges = self._read_sheet(ws, formula_ws)
                sheet_images = images_by_sheet.get(ws.title, [])
                if not grid and not sheet_images:
                    continue  # 全空工作表：不发标题、不占页码

                # 原生表对象：边界明确、表名即 caption，格子不再进连通块
                native_regions = []
                for table in getattr(ws, "tables", {}).values():
                    try:
                        c1, r1, c2, r2 = range_boundaries(table.ref)
                        native_regions.append(
                            _Region(r1, c1, r2, c2, caption=table.displayName, native=True)
                        )
                    except Exception:
                        logger.warning(f"原生表范围解析失败: {ws.title}.{table.displayName}")

                page_info = self._build_page(ws.title, grid, merges, sheet_images, native_regions)
                pages.append({"page_idx": len(pages), "page_info": page_info})
                full_text_parts.append(ws.title)
                full_text_parts.extend(
                    cell for row in grid for cell in row if cell
                )

            return {
                "status": "success",
                "total_pages": len(pages),
                "parser_source": "excel",
                "file_type": "xlsx" if suffix != ".xls" else "xls",
                "content": "\n".join(full_text_parts),
                "struct_content": {"root": pages},
            }
        finally:
            if xls_temp_dir is not None:
                shutil.rmtree(xls_temp_dir, ignore_errors=True)

    # ========== 工作表读取 ==========

    def _read_sheet(self, ws, formula_ws) -> Tuple[List[List[str]], List[Tuple[int, int, int, int]]]:
        """
        读成字符串矩阵（裁尾部全空行列，cap 保护）。

        Args:
            ws: data_only 工作表
            formula_ws: 懒加载闭包，返回公式版工作表（值缺失时回退公式文本）

        Returns:
            (grid, merges)：grid 0-based 矩阵；merges 为 (min_row, min_col,
            max_row, max_col) 1-based 合并区列表
        """
        max_row = min(ws.max_row or 0, _MAX_ROWS)
        max_col = min(ws.max_column or 0, _MAX_COLS)
        if (ws.max_row or 0) > _MAX_ROWS or (ws.max_column or 0) > _MAX_COLS:
            logger.warning(
                f"工作表 {ws.title} 超出解析上限（{ws.max_row}x{ws.max_column}），"
                f"截断为 {max_row}x{max_col}"
            )

        fw = None  # 公式版工作表，首个缺失值时加载

        def cell_str(r: int, c: int) -> str:
            nonlocal fw
            v = ws.cell(row=r, column=c).value
            if v is None:
                if fw is None:
                    fw = formula_ws(ws.title)
                fv = fw.cell(row=r, column=c).value
                if isinstance(fv, str) and fv.startswith("="):
                    return fv
                return ""
            return self._value_str(v)

        grid = [
            [cell_str(r, c) for c in range(1, max_col + 1)]
            for r in range(1, max_row + 1)
        ]

        while grid and not any(grid[-1]):
            grid.pop()
        width = max((len(r) for r in grid), default=0)
        grid = [r + [""] * (width - len(r)) for r in grid]

        merges = [
            (rng.min_row, rng.min_col, rng.max_row, rng.max_col)
            for rng in ws.merged_cells.ranges
        ]
        return grid, merges

    @staticmethod
    def _value_str(v) -> str:
        if hasattr(v, "isoformat"):  # datetime / date / time
            return v.isoformat()
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    # ========== 区域识别 ==========

    def _build_page(
        self,
        sheet_name: str,
        grid: List[List[str]],
        merges: List[Tuple[int, int, int, int]],
        images: List[_SheetImage],
        native_regions: List[_Region],
    ) -> List[Dict]:
        regions = self._detect_regions(grid, merges, native_regions)
        return self._emit_elements(sheet_name, regions, grid, merges, images)

    def _detect_regions(
        self,
        grid: List[List[str]],
        merges: List[Tuple[int, int, int, int]],
        native_regions: List[_Region],
    ) -> List[_Region]:
        """原生表优先 + 连通块 + note 归属 + 垂直合并校正"""
        rows = len(grid)
        cols = len(grid[0]) if grid else 0

        # 1) 原生表范围占位，格子不再进连通块
        occupied = [[False] * cols for _ in range(rows)]
        for reg in native_regions:
            for r in range(reg.min_row - 1, min(reg.max_row, rows)):
                for c in range(reg.min_col - 1, min(reg.max_col, cols)):
                    occupied[r][c] = True

        # 2) 剩余非空格子 4 邻接连通块
        blocks: List[_Region] = []
        visited = [[False] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if visited[r][c] or occupied[r][c] or not grid[r][c]:
                    continue
                stack = [(r, c)]
                visited[r][c] = True
                min_r = max_r = r
                min_c = max_c = c
                cells = 0
                while stack:
                    cr, cc = stack.pop()
                    cells += 1
                    min_r, max_r = min(min_r, cr), max(max_r, cr)
                    min_c, max_c = min(min_c, cc), max(max_c, cc)
                    for nr, nc in ((cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)):
                        if (
                            0 <= nr < rows and 0 <= nc < cols
                            and not visited[nr][nc]
                            and not occupied[nr][nc]
                            and grid[nr][nc]
                        ):
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                blocks.append(_Region(min_r+1, min_c+1, max_r+1, max_c+1, cells=cells))

        # 3) note 归属：仅"单行且只有单格有值"的块（合并标题/备注形态）才可归属；
        #    多格有值的单行是数据行，按独立表块参与后续合并校正
        def is_note(b: _Region) -> bool:
            return b.height == 1 and b.cells <= 1

        tables: List[_Region] = [b for b in blocks if not is_note(b)] + native_regions
        for b in blocks:
            if not is_note(b):
                continue
            text = " ".join(x for x in grid[b.min_row-1] if x)
            target, kind = self._attach_note(b, tables)
            if target is None:
                tables.append(b)  # 孤立文本行也按单行表保留，不丢内容
            elif kind == "caption" and not target.caption:
                target.caption = text
            elif kind == "footnote" and not target.footnote:
                target.footnote = text
            else:
                tables.append(b)  # 目标位置已被占用：保留为独立单行表

        # 4) 垂直合并校正：最多隔 1 空行、同列范围、下方无 caption、上方无
        #    footnote、首行非重复表头 → 判定为同表内空行，合并；
        #    （隔 ≥2 空行视为分表信号，不做合并）
        tables.sort(key=lambda t: (t.min_row, t.min_col))
        merged: List[_Region] = []
        for t in tables:
            prev = merged[-1] if merged else None
            if (
                prev is not None
                and not t.native
                and not prev.native
                and not t.caption
                and not prev.footnote
                and t.min_row - prev.max_row <= 2
                and t.min_col == prev.min_col
                and t.max_col == prev.max_col
                and self._first_row(grid, t) != self._first_row(grid, prev)
            ):
                prev.max_row = max(prev.max_row, t.max_row)
                prev.max_col = max(prev.max_col, t.max_col)
            else:
                merged.append(t)

        # 5) 紧贴式标题/备注提取：合并标题与表头间无空行时同属一个连通块，
        #    首行/尾行若是横跨全宽的合并单元格则提出为 caption/footnote
        for t in merged:
            self._extract_fullwidth_notes(t, grid, merges)
        return merged

    @staticmethod
    def _extract_fullwidth_notes(region: _Region, grid: List[List[str]], merges) -> None:
        if region.height < 2:
            return

        def fullwidth_row(row: int) -> bool:
            for (r0, c0, r1, c1) in merges:
                if r0 == r1 == row and c0 <= region.min_col and c1 >= region.max_col:
                    return True
            return False

        if not region.caption and fullwidth_row(region.min_row):
            region.caption = grid[region.min_row - 1][region.min_col - 1]
            region.body_min_row = region.min_row + 1
        if not region.footnote and fullwidth_row(region.max_row) and region.max_row > (region.body_min_row or region.min_row):
            region.footnote = grid[region.max_row - 1][region.min_col - 1]
            region.body_max_row = region.max_row - 1

    @staticmethod
    def _attach_note(note: _Region, tables: List[_Region]) -> Tuple[Optional[_Region], str]:
        """
        单行 note 找归属目标。

        Returns:
            (目标区域, "caption"|"footnote")；无归属返回 (None, "")
            caption：note 在目标上方（标题在表上方）
            footnote：note 在目标下方（备注在表下方）
        """
        best, best_kind, best_gap = None, "", _ADJACENT_GAP + 1
        for t in tables:
            col_overlap = not (note.max_col < t.min_col or note.min_col > t.max_col)
            if not col_overlap:
                continue
            for kind, gap in (
                ("caption", t.min_row - note.max_row),
                ("footnote", note.min_row - t.max_row),
            ):
                if 0 < gap <= _ADJACENT_GAP and gap < best_gap:
                    best, best_kind, best_gap = t, kind, gap
        return best, best_kind

    @staticmethod
    def _first_row(grid: List[List[str]], region: _Region) -> Tuple[str, ...]:
        return tuple(grid[region.min_row - 1][region.min_col - 1: region.max_col])

    # ========== 元素生成 ==========

    def _emit_elements(
        self,
        sheet_name: str,
        regions: List[_Region],
        grid: List[List[str]],
        merges: List[Tuple[int, int, int, int]],
        images: List[_SheetImage],
    ) -> List[Dict]:
        elements: List[Dict] = [
            {
                "type": "text",
                "text": sheet_name,
                "text_level": 1,
                "bbox": [],
                "element_index": 0,
            }
        ]

        items: List[Tuple[int, int, str, object]] = [
            (r.min_row, r.min_col, "table", r) for r in regions
        ]
        items += [(im.row + 1, im.col + 1, "image", im) for im in images]
        items.sort(key=lambda x: (x[0], x[1]))

        for idx, (_, _, kind, obj) in enumerate(items, start=1):
            if kind == "table":
                elements.append(self._table_element(obj, grid, merges, idx))
            else:
                elements.append(self._image_element(sheet_name, obj, regions, idx))
        return elements

    def _table_element(
        self,
        region: _Region,
        grid: List[List[str]],
        merges: List[Tuple[int, int, int, int]],
        element_index: int,
    ) -> Dict:
        trs = []
        # 仅使用完全落在区域内的合并区（跨界视为普通格子）
        inner = [
            m for m in merges
            if region.min_row <= m[0] and m[2] <= region.max_row
            and region.min_col <= m[1] and m[3] <= region.max_col
        ]
        for r in region.body_rows():
            tds = []
            for c in range(region.min_col, region.max_col + 1):
                attrs = ""
                skipped = False
                for (r0, c0, r1, c1) in inner:
                    if (r, c) == (r0, c0):
                        if r1 - r0 + 1 > 1:
                            attrs += f' rowspan="{r1 - r0 + 1}"'
                        if c1 - c0 + 1 > 1:
                            attrs += f' colspan="{c1 - c0 + 1}"'
                        break
                    if r0 <= r <= r1 and c0 <= c <= c1:
                        skipped = True  # 被合并吞掉的格子：不输出
                        break
                if not skipped:
                    tds.append(f"<td{attrs}>{html.escape(grid[r-1][c-1])}</td>")
            trs.append("<tr>" + "".join(tds) + "</tr>")

        return {
            "type": "table",
            "table_body": "<table>" + "".join(trs) + "</table>",
            "table_caption": [region.caption] if region.caption else [],
            "table_footnote": [region.footnote] if region.footnote else [],
            "bbox": [],
            "element_index": element_index,
        }

    def _image_element(
        self,
        sheet_name: str,
        image: _SheetImage,
        regions: List[_Region],
        element_index: int,
    ) -> Dict:
        owner = next((r for r in regions if r.contains(image.row, image.col)), None)
        where = owner.caption if owner and owner.caption else sheet_name
        return {
            "type": "image",
            "image_base64": base64.b64encode(image.data).decode("ascii"),
            "img_path": image.name,
            "image_caption": [f"{where}｜{self._cell_ref(image.row, image.col)} 插图"],
            "bbox": [],
            "element_index": element_index,
        }

    @staticmethod
    def _cell_ref(row0: int, col0: int) -> str:
        letters, col = "", col0 + 1
        while col:
            col, rem = divmod(col - 1, 26)
            letters = chr(65 + rem) + letters
        return f"{letters}{row0 + 1}"

    # ========== zip 层图片提取 ==========

    def _extract_images(self, xlsx_path: Path) -> Dict[str, List[_SheetImage]]:
        """sheet 名 → 锚点图片列表（workbook → sheet rels → drawing → media）"""
        result: Dict[str, List[_SheetImage]] = {}
        try:
            zf = zipfile.ZipFile(xlsx_path)
        except Exception as e:
            logger.warning(f"xlsx 图片提取失败（按无图处理）: {e}")
            return result

        try:
            wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
            wb_rels = self._load_rels(zf, "xl/_rels/workbook.xml.rels")

            for sheet_el in wb_root.iter(f"{{{_NS_MAIN}}}sheet"):
                sheet_name = sheet_el.get("name")
                rid = sheet_el.get(f"{{{_NS_R}}}id")
                sheet_target = wb_rels.get(rid)
                if not sheet_name or not sheet_target:
                    continue
                sheet_part = self._normalize("xl", sheet_target)
                drawing = self._sheet_drawing(zf, sheet_part)
                if not drawing:
                    continue
                for row, col, media in self._drawing_images(zf, drawing):
                    suffix = Path(media).suffix.lower()
                    if suffix not in _IMAGE_SUFFIXES:
                        logger.info(f"跳过不支持的嵌入图片格式: {media}")
                        continue
                    result.setdefault(sheet_name, []).append(
                        _SheetImage(row=row, col=col, name=Path(media).name, data=zf.read(media))
                    )
        except Exception as e:
            logger.warning(f"xlsx 图片关系解析失败（按无图处理）: {e}")
        finally:
            zf.close()
        return result

    @staticmethod
    def _load_rels(zf: zipfile.ZipFile, rels_part: str) -> Dict[str, str]:
        """rels XML → {rId: target}"""
        try:
            root = ET.fromstring(zf.read(rels_part))
        except KeyError:
            return {}
        return {
            el.get("Id"): el.get("Target")
            for el in root.iter(f"{{{_NS_PKG_REL}}}Relationship")
            if el.get("Id") and el.get("Target")
        }

    def _sheet_drawing(self, zf: zipfile.ZipFile, sheet_part: str) -> Optional[str]:
        """sheetN.xml 的 rels → drawing part 名（无则 None）"""
        base = posixpath.dirname(sheet_part)
        rels_part = posixpath.join(
            base, "_rels", posixpath.basename(sheet_part) + ".rels"
        )
        for _, target in self._load_rels(zf, rels_part).items():
            # Relationship Type 以 /drawing 结尾的即 drawing part
            if target and "drawings/" in target:
                return self._normalize(base, target)
        return None

    def _drawing_images(self, zf: zipfile.ZipFile, drawing_part: str) -> List[Tuple[int, int, str]]:
        """drawingN.xml → [(from_row, from_col, media_part)]"""
        base = posixpath.dirname(drawing_part)
        drawing_rels = self._load_rels(
            zf, posixpath.join(base, "_rels", posixpath.basename(drawing_part) + ".rels")
        )
        out = []
        root = ET.fromstring(zf.read(drawing_part))
        for anchor in root:
            if anchor.tag not in (f"{{{_NS_XDR}}}oneCellAnchor", f"{{{_NS_XDR}}}twoCellAnchor"):
                continue
            frm = anchor.find(f"{{{_NS_XDR}}}from")
            blip = anchor.find(f".//{{{_NS_A}}}blip")
            if frm is None or blip is None:
                continue
            col_el = frm.find(f"{{{_NS_XDR}}}col")
            row_el = frm.find(f"{{{_NS_XDR}}}row")
            rid = blip.get(f"{{{_NS_R}}}embed")
            if col_el is None or row_el is None or not rid:
                continue
            target = drawing_rels.get(rid)
            if not target:
                continue
            out.append((int(row_el.text), int(col_el.text), self._normalize(base, target)))
        return out

    @staticmethod
    def _normalize(base: str, target: str) -> str:
        """rels target（相对/绝对）→ zip 内 part 名"""
        if target.startswith("/"):
            return target.lstrip("/")
        return posixpath.normpath(posixpath.join(base, target))

    # ========== .xls 转换 ==========

    async def _convert_xls_to_xlsx(self, file_path: Path) -> Path:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._convert_xls_sync, file_path)

    def _convert_xls_sync(self, file_path: Path) -> Path:
        """LibreOffice headless 转 xlsx（独立 profile 防并发锁，同 base_parser 模式）"""
        import subprocess

        out_dir = Path(tempfile.mkdtemp(prefix="excel_convert_"))
        profile_dir = Path(tempfile.mkdtemp(prefix="lo_profile_"))
        try:
            cmd = [
                self.soffice_path,
                "--headless",
                "--nologo",
                "--nofirststartwizard",
                f"-env:UserInstallation=file://{profile_dir}",
                "--convert-to", "xlsx",
                "--outdir", str(out_dir),
                str(file_path),
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=self.convert_timeout, check=False,
                )
            except subprocess.TimeoutExpired:
                raise Exception(
                    f"LibreOffice 转换超时（{self.convert_timeout}s）: {file_path.name}"
                )
            if proc.returncode != 0:
                raise Exception(
                    f"LibreOffice 转换失败（returncode={proc.returncode}）: "
                    f"{(proc.stderr or '').strip()}"
                )
            xlsx_path = out_dir / (file_path.stem + ".xlsx")
            if not xlsx_path.exists():
                candidates = list(out_dir.glob("*.xlsx"))
                if not candidates:
                    raise Exception(f"LibreOffice 转换未生成 xlsx 产物: {file_path.name}")
                xlsx_path = candidates[0]
            return xlsx_path
        finally:
            shutil.rmtree(profile_dir, ignore_errors=True)
