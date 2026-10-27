#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Excel 解析器离线测试

覆盖：
1. 区域识别：合并标题/备注归属 caption/footnote、表内空行合并、
   空行分表（隔 2 空行）、重复表头保持分表、左右并列、合并单元格 span
2. 原生表对象：displayName 即 caption、范围格子不重复进连通块
3. 值序列化：float 整数归一/bool/datetime/公式缓存缺失回退公式文本
4. sheet 语义：隐藏跳过、全空跳过、page_idx 连续重编号
5. 图片：zip 链锚点解析、区域归属 caption、base64 契约
6. xls：soffice 可用时转换链路（无 soffice 跳过）
7. 离线 split E2E：payload → ParseEndMessage → split → section=sheet 名

用法：
    uv run python test/index/parser/test_excel_parser.py
"""

import asyncio
import base64
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XlImage
from openpyxl.worksheet.table import Table as XlTable
from PIL import Image as PILImage

from src.index.common_file_extract.parser.excel_parser import ExcelParser
from src.index.common_file_extract.parser.file_parser import FileParser

# 1x1 PNG（与 html 测试同款，zip 层可识别）
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _write_png(path: Path) -> Path:
    path.write_bytes(_PNG_BYTES)
    return path


def _build_main_workbook(path: Path, img_path: Path) -> Path:
    """综合样本：6 个行为各占一张 sheet + 隐藏/空 sheet"""
    wb = Workbook()

    # Sheet1: 合并标题 caption + 表 + 单格备注 footnote
    ws = wb.active
    ws.title = "一月销售"
    ws["A1"] = "产品明细"
    ws.merge_cells("A1:C1")
    ws.append([None, None, None])  # 占位对齐（append 从下一行起）
    ws.append(["产品", "数量", "单价"])   # A2:C2 → 实际行 3？改直接定位
    # 直接按坐标写，避免 append 行号漂移
    ws["A2"], ws["B2"], ws["C2"] = "产品", "数量", "单价"
    ws["A3"], ws["B3"], ws["C3"] = "电机A", 10, 5.5
    ws["A4"], ws["B4"], ws["C4"] = "电机B", 20, 7.0
    ws["A6"] = "备注：数据截至月底"

    # Sheet2: 表内空行（1 行）+ 多格数据行 → 合并为一张表
    ws = wb.create_sheet("表内空行")
    ws["A1"], ws["B1"] = "名称", "数量"
    ws["A2"], ws["B2"] = "甲", 1
    ws["A4"], ws["B4"] = "乙", 2  # 第 3 行空

    # Sheet3: 隔 2 空行 + 列范围不同 → 两张表
    ws = wb.create_sheet("空行分表")
    ws["A1"], ws["B1"] = "表头1", "值1"
    ws["A2"], ws["B2"] = "x", 1
    # 3、4 行空
    ws["A5"], ws["B5"], ws["C5"] = "表头2", "值2", "备注列"
    ws["A6"], ws["B6"], ws["C6"] = "y", 2, "r"

    # Sheet4: 重复表头（同列范围、首行相同）→ 保持两张表
    ws = wb.create_sheet("重复表头")
    ws["A1"], ws["B1"] = "产品", "数量"
    ws["A2"], ws["B2"] = "甲", 1
    # 第 3 行空
    ws["A4"], ws["B4"] = "产品", "数量"
    ws["A5"], ws["B5"] = "乙", 2

    # Sheet5: 左右并列（空列分隔）→ 两张表
    ws = wb.create_sheet("左右并列")
    ws["A1"], ws["B1"] = "甲表", "值"
    ws["A2"], ws["B2"] = "x", 1
    ws["D1"], ws["E1"] = "乙表", "值"
    ws["D2"], ws["E2"] = "y", 2

    # Sheet6: 原生表对象 + 范围外数据
    ws = wb.create_sheet("原生表")
    ws["A1"], ws["B1"] = "季度", "总额"
    ws["A2"], ws["B2"] = "Q1", 375
    ws["A3"], ws["B3"] = "Q2", 420
    ws.add_table(XlTable(displayName="SalesTable", ref="A1:B3"))

    # Sheet7: 值类型（float/bool/datetime/公式）
    ws = wb.create_sheet("值类型")
    ws["A1"], ws["B1"] = "类型", "值"
    ws["A2"], ws["B2"] = "浮点整数", 20.0
    ws["A3"], ws["B3"] = "布尔", True
    ws["A4"], ws["B4"] = "日期", datetime(2026, 1, 15)
    ws["A5"], ws["B5"] = "公式和", None
    ws["C1"], ws["C2"] = 1, 2
    ws["B5"] = "=SUM(C1:C2)"  # openpyxl 写入的公式无缓存值

    # Sheet8: 图片（表外锚点 B7）
    ws = wb.create_sheet("插图")
    ws["A1"], ws["B1"] = "名称", "值"
    ws["A2"], ws["B2"] = "甲", 1
    img = XlImage(str(img_path))
    img.width, img.height = 4, 4
    ws.add_image(img, "B7")

    # Sheet9: 合并单元格数据区（跨行跨列 span）
    ws = wb.create_sheet("合并单元格")
    ws["A1"] = "分类"
    ws.merge_cells("A1:A2")
    ws["B1"], ws["B2"] = "子项1", "子项2"
    ws["C1"], ws["C2"] = "值1", "值2"

    # 隐藏 sheet + 全空 sheet（应跳过且不占页码）
    ws = wb.create_sheet("隐藏表")
    ws["A1"] = "不应出现"
    ws.sheet_state = "hidden"
    wb.create_sheet("全空表")

    wb.save(path)
    return path


def _pages(result):
    return result["struct_content"]["root"]


def _page(result, idx):
    return _pages(result)[idx]["page_info"]


def _tables(page):
    return [e for e in page if e["type"] == "table"]


async def test_region_detection():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_path = _write_png(td / "tick.png")
        xlsx = _build_main_workbook(td / "sample.xlsx", img_path)
        result = await ExcelParser().parse(xlsx, "sample.xlsx")

    assert result["parser_source"] == "excel"
    assert result["total_pages"] == 9, f"隐藏/全空 sheet 应跳过: {result['total_pages']}"

    # Sheet1：标题 caption + 表体 3 行 + 备注 footnote
    p = _page(result, 0)
    assert p[0] == {
        "type": "text", "text": "一月销售", "text_level": 1,
        "bbox": [], "element_index": 0,
    }, p[0]
    tables = _tables(p)
    assert len(tables) == 1, f"应恰一张表: {len(tables)}"
    t = tables[0]
    assert t["table_caption"] == ["产品明细"], t["table_caption"]
    assert t["table_footnote"] == ["备注：数据截至月底"], t["table_footnote"]
    assert t["table_body"] == (
        "<table>"
        "<tr><td>产品</td><td>数量</td><td>单价</td></tr>"
        "<tr><td>电机A</td><td>10</td><td>5.5</td></tr>"
        "<tr><td>电机B</td><td>20</td><td>7</td></tr>"
        "</table>"
    ), t["table_body"]
    print("✅ Sheet1 标题/备注归属 + 表体序列化")


async def test_blank_row_merge_and_split():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_path = _write_png(td / "tick.png")
        xlsx = _build_main_workbook(td / "sample.xlsx", img_path)
        result = await ExcelParser().parse(xlsx, "sample.xlsx")

    # Sheet2：表内 1 空行 → 合并，空行保留为空 tr
    t = _tables(_page(result, 1))
    assert len(t) == 1, f"表内空行应合并: {len(t)}"
    assert t[0]["table_body"].count("<tr>") == 4
    assert "<tr><td></td><td></td></tr>" in t[0]["table_body"], t[0]["table_body"]
    print("✅ Sheet2 表内空行合并（空 tr 保留）")

    # Sheet3：隔 2 空行 + 列范围不同 → 两张表
    t = _tables(_page(result, 2))
    assert len(t) == 2, f"隔 2 空行且列数不同应分表: {len(t)}"
    print("✅ Sheet3 空行分表")

    # Sheet4：同列范围但首行重复表头 → 保持两张表
    t = _tables(_page(result, 3))
    assert len(t) == 2, f"重复表头应保持分表: {len(t)}"
    assert t[1]["table_body"].startswith("<table><tr><td>产品</td><td>数量</td></tr>")
    print("✅ Sheet4 重复表头保持分表")

    # Sheet5：左右并列 → 两张表，先左后右
    t = _tables(_page(result, 4))
    assert len(t) == 2, f"空列分隔应分表: {len(t)}"
    assert "甲表" in t[0]["table_body"] and "乙表" in t[1]["table_body"]
    print("✅ Sheet5 左右并列分表")


async def test_native_table_and_values():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_path = _write_png(td / "tick.png")
        xlsx = _build_main_workbook(td / "sample.xlsx", img_path)
        result = await ExcelParser().parse(xlsx, "sample.xlsx")

    # Sheet6：原生表 displayName 即 caption
    t = _tables(_page(result, 5))
    assert len(t) == 1
    assert t[0]["table_caption"] == ["SalesTable"], t[0]["table_caption"]
    assert "Q1" in t[0]["table_body"]
    print("✅ Sheet6 原生表对象 caption")

    # Sheet7：值序列化
    body = _tables(_page(result, 6))[0]["table_body"]
    assert "<td>20</td>" in body, "float 20.0 应归一为 20"
    assert "<td>TRUE</td>" in body, "bool 应大写"
    assert "<td>2026-01-15T00:00:00</td>" in body, "datetime 应 isoformat"
    assert "<td>=SUM(C1:C2)</td>" in body, "无缓存公式应回退公式文本"
    print("✅ Sheet7 值序列化（float/bool/datetime/公式回退）")


async def test_image_and_merge_cells():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_path = _write_png(td / "tick.png")
        xlsx = _build_main_workbook(td / "sample.xlsx", img_path)
        result = await ExcelParser().parse(xlsx, "sample.xlsx")

    # Sheet8：图片元素（锚点表外 → sheet 级 caption）
    page = _page(result, 7)
    images = [e for e in page if e["type"] == "image"]
    assert len(images) == 1, f"应解析出 1 张图: {len(images)}"
    im = images[0]
    assert base64.b64decode(im["image_base64"]) == _PNG_BYTES, "base64 应为图片原字节"
    assert im["img_path"] == "image1.png", im["img_path"]
    assert im["image_caption"] == ["插图｜B7 插图"], im["image_caption"]
    assert im["element_index"] == 2, "元素顺序应为 标题→表→图"
    print("✅ Sheet8 图片 zip 链锚点解析 + 区域归属")

    # Sheet9：合并单元格 span（page 8）
    t = _tables(_page(result, 8))[0]
    assert '<td rowspan="2">分类</td>' in t["table_body"], t["table_body"]
    assert t["table_body"].count("<td") == 5, "被吞格子不应输出 td"
    print("✅ Sheet9 合并单元格 rowspan + 被吞格跳过")


async def test_xls_conversion():
    soffice = shutil.which("soffice") or (
        "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    )
    if not Path(soffice).exists():
        print("⏭️ 无 LibreOffice，跳过 xls 转换测试")
        return

    import subprocess
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_path = _write_png(td / "tick.png")
        xlsx = _build_main_workbook(td / "sample.xlsx", img_path)
        # 仅保留单 sheet 简化断言（soffice 对表格对象/图片的兼容性不参与本断言）
        wb = Workbook()
        ws = wb.active
        ws.title = "销售"
        ws["A1"], ws["B1"] = "产品", "数量"
        ws["A2"], ws["B2"] = "甲", 1
        wb.save(xlsx)

        proc = subprocess.run(
            [soffice, "--headless", "--convert-to", "xls", "--outdir", str(td), str(xlsx)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        xls = td / "sample.xls"
        assert proc.returncode == 0 and xls.exists(), f"soffice 转 xls 失败: {proc.stderr}"

        result = await ExcelParser(soffice_path=soffice, convert_timeout=60).parse(xls, "sample.xls")

    assert result["file_type"] == "xls"
    assert result["total_pages"] == 1
    page = _page(result, 0)
    assert page[0]["text"] == "销售"
    assert "<td>甲</td>" in _tables(page)[0]["table_body"]
    print("✅ xls → soffice 转 xlsx → 同链路解析")


async def test_offline_split_and_tree():
    import src.db.kafka  # noqa: F401 - 先初始化 kafka 包，规避 types.messages 循环导入
    from src.types.messages.index import ParseEndMessage
    from src.service.knowledge.components.text_splitter_service import TextSplitterService
    from src.index.common_file_extract.extract.section_context import (
        build_sections_from_payload,
    )
    from src.index.common_file_extract.extract.section_tree import (
        build_section_tree,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_path = _write_png(td / "tick.png")
        xlsx = _build_main_workbook(td / "sample.xlsx", img_path)
        result = await ExcelParser().parse(xlsx, "sample.xlsx")

    elements_payload = []
    for page in _pages(result):
        for e in page["page_info"]:
            payload = {
                "element_id": f"element-{len(elements_payload)}",
                "element_index": e["element_index"],
                "page_index": page["page_idx"],
                "element_type": e["type"],
                "page_position": None,
                "text_level": e.get("text_level"),
                "text": e.get("text", ""),
                "image_caption": e.get("image_caption"),
                "image_footnote": e.get("image_footnote"),
                "table_body": e.get("table_body", ""),
                "table_caption": e.get("table_caption"),
                "table_footnote": e.get("table_footnote"),
            }
            if e["type"] == "image":
                payload["image_file_name"] = e.get("img_path")
                payload["image_file_type"] = (e.get("img_path") or "").rsplit(".", 1)[-1]
            elements_payload.append(payload)

    message = ParseEndMessage(
        user_id="u-test",
        file_id="file-test",
        filename="sample.xlsx",
        status="success",
        total_pages=result["total_pages"],
        total_chars=len(result.get("content", "")),
        parse_tool=result["parser_source"],
        document_language="zh",
        document_id="doc-test",
        elements=elements_payload,
    )

    service = TextSplitterService()
    parse_result = service.build_parse_result_from_payload(message)
    assert parse_result.parse_tool == "excel"
    split_result = await service.split_document(parse_result, "doc-test")

    sections = split_result.sections
    titles = [s.content for s in sections]
    assert titles == [
        "一月销售", "表内空行", "空行分表", "重复表头", "左右并列",
        "原生表", "值类型", "插图", "合并单元格",
    ], f"section 应为工作表名: {titles}"

    table_chunks = [c for c in split_result.chunks if c.chunk_type == "table"]
    assert len(table_chunks) >= 9, f"每 sheet 至少 1 个 table chunk: {len(table_chunks)}"
    first = next(c for c in table_chunks if "电机A" in (c.get_text_content() or ""))
    assert "产品明细" in (first.get_text_content() or ""), "caption 应随 table chunk"

    image_chunks = [c for c in split_result.chunks if c.chunk_type == "image"]
    assert len(image_chunks) == 1
    assert image_chunks[0].image_caption == "插图｜B7 插图", \
        f"caption 应 list→str 归一: {image_chunks[0].image_caption!r}"

    chunks_payload = [
        {
            "chunk_id": c.chunk_id, "section_id": c.section_id,
            "chunk_type": c.chunk_type, "text": c.get_text_content() or "",
            "image_caption": c.image_caption, "image_footnote": c.image_footnote,
            "page_index": c.page_index, "language": c.language,
        }
        for c in split_result.chunks
    ]
    sections_payload = [
        {
            "section_id": s.section_id, "title": s.content, "level": s.level,
            "page_index": s.page_index, "chunk_id_list": list(s.chunk_id_list),
        }
        for s in sections
    ]
    swc = build_sections_from_payload(sections_payload, chunks_payload)
    roots = build_section_tree(swc, strategy="heading_level")
    assert len(roots) == 9 and all(r.title in titles for r in roots), \
        [r.title for r in roots]
    print(f"✅ 离线 split + 建树: sections={len(sections)}, chunks={len(split_result.chunks)}")


async def main():
    await test_region_detection()
    await test_blank_row_merge_and_split()
    await test_native_table_and_values()
    await test_image_and_merge_cells()
    await test_xls_conversion()
    await test_offline_split_and_tree()

    # FileParser 类型注册
    assert FileParser.detect_file_type(Path("a.xlsx")) == "excel"
    assert FileParser.detect_file_type(Path("a.XLS")) == "excel"
    print("✅ FileParser 类型注册: xlsx/xls")

    print("\n🎉 Excel 解析器全部测试通过")


if __name__ == "__main__":
    asyncio.run(main())
