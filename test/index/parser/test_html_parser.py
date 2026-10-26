#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
HTML 解析器离线测试

覆盖：
1. 元素流断言：综合 HTML 样本逐元素对照（标题层级/段落/列表/表格洗白/
   代码围栏/MathJax/KaTeX 公式/图片三分流/噪音剔除/散落文本）
2. 编码链：meta charset GBK / UTF-8 BOM / 无声明 GBK / 声明与实际不符
3. 边界：无 html/body 的片段 / 未闭合标签 / 空文档 / .htm 别名
4. 离线 split E2E：元素 payload → ParseEndMessage → split_document →
   建树（heading_level 策略）验证 h1→h2→h3 层级链

用法：
    uv run python test/index/parser/test_html_parser.py
"""

import asyncio
import base64
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.index.common_file_extract.parser.html_parser import HtmlParser
from src.index.common_file_extract.parser.file_parser import FileParser

# 1x1 PNG（魔数嗅探可识别）
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>测试文档</title>
<script>var noise = 1;</script>
<style>.a { color: red; }</style>
</head>
<body>
<h1>产品规格书</h1>
<p>本规范定义了<b>电气特性</b>与<i>机械特性</i>，参见<a href="#a">附录A</a>。</p>
<!-- 注释应被丢弃 -->
<div>散落的导言文本</div>
<h2>电气特性</h2>
<p>工作温度 -40~85℃。<br>存储温度 -60~100℃。</p>
<ul>
  <li>输入电压 5V</li>
  <li>输出电压 3.3V
    <ul><li>纹波 &lt; 50mV</li></ul>
  </li>
</ul>
<ol><li>上电自检</li><li>进入工作态</li></ol>
<dl><dt>VCC</dt><dd>供电正极</dd><dt>GND</dt><dd>地</dd></dl>
<table>
  <caption>引脚定义</caption>
  <tr><th>引脚</th><th>名称</th><th>功能</th></tr>
  <tr><td>1</td><td>VCC</td><td>供电</td></tr>
  <tr><td rowspan="2">2</td><td>IO0</td><td onclick="evil()">输入 &lt;0.3V</td></tr>
</table>
<pre><code class="language-python">def hello():
    print("hi")</code></pre>
<p>公式行内 <script type="math/tex">E=mc^2</script> 能量守恒。</p>
<script type="math/tex; mode=display">\\int_0^1 x dx = \\frac{1}{2}</script>
<div class="katex-display"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>x</mi></mrow><annotation encoding="application/x-tex">a^2+b^2=c^2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true">rendered-noise</span></span></div>
<p>拓扑如下图：</p>
<figure>
  <img src="https://example.com/figures/topo.png" alt="备用说明">
  <figcaption>系统拓扑结构图</figcaption>
</figure>
<img src="images/relative.png" alt="相对路径图">
<p>内嵌小图 <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="> 完成。</p>
<div hidden>隐藏属性内容不应出现</div>
<div style="display:none">css隐藏内容不应出现</div>
<nav><span>首页 产品 关于</span></nav>
<blockquote><h3>引用内标题</h3><p>引用正文</p></blockquote>
</body>
</html>
"""


class _FakeFetcher:
    """离线替身：只放行 topo.png，其余 URL 一律失败（走回退）。"""

    def __init__(self):
        self._ok = {"https://example.com/figures/topo.png": (_PNG_BYTES, "png")}

    async def fetch_all(self, urls):
        return {u: self._ok.get(u) for u in urls}


def _write_tmp(content: bytes, suffix: str = ".html") -> Path:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


def _elements(result):
    return result["struct_content"]["root"][0]["page_info"]


# ==================== 1. 元素流断言 ====================

async def test_element_stream():
    parser = HtmlParser(image_fetcher=_FakeFetcher())
    path = _write_tmp(SAMPLE_HTML.encode("utf-8"))
    result = await parser.parse(path, "sample.html")
    els = _elements(result)

    assert result["parser_source"] == "html"
    assert result["total_pages"] == 1
    assert result["file_type"] == "html"
    assert [e["element_index"] for e in els] == list(range(len(els))), \
        "element_index 必须连续从 0 编号"
    assert all(e["bbox"] == [] for e in els)

    i = 0

    def expect(**kwargs):
        nonlocal i
        el = els[i]
        for key, value in kwargs.items():
            actual = el.get(key)
            assert actual == value, f"元素[{i}] 字段 {key}: 期望 {value!r}, 实际 {actual!r} (元素: {el})"
        i += 1

    # 0: h1
    expect(type="text", text_level=1, text="产品规格书")
    # 1: 段落行内展平
    expect(type="text", text="本规范定义了电气特性与机械特性，参见附录A。")
    # 2: div 散落文本
    expect(type="text", text="散落的导言文本")
    # 3: h2
    expect(type="text", text_level=2, text="电气特性")
    # 4: br 保留换行
    expect(type="text", text="工作温度 -40~85℃。\n存储温度 -60~100℃。")
    # 5: ul 嵌套缩进 + 实体解码
    expect(type="text", text="- 输入电压 5V\n- 输出电压 3.3V\n  - 纹波 < 50mV")
    # 6: ol 序号
    expect(type="text", text="1. 上电自检\n2. 进入工作态")
    # 7: dl
    expect(type="text", text="VCC\n  供电正极\nGND\n  地")
    # 8: 表格洗白（受限标签 + rowspan 保留 + 属性剥除 + 文本转义）
    expect(
        type="table",
        table_caption=["引脚定义"],
        table_body=(
            "<table>"
            "<tr><th>引脚</th><th>名称</th><th>功能</th></tr>"
            "<tr><td>1</td><td>VCC</td><td>供电</td></tr>"
            '<tr><td rowspan="2">2</td><td>IO0</td><td>输入 &lt;0.3V</td></tr>'
            "</table>"
        ),
    )
    # 9: 代码块围栏标准形态
    expect(
        type="code_block",
        text='```python\ndef hello():\n    print("hi")\n```',
        code_language="python",
    )
    # 10: 行内 math/tex → $latex$ 文本
    expect(type="text", text="公式行内 $E=mc^2$ 能量守恒。")
    # 11: display math/tex → equation
    expect(type="equation", text="\\int_0^1 x dx = \\frac{1}{2}", text_format="latex")
    # 12: KaTeX annotation → equation（katex-html aria-hidden 噪音不出现）
    expect(type="equation", text="a^2+b^2=c^2", text_format="latex")
    # 13: 引导段落
    expect(type="text", text="拓扑如下图：")
    # 14: figure 图片下载成功 + figcaption 优先于 alt
    expect(
        type="image",
        img_path="topo.png",
        image_caption=["系统拓扑结构图"],
    )
    assert els[14]["image_base64"], "外链图片应下载转 base64"
    # 15: 相对路径回退为 text 原文
    expect(type="text", text='<img src="images/relative.png" alt="相对路径图">')
    # 16/17: 混排段落先 text 后 image（data URI 解码）
    expect(type="text", text="内嵌小图 完成。")
    expect(type="image", img_path="html-image.png", image_caption=[])
    assert els[17]["image_base64"]
    # 18: nav 保守保留（只剥确定噪音）
    expect(type="text", text="首页 产品 关于")
    # 19/20: blockquote 递归
    expect(type="text", text_level=3, text="引用内标题")
    expect(type="text", text="引用正文")

    assert i == len(els), f"元素总数: 期望 {i}, 实际 {len(els)}"

    # 全文不含噪音
    all_text = " ".join(str(e.get("text", "")) + str(e.get("table_body", "")) for e in els)
    for noise in ("noise", "color", "隐藏", "rendered-noise", "evil()"):
        assert noise not in all_text, f"噪音泄漏: {noise}"

    print(f"✅ 元素流断言通过: {len(els)} 元素")


# ==================== 2. 编码链 ====================

async def test_encodings():
    parser = HtmlParser()

    # meta charset 声明 GBK
    gbk_html = (
        '<html><head><meta charset="gbk"></head>'
        "<body><h1>中文标题</h1><p>温度范围 -40~85℃。</p></body></html>"
    ).encode("gbk")
    result = await parser.parse(_write_tmp(gbk_html), "gbk.html")
    els = _elements(result)
    assert els[0]["text"] == "中文标题", f"meta gbk 解码失败: {els[0]['text']!r}"
    assert els[1]["text"] == "温度范围 -40~85℃。"

    # UTF-8 BOM
    bom_html = ("<html><body><h1>BOM标题</h1></body></html>").encode("utf-8")
    result = await parser.parse(_write_tmp(b"\xef\xbb\xbf" + bom_html), "bom.html")
    assert _elements(result)[0]["text"] == "BOM标题"

    # 无声明 GBK（降级链命中）
    result = await parser.parse(_write_tmp(gbk_html.replace(b'charset="gbk"', b"")), "nogbk.html")
    assert _elements(result)[0]["text"] == "中文标题", "无声明 GBK 降级失败"

    # 声明与实际不符（声明 utf-8 实为 GBK）→ 声明解码失败落降级链
    mismatch = (
        '<html><head><meta charset="utf-8"></head><body><p>温度 -40~85℃</p></body></html>'
    ).encode("gbk")
    result = await parser.parse(_write_tmp(mismatch), "mismatch.html")
    assert _elements(result)[0]["text"] == "温度 -40~85℃"

    print("✅ 编码链断言通过: meta/BOM/降级/声明不符")


# ==================== 3. 边界 ====================

async def test_edges():
    parser = HtmlParser()

    # 片段（无 html/body/head）
    result = await parser.parse(_write_tmp("<h1>frag</h1><p>para</p>".encode()), "frag.html")
    els = _elements(result)
    assert [e["text"] for e in els] == ["frag", "para"]

    # 未闭合/嵌套错乱标签
    result = await parser.parse(
        _write_tmp("<div><p>未闭合<b>加粗<span>嵌套".encode()), "broken.html"
    )
    assert _elements(result)[0]["text"] == "未闭合加粗嵌套"

    # 空文档
    result = await parser.parse(_write_tmp(b"<html><body></body></html>"), "empty.html")
    assert _elements(result) == []
    assert result["status"] == "success"

    # .htm 别名
    result = await parser.parse(_write_tmp("<p>htm 内容</p>".encode(), ".htm"), "alias.htm")
    assert result["file_type"] == "htm"
    assert result["parser_source"] == "html"
    assert _elements(result)[0]["text"] == "htm 内容"

    # FileParser 类型注册
    assert FileParser.detect_file_type(Path("a.html")) == "html"
    assert FileParser.detect_file_type(Path("a.htm")) == "html"
    assert FileParser.detect_file_type(Path("a.HTML")) == "html"

    print("✅ 边界断言通过: 片段/未闭合/空文档/.htm/类型注册")


# ==================== 4. 离线 split E2E + 建树 ====================

async def test_offline_split_and_tree():
    import src.db.kafka  # noqa: F401 - 先初始化 kafka 包，规避 types.messages 循环导入
    from src.types.messages.index import ParseEndMessage
    from src.service.knowledge.components.text_splitter_service import TextSplitterService
    from src.index.common_file_extract.extract.section_context import (
        build_sections_from_payload,
    )
    from src.index.common_file_extract.extract.section_tree import (
        build_section_tree,
        post_order,
    )

    parser = HtmlParser(image_fetcher=_FakeFetcher())
    result = await parser.parse(_write_tmp(SAMPLE_HTML.encode("utf-8")), "sample.html")

    # 元素 payload 构造（镜像 file_parser_service._process_parse_result，
    # 离线省去图片上传：image_file_name 直接取 img_path）
    elements_payload = []
    for idx, e in enumerate(_elements(result)):
        payload = {
            "element_id": f"element-{idx}",
            "element_index": e["element_index"],
            "page_index": 0,
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
            payload["image_file_type"] = e.get("img_path", "").rsplit(".", 1)[-1]
        elements_payload.append(payload)

    message = ParseEndMessage(
        user_id="u-test",
        file_id="file-test",
        filename="sample.html",
        status="success",
        total_pages=1,
        total_chars=len(result.get("content", "")),
        parse_tool=result["parser_source"],
        document_language="zh",
        document_id="doc-test",
        elements=elements_payload,
    )

    service = TextSplitterService()
    parse_result = service.build_parse_result_from_payload(message)
    assert parse_result.parse_tool == "html"
    split_result = await service.split_document(parse_result, "doc-test")

    sections = split_result.sections
    chunks = split_result.chunks
    section_titles = [s.content for s in sections]
    assert section_titles == ["产品规格书", "电气特性", "引用内标题"], section_titles

    code_chunks = [c for c in chunks if c.is_code_block()] if hasattr(chunks[0], "is_code_block") \
        else [c for c in chunks if c.chunk_type == "code_block"]
    assert len(code_chunks) == 1
    assert code_chunks[0].metadata.get("code_language") == "python", \
        f"code_language 应从围栏重解析: {code_chunks[0].metadata}"
    assert 'print("hi")' in (code_chunks[0].get_text_content() or "")

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert "引脚定义" in (table_chunks[0].get_text_content() or "")

    image_chunks = [c for c in chunks if c.chunk_type == "image"]
    assert len(image_chunks) == 2, f"topo + data-uri 两张图: {len(image_chunks)}"
    assert image_chunks[0].image_caption == "系统拓扑结构图", \
        f"figcaption 应进 image_caption: {image_chunks[0].image_caption!r}"
    assert "系统拓扑结构图" in (image_chunks[0].display_text or ""), \
        "figcaption 应进 image 占位展示文本"

    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert text_chunks, "应有文本 chunk"
    merged_text = "\n".join(c.get_text_content() or "" for c in text_chunks)
    assert "$E=mc^2$" in merged_text, "行内公式应并入文本流"
    assert "\\int_0^1" in merged_text, "equation 元素应并入文本流（免清洗）"
    assert "1. 上电自检" in merged_text

    # 建树（heading_level 策略，h1→h2→h3 层级链）
    chunks_payload = [
        {
            "chunk_id": c.chunk_id,
            "section_id": c.section_id,
            "chunk_type": c.chunk_type,
            "text": c.get_text_content() or "",
            "image_caption": c.image_caption,
            "image_footnote": c.image_footnote,
            "page_index": c.page_index,
            "language": c.language,
        }
        for c in chunks
    ]
    sections_payload = [
        {
            "section_id": s.section_id,
            "title": s.content,
            "level": s.level,
            "page_index": s.page_index,
            "chunk_id_list": list(s.chunk_id_list),
        }
        for s in sections
    ]
    swc = build_sections_from_payload(sections_payload, chunks_payload)
    roots = build_section_tree(swc, strategy="heading_level")
    assert len(roots) == 1 and roots[0].title == "产品规格书"
    children = roots[0].children
    assert len(children) == 1 and children[0].title == "电气特性"
    assert len(children[0].children) == 1 and children[0].children[0].title == "引用内标题"
    # 后序遍历：叶子先于父
    order = [n.title for n in post_order(roots)]
    assert order.index("引用内标题") < order.index("电气特性") < order.index("产品规格书")

    total = len(sections) + len(chunks)
    print(f"✅ 离线 split + 建树断言通过: sections={len(sections)}, chunks={len(chunks)} (共 {total})")


async def main():
    await test_element_stream()
    await test_encodings()
    await test_edges()
    await test_offline_split_and_tree()
    print("\n🎉 全部通过")


if __name__ == "__main__":
    asyncio.run(main())
