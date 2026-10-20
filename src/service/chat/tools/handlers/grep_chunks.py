#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""grep_chunks 工具 — 字面 / 正则 / 布尔检索。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Literal, Optional

from loguru import logger

from src.client.llm.types import ToolSchema
from src.retrieve.types.enums import MatchMode
from src.retrieve.types.query import LexicalQuery, MetadataFilter
from src.retrieve.types.result import ChunkItem
from src.service.chat.tools.base import ToolDefinition
from src.service.chat.tools.helpers import format_grep_chunks_for_llm

if TYPE_CHECKING:
    from src.service.chat.tools.kit import KnowledgeNavToolKit

NAME = "grep_chunks"

_MAX_PATTERN_LEN = 500
_MAX_TOP_K = 30

_BOOLEAN_HINT_RE = re.compile(
    r"\b(AND|OR|NOT)\b",
    re.IGNORECASE,
)

SCHEMA: ToolSchema = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "在已索引 chunk 中做**字面穷举定位**（类似 Cursor 的 grep），收集某词/符号/型号"
            "在文档中的**全部出现位置**，为后续 `read_chunks` 与作答提供证据。"
            "适用场景：需要**穷举所有出现、对比多处、确认精确数值/配置、或精确引用某术语**。"
            "仅判断「有没有 / 是否出现」或概念性问题不必用本工具——那种情况用 `search_knowledge_base`。"
            "与 `search_knowledge_base` 互补：语义检索给 Top-K 相关段；本工具给字面全命中。"
            "返回 chunk 引用号 alias（`c1` / `c2`）与命中 snippet；全文用 `read_chunks`。"
            "可用 `document_id` 限定单篇文档。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "要定位的字面锚点：术语、指标名、符号、表号、型号、错误码等。"
                        "mode=literal 时为子串；mode=regex 时为 Python 正则；"
                        "mode=boolean 时为布尔表达式，如 "
                        "'\"GFLOPs\" OR \"FLOPs\"'、'\"λ_1\" AND \"λ_2\"'。"
                    ),
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "匹配模式：literal=大小写不敏感子串（默认）；"
                        "regex=正则；boolean=AND/OR/NOT 布尔表达式。"
                    ),
                    "enum": ["literal", "regex", "boolean"],
                    "default": "literal",
                },
                "document_id": {
                    "type": "string",
                    "description": (
                        "可选：限定在单篇文档内搜索。"
                        "不传则在当前会话 scope（知识库或文件夹）内搜索。"
                    ),
                },
                "chunk_type": {
                    "type": "string",
                    "description": (
                        "可选：过滤 chunk 类型：text / image / table / code_block"
                    ),
                    "enum": ["text", "image", "table", "code_block"],
                },
                "top_k": {
                    "type": "integer",
                    "description": "最多返回多少条命中 chunk",
                    "default": 15,
                    "minimum": 1,
                    "maximum": _MAX_TOP_K,
                },
                "context_chars": {
                    "type": "integer",
                    "description": "命中位置前后各展示多少字符（snippet 上下文）",
                    "default": 80,
                    "minimum": 20,
                    "maximum": 300,
                },
            },
            "required": ["query"],
        },
    },
}


def _build_filters(
    kit: KnowledgeNavToolKit,
    *,
    document_id: Optional[str],
    chunk_type: Optional[str],
) -> MetadataFilter:
    from src.service.knowledge.kb_access import build_kb_scoped_filter
    from src.service.knowledge.tombstone import exclude_deleted

    # 可见性分流：共享库（team/public）向量按 kb 隔离，不能带 user_id 过滤
    kb_id = kit.knowledge_base_ids[0] if kit.knowledge_base_ids else None
    filters = build_kb_scoped_filter(
        kit.user_id, kb_id, chunk_type=chunk_type
    )
    if document_id:
        filters.document_id = document_id.strip()
    elif kit.scope_document_ids:
        filters.document_ids = list(kit.scope_document_ids)
    # grep 直接调 ExactMatch / BooleanSearch，不走 RetrieveService，得自己排除墓碑
    return exclude_deleted(filters)


def _resolve_mode(
    query: str,
    mode: str,
) -> Literal["literal", "regex", "boolean"]:
    normalized = (mode or "literal").strip().lower()
    if normalized not in ("literal", "regex", "boolean"):
        return "literal"
    if normalized == "literal" and _BOOLEAN_HINT_RE.search(query):
        return "boolean"
    return normalized  # type: ignore[return-value]


async def _run_lexical_search(
    *,
    resolved_mode: str,
    query: str,
    filters: MetadataFilter,
    top_k: int,
) -> List[ChunkItem]:
    if resolved_mode == "boolean":
        from src.retrieve.capabilities.lexical.boolean_search import BooleanSearch

        lexical_query = LexicalQuery(
            bool_expression=query,
            top_k=top_k,
            filters=filters,
        )
        result = await BooleanSearch().execute(query=lexical_query)
        return list(result.items or [])

    from src.retrieve.capabilities.lexical.exact_match import ExactMatch

    match_mode = MatchMode.REGEX if resolved_mode == "regex" else MatchMode.FUZZY
    lexical_query = LexicalQuery(
        keywords=[query],
        match_mode=match_mode,
        top_k=top_k,
        filters=filters,
    )
    result = await ExactMatch().execute(query=lexical_query)
    return list(result.items or [])


async def _enrich_chunk_items(
    kit: KnowledgeNavToolKit,
    items: List[ChunkItem],
) -> None:
    if not items:
        return
    from src.service.chat.chunk_enricher import enrich_chunks

    kb_id = kit.knowledge_base_ids[0] if kit.knowledge_base_ids else None
    try:
        meta_map = await enrich_chunks(
            items,
            user_id=kit.user_id,
            knowledge_base_id=kb_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"grep_chunks enrich 失败: {e}")
        return

    for item in items:
        meta = meta_map.get(item.chunk_id)
        if meta is None:
            continue
        if not item.document_id and meta.document_id:
            item.document_id = meta.document_id
        if not item.section_id and meta.section_id:
            item.section_id = meta.section_id


async def handle(
    kit: KnowledgeNavToolKit,
    query: str,
    mode: str = "literal",
    document_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
    top_k: int = 15,
    context_chars: int = 80,
) -> str:
    raw_query = (query or "").strip()
    if not raw_query:
        return "grep_chunks: query 不能为空。"
    if len(raw_query) > _MAX_PATTERN_LEN:
        return (
            f"grep_chunks: query 过长（>{_MAX_PATTERN_LEN} 字符），"
            "请缩短或拆分多次检索。"
        )

    resolved_mode = _resolve_mode(raw_query, mode)
    if resolved_mode == "regex":
        try:
            re.compile(raw_query)
        except re.error as e:
            return f"grep_chunks: 正则表达式无效: {e}"

    top_k = max(1, min(int(top_k), _MAX_TOP_K))
    context_chars = max(20, min(int(context_chars), 300))
    try:
        filters = _build_filters(
            kit,
            document_id=document_id,
            chunk_type=chunk_type,
        )
    except Exception:
        return "grep_chunks: 当前会话绑定的知识库已不可访问。"

    try:
        items = await _run_lexical_search(
            resolved_mode=resolved_mode,
            query=raw_query,
            filters=filters,
            top_k=top_k,
        )
    except ValueError as e:
        return f"grep_chunks: 参数错误: {e}"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"grep_chunks 执行异常: {e}")
        return f"grep_chunks 失败: {e}"

    await _enrich_chunk_items(kit, items)
    kit.supplemented.extend(items)
    kit.note_result_count(len(items))

    scope_note = ""
    if document_id:
        doc_label = (
            kit.alias_map.alias_for_document(document_id.strip())
            if kit.alias_map
            else document_id.strip()
        )
        scope_note = f"，document_id={doc_label}"
    elif kit.scope_kind == "folder":
        scope_note = f"，folder scope（{len(kit.scope_doc_id_set)} 篇文档）"

    header = (
        f"grep_chunks: mode={resolved_mode}, 命中 {len(items)} 条"
        f"{scope_note}。"
    )
    if not items:
        return (
            f"{header}\n"
            "未找到匹配 chunk。可尝试：\n"
            "- 换 mode（literal / regex / boolean）\n"
            "- 放宽 query 或去掉 document_id 限制\n"
            "- 语义相关问题改用 search_knowledge_base"
        )

    body = format_grep_chunks_for_llm(
        items,
        query=raw_query,
        mode=resolved_mode,
        context_chars=context_chars,
        alias_map=kit.alias_map,
    )
    logger.debug(
        f"grep_chunks({raw_query!r}, mode={resolved_mode}) → {len(items)} chunks",
    )
    return f"{header}\n{body}"


DEFINITION = ToolDefinition(name=NAME, schema=SCHEMA, handler=handle)
