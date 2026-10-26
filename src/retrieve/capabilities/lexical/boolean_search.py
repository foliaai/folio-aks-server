#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : boolean_search.py
@Author  : caixiongjiang
@Date    : 2026/03/02
@Function: 
    AND / OR / NOT 布尔逻辑检索原子能力
    
    核心流程:
      bool_expression 字符串 → BooleanExpressionParser 解析为 AST
      → AST.to_mongo_query() 转化为 MongoDB 查询
      → ChunkData 结果 → RetrieveResult[ChunkItem]
    
    依赖:
      - MongoDB ChunkData 模型
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from src.db.mongodb.models.chunk_data import ChunkData
from src.types.utils.chunk_search_text import resolve_chunk_display_text
from src.retrieve.capabilities.base import BaseCapability, CapabilityDescriptor
from src.retrieve.capabilities.lexical._filter_helper import (
    filter_has_chunk_scope,
    resolve_chunk_ids_from_filters,
)
from src.retrieve.types.query import LexicalQuery
from src.retrieve.types.result import ChunkItem, RetrieveResult


# ==================== AST 节点定义 ====================


def _keyword_field_or_query(keyword: str) -> Dict[str, Any]:
    """在 search_text 与结构化元数据字段上做包含匹配。"""
    escaped = re.escape(keyword)
    regex = {"$regex": escaped, "$options": "i"}
    return {
        "$or": [
            {"search_text": regex},
            {"text_meta.image_caption": regex},
            {"text_meta.table_caption": regex},
            {"text_meta.text": regex},
        ],
    }


@dataclass
class TermNode:
    """叶子节点：单个关键词"""
    keyword: str

    def to_mongo_query(self) -> Dict[str, Any]:
        return _keyword_field_or_query(self.keyword)


@dataclass
class AndNode:
    """AND 节点"""
    left: ASTNode
    right: ASTNode

    def to_mongo_query(self) -> Dict[str, Any]:
        return {"$and": [self.left.to_mongo_query(), self.right.to_mongo_query()]}


@dataclass
class OrNode:
    """OR 节点"""
    left: ASTNode
    right: ASTNode

    def to_mongo_query(self) -> Dict[str, Any]:
        return {"$or": [self.left.to_mongo_query(), self.right.to_mongo_query()]}


@dataclass
class NotNode:
    """NOT 节点"""
    child: ASTNode

    def to_mongo_query(self) -> Dict[str, Any]:
        return {"$nor": [self.child.to_mongo_query()]}


ASTNode = Union[TermNode, AndNode, OrNode, NotNode]


# ==================== 词法分析 ====================


class _TokenType:
    TERM = "TERM"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    EOF = "EOF"


@dataclass
class _Token:
    type: str
    value: str


class _Lexer:
    """将布尔表达式字符串拆分为 Token 序列"""

    _KEYWORDS = {"AND": _TokenType.AND, "OR": _TokenType.OR, "NOT": _TokenType.NOT}

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0

    def tokenize(self) -> List[_Token]:
        tokens: List[_Token] = []
        while self._pos < len(self._text):
            ch = self._text[self._pos]

            if ch.isspace():
                self._pos += 1
                continue

            if ch == "(":
                tokens.append(_Token(_TokenType.LPAREN, "("))
                self._pos += 1
                continue

            if ch == ")":
                tokens.append(_Token(_TokenType.RPAREN, ")"))
                self._pos += 1
                continue

            if ch == '"' or ch == "'":
                tokens.append(self._read_quoted_string(ch))
                continue

            word = self._read_word()
            token_type = self._KEYWORDS.get(word.upper(), _TokenType.TERM)
            tokens.append(_Token(token_type, word))

        tokens.append(_Token(_TokenType.EOF, ""))
        return tokens

    def _read_word(self) -> str:
        start = self._pos
        while self._pos < len(self._text) and not self._text[self._pos].isspace() \
                and self._text[self._pos] not in ("(", ")"):
            self._pos += 1
        return self._text[start:self._pos]

    def _read_quoted_string(self, quote: str) -> _Token:
        self._pos += 1
        start = self._pos
        while self._pos < len(self._text) and self._text[self._pos] != quote:
            self._pos += 1
        value = self._text[start:self._pos]
        if self._pos < len(self._text):
            self._pos += 1
        return _Token(_TokenType.TERM, value)


# ==================== 语法分析（递归下降） ====================


class BooleanExpressionParser:
    """将布尔表达式字符串解析为 AST

    语法:
        expr     → or_expr
        or_expr  → and_expr ( 'OR' and_expr )*
        and_expr → not_expr ( 'AND' not_expr )*
        not_expr → 'NOT' not_expr | primary
        primary  → TERM | '(' expr ')'

    运算符优先级 (从高到低): NOT > AND > OR
    """

    def __init__(self) -> None:
        self._tokens: List[_Token] = []
        self._pos = 0

    def parse(self, expression: str) -> ASTNode:
        self._tokens = _Lexer(expression).tokenize()
        self._pos = 0
        node = self._or_expr()
        if self._current().type != _TokenType.EOF:
            raise ValueError(
                f"布尔表达式解析错误: 意外的 Token '{self._current().value}' "
                f"在位置 {self._pos}"
            )
        return node

    def _current(self) -> _Token:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return _Token(_TokenType.EOF, "")

    def _consume(self, expected_type: str) -> _Token:
        token = self._current()
        if token.type != expected_type:
            raise ValueError(
                f"布尔表达式解析错误: 期望 {expected_type}，"
                f"实际 {token.type}('{token.value}')"
            )
        self._pos += 1
        return token

    def _or_expr(self) -> ASTNode:
        node = self._and_expr()
        while self._current().type == _TokenType.OR:
            self._pos += 1
            right = self._and_expr()
            node = OrNode(left=node, right=right)
        return node

    def _and_expr(self) -> ASTNode:
        node = self._not_expr()
        while self._current().type == _TokenType.AND:
            self._pos += 1
            right = self._not_expr()
            node = AndNode(left=node, right=right)
        return node

    def _not_expr(self) -> ASTNode:
        if self._current().type == _TokenType.NOT:
            self._pos += 1
            child = self._not_expr()
            return NotNode(child=child)
        return self._primary()

    def _primary(self) -> ASTNode:
        token = self._current()

        if token.type == _TokenType.LPAREN:
            self._pos += 1
            node = self._or_expr()
            self._consume(_TokenType.RPAREN)
            return node

        if token.type == _TokenType.TERM:
            self._pos += 1
            return TermNode(keyword=token.value)

        raise ValueError(
            f"布尔表达式解析错误: 意外的 Token '{token.value}' "
            f"(类型 {token.type})"
        )


# ==================== 原子能力 ====================


class BooleanSearch(BaseCapability):
    """AND / OR / NOT 布尔逻辑检索

    将布尔表达式字符串解析为 AST，转化为 MongoDB $and/$or/$not 查询，
    对 chunk_data 的 text 字段执行布尔逻辑匹配。

    表达式语法示例:
      - "Transformer AND Attention"
      - "(YOLO AND 实时) OR (SSD NOT 低精度)"
      - "深度学习 NOT 入门教程"

    对应 MongoDB Collection: chunk_data
    """

    def __init__(self) -> None:
        super().__init__()
        self._parser = BooleanExpressionParser()

    async def _do_execute(self, **kwargs: Any) -> RetrieveResult:
        query: LexicalQuery = kwargs["query"]

        if not query.bool_expression:
            raise ValueError("BooleanSearch 需要 bool_expression 参数")

        ast = self._parser.parse(query.bool_expression)
        mongo_query = ast.to_mongo_query()
        mongo_query["deleted"] = 0

        # 透传 MetadataFilter：先在 MySQL 解析 chunk_id 集合，再注入 _id $in 条件
        # （2026-10-26 起解析失败 fail-closed 返回空集，不再回退无限制查询）
        if filter_has_chunk_scope(query.filters):
            allowed_ids = resolve_chunk_ids_from_filters(query.filters)
            if not allowed_ids:
                # None 理论不可达（前置 filter_has_chunk_scope 已保证）；
                # 空集 = 无命中或解析失败，均按空结果处理
                return RetrieveResult(items=[], total_count=0)
            mongo_query["_id"] = {"$in": allowed_ids}

        results = await ChunkData.find(
            mongo_query,
        ).limit(query.top_k).to_list()

        items = self._build_result_items(results)

        return RetrieveResult(
            items=items,
            total_count=len(items),
        )

    @staticmethod
    def _build_result_items(docs: List[ChunkData]) -> List[ChunkItem]:
        items: List[ChunkItem] = []
        for doc in docs:
            items.append(ChunkItem(
                chunk_id=str(doc.id),
                score=1.0,
                text=resolve_chunk_display_text(doc),
                metadata={
                    "chunk_type": doc.chunk_type,
                },
            ))
        return items

    def describe(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            name="boolean_search",
            display_name="布尔逻辑检索",
            description=(
                "使用 AND/OR/NOT 布尔运算符组合多个关键词，"
                "在 MongoDB chunk_data 中执行精确的多条件复合查询。"
                "支持括号分组控制优先级。"
            ),
            input_schema={
                "bool_expression": "str - 布尔表达式（如 'A AND B NOT C'）",
                "top_k": "int - 返回数量上限，默认 10",
                "filters": "MetadataFilter - 元数据过滤条件（可选）",
            },
            output_type="RetrieveResult[ChunkItem]",
        )
