#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : session.py
@Author  : caixiongjiang
@Date    : 2026/05/11
@Function:
    Chat REST API 请求 / 响应模型

    设计要点
    --------
    - 与 ``src.db.mysql.models.conversation.chat_session.ChatSession`` 字段
      命名对齐，便于 DTO → ORM 直接构造；
    - 嵌套子模型（``Citation`` / ``ToolCallRecord`` / ``TokenUsageRecord``）
      照搬 MongoDB 端 Pydantic 定义的关键字段，避免前端跨库认知；
    - 列表返回统一用 ``PaginationResponse[T]`` 风格（``total / page / page_size``）。
@Modify History:
    2026-05-11 - 首版（Phase 4）
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 请求模型
# ============================================================


class ChatSessionCreateRequest(BaseModel):
    """创建会话请求"""

    title: str = Field("新会话", max_length=255, description="会话标题（占位，首轮后异步覆盖）")
    knowledge_base_ids: List[str] = Field(
        default_factory=list,
        description="本会话允许检索的知识库 ID 列表；空表示用户全量 KB",
    )
    folder_id: Optional[str] = Field(
        None,
        description=(
            "可选：会话绑定的文件夹 ID（来自 workspace_folder.folder_id）。"
            "传入后启用 folder scope，每轮检索范围限定在该文件夹下文档；"
            "session 创建时会校验 folder 所属 KB 必须在 knowledge_base_ids 中"
        ),
    )
    include_subfolders: bool = Field(
        True,
        description=(
            "folder scope 下是否递归包含子文件夹的文档，默认 True；"
            "仅当 folder_id 非空时有意义"
        ),
    )
    model_preset: str = "fast"
    model: Optional[str] = Field(
        None,
        description=(
            "LiteLLM 模型字符串（如 'openai/gpt-4o-mini'）；优先级高于 "
            "model_preset。None 表示由 model_preset 决定"
        ),
    )
    mode: str = Field("agent", description="会话交互模式（agent / plan 等）")
    thinking_level: str = Field(
        "off",
        description=(
            "思考强度档位（pi 标准 7 档：off/minimal/low/medium/high/xhigh/max）。"
            "不支持思考的模型忽略此字段；后端会按模型 thinking_levels 钳位。"
        ),
    )
    enable_multimodal: bool = Field(False, description="是否默认启用多模态读图")
    max_tool_rounds: int = Field(5, ge=1, le=20, description="Agent 工具批次上限")
    system_prompt: Optional[str] = Field(
        None, description="用户自定义 system_prompt；None 表示用模块默认",
    )

    # Pydantic v2 默认把 ``model_*`` 视作受保护命名空间；我们需要一个真叫
    # ``model`` 的字段，所以这里手动放空 protected_namespaces
    model_config = ConfigDict(protected_namespaces=())


class ChatSessionRenameRequest(BaseModel):
    """重命名会话请求"""

    title: str = Field(..., min_length=1, max_length=255, description="新标题")


# ============================================================
# 嵌套子模型
# ============================================================


class CitationItem(BaseModel):
    """命中片段引用（与 MongoDB Citation 子文档对齐）"""

    chunk_id: str
    document_id: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    score: float = 0.0

    # Phase A：UI 渲染扩展字段
    chunk_type: Optional[str] = None
    page_index: Optional[int] = None
    section_title: Optional[str] = None
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    preview: Optional[str] = None

    # Phase B：session 级 chunk alias（cN）
    alias: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class ToolCallItem(BaseModel):
    """assistant 发起的单次工具调用记录（与 MongoDB ToolCallRecord 对齐）"""

    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result_brief: Optional[str] = None
    items_added: int = 0
    # 检索工具专用：持久化检索结果，供历史回放时前端渲染"查看"按钮
    retrieval_chunks: Optional[List[Dict[str, Any]]] = None
    retrieval_params: Optional[Dict[str, Any]] = None
    # 召回链路统计（独立于查询参数）：每路 recalled/aligned/final + 融合/rerank 计数 + chunk_id 截断列表
    recall_stats: Optional[Dict[str, Any]] = None
    time_ms: Optional[float] = None
    execution_model: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class TokenUsageItem(BaseModel):
    """单条 assistant 消息的 token 用量"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: Optional[int] = None
    total_tokens: int = 0
    # 前缀缓存命中数（命中率 = cached/prompt）
    cached_tokens: int = 0

    model_config = ConfigDict(extra="ignore")


# ============================================================
# 响应模型
# ============================================================


class ChatSessionInfo(BaseModel):
    """会话详情（创建 / 重命名 / 单条 get 共用）"""

    session_id: str
    user_id: str
    title: str
    knowledge_base_ids: List[str] = Field(default_factory=list)
    folder_id: Optional[str] = Field(
        None,
        description="会话绑定的文件夹 ID；NULL=KB scope，非 NULL=folder scope",
    )
    include_subfolders: bool = Field(
        True,
        description="folder scope 下是否递归含子文件夹（仅当 folder_id 非空时生效）",
    )
    model_preset: str = "fast"
    model: Optional[str] = Field(
        None,
        description=(
            "LiteLLM 模型字符串；用户在前端选定后写回。NULL → 走 model_preset"
        ),
    )
    mode: str = "agent"
    thinking_level: Optional[str] = None
    enable_multimodal: bool = False
    max_tool_rounds: int = 5
    system_prompt: Optional[str] = None
    message_count: int = 0
    last_message_at: Optional[datetime] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None

    # 含 ``model`` 字段，需要解除 Pydantic v2 的保护命名空间
    model_config = ConfigDict(
        from_attributes=True, extra="ignore", protected_namespaces=(),
    )


class ChatSessionUpdateResponse(BaseModel):
    """重命名 / 软删除 的统一返回"""

    session_id: str
    success: bool = True
    message: str = "ok"


class ChatSessionListItem(BaseModel):
    """会话列表中的轻量条目（不含 system_prompt 等长字段）

    必须保留 ``knowledge_base_ids``：前端按 KB 维度组织对话面板，需要据此把
    "本知识库的会话"从用户全量会话里筛出来；缺字段会导致前端过滤后总是空，
    每次刷新都触发 ensureSession → createChatSession，旧会话视觉上"消失"。
    """

    session_id: str
    title: str
    knowledge_base_ids: List[str] = Field(default_factory=list)
    folder_id: Optional[str] = None
    include_subfolders: bool = True
    model_preset: str = "fast"
    model: Optional[str] = None
    mode: str = "agent"
    thinking_level: Optional[str] = None
    message_count: int = 0
    last_message_at: Optional[datetime] = None
    create_time: Optional[datetime] = None

    # 含 ``model`` 字段，需要解除 Pydantic v2 的保护命名空间
    model_config = ConfigDict(
        from_attributes=True, extra="ignore", protected_namespaces=(),
    )


class ChatSessionListResponse(BaseModel):
    """会话列表分页响应"""

    items: List[ChatSessionListItem] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


# ============================================================
# 消息历史响应
# ============================================================


class ChatMessageItem(BaseModel):
    """消息列表条目（与 MongoDB ChatMessage 字段对齐）"""

    message_id: str = Field(..., description="消息 ID（即 MongoDB _id）")
    role: str = Field(..., description="system / user / assistant / tool")
    content: str = ""
    thinking: Optional[str] = None
    thinking_ms: Optional[float] = None
    tool_calls: List[ToolCallItem] = Field(default_factory=list)
    tool_call_id: Optional[str] = None
    citations: List[CitationItem] = Field(default_factory=list)
    usage: Optional[TokenUsageItem] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    create_time: Optional[datetime] = None

    model_config = ConfigDict(extra="ignore")


class ChatMessageListResponse(BaseModel):
    """会话消息列表（按 create_time 正序，便于前端直接拼）"""

    session_id: str
    items: List[ChatMessageItem] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 50


class ContextStatusBreakdown(BaseModel):
    """上下文用量分项。"""

    system: int = 0
    skills: int = 0
    tools_schema: int = 0
    summary: int = 0
    history: int = 0
    user: int = 0
    reserved_output: int = 0


class ContextCompactionInfo(BaseModel):
    """最近一次压缩的可观测摘要。"""

    at: Optional[str] = None
    trigger: Optional[str] = None
    input_tokens: Optional[int] = None
    summary_tokens: Optional[int] = None


class ContextStatusResponse(BaseModel):
    """GET /sessions/{id}/context-status 响应。"""

    session_id: str
    model: Optional[str] = None
    max_context: int
    reserved_output: int
    used_tokens: int
    soft_limit: int
    ratio: float
    threshold_ratio: float
    will_compact_at: int
    counting: str
    breakdown: ContextStatusBreakdown
    last_compaction: Optional[ContextCompactionInfo] = None
    summary_count: int = 0
