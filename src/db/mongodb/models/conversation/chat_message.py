#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : chat_message.py
@Author  : caixiongjiang
@Date    : 2026/05/09
@Function:
    ChatMessage Document 定义

    对话消息文档：每个会话里产生的每一条消息（user / assistant / tool /
    可选 system）都是一条独立 Document。

    与 MySQL ``ChatSession`` 的关系
    --------------------------------
    通过 ``session_id`` 字符串关联，一对多：

        MySQL.chat_session.session_id (PK)
            ←── chat_message.session_id (索引) ──── 多条 ChatMessage

    与 OpenAI / LiteLLM messages 协议的对应
    ----------------------------------------
    ``role / content / tool_calls / tool_call_id`` 字段命名与传入
    ``LLMClient.agenerate(messages=...)`` 的格式保持一致，
    可零成本反向构造历史 messages 注入下一轮 LLM 请求。

    设计要点
    --------
    - 主键格式：``chatmsg_<uuid>``，与项目里 ``document_<uuid>`` /
      ``chunk_<uuid>`` 风格保持一致；
    - 通过 ``id: str = Field(..., alias="_id")`` 让 Beanie 接受字符串主键；
    - ``tool_calls`` / ``citations`` 用嵌套 Pydantic 模型，存为 BSON 子文档；
    - 索引：
      - ``(session_id, create_time)`` 是高频访问路径（按时间序列拉历史）；
      - ``(user_id, create_time desc)`` 用于"我所有消息"的审计查询；
      - ``(deleted, create_time)`` 兼容默认软删除过滤。
@Modify History:
    2026-05-09 - 首版（Phase 1）
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field
from pymongo import ASCENDING, DESCENDING, IndexModel

from src.db.mongodb.models.base_model import BaseDocument


# ==================== 角色枚举 ====================


class ChatRole(str, Enum):
    """OpenAI 风格的对话角色"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SUMMARY = "summary"  # 总结消息，用于上下文压缩


# ==================== 嵌套子模型 ====================


class Citation(PydanticBaseModel):
    """命中片段引用（assistant 消息引用了哪些 chunk）

    历史保留字段 (chunk_id / document_id / knowledge_base_id / score) 用于检索路径
    溯源；扩展字段 (chunk_type / page_index / section_title / file_name / file_id /
    preview) 服务于前端"内联可点击 + 悬浮预览"渲染，全部 Optional 以保持向后兼容
    （历史会话的 citations 没有这些字段，前端会按 chunk_id 兜底降级显示）。
    """

    chunk_id: str = Field(..., description="命中的 chunk ID（真实，落库稳定）")
    document_id: Optional[str] = Field(None, description="所属文档 ID")
    knowledge_base_id: Optional[str] = Field(None, description="所属知识库 ID")
    score: float = Field(0.0, description="检索得分（fusion / rerank 后）")
    alias: Optional[str] = Field(
        None,
        description=(
            "Phase B: session 级短 alias（cN 形式）；前端把 LLM 输出里的 [cN] 解析"
            "成 chip 时按 alias 反查本 citation。历史会话没有该字段时为 None。"
        ),
    )

    # ========== UI 渲染扩展字段（Phase A: 内联引用） ==========
    chunk_type: Optional[str] = Field(
        None, description="chunk 类型：text / table / image（equation 走 text + LaTeX）",
    )
    page_index: Optional[int] = Field(
        None, description="所在页码，从 0 开始；前端展示时 +1",
    )
    section_title: Optional[str] = Field(
        None, description="所属 section 的标题文本（SectionData.text）",
    )
    file_id: Optional[str] = Field(
        None, description="业务层 file_id（用于跳转 /knowledge/file/<file_id>）",
    )
    file_name: Optional[str] = Field(
        None, description="原始文件名（如 FRT075-33F.pdf）",
    )
    preview: Optional[str] = Field(
        None, description="片段正文摘要（截断至 200 字符）",
    )

    model_config = ConfigDict(extra="ignore")


class ToolCallRecord(PydanticBaseModel):
    """assistant 发起的单次工具调用记录（持久化用，非传输用）"""

    id: str = Field(..., description="LLM 给出的调用 ID（与 role=tool 的 tool_call_id 关联）")
    name: str = Field(..., description="工具名")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="参数（已解析，按模型原样落库——模型写 alias（c1/s1/d1）就存 alias，"
        "与流式帧/下一轮上下文装配/历史回放同一口径；真实 id 由 metadata deltas 重建 alias map 反查）",
    )
    result_brief: Optional[str] = Field(
        None, description="工具执行结果的文本摘要（前几百字）",
    )
    items_added: int = Field(
        0, description="本次工具补全到 supplemented_items 的数量",
    )
    # 检索工具专用：持久化检索结果，供历史回放时前端渲染"查看"按钮
    retrieval_chunks: Optional[List[Dict[str, Any]]] = Field(
        None, description="检索命中的 chunk 简要列表（chunk_id, score, preview 等）",
    )
    retrieval_params: Optional[Dict[str, Any]] = Field(
        None, description="检索参数（query_text, top_k, route_plan 等）",
    )
    # 召回结果统计（独立于检索参数）：每路召回/对齐/最终计数 + chunk_id 截断列表，供前端「召回链路」栏目
    recall_stats: Optional[Dict[str, Any]] = Field(
        None, description="召回链路统计（每路 recalled/aligned/final 计数 + 融合/rerank 计数 + chunk_id 截断列表）",
    )
    time_ms: Optional[float] = Field(
        None, description="工具执行耗时（毫秒）",
    )
    execution_model: Optional[str] = Field(
        None,
        description="工具内部调用的子模型（如 read_image_chunks 的 VLM）",
    )

    model_config = ConfigDict(extra="ignore")


class TokenUsageRecord(PydanticBaseModel):
    """单条 assistant 消息的 token 用量"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: Optional[int] = None
    total_tokens: int = 0
    # 前缀缓存命中数（命中率 = cached/prompt；session 级按 token 加权平均）
    cached_tokens: int = 0

    model_config = ConfigDict(extra="ignore")


# ==================== 主文档 ====================


class ChatMessage(BaseDocument):
    """
    对话消息文档

    role 与字段约定
    ---------------
    - ``role=user``      : 仅 ``content``（用户输入）
    - ``role=assistant`` : ``content`` + 可选 ``thinking`` + 可选 ``thinking_ms``
                           + 可选 ``tool_calls`` + 可选 ``citations``
                           + 可选 ``usage`` + ``finish_reason``
    - ``role=tool``      : ``content`` 为工具结果文本；
                           ``tool_call_id`` 必填，关联同会话内最近一条
                           assistant 消息中的某个 ``tool_calls[*].id``
    - ``role=system``    : 仅 ``content``（自定义 system prompt 落盘留底）

    主键
    ----
    ``id`` 为 ``chatmsg_<uuid>``；前缀方便日志中识别消息来源。
    """

    # ========== 主键（对齐项目其他 MongoDB 文档的字符串主键风格） ==========
    id: str = Field(
        ...,
        alias="_id",
        description="消息唯一标识（建议格式：chatmsg_<uuid>）",
    )

    # ========== 关联 ==========
    session_id: str = Field(
        ...,
        max_length=64,
        description="所属会话 ID（与 MySQL.chat_session.session_id 一致）",
    )

    user_id: str = Field(
        ...,
        max_length=64,
        description="所属用户 ID（冗余存储，便于跨会话审计）",
    )

    # ========== 角色与正文 ==========
    role: Literal["system", "user", "assistant", "tool", "summary"] = Field(
        ...,
        description="OpenAI 风格角色（summary 用于上下文压缩）",
    )

    content: str = Field(
        default="",
        description=(
            "消息正文。assistant 在仅 tool_calls 无文本时可为空字符串，"
            "tool 消息则填工具执行的文本结果"
        ),
    )

    # ========== assistant 专属 ==========
    thinking: Optional[str] = Field(
        None,
        description="思考链（仅 assistant；deepseek-reasoner 等模型给出时填）",
    )

    thinking_ms: Optional[float] = Field(
        None,
        description=(
            "本轮思考墙钟耗时（毫秒）。从首个 thinking.delta 计到 "
            "content / tool_call 开始或本轮结束；无思考则为 None。"
        ),
    )

    tool_calls: List[ToolCallRecord] = Field(
        default_factory=list,
        description="assistant 本轮发起的工具调用列表（持久化版）",
    )

    citations: List[Citation] = Field(
        default_factory=list,
        description="assistant 引用的命中片段（chunk_id + 元信息）",
    )

    usage: Optional[TokenUsageRecord] = Field(
        None,
        description="本条 assistant 消息对应的一次 LLM 推理用量",
    )

    finish_reason: Optional[str] = Field(
        None,
        description="LLM 结束原因（stop / length / tool_calls / content_filter / error）",
    )

    # ========== tool 专属 ==========
    tool_call_id: Optional[str] = Field(
        None,
        description=(
            "role=tool 时必填；指向同会话内某条 assistant 消息的 tool_calls[*].id"
        ),
    )

    # ========== 通用元数据 ==========
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据（model 名、provider、event_id、AB 实验标记 等）",
    )

    # ========== Pydantic 配置 ==========
    model_config = ConfigDict(
        populate_by_name=True,  # 允许字段名 / alias 互通
        arbitrary_types_allowed=True,
    )

    # ========== Beanie 配置 ==========
    class Settings:
        name = "chat_message"
        use_state_management = True
        validate_on_save = True
        indexes = [
            # 高频：拉某会话历史（按时间正序）
            IndexModel(
                [("session_id", ASCENDING), ("create_time", ASCENDING)],
                name="idx_session_create_time",
            ),
            # 审计：用户最近消息（按时间倒序）
            IndexModel(
                [("user_id", ASCENDING), ("create_time", DESCENDING)],
                name="idx_user_create_time",
            ),
            # 软删除过滤
            IndexModel(
                [("deleted", ASCENDING), ("create_time", DESCENDING)],
                name="idx_deleted_create_time",
            ),
        ]
