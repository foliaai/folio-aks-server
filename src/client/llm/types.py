#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : types.py
@Author  : caixiongjiang
@Date    : 2026/04/21
@Function:
    LLM Client 数据结构定义（LiteLLM 适配版本）

    设计原则:
        - 仅保留业务真正消费的字段，避免复刻一份 LiteLLM 已有的复杂结构
        - 与原 ``LLMResponse`` 字段名保持兼容（content / usage / model /
          finish_reason / thinking）以最小化业务侧改动
        - 输入侧统一使用 OpenAI 风格 message dict，由 LiteLLM 直接消费，
          因此不再需要 Pydantic Message 类
        - 工具调用字段直接复用 OpenAI/LiteLLM 的 ``tool_calls`` 形态
@Modify History:
    2026/04/21 - 迁移至 LiteLLM，删除 ContentPart/Message/StreamChunk 等冗余类型
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ==================== 输出结构 ====================


class ThinkingContent(BaseModel):
    """模型推理 / 思考内容（DeepSeek-Reasoner / Gemini Thinking 等）"""

    reasoning: str = Field(..., description="模型显式思考链原文")
    tokens_used: Optional[int] = Field(
        None, ge=0, description="思考阶段消耗的 token 数（若上游给出）",
    )

    model_config = ConfigDict(extra="ignore")


class TokenUsage(BaseModel):
    """Token 使用统计（与 OpenAI/LiteLLM 字段一致）"""

    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    thinking_tokens: Optional[int] = Field(None, ge=0)
    total_tokens: int = Field(0, ge=0)
    # 前缀缓存命中数：model_lake/litellm 均透传 prompt_tokens_details.cached_tokens；
    # 兜底 DeepSeek 原生 prompt_cache_hit_tokens。命中率 = cached/prompt
    cached_tokens: int = Field(0, ge=0)

    model_config = ConfigDict(extra="ignore")


class ToolCall(BaseModel):
    """OpenAI 风格的工具调用结构（透传 LiteLLM 返回的 tool_calls）"""

    id: str = Field(..., description="LLM 给出的本次调用 ID，回填 ToolMessage 用")
    name: str = Field(..., description="工具函数名")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="参数；LiteLLM 给出的是 JSON 字符串，本字段已解析为 dict",
    )

    model_config = ConfigDict(extra="ignore")


class LLMResponse(BaseModel):
    """统一 LLM 响应结构

    与旧客户端字段保持兼容；新增 ``tool_calls`` 直通 LiteLLM 工具调用结构。
    """

    content: str = Field("", description="模型主文本输出（无 tool_calls 时即最终回答）")
    thinking: Optional[ThinkingContent] = Field(
        None,
        description="思考链（DeepSeek-Reasoner 等模型；普通模型为 None）",
    )
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="若模型本轮发起了工具调用，按 OpenAI 协议透传到此字段",
    )
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = Field(..., description="LiteLLM 实际路由到的模型字符串")
    finish_reason: Literal[
        "stop", "length", "tool_calls", "content_filter", "error",
    ] = Field("stop")
    raw: Dict[str, Any] = Field(
        default_factory=dict,
        description="LiteLLM 原始响应（已 model_dump 为 dict，调试用）",
    )

    model_config = ConfigDict(extra="ignore")

    def __str__(self) -> str:
        return (
            f"LLMResponse(model={self.model}, "
            f"tokens={self.usage.total_tokens}, "
            f"finish={self.finish_reason}, "
            f"tool_calls={len(self.tool_calls)})"
        )


# ==================== 流式输出 ====================


class ToolCallDelta(BaseModel):
    """流式输出中工具调用的一个增量块

    OpenAI / LiteLLM 流式协议：``delta.tool_calls`` 是一个数组，每个元素带 ``index``
    标识本回复内的第几个工具调用；首个增量块会带上 ``id`` 与 ``function.name``，
    后续增量块只携带 ``function.arguments`` 的字符串增量。

    Anthropic / 部分供应商可能一次性给出完整 tool_call，本类型同样兼容
    （``arguments_delta`` 直接为完整 JSON 字符串即可）。
    """

    index: int = Field(..., ge=0, description="本回复中工具调用的序号")
    id: Optional[str] = Field(
        None, description="工具调用 ID；通常仅首块携带",
    )
    name: Optional[str] = Field(
        None, description="工具函数名；通常仅首块携带",
    )
    arguments_delta: Optional[str] = Field(
        None,
        description="参数 JSON 字符串增量；多块拼接后才是完整 JSON",
    )

    model_config = ConfigDict(extra="ignore")


class StreamChunk(BaseModel):
    """流式输出的一个增量块

    五种互斥语义（同一块只命中其一）：

    1. **正文增量**：``delta`` 非空、``is_thought=False``、``tool_call_delta=None``
    2. **思考增量**：``delta`` 非空、``is_thought=True``
    3. **工具调用增量**：``tool_call_delta`` 非空（``delta`` 为空字符串）
    4. **流结束信号**：``finish_reason`` 非 None（``delta`` 为空字符串）
    5. **usage 块**：``usage`` 非 None（OpenAI / LiteLLM 在 ``stream_options.include_usage=True``
       时，会在流结束后再发一个 ``choices=[]`` 但携带顶层 ``usage`` 的尾块；
       本字段用来透传该尾块，便于 ``StreamAccumulator.finalize()`` 拿到真实
       token 计数，而不是退化为全 0）
    """

    delta: str = Field("", description="本块新增的文本内容（正文或思考）")
    is_thought: bool = Field(False, description="是否为 reasoning 块（思考链）")
    tool_call_delta: Optional[ToolCallDelta] = Field(
        None,
        description="工具调用增量；非 None 时表示本块属于 tool_calls 流",
    )
    finish_reason: Optional[str] = Field(None, description="非 None 即流结束")
    usage: Optional[TokenUsage] = Field(
        None,
        description="流尾的 usage 块（来自 stream_options.include_usage）",
    )
    model: Optional[str] = Field(None)

    model_config = ConfigDict(extra="ignore")


# ==================== 类型别名 ====================

#: 与 LiteLLM/OpenAI 完全一致的 messages 入参形态
MessageDict = Dict[str, Any]
MessageList = List[MessageDict]

#: tool schema dict（OpenAI function-calling tools 协议）
ToolSchema = Dict[str, Any]


__all__ = [
    "ThinkingContent",
    "TokenUsage",
    "ToolCall",
    "LLMResponse",
    "StreamChunk",
    "ToolCallDelta",
    "MessageDict",
    "MessageList",
    "ToolSchema",
]


def parse_litellm_response(resp: Any) -> LLMResponse:
    """把 ``litellm.acompletion`` / ``litellm.completion`` 的返回对象解析为 ``LLMResponse``

    LiteLLM 的返回是 OpenAI 风格的 ``ModelResponse``，``choices[0].message`` 上
    可能挂以下扩展字段（按厂商不同）：

    - ``content``: 主文本
    - ``reasoning_content``: 思考链（DeepSeek-Reasoner、部分 Anthropic、Gemini Thinking 等）
    - ``tool_calls``: 函数调用列表
    - ``usage``: token 统计；DeepSeek-Reasoner 在 ``completion_tokens_details.reasoning_tokens``
    """
    import json as _json

    raw_dump: Dict[str, Any]
    try:
        raw_dump = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    except Exception:
        raw_dump = {"_repr": repr(resp)[:8000]}

    choices = raw_dump.get("choices") or []
    msg = (choices[0] if choices else {}).get("message", {}) or {}
    content = msg.get("content") or ""

    # ---- 思考链 ----
    reasoning_text = msg.get("reasoning_content") or msg.get("reasoning") or ""
    thinking_obj: Optional[ThinkingContent] = None
    if reasoning_text:
        thinking_obj = ThinkingContent(reasoning=reasoning_text)

    # ---- 工具调用 ----
    raw_tool_calls = msg.get("tool_calls") or []
    parsed_tool_calls: List[ToolCall] = []
    for tc in raw_tool_calls:
        try:
            fn = (tc or {}).get("function") or {}
            args_raw = fn.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args = _json.loads(args_raw) if args_raw.strip() else {}
                except Exception:
                    args = {"_raw": args_raw}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            parsed_tool_calls.append(
                ToolCall(
                    id=str(tc.get("id") or ""),
                    name=str(fn.get("name") or ""),
                    arguments=args,
                ),
            )
        except Exception:  # noqa: BLE001
            continue

    # ---- usage ----
    usage_raw = raw_dump.get("usage") or {}
    completion_details = usage_raw.get("completion_tokens_details") or {}
    thinking_tokens = completion_details.get("reasoning_tokens")
    if thinking_tokens is None and thinking_obj is not None:
        thinking_tokens = None
    usage = TokenUsage(
        prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
        completion_tokens=int(usage_raw.get("completion_tokens") or 0),
        thinking_tokens=int(thinking_tokens) if thinking_tokens is not None else None,
        total_tokens=int(usage_raw.get("total_tokens") or 0),
    )
    if thinking_obj is not None and thinking_tokens is not None:
        thinking_obj.tokens_used = int(thinking_tokens)

    finish_reason = (choices[0] if choices else {}).get("finish_reason") or "stop"
    finish_reason = _normalize_finish_reason(finish_reason)

    return LLMResponse(
        content=content,
        thinking=thinking_obj,
        tool_calls=parsed_tool_calls,
        usage=usage,
        model=str(raw_dump.get("model") or ""),
        finish_reason=finish_reason,
        raw=raw_dump,
    )


def _normalize_finish_reason(raw: Union[str, None]) -> str:
    """LiteLLM 的 finish_reason 在不同厂商有 ``end_turn``、``MAX_TOKENS`` 等差异。"""
    if not raw:
        return "stop"
    s = str(raw).lower()
    if s in ("end_turn", "stop"):
        return "stop"
    if s in ("max_tokens", "length"):
        return "length"
    if s in ("tool_use", "function_call", "tool_calls"):
        return "tool_calls"
    if s in ("content_filter", "safety", "blocked"):
        return "content_filter"
    if s in ("error",):
        return "error"
    return "stop"
