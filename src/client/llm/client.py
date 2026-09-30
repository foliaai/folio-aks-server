#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : client.py
@Author  : caixiongjiang
@Date    : 2026/04/21
@Function:
    LiteLLM 统一客户端封装，同时支持 LiteLLM Proxy 与 Model Lake（OpenAI 兼容网关）

    设计要点
    --------
    1. **薄封装、零业务**：LiteLLM 已经处理了 provider 路由、重试、token 统计；
       这里只负责把 ``components.json`` 的配置 + 业务 messages → ``litellm.acompletion``
       的入参，并把响应解析成本项目内部的 ``LLMResponse``。
    2. **统一一个 model 字符串**：完全采用 LiteLLM 的 ``"<provider>/<model>"`` 形式，
       例如 ``"deepseek/deepseek-chat"``、``"openai/gpt-4o-mini"``、
       ``"litellm_proxy/<virtual_name>"``（指向 LiteLLM Proxy），或 ``"openai/<model_lake_name>"``。
    3. **多网关无缝切换（LiteLLM Proxy / Model Lake）**：
       - LiteLLM Proxy 模式：裸名（如 ``deepseek-v4-flash``）自动归一化为 ``litellm_proxy/<name>``。
       - Model Lake 模式（OpenAI 兼容）：自动将模型归一化为 ``openai/<name>``，调用指定 endpoint。
       - 切换只需配置环境变量 ``MODEL_GATEWAY_TYPE``，无需多分支维护。
    4. **多模态原生支持**：``messages`` 里直接传 OpenAI 风格的 multi-content
       结构（``{"type":"text","text":...}`` / ``{"type":"image_url",...}``），
       LiteLLM 会负责按 provider 转换。
    5. **思考强度（pi 标准 7 档，统一经 LiteLLM 翻译）**：调用方传 ``reasoning_effort``
       字符串（pi 档位 ``off/minimal/low/medium/high/xhigh/max`` 经 registry 翻译后的厂商
       原生值，如 ``"high"`` / ``"max"``），由 LiteLLM / Proxy / ThinkingAdapter 转成原生参数。
    6. **观测**：用户运行的模型网关把日志写入数据库/网关端，本地客户端仅用 loguru 输出关键 metrics。
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.client.llm.types import (
    LLMResponse,
    MessageList,
    StreamChunk,
    TokenUsage,
    ToolCallDelta,
    ToolSchema,
    parse_litellm_response,
)

# 延迟 import litellm，避免启动期立即拉依赖
_LITELLM_INITIALIZED = False

# reasoning_effort 哨兵：区分「调用方未传」（沿用 cfg.default_reasoning_effort）
# 与「调用方显式传 None」（off → 不下发 reasoning_effort）。None 本身无法
# 区分这两种语义，故用哨兵作默认值。
_REASONING_UNSET = object()


def _ensure_litellm_initialized() -> None:
    """全局只跑一次：禁用 LiteLLM 的网络遥测、设默认日志级别。"""
    global _LITELLM_INITIALIZED
    if _LITELLM_INITIALIZED:
        return

    import litellm  # noqa: WPS433 (deferred import)

    litellm.suppress_debug_info = True
    litellm.set_verbose = False  # type: ignore[attr-defined]
    litellm.drop_params = False  # 关闭自动过滤，允许参数完整透传以便测试真实生效情况
    litellm.telemetry = False    # 关闭 LiteLLM 的匿名遥测
    litellm.modify_params = True  # 容许 LiteLLM 微调入参（例如 anthropic system 拼接）

    _LITELLM_INITIALIZED = True
    logger.debug("LiteLLM 全局初始化完成（telemetry=False, drop_params=False）")


# ==================== 客户端配置 ====================


class LLMClientConfig(BaseModel):
    """单个 LLM 客户端的运行时配置（由 ComponentConfigManager 构造）"""

    model: str = Field(
        ...,
        description="LiteLLM 模型字符串，形如 'deepseek/deepseek-chat'、'openai/gpt-4o-mini'、'litellm_proxy/qwen3.7-flash'",
    )
    api_base: Optional[str] = Field(
        None,
        description="覆盖 provider 默认 endpoint，模型网关（LiteLLM Proxy 或 Model Lake）时填写",
    )
    api_key: Optional[str] = Field(
        None,
        description="provider / proxy 的 API Key；为空则走环境变量",
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1)
    timeout: float = Field(60.0, gt=0.0)
    max_retries: int = Field(2, ge=0)
    default_reasoning_effort: Optional[str] = Field(
        None,
        description=(
            "本客户端的默认思考强度（已翻译为厂商原生 reasoning_effort 字符串，"
            "如 'high' / 'none'）。None=不主动声明（按上游默认）；"
            "单次调用可被 ``astream(reasoning_effort=...)`` 覆盖。"
            "由工厂 ``create_llm_client_from_preset`` 通过 "
            "``LiteLLMRegistry.resolve_reasoning_effort`` 翻译 pi 档位得到。"
        ),
    )
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="透传给 litellm.acompletion 的额外参数（例如 top_p / response_format）",
    )

    model_config = ConfigDict(extra="ignore")


# ==================== 客户端实现 ====================


class LLMClient:
    """LiteLLM 统一客户端

    生命周期
    --------
    - 进程内可创建多个实例（每个组件持有自己的实例），互不影响。
    - 不持有长连接，无需显式 close（保留 ``aclose()`` 供 explicit 清理）。
    """

    def __init__(self, config: LLMClientConfig) -> None:
        _ensure_litellm_initialized()
        self.config = config

    # ---- 兼容字段（旧代码读取 client.provider / client.model_name 等） ----
    @property
    def model_name(self) -> str:
        return self.config.model

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def provider(self) -> str:
        if "/" in self.config.model:
            return self.config.model.split("/", 1)[0]
        return ""

    @property
    def api_base(self) -> Optional[str]:
        return self.config.api_base

    @property
    def api_key(self) -> Optional[str]:
        return self.config.api_key

    # ---- 同步入口（仅保留必要场景，主流统一用 agenerate） ----
    def generate(
        self,
        messages: MessageList,
        *,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Any = _REASONING_UNSET,
        **kwargs: Any,
    ) -> LLMResponse:
        """同步请求，内部仍走 LiteLLM 的 ``completion``。"""
        import litellm

        params = self._build_params(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra=kwargs,
        )
        t0 = time.perf_counter()
        try:
            resp = litellm.completion(**params)
        except Exception as e:
            logger.error(f"[LLM] {self.config.model} sync generate 失败: {e}")
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        parsed = parse_litellm_response(resp)
        self._log_metrics("sync", parsed, elapsed_ms)
        return parsed

    # ---- 异步入口（业务主流） ----
    async def agenerate(
        self,
        messages: MessageList,
        *,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Any = _REASONING_UNSET,
        **kwargs: Any,
    ) -> LLMResponse:
        """异步请求。"""
        import litellm

        params = self._build_params(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra=kwargs,
        )
        t0 = time.perf_counter()
        try:
            resp = await litellm.acompletion(**params)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[LLM] {self.config.model} async generate 失败: {e}")
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        parsed = parse_litellm_response(resp)
        self._log_metrics("async", parsed, elapsed_ms)
        return parsed

    # ---- 流式（同步） ----
    def stream(
        self,
        messages: MessageList,
        *,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Any = _REASONING_UNSET,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """同步流式生成。支持 ``tools / tool_choice`` 透传，便于 Agent 模式
        在流式过程中也能拿到 ``tool_calls`` 增量。
        """
        import litellm

        params = self._build_params(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra=kwargs,
        )
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        stream = litellm.completion(**params)
        for chunk in stream:
            yield from _yield_stream_chunks(chunk)

    # ---- 流式（异步） ----
    async def astream(
        self,
        messages: MessageList,
        *,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Any = _REASONING_UNSET,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """异步流式生成。支持 ``tools / tool_choice`` 透传。

        在 ChatService 的 Agent 模式中，每轮都通过本方法拉流：
        - 文本增量按 ``StreamChunk(delta=..., is_thought=False)`` 透出
        - 思考链按 ``StreamChunk(delta=..., is_thought=True)`` 透出
        - 工具调用按 ``StreamChunk(tool_call_delta=ToolCallDelta(...))`` 透出
        - 流结束按 ``StreamChunk(finish_reason=..., delta="")`` 透出
        """
        import litellm

        params = self._build_params(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra=kwargs,
        )
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        resp = await litellm.acompletion(**params)
        async for chunk in resp:  # type: ignore[union-attr]
            for sc in _yield_stream_chunks(chunk):
                yield sc

    # ---- 资源清理 ----
    async def aclose(self) -> None:
        """LiteLLM 自身无需 close；保留方法便于上层统一调用。"""
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ---- 内部 ----
    def _build_params(
        self,
        messages: MessageList,
        *,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Any = _REASONING_UNSET,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        # 消息预处理：确保 assistant 带有 tool_calls 且无正文时 content 为 None（符合 OpenAI 标准协议，防止上游 Rust 网关反序列化报错）
        sanitized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                m_role = msg.get("role")
                if m_role == "assistant" and msg.get("tool_calls"):
                    if not msg.get("content"):
                        msg = {**msg, "content": None}
            sanitized_messages.append(msg)

        params: Dict[str, Any] = {
            "model": cfg.model,
            "messages": sanitized_messages,
            "temperature": temperature if temperature is not None else cfg.temperature,
            "max_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
            "timeout": cfg.timeout,
            "num_retries": cfg.max_retries,
        }
        if cfg.api_base:
            params["api_base"] = cfg.api_base
        if cfg.api_key:
            params["api_key"] = cfg.api_key

        if tools:
            params["tools"] = tools
            if tool_choice is not None:
                params["tool_choice"] = tool_choice

        if cfg.extra_params:
            for k, v in cfg.extra_params.items():
                params.setdefault(k, v)
        if extra:
            for k, v in extra.items():
                if v is None:
                    continue
                params[k] = v

        # 关闭自动过滤参数，允许自定义与思考强度参数完整透传至网关/上游
        params["drop_params"] = False

        # 允许扩展参数（如思考强度等）透传至 OpenAI 兼容网关（如 Model Lake），防止被 LiteLLM 本地校验拦截
        default_allowed = [
            "reasoning_effort",
            "thinking",
            "enable_thinking",
            "thinking_budget",
            "thinking_config",
        ]
        if "allowed_openai_params" in params and isinstance(params["allowed_openai_params"], list):
            params["allowed_openai_params"] = list(
                dict.fromkeys(params["allowed_openai_params"] + default_allowed)
            )
        else:
            params["allowed_openai_params"] = default_allowed

        # 思考强度参数适配（Thinking Adapter）：
        # 调用层 reasoning_effort 优先；显式 None = 不主动下发参数；
        # 哨兵 _REASONING_UNSET = 调用方未传，沿用 cfg.default_reasoning_effort。
        if reasoning_effort is _REASONING_UNSET:
            effective_effort = cfg.default_reasoning_effort
        elif reasoning_effort is None:
            effective_effort = None
        else:
            effective_effort = reasoning_effort

        if effective_effort is not None:
            from src.client.llm.thinking_adapter import (
                get_thinking_adapter,
                merge_thinking_params,
            )

            adapter = get_thinking_adapter(cfg.model)
            adapted = adapter.adapt(
                model=cfg.model,
                level_or_effort=effective_effort,
                max_tokens=params.get("max_tokens"),
            )
            merge_thinking_params(params, adapted)

        return params

    def _log_metrics(self, mode: str, resp: LLMResponse, elapsed_ms: float) -> None:
        usage = resp.usage
        logger.debug(
            "[LLM] {mode} {model} {elapsed:.0f}ms tokens={total} "
            "(prompt={p}, completion={c}, reasoning={r}) finish={fr} tools={n}",
            mode=mode,
            model=resp.model or self.config.model,
            elapsed=elapsed_ms,
            total=usage.total_tokens,
            p=usage.prompt_tokens,
            c=usage.completion_tokens,
            r=usage.thinking_tokens or 0,
            fr=resp.finish_reason,
            n=len(resp.tool_calls),
        )


# ==================== 工厂函数 ====================


def _proxy_defaults() -> Dict[str, Any]:
    """从 ``ConfigManager`` + ``EnvManager`` 读取 LLM 大模型网关默认配置。

    优先级：
      1) 组件 / preset 显式 ``api_base`` / ``api_key``
      2) ``ConfigManager.get_llm_gateway_full_config(env_manager)``：
         - ``gateway_type`` 取 ``.env: MODEL_GATEWAY_TYPE``（默认 litellm）
         - ``api_base`` 取 ``.env: MODEL_LAKE_BASE / LITELLM_PROXY_URL``
         - ``api_key``  取 ``.env: MODEL_LAKE_API_KEY / LITELLM_PROXY_KEY``
           （ModelNexus 网关为静态 mn-sk- Key，直接作 Bearer 使用）
         - ``timeout`` / ``max_retries`` 取 ``.env: MODEL_GATEWAY_TIMEOUT / MODEL_GATEWAY_MAX_RETRIES``
    单例失败时降级为返回空字典，避免阻断单元测试 / 离线场景。
    """
    try:
        from src.utils.config_manager import get_config_manager
        from src.utils.env_manager import get_env_manager

        return get_config_manager().get_llm_gateway_full_config(get_env_manager())
    except Exception as e:  # pragma: no cover - 配置缺失时不阻断
        logger.debug(f"读取模型网关默认配置失败，使用空默认值: {e}")
        return {}


_llm_gateway_defaults = _proxy_defaults


def _get_default_gateway_type() -> str:
    proxy = _proxy_defaults()
    return (proxy.get("gateway_type") or "litellm").strip().lower()


_LITELLM_PROXY_PREFIX = "litellm_proxy/"
_OPENAI_PREFIX = "openai/"


def _ensure_gateway_routable(model: str, gateway_type: Optional[str] = None) -> str:
    """对接 LiteLLM Proxy 或 Model Lake（OpenAI 兼容网关）的前缀归一化。

    - 当 gateway_type 为 'litellm'（默认）时：
      - 裸名（如 'deepseek-v4-flash'）补上 'litellm_proxy/' 前缀，告诉 LiteLLM SDK 走透传分支。
      - 已有 'litellm_proxy/' 或其他已知 provider 前缀保持不变。
    - 当 gateway_type 为 'model_lake' / 'openai' / 'openai_compatible' 时：
      - 任何模型（无论是裸名 'deepseek-v4-flash' 还是带有 'litellm_proxy/' 前缀的预设），
        统一归一化为 'openai/<bare_model>'，告诉 LiteLLM SDK 调用 OpenAI 兼容网关。
      - 已是 'openai/xxx' 的保持不变。
    """
    if not model:
        return model

    if not gateway_type:
        gateway_type = _get_default_gateway_type()

    gw = (gateway_type or "litellm").lower().strip()

    if gw in ("model_lake", "openai", "openai_compatible"):
        clean_model = model.strip()
        if clean_model.startswith(_LITELLM_PROXY_PREFIX):
            clean_model = clean_model[len(_LITELLM_PROXY_PREFIX):]
        if not clean_model.startswith(_OPENAI_PREFIX):
            return f"{_OPENAI_PREFIX}{clean_model}"
        return clean_model
    else:
        if "/" in model:
            return model
        return f"{_LITELLM_PROXY_PREFIX}{model}"


def _ensure_proxy_routable(model: str) -> str:
    """向后兼容接口，自动根据当前生效的网关类型归一化路由前缀。"""
    return _ensure_gateway_routable(model)


def create_llm_client(
    *,
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    default_reasoning_effort: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> LLMClient:
    """显式参数构造；通常由 ``ComponentConfigManager`` 调用。

    ``api_base`` / ``api_key`` / ``timeout`` / ``max_retries`` 未显式提供时，
    自动回落到 ``.env`` 的模型网关默认值。
    """
    proxy = _proxy_defaults()
    gw_type = proxy.get("gateway_type", "litellm")
    routable_model = _ensure_gateway_routable(model, gw_type)
    if routable_model != model:
        logger.debug(
            f"create_llm_client: '{model}' → '{routable_model}' (gateway={gw_type})"
        )
    cfg = LLMClientConfig(
        model=routable_model,
        api_base=api_base or proxy.get("api_base"),
        api_key=api_key or proxy.get("api_key"),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout if timeout is not None else float(proxy.get("default_timeout", 60.0)),
        max_retries=max_retries if max_retries is not None else int(proxy.get("default_max_retries", 2)),
        default_reasoning_effort=default_reasoning_effort,
        extra_params=extra_params or {},
    )
    return LLMClient(cfg)


def create_llm_client_from_model(
    *,
    model: str,
    chat_template_preset: str = "fast",
) -> LLMClient:
    """按"具体模型字符串 + 采样模板 preset"组装 LLMClient

    使用场景：用户在前端从 ``/api/chat/models`` 选了一个模型字符串
    （如 ``openai/deepseek-v4-flash`` 或 ``litellm_proxy/deepseek-v4-flash``），后端
    不再走 preset 的 ``model`` 字段，但仍希望复用 preset 里调好的
    ``temperature / max_tokens / extra_params`` 这些采样
    参数——这就是 ``chat_template_preset`` 的用途。

    注：思考强度由 ``ChatService`` 按用户当轮 ``thinking_level`` 解析后通过
    ``astream(reasoning_effort=...)`` 逐轮传入，不在此 preset 模板里固化。

    优先级：
        - ``model`` ← 入参（覆盖 preset.model）；自动根据当前网关类型归一化前缀
        - 其他字段 ← preset；preset 缺失 / 字段未设 → 走默认值

    ``api_base`` / ``api_key`` 始终走模型网关配置，与 ``create_llm_client_from_preset`` 一致。
    """
    from src.utils.config_manager import get_config_manager

    cm = get_config_manager()
    p = cm.get_llm_preset(chat_template_preset) or {}
    if not p:
        logger.warning(
            f"chat_template_preset '{chat_template_preset}' 未配置，"
            f"使用默认采样参数",
        )

    proxy = _proxy_defaults()
    gw_type = proxy.get("gateway_type", "litellm")
    routable_model = _ensure_gateway_routable(model, gw_type)
    if routable_model != model:
        logger.debug(
            f"create_llm_client_from_model: '{model}' → '{routable_model}' (gateway={gw_type})"
        )

    return create_llm_client(
        model=routable_model,
        api_base=p.get("api_base"),
        api_key=p.get("api_key"),
        temperature=p.get("temperature", 0.7),
        max_tokens=p.get("max_tokens", 2048),
        timeout=p.get("timeout"),
        max_retries=p.get("max_retries"),
        extra_params=p.get("extra_params") or {},
    )


def create_llm_client_from_preset(preset_name: str) -> LLMClient:
    """从 ``config/config.toml`` 的 ``[llm.presets.<preset_name>]`` 构造客户端。

    preset 字段约定：

    .. code-block:: toml

       [llm.presets.fast]
       model = "deepseek/deepseek-chat"
       temperature = 0.3
       max_tokens = 2048
       timeout = 60
       # 可选: thinking_level, api_base, api_key, max_retries, extra_params

    ``thinking_level`` 取 pi 标准 7 档之一（off/minimal/low/medium/high/xhigh/max），
    由 ``LiteLLMRegistry.resolve_reasoning_effort`` 翻译成厂商原生 reasoning_effort
    字符串后作为该客户端的默认思考强度；模型不支持思考或档位不合法时降级为 None。

    ``api_base`` / ``api_key`` 默认走模型网关配置；
    单个 preset 也可在自身字段中强制覆盖。
    """
    from src.utils.config_manager import get_config_manager

    cm = get_config_manager()
    p = cm.get_llm_preset(preset_name)
    if not p:
        presets = cm.get_llm_presets()
        available = ", ".join(sorted(presets.keys())) or "(empty)"
        raise ValueError(f"未知 LLM preset '{preset_name}'，可用: {available}")

    raw_model = p.get("model")
    if not raw_model:
        from src.utils.config_profile import resolve_config_profile
        raise ValueError(
            f"LLM preset '{preset_name}' 未绑定 model，"
            f"请在 config/profiles/{resolve_config_profile()}/models.toml 的 [presets] 中配置"
        )
    proxy = _proxy_defaults()
    gw_type = proxy.get("gateway_type", "litellm")
    routable_model = _ensure_gateway_routable(raw_model, gw_type)
    if routable_model != raw_model:
        logger.debug(
            f"create_llm_client_from_preset[{preset_name}]: "
            f"'{raw_model}' → '{routable_model}' (gateway={gw_type})"
        )

    # pi 档位 → 厂商原生 reasoning_effort 字符串（模型不支持思考时返回 None）
    default_reasoning_effort: Optional[str] = None
    thinking_level = p.get("thinking_level")
    if thinking_level:
        try:
            from src.client.llm.registry import get_litellm_registry
            default_reasoning_effort = get_litellm_registry().resolve_reasoning_effort(
                routable_model, str(thinking_level),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(
                f"create_llm_client_from_preset[{preset_name}] 解析 thinking_level "
                f"'{thinking_level}' 失败，忽略: {e}"
            )

    return create_llm_client(
        model=routable_model,
        api_base=p.get("api_base"),
        api_key=p.get("api_key"),
        temperature=p.get("temperature", 0.7),
        max_tokens=p.get("max_tokens", 2048),
        timeout=p.get("timeout"),
        max_retries=p.get("max_retries"),
        default_reasoning_effort=default_reasoning_effort,
        extra_params=p.get("extra_params") or {},
    )


# ==================== 流式辅助 ====================


def _yield_stream_chunks(chunk: Any) -> Iterator[StreamChunk]:
    """把 LiteLLM 流式输出的一个 chunk 转换为若干 StreamChunk。

    解析的 4 类增量（同一个 LiteLLM chunk 内可能同时出现）：

    1. ``delta.content``           → 正文增量（``is_thought=False``）
    2. ``delta.reasoning_content`` → 思考链增量（``is_thought=True``）
    3. ``delta.tool_calls[*]``     → 工具调用增量（``tool_call_delta`` 非空）
    4. ``finish_reason``           → 流结束信号

    工具调用增量的兼容性说明：
        - OpenAI / DeepSeek / 国产 Qwen 等：按 ``index`` 分多块，首块带 ``id``
          与 ``function.name``，后续块只带 ``function.arguments`` 字符串增量；
        - Anthropic / 部分供应商：一次性给完整 tool_call（首块即包含完整
          ``arguments``）；本函数对两种形态均做规范化输出，调用方只需按
          ``index`` 聚合即可拿到完整 ``ToolCallDelta`` 序列。
    """
    try:
        data = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
    except Exception:
        return
    choices = data.get("choices") or []
    model = data.get("model")

    # OpenAI / LiteLLM 在 stream_options.include_usage=True 下，会在流末尾追发
    # 一个 choices=[] 但带顶层 usage 的尾块。这里把它独立透出，让
    # StreamAccumulator.finalize() 能拿到真实 token 计数（否则展示成全 0）。
    usage_raw = data.get("usage") or {}
    if usage_raw:
        completion_details = usage_raw.get("completion_tokens_details") or {}
        thinking_tokens = completion_details.get("reasoning_tokens")
        usage_obj = TokenUsage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            thinking_tokens=(
                int(thinking_tokens) if thinking_tokens is not None else None
            ),
            total_tokens=int(usage_raw.get("total_tokens") or 0),
        )
        yield StreamChunk(
            delta="", is_thought=False, finish_reason=None, model=model,
            usage=usage_obj,
        )

    if not choices:
        return
    ch = choices[0]
    delta = ch.get("delta") or {}
    finish = ch.get("finish_reason")

    text = delta.get("content")
    if text:
        yield StreamChunk(delta=text, is_thought=False, finish_reason=None, model=model)

    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
    if reasoning:
        yield StreamChunk(delta=reasoning, is_thought=True, finish_reason=None, model=model)

    raw_tool_calls = delta.get("tool_calls") or []
    for tc in raw_tool_calls:
        try:
            tcd = _parse_tool_call_delta(tc)
        except Exception:  # noqa: BLE001 — 单个增量解析失败不阻断整流
            continue
        if tcd is None:
            continue
        yield StreamChunk(
            delta="",
            is_thought=False,
            tool_call_delta=tcd,
            finish_reason=None,
            model=model,
        )

    if finish:
        yield StreamChunk(delta="", is_thought=False, finish_reason=finish, model=model)


def _parse_tool_call_delta(tc: Any) -> Optional[ToolCallDelta]:
    """把 LiteLLM 的单个 tool_call 增量字典转换为 ``ToolCallDelta``。

    返回 ``None`` 表示该增量无任何可用信息（例如完全为空的 placeholder）。
    """
    if not isinstance(tc, dict):
        return None

    index = tc.get("index")
    if index is None:
        # 部分供应商在非流式 tool_calls 拼回时可能不带 index，按 0 兜底
        index = 0
    try:
        index_int = int(index)
    except (TypeError, ValueError):
        return None
    if index_int < 0:
        return None

    fn = tc.get("function") or {}
    name = fn.get("name")
    args = fn.get("arguments")

    # 仅当至少有一个有效字段时才产出
    if (
        tc.get("id") is None
        and not name
        and (args is None or args == "")
    ):
        return None

    return ToolCallDelta(
        index=index_int,
        id=tc.get("id"),
        name=name if name else None,
        arguments_delta=args if isinstance(args, str) and args else None,
    )
