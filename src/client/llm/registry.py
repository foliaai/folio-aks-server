#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : registry.py
@Author  : caixiongjiang
@Date    : 2026/05/19
@Function:
    模型注册中心（支持 LiteLLM Proxy 与 Model Lake / OpenAI 兼容网关）

    负责把模型网关的 ``/v1/models`` 暴露给业务代码 / 前端，做以下工作：

    1. **拉取真相源**：调用 ``<gateway_base>/v1/models``。LiteLLM 的
       ``/v1/model/info`` 只补上下文长度，不再按 ``mode`` 过滤。
    2. **前端可见性**：litellm 档案用 ``visible`` 白名单；model_lake（ModelNexus）
       从 ``/v1/models`` 富元数据（OpenRouter 风格模型卡片）**自动合成**能力声明
       ——思考开关/档位、多模态、上下文长度，默认全部可见，仅受 ``[hidden]``
       黑名单与 ``deprecated_at`` 过滤；``[presets]`` 只服务后台组件并做存在性校验。
       合成的思考声明会整体替换运行时翻译字典（``resolve_reasoning_effort``
       等接口的数据源），前端展示与后端翻译天然同源。
    3. **TTL 缓存**：5 分钟内复用上一次结果；网关失败时复用上次成功结果，
       否则返回空列表，不编造兜底模型。

    本模块只做"读模型清单"。具体调用模型（chat / embedding / rerank）依然
    走 ``LLMClient`` / ``EmbeddingClient`` / ``RerankerClient``——它们各
    自把 ``api_base`` / ``api_key`` 注入给底层 SDK。
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 思考强度（pi 风格的标准 7 档词汇）
# ============================================================

# pi 的标准档位顺序（从低到高）。所有模型共用这套词汇，每模型通过
# ``thinkingLevelMap`` 声明各自支持哪些档位、以及档位到厂商原生字符串的映射。
# 详见 config/thinking_models.json。
EXTENDED_THINKING_LEVELS: List[str] = [
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]

# ``off`` 之外的可选档位（用于 resolve 时区分 off 与其余档位的缺省语义）
THINKING_LEVELS_NO_OFF: List[str] = [
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]


class ThinkingModelSpec(BaseModel):
    """``thinking_models.json`` 中单个模型的思考声明。

    - ``reasoning``：是否支持思考链。
    - ``supports_thinking_effort``：是否支持思考强度。false 时只开关，忽略 map。
    - ``default`` / ``thinking_level_map``：仅强度模型使用。
    """

    reasoning: bool = True
    supports_thinking_effort: bool = False
    default: str = "medium"
    thinking_level_map: Dict[str, Optional[str]] = Field(
        default_factory=dict, alias="thinkingLevelMap",
    )

    model_config = ConfigDict(
        extra="ignore", protected_namespaces=(), populate_by_name=True,
    )

    def resolve_reasoning_effort(self, level: str) -> Optional[str]:
        """把 pi 标准档位翻译成要下发给 LiteLLM 的 ``reasoning_effort`` 字符串。

        返回 ``None`` 表示"不下发该参数"；返回字符串表示透传。调用方应先用
        ``clamp_thinking_level`` 把 level 归位到该模型支持的档位。

        翻译规则（统一走 ``thinking_level_map``，off 不特判）：
        - ``off`` → ``thinking_level_map["off"]``：字符串（如 ``"none"``）透传，
          显式关闭思考链；缺省 → ``None``（不下发，让模型按默认）；
          显式 ``null`` = 不能关思考（``clamp`` 已归位，不会到达）。
        - ``minimal..high`` → 字符串透传；缺省 → 透传档位名本身。
        - ``xhigh/max`` → 同上（``clamp`` 已保证有非空字符串映射）。
        """
        if not self.reasoning:
            return None
        if not self.supports_thinking_effort:
            return "none" if level == "off" else "enabled"
        if level not in self.thinking_level_map:
            return None if level == "off" else level
        mapped = self.thinking_level_map[level]
        if mapped is None:
            return None  # 显式 null = 不支持（防御式）
        return mapped


def get_supported_thinking_levels(spec: Optional[ThinkingModelSpec]) -> List[str]:
    """根据 ``ThinkingModelSpec`` 计算该模型支持的档位列表（保持 EXTENDED 顺序）。

    - 无 spec 或 ``reasoning=False`` → ``["off"]``。
    - ``supports_thinking_effort=False`` → ``["off", "medium"]``（前端画关/开）。
    - 强度模型按 ``thinking_level_map``：非空字符串支持；``null`` 跳过；
      缺省档默认支持，但 ``xhigh``/``max`` 必须显式声明。
    """
    if spec is None or not spec.reasoning:
        return ["off"]
    if not spec.supports_thinking_effort:
        return ["off", "medium"]

    out: List[str] = []
    for level in EXTENDED_THINKING_LEVELS:
        if level in spec.thinking_level_map:
            mapped = spec.thinking_level_map[level]
            if mapped is None:
                # 显式 null → 不支持（含 off:null 表示不能关思考）
                continue
            out.append(level)
        else:
            # 缺省：xhigh/max 是 opt-in（缺省不支持）；其余档位默认支持
            if level in ("xhigh", "max"):
                continue
            out.append(level)
    return out


def clamp_thinking_level(spec: Optional[ThinkingModelSpec], level: str) -> str:
    """把任意请求档位归位到该模型实际支持的最近档位（pi ``clampThinkingLevel``）。

    先向上（更高强度）找，再向下（更低强度）找；都找不到回落到 ``"off"``。
    """
    available = get_supported_thinking_levels(spec)
    if level in available:
        return level
    idx = EXTENDED_THINKING_LEVELS.index(level) if level in EXTENDED_THINKING_LEVELS else -1
    if idx >= 0:
        for i in range(idx, len(EXTENDED_THINKING_LEVELS)):
            if EXTENDED_THINKING_LEVELS[i] in available:
                return EXTENDED_THINKING_LEVELS[i]
        for i in range(idx - 1, -1, -1):
            if EXTENDED_THINKING_LEVELS[i] in available:
                return EXTENDED_THINKING_LEVELS[i]
    return available[0] if available else "off"


# ============================================================
# 输出结构
# ============================================================


class LLMModelInfo(BaseModel):
    """前端可见的最小模型描述

    设计原则：**只暴露用于渲染下拉的字段**。能力 / 价格 / 内部 alias 不出现，
    避免把网关的内部命名规则泄露给客户端。
    """

    id: str = Field(
        ...,
        description="模型标识字符串（如 'litellm_proxy/qwen3.7-flash' 或 'openai/deepseek-v4-flash'），同时是 API 入参",
    )
    label: str = Field(..., description="UI 上显示的友好名称（去掉 provider/网关 前缀）")
    provider: str = Field(..., description="provider 名（用于前端按 provider 分组）")
    supports_thinking: bool = Field(
        default=False,
        description="模型是否支持思考链 / reasoning（来自 config/thinking_models.json 白名单）",
    )
    thinking_levels: List[str] = Field(
        default_factory=list,
        description=(
            "该模型支持的思考档位。"
            "不支持思考：['off']；只开关：['off', 'medium']；支持强度：map 声明的档位。"
        ),
    )
    default_thinking_level: Optional[str] = Field(
        default=None,
        description=(
            "新建会话时默认选中的档位（必须在 thinking_levels 内）。"
            "supports_thinking=False 时为 None。"
        ),
    )
    supports_multimodal: bool = Field(
        default=False,
        description="模型是否支持多模态读图（来自 config/multimodal_models.json 白名单）",
    )
    max_context: Optional[int] = Field(
        default=None,
        description=(
            "模型自身声明的最大上下文长度（tokens）：优先取 "
            "config/long_context_models.json，其次取网关 /v1/model/info 的 "
            "max_input_tokens；None 表示两处都未提供。"
            "注意这是模型能力值，实际参与预算的窗口是 "
            "min(该值, [chat.context] max_context_cap)，见 ModelContextCatalog。"
        ),
    )

    # 非公开：档位 → 厂商原生 reasoning_effort 字符串的映射，供
    # ``resolve_reasoning_effort`` 在后端调用 LLM 前做翻译。不暴露给前端。
    thinking_level_map: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="档位到厂商原生 reasoning_effort 的映射（内部用，不下发前端）",
    )

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    def resolve_reasoning_effort(self, level: str) -> Optional[str]:
        """把 pi 标准档位翻译成要下发给 LiteLLM 的 ``reasoning_effort`` 字符串。

        返回 ``None`` 表示"不下发该参数"；返回字符串表示透传该字符串。调用方应
        先用 ``clamp_thinking_level`` 把 level 归位到该模型支持的档位，再调用本方法。

        翻译规则（统一走 ``thinking_level_map``，off 不再特判 None）：
        - ``off`` → ``thinking_level_map["off"]``：字符串（如 ``"none"``）则透传，
          显式下发以关闭思考链；缺省则返回 ``None``（不下发，让模型按默认）；
          显式 ``null`` 表示该模型不能关思考（``clamp`` 已归位，不会到达）。
        - ``minimal..high`` → ``thinking_level_map[level]``：字符串透传；缺省则
          透传档位名本身（如 ``"medium"``）。
        - ``xhigh/max`` → 同上（``clamp`` 已保证 ``map`` 有非空字符串映射）。
        - 不支持的档位（map 显式 null 且被传入）→ 返回 ``None``（防御式）。
        """
        if not self.supports_thinking:
            return None
        if level not in self.thinking_level_map:
            # 缺省：off → None（不下发）；minimal..high → 透传档位名本身
            return None if level == "off" else level
        mapped = self.thinking_level_map[level]
        if mapped is None:
            # 显式 null = 不支持（clamp 已归位，正常不到达；防御式返回 None）
            return None
        return mapped  # 字符串透传（含 off 的 "none" 或厂商原生 "max"）


# ============================================================
# 注册中心
# ============================================================


class LiteLLMRegistry:
    """LiteLLM / Model Lake 模型清单的进程内注册中心

    生命周期
    --------
    通常作为模块级单例使用（见 ``get_litellm_registry()``）。线程安全：内部
    用一把简单锁保护 cache，缓存命中读不加锁，未命中拉网关时序列化。
    """

    DEFAULT_TTL_SECONDS = 300  # 5 min
    HTTP_TIMEOUT_SECONDS = 5.0

    @staticmethod
    def _capability_path(filename: str) -> Path:
        from src.utils.config_profile import resolve_profile_file
        return resolve_profile_file(filename)

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = max(0, int(ttl_seconds))
        self._lock = threading.Lock()
        self._cache: Optional[List[LLMModelInfo]] = None
        self._cache_at: float = 0.0
        # 失败兜底缓存（不参与 TTL，刷新失败时降级用）
        self._fallback_cache: List[LLMModelInfo] = []
        # thinking 模型白名单（从 config/thinking_models.json 加载）
        # thinking 模型声明（从 config/thinking_models.json 加载：bare_name -> spec）
        self._thinking_models: Dict[str, ThinkingModelSpec] = self._load_thinking_models()
        # 兼容旧调用方：裸名集合（supports_thinking 判断用）
        self._thinking_names: Set[str] = set(self._thinking_models.keys())
        # multimodal 模型白名单（从 config/multimodal_models.json 加载）
        self._multimodal_models: Set[str] = self._load_multimodal_models()
        # 长上下文模型声明（从 config/long_context_models.json 加载：model -> max_context tokens）
        self._long_context_map: Dict[str, int] = self._load_long_context_models()

    # ---- 公共 API ----

    def list_models(self, *, force_refresh: bool = False) -> List[LLMModelInfo]:
        """返回当前可见的 chat 模型清单。

        Args:
            force_refresh: 强制忽略 TTL 立即拉一次网关

        Returns:
            当前档案 ``visible`` 白名单与网关实际库存的交集；
            网关不可达时复用上次成功结果，否则返回空列表。
        """
        if not force_refresh and self._is_fresh():
            assert self._cache is not None
            return list(self._cache)

        with self._lock:
            # double-check：可能已有别的线程刷新过了
            if not force_refresh and self._is_fresh():
                assert self._cache is not None
                return list(self._cache)

            try:
                fresh = self._fetch_from_proxy()
                self._cache = fresh
                self._cache_at = time.monotonic()
                self._fallback_cache = list(fresh)  # 同时刷新兜底
                logger.info(f"[LiteLLMRegistry] 刷新成功，共 {len(fresh)} 个可见模型")
                return list(fresh)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[LiteLLMRegistry] 拉取 /v1/models 失败: {e}")
                if self._fallback_cache:
                    return list(self._fallback_cache)
                self._cache = []
                self._cache_at = time.monotonic() - max(0, self._ttl - 30)
                return []

    def peek_max_context(self, bare_name: str) -> Optional[int]:
        """从**已有缓存**里查模型声明的上下文长度；缓存为空时返回 None。

        专供 ``ModelContextCatalog`` 在对话主链路上调用：绝不触发网关拉取，
        避免把 5s 超时的 HTTP 往返引入每轮预算计量。
        """
        cache = self._cache or self._fallback_cache
        if not cache:
            return None
        target = (bare_name or "").strip()
        if not target:
            return None
        for m in cache:
            if self._bare_model_name(m.id) == target and m.max_context:
                return int(m.max_context)
        return None

    def peek_thinking_spec(self, bare_name: str) -> Optional[ThinkingModelSpec]:
        """从配置里查模型的思考强度声明（不依赖网关缓存，不触发拉取）。

        供 ``ChatService`` 在调用 LLM 前把用户选的 ``thinking_level`` 翻译成
        ``reasoning_effort`` 字符串。入参可以是最后一段裸名、``channel/model``
        路由，或带 ``litellm_proxy/`` / ``openai/`` 前缀的 SDK id；找不到返回
        ``None``（模型不支持思考）。
        """
        return self._lookup_thinking_spec(self._thinking_models, bare_name)

    def resolve_reasoning_effort(
        self, model: str, thinking_level: str
    ) -> Optional[str]:
        """把 pi 标准档位翻译成要下发给 LiteLLM 的 ``reasoning_effort`` 字符串。

        入参 ``model`` 可以是裸名、``channel/model`` 路由，或带 ``litellm_proxy/``
        / ``openai/`` 前缀的 SDK id。流程：
        1. 按多种命名形态查 ``ThinkingModelSpec``；
        2. 命中则 ``clamp`` 后映射为厂商 ``reasoning_effort``；
        3. 未命中时：用户档位为 off → ``None``（不下发，避免部分厂商
           ``reasoning_effort='none'`` 禁用工具）；用户打开了思考 → 原样透传，
           交给 ``ThinkingAdapter`` 按模型名生成参数。Model Lake 的
           ``openai/channel/model`` 若档案键对不上，不能把开关/强度静默丢掉。

        返回 ``None`` 表示"不下发 ``reasoning_effort``"（档位为 off，或模型
        未声明思考且用户也未打开）。
        """
        spec = self._lookup_thinking_spec(self._thinking_models, model)
        if spec is None or not spec.reasoning:
            raw = (thinking_level or "").strip()
            if raw.lower() in ("", "off", "none"):
                return None
            logger.warning(
                "[LiteLLMRegistry] 思考声明未命中，按用户档位透传给 ThinkingAdapter: "
                f"model={model!r} level={raw!r}"
            )
            return raw
        clamped = clamp_thinking_level(spec, thinking_level)
        return spec.resolve_reasoning_effort(clamped)

    def clamp_thinking_level(self, model: str, thinking_level: str) -> str:
        """把请求档位归位到该模型实际支持的最近 pi 档位（返回标准档位名，非 effort 字符串）。

        档案未命中时不再强制 ``"off"``：用户打开的档位原样保留，避免 Model Lake
        路由 id 对不上声明键时，前端开关/强度被后端静默关掉。
        """
        spec = self._lookup_thinking_spec(self._thinking_models, model)
        if spec is None or not spec.reasoning:
            raw = (thinking_level or "").strip() or "off"
            if raw.lower() not in ("off", "none"):
                logger.warning(
                    "[LiteLLMRegistry] 思考声明未命中，保留用户档位: "
                    f"model={model!r} level={raw!r}"
                )
            return "off" if raw.lower() == "none" else raw
        return clamp_thinking_level(spec, thinking_level)

    def invalidate(self) -> None:
        """清空缓存；下一次 ``list_models`` 强制刷新。"""
        with self._lock:
            self._cache = None
            self._cache_at = 0.0

    # ---- 内部 ----

    @classmethod
    def _load_thinking_models(cls) -> Dict[str, ThinkingModelSpec]:
        """从 ``config/thinking_models.json`` 加载思考强度声明。

        格式::

            {
              "models": {
                "deepseek-v4-pro": {
                  "reasoning": true,
                  "supports_thinking_effort": true,
                  "default": "high",
                  "thinkingLevelMap": {"off": "none", "low": "low", "high": "high", "max": "max"}
                },
                "glm-5.1": {
                  "reasoning": true,
                  "supports_thinking_effort": false
                }
              }
            }

        旧裸数组 ``{ "models": ["glm-5.1", ...] }`` 升级为只支持思考开关。

        返回 ``{bare_name: ThinkingModelSpec}``；文件不存在或解析失败时返回空字典
        （降级为全部不支持思考）。
        """
        path = LiteLLMRegistry._capability_path("thinking_models.json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.debug(f"[LiteLLMRegistry] thinking 声明文件不存在: {path}")
            return {}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[LiteLLMRegistry] 加载 thinking 声明失败: {e}")
            return {}

        raw = data.get("models")
        result: Dict[str, ThinkingModelSpec] = {}
        if isinstance(raw, dict):
            for name, spec in raw.items():
                if not isinstance(name, str) or not name.strip():
                    continue
                if isinstance(spec, dict):
                    tlm = spec.get("thinkingLevelMap") or spec.get("thinking_level_map") or {}
                    norm_map: Dict[str, Optional[str]] = {}
                    for k, v in (tlm or {}).items():
                        if not isinstance(k, str):
                            continue
                        norm_map[k] = v if (v is None or isinstance(v, str)) else None
                    result[name.strip()] = ThinkingModelSpec(
                        reasoning=bool(spec.get("reasoning", True)),
                        supports_thinking_effort=bool(
                            spec.get("supports_thinking_effort", False)
                        ),
                        default=str(spec.get("default", "medium")),
                        thinking_level_map=norm_map,
                    )
                else:
                    result[name.strip()] = ThinkingModelSpec()
        elif isinstance(raw, list):
            # 旧裸数组格式：升级为默认 spec
            for m in raw:
                if isinstance(m, str) and m.strip():
                    result[m.strip()] = ThinkingModelSpec()
        logger.debug(f"[LiteLLMRegistry] 加载 thinking 声明: {len(result)} 个模型")
        return result

    @classmethod
    def _load_multimodal_models(cls) -> Set[str]:
        """从 ``config/multimodal_models.json`` 加载支持多模态读图的模型白名单。

        JSON 格式::

            { "models": ["qwen3.7-flash", "qwen3.7-plus", ...] }

        返回模型名的 set；文件不存在或解析失败时返回空集（降级为全部不支持）。
        """
        path = LiteLLMRegistry._capability_path("multimodal_models.json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            models = data.get("models") or []
            result = {str(m).strip() for m in models if isinstance(m, str) and m.strip()}
            logger.debug(f"[LiteLLMRegistry] 加载 multimodal 白名单: {len(result)} 个模型")
            return result
        except FileNotFoundError:
            logger.debug(f"[LiteLLMRegistry] multimodal 白名单文件不存在: {path}")
            return set()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[LiteLLMRegistry] 加载 multimodal 白名单失败: {e}")
            return set()

    @classmethod
    def _load_long_context_models(cls) -> Dict[str, int]:
        """从 ``config/long_context_models.json`` 加载模型上下文长度声明。

        JSON 格式（向后兼容）::

            {
              "models": {
                "deepseek-v4-pro": 1000000,
                "qwen3.7-plus": {"max_context": 262144, "max_output": 8192}
              }
            }

        返回 ``{model_name: max_context_tokens}`` 字典；对象形式只取
        ``max_context``（``max_output`` 由 ModelContextCatalog 使用）。
        文件不存在或解析失败时返回空字典。
        """
        path = LiteLLMRegistry._capability_path("long_context_models.json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data.get("models") or {}
            result: Dict[str, int] = {}
            for k, v in raw.items():
                if not isinstance(k, str) or not k.strip():
                    continue
                name = k.strip()
                if isinstance(v, int) and v > 0:
                    result[name] = v
                elif isinstance(v, dict):
                    mc = v.get("max_context")
                    if isinstance(mc, int) and mc > 0:
                        result[name] = mc
            logger.debug(f"[LiteLLMRegistry] 加载 long_context 声明: {len(result)} 个模型")
            return result
        except FileNotFoundError:
            logger.debug(f"[LiteLLMRegistry] long_context 声明文件不存在: {path}")
            return {}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[LiteLLMRegistry] 加载 long_context 声明失败: {e}")
            return {}

    def _is_fresh(self) -> bool:
        if self._cache is None:
            return False
        return (time.monotonic() - self._cache_at) < self._ttl

    def _resolve_proxy_config(self) -> Dict[str, Any]:
        """读取模型网关配置。"""
        try:
            from src.utils.config_manager import get_config_manager
            from src.utils.env_manager import get_env_manager

            return get_config_manager().get_llm_gateway_full_config(get_env_manager())
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[LiteLLMRegistry] 读取网关配置失败: {e}")
            return {}

    def _fetch_from_proxy(self) -> List[LLMModelInfo]:
        cfg = self._resolve_proxy_config()
        api_base = (cfg.get("api_base") or "").strip()
        api_key = (cfg.get("api_key") or "").strip()
        gateway_type = (cfg.get("gateway_type") or "litellm").strip().lower()
        if not api_base:
            raise RuntimeError(
                "未配置 LLM 模型网关 api_base（检查 .env: MODEL_LAKE_BASE / LITELLM_PROXY_URL）"
            )

        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/models"
            info_url = f"{base}/model/info"
        else:
            url = f"{base}/v1/models"
            info_url = f"{base}/v1/model/info"

        headers: Dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with httpx.Client(timeout=self.HTTP_TIMEOUT_SECONDS) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        # LiteLLM 增强字段只用于补上下文长度，不再按 mode 过滤
        info_map: Dict[str, Dict[str, Any]] = {}
        try:
            with httpx.Client(timeout=self.HTTP_TIMEOUT_SECONDS) as client:
                info_resp = client.get(info_url, headers=headers)
                if info_resp.status_code == 200:
                    info_payload = info_resp.json()
                    for item in info_payload.get("data") or []:
                        mid = item.get("model_name") or item.get("id")
                        if mid:
                            info_map[mid] = item.get("model_info") or {}
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[LiteLLMRegistry] {info_url} 不可用: {e}")

        if gateway_type in LiteLLMRegistry.MODEL_LAKE_GATEWAY_TYPES:
            return self._fetch_model_lake(payload, info_map, gateway_type)

        from src.utils.config_profile import load_profile_visible_models
        parsed = self._parse_models_response(
            payload,
            info_map,
            self._thinking_models,
            self._multimodal_models,
            self._long_context_map,
            gateway_type=gateway_type,
        )
        return self._filter_visible_models(parsed, load_profile_visible_models())

    def _fetch_model_lake(
        self,
        payload: Any,
        info_map: Dict[str, Dict[str, Any]],
        gateway_type: str,
    ) -> List[LLMModelInfo]:
        """ModelNexus：能力声明从 ``/v1/models`` 富字段合成，档案只留 presets/hidden。

        富字段（OpenRouter 风格模型卡片）与能力声明的对应：
        - ``supported_parameters`` 含 ``reasoning`` / ``reasoning_effort`` -> 思考开关 / 强度
          （档位枚举网关不提供，由 ``thinking_adapter.infer_thinking_level_map``
          按厂商规则推导；仅开关的模型用 off/medium 两档）
        - ``architecture.input_modalities`` 含 ``image`` -> 多模态
        - ``context_length`` -> 最大上下文
        - ``deprecated_at`` 非空 -> 已弃用，直接过滤（不进前端也不进合成）
        """
        from src.client.llm.thinking_adapter import (
            infer_default_thinking_level,
            infer_thinking_level_map,
        )
        from src.utils.config_profile import load_profile_hidden_models

        thinking_specs: Dict[str, ThinkingModelSpec] = {}
        multimodal_set: Set[str] = set()
        long_ctx_map: Dict[str, int] = {}
        valid_ids: List[str] = []
        fresh_payload: Dict[str, Any] = {"data": []}
        for it in (payload or {}).get("data") or []:
            if not isinstance(it, dict):
                continue
            mid = it.get("id")
            if not isinstance(mid, str) or not mid.strip():
                continue
            if it.get("deprecated_at"):
                logger.warning(
                    f"[LiteLLMRegistry] 跳过已弃用模型: {mid} "
                    f"(deprecated_at={it.get('deprecated_at')})"
                )
                continue
            valid_ids.append(mid)
            fresh_payload["data"].append(it)

            # 声明 key 双写：网关原始路由（channel/model）+ 最后一段裸名，
            # 与 _lookup_thinking_spec 的多形态查找兼容
            bare = LiteLLMRegistry._bare_model_name(mid)
            sp = it.get("supported_parameters") or []
            if "reasoning" in sp or "reasoning_effort" in sp:
                # 本地厂商规则优先：网关声明 reasoning_effort 只是必要条件，
                # infer 返回 None 表示本地明确判定该模型仅支持思考开关
                # （如 Qwen 3.5/3.7、GLM 5.1 及以下）——即使网关标了
                # effort 也按开关式处理，展示与 adapter 实际行为保持一致。
                level_map = (
                    infer_thinking_level_map(mid)
                    if "reasoning_effort" in sp
                    else None
                )
                if level_map is not None:
                    spec = ThinkingModelSpec(
                        reasoning=True,
                        supports_thinking_effort=True,
                        default=infer_default_thinking_level(level_map) or "medium",
                        thinking_level_map=level_map,
                    )
                else:
                    spec = ThinkingModelSpec(
                        reasoning=True, supports_thinking_effort=False,
                    )
                thinking_specs[mid] = spec
                if bare and bare != mid:
                    thinking_specs[bare] = spec
            arch = it.get("architecture") or {}
            if "image" in (arch.get("input_modalities") or []):
                multimodal_set.add(mid)
                if bare and bare != mid:
                    multimodal_set.add(bare)
            ctx = it.get("context_length")
            if isinstance(ctx, int) and ctx > 0:
                long_ctx_map[bare] = ctx
                long_ctx_map[mid] = ctx

        # 运行时翻译数据源切换：整体替换（CPython 属性赋值原子，读侧
        # peek/resolve 拿到旧或新 dict 均自洽）。注意 list_models 持有
        # self._lock 期间调用本方法，这里不能再抢同一把不可重入锁。
        self._thinking_models = thinking_specs
        self._thinking_names = set(thinking_specs.keys())
        self._multimodal_models = multimodal_set
        self._long_context_map = long_ctx_map

        parsed = self._parse_models_response(
            fresh_payload, info_map, thinking_specs, multimodal_set, long_ctx_map,
            gateway_type=gateway_type,
        )
        visible = self._filter_hidden_models(parsed, load_profile_hidden_models())
        self._validate_presets(valid_ids)
        return visible

    def _validate_presets(self, valid_ids: List[str]) -> None:
        """校验档案 presets 指向的模型在网关清单中（存在且未弃用）。

        失配不抛异常阻断服务，只打 error 日志——把内部 pipeline 的运行时
        失败提前暴露为部署期可见的日志。
        """
        from src.utils.config_profile import load_profile_preset_models

        try:
            presets = load_profile_preset_models()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[LiteLLMRegistry] 读取 presets 失败，跳过校验: {e}")
            return
        valid = {LiteLLMRegistry._strip_gateway_prefix(v) for v in valid_ids}
        for role, model in presets.items():
            key = LiteLLMRegistry._strip_gateway_prefix(model)
            if key not in valid:
                logger.error(
                    f"[LiteLLMRegistry] preset '{role}' 指向的模型 '{model}' "
                    f"不在网关模型清单（不存在或已弃用），相关内部调用将失败"
                )

    # LiteLLM SDK 走 proxy 的官方前缀
    PROXY_MODEL_PREFIX = "litellm_proxy/"

    # OpenAI 兼容网关（ModelNexus）类型集合：走富元数据自动合成分支
    MODEL_LAKE_GATEWAY_TYPES = frozenset({"model_lake", "openai", "openai_compatible"})

    @staticmethod
    def _parse_models_response(
        payload: Any,
        info_map: Optional[Dict[str, Dict[str, Any]]] = None,
        thinking_models: Optional[Dict[str, ThinkingModelSpec]] = None,
        multimodal_models: Optional[Set[str]] = None,
        long_context_map: Optional[Dict[str, int]] = None,
        gateway_type: str = "litellm",
    ) -> List[LLMModelInfo]:
        """解析 ``/v1/models`` 响应：只丢掉没有 ``id`` 的项，再归一化给 SDK。

        前端可见范围由 ``_filter_visible_models`` 按档案 ``visible`` 再裁一次。
        """
        items = (payload or {}).get("data") or []
        info_map = info_map or {}
        thinking_specs = thinking_models or {}
        multimodal_set = multimodal_models or set()
        long_ctx_map = long_context_map or {}

        out: List[LLMModelInfo] = []
        for it in items:
            mid = (it or {}).get("id")
            if not isinstance(mid, str) or not mid:
                continue

            info = info_map.get(mid) or {}
            normalized_id, label, provider = LiteLLMRegistry._normalize_proxy_id(
                mid, gateway_type=gateway_type
            )
            bare = LiteLLMRegistry._bare_model_name(mid)
            spec = LiteLLMRegistry._lookup_thinking_spec(thinking_specs, mid)
            if spec is not None:
                tlevels = get_supported_thinking_levels(spec)
                tdefault = spec.default if spec.default in tlevels else (
                    tlevels[0] if tlevels else "off"
                )
                tmap = (
                    dict(spec.thinking_level_map)
                    if spec.supports_thinking_effort
                    else {"off": "none", "medium": "enabled"}
                )
            else:
                tlevels = ["off"]
                tdefault = None
                tmap = {}
            out.append(
                LLMModelInfo(
                    id=normalized_id,
                    label=label,
                    provider=provider,
                    supports_thinking=spec is not None and spec.reasoning,
                    thinking_levels=tlevels,
                    default_thinking_level=tdefault,
                    supports_multimodal=bare in multimodal_set or mid in multimodal_set,
                    max_context=LiteLLMRegistry._resolve_declared_context(
                        bare, long_ctx_map, info,
                    ),
                    thinking_level_map=tmap,
                ),
            )

        # 展示名统一用去掉 channel 前缀的裸模型名（如 deepseek/deepseek-v4.1-flash
        # -> deepseek-v4.1-flash）；排序先按 provider（channel）分组、组内按
        # 名称排序——同厂商模型相邻展示。
        for m in out:
            bare = LiteLLMRegistry._bare_model_name(m.id)
            if bare:
                m.label = bare
        out.sort(key=lambda m: (m.provider.lower(), m.label.lower()))
        return out

    @staticmethod
    def _strip_gateway_prefix(model_id: str) -> str:
        name = (model_id or "").strip()
        for prefix in (LiteLLMRegistry.PROXY_MODEL_PREFIX, "openai/"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    @staticmethod
    def _lookup_thinking_spec(
        thinking_specs: Dict[str, ThinkingModelSpec],
        model: str,
    ) -> Optional[ThinkingModelSpec]:
        """按多种命名形态查思考声明，与声明 key 的双写形态（routed + bare）兼容。

        model_lake（ModelNexus）声明由接口合成，key 双写 ``channel/model``
        （如 ``deepseek/deepseek-v4.1-flash``）与最后一段裸名（如 ``glm-5.3``）。
        调用侧则可能传入 ``openai/<channel>/<model>``、``channel/model``
        或最后一段裸名。只查 ``_bare_model_name`` 会让 channel 键全部 miss，
        前端仍显示支持思考，实际请求却被钳成 off。
        """
        raw = (model or "").strip()
        if not raw:
            return None
        routed = LiteLLMRegistry._strip_gateway_prefix(raw)
        bare = LiteLLMRegistry._bare_model_name(raw)
        for key in (raw, routed, bare):
            if key and key in thinking_specs:
                return thinking_specs[key]
        return None

    @staticmethod
    def _filter_visible_models(
        models: List[LLMModelInfo],
        allow_list: Optional[List[str]] = None,
    ) -> List[LLMModelInfo]:
        """按档案 ``visible`` 白名单过滤，保持解析层的分组排序。

        展示顺序统一由 ``_parse_models_response`` 决定（provider 分组、
        组内按名称）；白名单只做可见性筛选，不再决定顺序。
        """
        if not allow_list:
            return []
        allow = {
            LiteLLMRegistry._strip_gateway_prefix(str(raw))
            for raw in allow_list
            if str(raw).strip()
        }
        return [
            m for m in models
            if LiteLLMRegistry._strip_gateway_prefix(m.id) in allow
        ]

    @staticmethod
    def _filter_hidden_models(
        models: List[LLMModelInfo],
        hidden_list: Optional[List[str]] = None,
    ) -> List[LLMModelInfo]:
        """model_lake：默认全部可见，仅剔除档案 ``[hidden]`` 黑名单（空=全可见）。

        与 ``_filter_visible_models`` 的白名单语义相对：网关 ``/v1/models``
        即真相源，档案只做"不想暴露"的减法，不再逐个登记。
        """
        if not hidden_list:
            return list(models)
        hidden = {
            LiteLLMRegistry._strip_gateway_prefix(str(h))
            for h in hidden_list
            if str(h).strip()
        }
        return [
            m for m in models
            if LiteLLMRegistry._strip_gateway_prefix(m.id) not in hidden
        ]

    @staticmethod
    def _infer_provider(bare_name: str, default: str = "litellm_proxy") -> str:
        """从裸模型名启发式推断展示用 provider 分组。

        规则：对 lowercase 后的裸名做**子串关键词匹配，按固定顺序取第一个命中的**
        （顺序即优先级：更具体的名字放前面）。都没命中时返回 ``default``。
        注意这是"看名字猜厂商"——同名字不同源会误判；想改分组只能改网关模型名
        或在本表加关键词。
        """
        lowered = (bare_name or "").lower()
        # —— 已有主力厂商（保持原有判定顺序不变）——
        if "deepseek" in lowered:
            return "deepseek"
        if "qwen" in lowered or "qwq" in lowered or "tongyi" in lowered:
            return "qwen"
        if "glm" in lowered or "chatglm" in lowered or "zhipu" in lowered:
            return "glm"
        if "claude" in lowered or "anthropic" in lowered:
            return "anthropic"
        if "gpt" in lowered or lowered.startswith(("o1", "o3", "o4")):
            return "openai"
        if "gemini" in lowered:
            return "gemini"
        if "kimi" in lowered or "moonshot" in lowered:
            return "moonshot"
        if "baichuan" in lowered:
            return "baichuan"
        if "minimax" in lowered or "abab" in lowered:
            return "minimax"
        # —— 国内其他厂商 ——
        if "mimo" in lowered:
            return "mimo"
        if "ernie" in lowered:
            return "ernie"
        if "hunyuan" in lowered:
            return "hunyuan"
        if "doubao" in lowered or "skylark" in lowered:
            return "doubao"
        if "spark" in lowered:
            return "spark"
        if "internlm" in lowered:
            return "internlm"
        if "yi-" in lowered or lowered == "yi":
            return "yi"
        if "stepfun" in lowered or "step-" in lowered:
            return "step"
        if "sensenova" in lowered or "sensechat" in lowered:
            return "sensenova"
        # —— 国际其他厂商 ——
        if "mistral" in lowered or "mixtral" in lowered or "ministral" in lowered or "codestral" in lowered:
            return "mistral"
        if "llama" in lowered:
            return "meta"
        if "grok" in lowered:
            return "xai"
        if "cohere" in lowered or "command-r" in lowered or "command-a" in lowered:
            return "cohere"
        if "phi-" in lowered or lowered == "phi":
            return "microsoft"
        if "nemotron" in lowered:
            return "nvidia"
        if "jamba" in lowered:
            return "ai21"
        if "minicpm" in lowered:
            return "openbmb"
        return default

    @staticmethod
    def _normalize_proxy_id(raw_id: str, gateway_type: str = "litellm") -> tuple[str, str, str]:
        """Normalize a gateway model id into ``(sdk_id, ui_label, provider)``.

        Model Lake keeps the original ``channel/model`` route and only adds the
        LiteLLM-required ``openai/`` prefix. Distinct channel routes therefore
        remain distinct choices in the model selector.
        """
        raw = raw_id.strip()
        bare = LiteLLMRegistry._bare_model_name(raw)
        gw = (gateway_type or "litellm").lower().strip()

        if gw in ("model_lake", "openai", "openai_compatible"):
            model_lake_id = raw[len("openai/"):] if raw.startswith("openai/") else raw
            sdk_id = f"openai/{model_lake_id}"
            label = model_lake_id
            provider = model_lake_id.split("/", 1)[0] if "/" in model_lake_id else (
                LiteLLMRegistry._infer_provider(bare, default="model_lake")
            )
        else:
            sdk_id = f"{LiteLLMRegistry.PROXY_MODEL_PREFIX}{bare}"
            label = bare
            provider = LiteLLMRegistry._infer_provider(bare, default="litellm_proxy")

        return sdk_id, label, provider

    @staticmethod
    def _bare_model_name(raw_id: str) -> str:
        """剥离网关前缀（litellm_proxy/、openai/ 等），返回用于查配置和展示的裸名。"""
        if not raw_id:
            return ""
        name = raw_id.strip()
        for prefix in (LiteLLMRegistry.PROXY_MODEL_PREFIX, "openai/"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if "/" in name:
            # 若有其他 provider 前缀（如 anthropic/claude-3-5-sonnet），返回模型名部分
            name = name.split("/", 1)[1]
        return name

    @staticmethod
    def _resolve_declared_context(
        bare: str,
        long_ctx_map: Dict[str, int],
        model_info: Dict[str, Any],
    ) -> Optional[int]:
        """解析模型自身声明的上下文长度：本地声明优先，其次 proxy 上报。

        小于统一上限的模型**不再被剔除**——它们照常出现在选择器里，
        由 ``ModelContextCatalog`` 按 ``min(声明值, max_context_cap)`` 计量。
        """
        declared = long_ctx_map.get(bare)
        if isinstance(declared, int) and declared > 0:
            return declared
        for key in ("max_input_tokens", "max_tokens"):
            v = (model_info or {}).get(key)
            if isinstance(v, int) and v > 0:
                return v
        return None

# ============================================================
# 单例
# ============================================================


_registry: Optional[LiteLLMRegistry] = None
_registry_lock = threading.Lock()


def get_litellm_registry() -> LiteLLMRegistry:
    """模块级单例工厂"""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = LiteLLMRegistry()
    return _registry


__all__ = [
    "LLMModelInfo",
    "LiteLLMRegistry",
    "get_litellm_registry",
    "ThinkingModelSpec",
    "EXTENDED_THINKING_LEVELS",
    "THINKING_LEVELS_NO_OFF",
    "get_supported_thinking_levels",
    "clamp_thinking_level",
]
