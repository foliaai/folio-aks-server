#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : chat_service.py
@Author  : caixiongjiang
@Date    : 2026/05/11
@Function:
    ChatService ── 知识库对话核心编排

    模块定位
    --------
    把 Phase 0~2 的所有能力（流式累积、Mongo/MySQL 持久化、KnowledgeNavToolKit、
    Chat prompts、History 压缩）串联成"一次完整 chat 轮次"的事件流。
    本服务是个 **async generator**：``chat_stream(request)`` 产出 ``ChatEvent``，
    供上层（Phase 4 WS / SSE 端点）直接迭代转发。

    执行路径：Agent 工具循环
    ------------------------
    - Agent 拥有完整工具集（``search_knowledge_base`` + 导航工具）；
    - Agent 自主决定何时检索、用什么查询、是否需要导航补充；
    - 每轮 ``astream`` 流出文本/思考/tool_call 增量；
    - 若收到 tool_calls，并行 ``asyncio.gather`` 执行后把结果以
      ``role=tool`` 拼回 messages，进入下一轮；
    - **不限制工具循环轮数**：循环一直进行，直到模型自主决定不再调用
      工具（返回纯文本）为止，由 LLM 自己负责"何时停止检索"。

    与现有模块的协作
    ----------------
    - 历史加载：``ChatSessionService.load_history`` →
      ``compose_chat_messages``（含 history 反序列化）
    - 检索：``RetrieveService.retrieve(RetrieveRequest)``
    - 流式：``LLMClient.astream`` + ``StreamAccumulator`` 翻译事件
    - 工具：``KnowledgeNavToolKit`` 提供 schemas / dispatch / supplemented_items
    - 持久化：每轮 assistant + 每条 tool 写 MongoDB；末尾 ``touch`` MySQL
    - 标题：首轮结束后 ``TitleService.schedule_in_background`` 异步起标题

    设计取舍
    --------
    - **citations 合并**：初始检索命中 + 工具补充统一汇入
      ``supplemented_items``；写最后一条 assistant 的 ``citations`` 时去重；
      中间轮 assistant 的 ``citations`` 仅写"本轮工具新增的"，可观测但不重复。
    - **错误隔离**：检索失败、工具执行失败都不中断会话——以
      ``ChatEvent(ERROR)`` 透出，主流程继续；只有 LLM 调用本身彻底崩溃才
      抛出（由 WS 层捕获）。
    - **持久化粒度**：每条消息单独写一次 MongoDB，``touch`` 在轮次末尾
      一次性更新计数；避免每条消息触发 MySQL update。
    - **背压**：本 generator 是"边算边出"，与上层消费速度耦合；若上层
      WS 写入慢会自动反压到 LLM 拉流，无需额外缓冲。
@Modify History:
    2026-05-11 - 首版（Phase 3）
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from loguru import logger

from src.chat.stream_buffer import (
    StreamAccumulator,
    StreamEvent,
    StreamEventType,
)
from src.client.llm import LLMClient
from src.client.llm.types import LLMResponse, ToolCall
from src.db.mongodb.models.conversation.chat_message import (
    ChatMessage,
    ChatRole,
    Citation,
    ToolCallRecord,
    TokenUsageRecord,
)
from src.db.mongodb.repositories.conversation import chat_message_repo
from src.service.chat.chunk_alias_map import (
    NavAliasMap,
    METADATA_ALIAS_ADDITIONS_KEY,
    METADATA_SECTION_ALIAS_ADDITIONS_KEY,
    METADATA_DOCUMENT_ALIAS_ADDITIONS_KEY,
    rebuild_alias_map_from_history,
)
from src.service.chat.chunk_enricher import ChunkMeta, TurnEnrichCache
from src.prompts.chat import (
    DEFAULT_CHAT_SYSTEM,
    apply_token_window,
    build_chat_system_prompt,
    compose_chat_messages,
    drop_assistant_tool_dangling,
)
from src.service.chat.context import (
    ContextBudgetInput,
    ContextBudgeter,
    HierarchicalSummarizer,
    get_model_context_catalog,
    truncate_tool_output,
)
from src.retrieve.types.result import ChunkItem
from src.service.chat.session_service import (
    ChatSessionService,
    generate_message_id,
)
from src.service.chat.title_service import TitleService
from src.service.chat.tools import KnowledgeNavToolKit
from src.service.chat.tools.registry import AGENT_ENABLED_TOOLS, agent_tools_schema
from src.service.chat.types import (
    ChatEvent,
    ChatEventType,
    ChatRequest,
    ChatTurnContext,
    ChatTurnResult,
)


# ============================================================
# 配置
# ============================================================


class ChatServiceConfig:
    """ChatService 运行时配置

    设计原则
    --------
    - **测试友好**：直接 ``ChatServiceConfig()`` 返回硬编码默认值，不读 toml；
    - **生产装配**：API 端点 / WS 路由通过 ``from_config_manager()`` 从
      ``config.toml [chat]`` 节加载，覆盖任何字段；
    - **不读 components.json**：``components.json`` 服务于 RAG 抽取 Pipeline
      的 Kafka Worker 组件，与 chat 这种在线请求路径无关。Chat 直接通过
      ``agent_model_preset`` / ``title_model_preset`` 引用 ``[llm.presets.*]``。
    """

    # ---------- 检索 ----------
    retrieve_top_k: int = 8
    """初始服务端检索 top_k"""

    # ---------- 上下文（Cursor 式：全量加载 + token 窗口 + 持久化摘要）----------
    history_load_limit: int = 100000
    """全量加载历史消息的防御性上限（条数）；正常会话远不会触及"""

    context_window_tokens: int = 200000
    """上下文窗口 token 预算（system + history + tools schema + 本轮 user + 输出预留）"""

    summary_compress_threshold_ratio: float = 0.9
    """完整请求 token 占软上限（统一上限 - 输出预留）超过该比例时触发持久化摘要压缩"""

    summary_keep_recent_turns: int = 1
    """摘要压缩后保留的最近轮数（1 轮 = 一次用户交互，含其完整工具链）"""

    # ---------- 上下文预算（[chat.context]）----------
    reserved_output_fallback: int = 8192
    tool_output_hard_cap_tokens: int = 32000
    tool_loop_budget_check: bool = True
    tool_output_floor_tokens: int = 512
    summary_chunk_tokens: int = 24000
    summary_target_tokens: int = 1500
    summary_map_concurrency: int = 4
    context_status_cache_seconds: int = 15
    heuristic_safety_factor: float = 1.2

    # ---------- 输出 ----------
    max_completion_tokens: Optional[int] = None
    """单轮 assistant 输出 token 上限；``None`` 表示由模型 / preset 决定"""

    # ---------- LLM 选型 ----------
    agent_model_preset: str = "fast"
    """主对话 LLM 的 preset 名（``ChatRequest`` / ``ChatSession`` 都未指定时使用）"""

    title_model_preset: str = "fast"
    """异步起标题的 preset 名（一次性短调用，性价比优先）"""

    user_chat_template_preset: str = "fast"
    """用户从前端 ``/api/chat/models`` 选定具体 ``model`` 时，作为采样参数模板的 preset 名

    业务语义：preset 仍然是后台抽取/起标题/摘要的真相源（model + 调好的
    temperature / max_tokens 等）；当用户在前端选了某个具体模型字符串时，
    ``model`` 由用户提供，但 temperature 之类继续从这个 template preset 里读，
    保持 Agent 单轮的体验稳定。思考强度由前端 ``thinking_level`` 逐轮传入，
    不在此 preset 模板里固化。
    """

    multimodal_model_preset: str = "multimodal"
    """``read_image_chunks`` 工具内 VLM 的 preset 名（引用 ``[llm.presets.multimodal]``）"""

    # ---------- session 默认（仅当 ChatRequest 与 ChatSession 都未指定时使用） ----------
    default_mode: str = "agent"
    default_thinking_level: str = "off"
    """新建会话默认思考强度档位（pi 标准 7 档之一）；仅 reasoning 类模型生效，
    后端会按模型 thinking_levels 钳位。前端不传时用该默认。"""
    default_max_tool_rounds: int = 5

    # ---------- @ 内联引用注入（方案A + 方案C）----------
    mention_inject_max_chars: int = 6000
    """方案A：单个 @ 文件全量注入正文的字符上限；超过则退化为方案C（仅提示 document_id）"""

    mention_inject_total_budget: int = 12000
    """方案A：单轮所有 @ 文件全量注入正文的合计字符预算；超预算的文件退化为方案C"""

    # ============================================================
    # 配置加载
    # ============================================================

    @classmethod
    def from_config_manager(cls, manager: Optional[Any] = None) -> "ChatServiceConfig":
        """从 ``config.toml [chat]`` 节加载；缺失字段使用类默认值。

        Args:
            manager: ``ConfigManager`` 实例；不传则使用全局单例。

        Returns:
            ``ChatServiceConfig`` 实例
        """
        try:
            if manager is None:
                from src.utils.config_manager import get_config_manager
                manager = get_config_manager()
            chat_cfg: Dict[str, Any] = manager.get_section("chat") or {}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"加载 [chat] 配置失败，使用默认值: {e}")
            return cls()

        retrieval = chat_cfg.get("retrieval", {}) or {}
        history = chat_cfg.get("history", {}) or {}
        context = chat_cfg.get("context", {}) or {}

        inst = cls()

        # 顶层
        max_ct = chat_cfg.get("max_completion_tokens")
        # 0 / None → 透传 None（让模型/preset 决定）
        inst.max_completion_tokens = (
            int(max_ct) if isinstance(max_ct, int) and max_ct > 0 else None
        )
        inst.agent_model_preset = str(
            chat_cfg.get("agent_model_preset", inst.agent_model_preset),
        )
        inst.title_model_preset = str(
            chat_cfg.get("title_model_preset", inst.title_model_preset),
        )
        inst.user_chat_template_preset = str(
            chat_cfg.get(
                "user_chat_template_preset", inst.user_chat_template_preset,
            ),
        )
        inst.multimodal_model_preset = str(
            chat_cfg.get(
                "multimodal_model_preset", inst.multimodal_model_preset,
            ),
        )
        inst.default_mode = str(
            chat_cfg.get("default_mode", inst.default_mode),
        )
        inst.default_thinking_level = str(
            chat_cfg.get("default_thinking_level", inst.default_thinking_level),
        )
        inst.default_max_tool_rounds = int(
            chat_cfg.get("default_max_tool_rounds", inst.default_max_tool_rounds),
        )
        inst.mention_inject_max_chars = int(
            chat_cfg.get("mention_inject_max_chars", inst.mention_inject_max_chars),
        )
        inst.mention_inject_total_budget = int(
            chat_cfg.get(
                "mention_inject_total_budget", inst.mention_inject_total_budget,
            ),
        )

        # [chat.retrieval]
        inst.retrieve_top_k = int(retrieval.get("top_k", inst.retrieve_top_k))

        # [chat.history]（Cursor 式：全量加载 + token 窗口 + 持久化摘要）
        inst.history_load_limit = int(
            history.get("max_messages", inst.history_load_limit),
        )
        inst.context_window_tokens = int(
            history.get("max_tokens", inst.context_window_tokens),
        )
        inst.summary_compress_threshold_ratio = float(
            history.get(
                "summary_compress_threshold_ratio",
                inst.summary_compress_threshold_ratio,
            ),
        )
        inst.summary_keep_recent_turns = int(
            history.get(
                "summary_keep_recent_turns", inst.summary_keep_recent_turns,
            ),
        )

        # [chat.context]
        inst.reserved_output_fallback = int(
            context.get("reserved_output_fallback", inst.reserved_output_fallback),
        )
        inst.tool_output_hard_cap_tokens = int(
            context.get("tool_output_hard_cap_tokens", inst.tool_output_hard_cap_tokens),
        )
        inst.tool_loop_budget_check = bool(
            context.get("tool_loop_budget_check", inst.tool_loop_budget_check),
        )
        inst.tool_output_floor_tokens = int(
            context.get("tool_output_floor_tokens", inst.tool_output_floor_tokens),
        )
        inst.summary_chunk_tokens = int(
            context.get("summary_chunk_tokens", inst.summary_chunk_tokens),
        )
        inst.summary_target_tokens = int(
            context.get("summary_target_tokens", inst.summary_target_tokens),
        )
        inst.summary_map_concurrency = int(
            context.get("summary_map_concurrency", inst.summary_map_concurrency),
        )
        inst.context_status_cache_seconds = int(
            context.get(
                "context_status_cache_seconds", inst.context_status_cache_seconds,
            ),
        )
        inst.heuristic_safety_factor = float(
            context.get("heuristic_safety_factor", inst.heuristic_safety_factor),
        )

        logger.info(
            f"ChatServiceConfig 已从 [chat] 加载: "
            f"agent_preset={inst.agent_model_preset}, "
            f"title_preset={inst.title_model_preset}, "
            f"top_k={inst.retrieve_top_k}, "
            f"context_window_tokens={inst.context_window_tokens}, "
            f"summary_threshold_ratio={inst.summary_compress_threshold_ratio}, "
            f"default_thinking_level={inst.default_thinking_level}"
        )
        return inst


# ============================================================
# ChatService
# ============================================================


class ChatService:
    """知识库对话主编排（async generator 风格）

    用法::

        service = ChatService(
            session_service=ChatSessionService(),
            retrieve_service=RetrieveService(),
            title_service=TitleService(),
        )
        async for ev in service.chat_stream(req):
            ws.send(ev)

    """

    def __init__(
        self,
        *,
        session_service: Optional[ChatSessionService] = None,
        retrieve_service: Optional[Any] = None,  # RetrieveService（避免循环 import）
        title_service: Optional[TitleService] = None,
        config: Optional[ChatServiceConfig] = None,
    ) -> None:
        self._cfg = config or ChatServiceConfig()
        self._session_service = session_service or ChatSessionService()
        self._retrieve_service = retrieve_service  # 延迟加载，见 _get_retrieve_service
        # TitleService 用 config 里指定的 preset；显式注入优先
        self._title_service = title_service or TitleService(
            model_preset=self._cfg.title_model_preset,
        )
        # LLM 客户端按 preset 名缓存
        self._client_cache: Dict[str, LLMClient] = {}

    # ============================================================
    # 依赖装配（懒加载）
    # ============================================================

    def _get_retrieve_service(self):
        if self._retrieve_service is None:
            from src.service.knowledge.retrieve_service import RetrieveService

            self._retrieve_service = RetrieveService()
        return self._retrieve_service

    def _get_llm_client(
        self,
        *,
        model: Optional[str] = None,
        model_preset: Optional[str] = None,
    ) -> LLMClient:
        """获取 LLMClient（按"最终 model 字符串"缓存）。

        分发逻辑
        --------
        - ``model`` 非空：用户在前端显式选定了具体模型字符串，走
          ``create_llm_client_from_model``——以 ``user_chat_template_preset``
          作为采样参数模板，``model`` 字段被入参覆盖。
        - 否则：按 ``model_preset`` 走 ``create_llm_client_from_preset``，
          与历史行为完全一致。

        缓存键统一用最终的 ``client.model`` 字符串：preset 路径上多个 preset
        指向同一 model 时也能命中同一个 client；用户 model 路径上不同 preset
        模板 + 同 model 时缓存粒度按 model 区分（业务上够用）。

        chat 模块**不走 ``components.json``**：``components.json`` 是 RAG 抽取
        Pipeline（Kafka Worker）的中央配置簿，与 chat 这种在线请求路径无关。
        """
        from src.client.llm import (
            create_llm_client_from_model,
            create_llm_client_from_preset,
        )

        if model:
            cache_key = f"model::{model}"
            cached = self._client_cache.get(cache_key)
            if cached is not None:
                return cached
            client = create_llm_client_from_model(
                model=model,
                chat_template_preset=self._cfg.user_chat_template_preset,
            )
            self._client_cache[cache_key] = client
            logger.debug(
                f"ChatService LLMClient: explicit model={client.model}, "
                f"template_preset={self._cfg.user_chat_template_preset}"
            )
            return client

        preset = model_preset or self._cfg.agent_model_preset
        cache_key = f"preset::{preset}"
        cached = self._client_cache.get(cache_key)
        if cached is not None:
            return cached
        client = create_llm_client_from_preset(preset)
        self._client_cache[cache_key] = client
        logger.debug(
            f"ChatService LLMClient: preset={preset}, model={client.model}"
        )
        return client

    # ============================================================
    # 配置合并：ChatRequest + ChatSession → ChatTurnContext
    # ============================================================

    def _resolve_turn_context(
        self,
        request: ChatRequest,
        session,
    ) -> ChatTurnContext:
        """把请求级覆盖与会话默认合并成一份"本轮有效配置"。"""
        # ===== scope 摘要（v0.8.0）=====
        # 注意：sys_prompt 构造在 scope 解析之前，但 scope dict 构造需要解析结果。
        # 所以这里先把 sys_prompt 的最终构造放到 scope 解析之后；上面只准备 desc。
        # Agent 模式：始终启用工具（与上下文计量共用 AGENT_ENABLED_TOOLS）
        _enabled_tools = AGENT_ENABLED_TOOLS
        # model 字段优先级：request > session > None（None 时由 model_preset 决定）
        # 注意：``model`` 与 ``model_preset`` 并存——``model`` 选定具体模型，
        # ``model_preset`` 仍作为采样参数模板（temperature / max_tokens 等）。
        effective_model = request.model or session.model or None

        # ===== scope 解析（v0.8.0）=====
        # 优先级：request.folder_id > session.folder_id > kb
        # 注：@ 文件走 mentions 软引用（见 _resolve_mentions），不再经 scope 硬锁。
        effective_folder_id: Optional[str] = (
            request.folder_id
            if request.folder_id is not None
            else getattr(session, "folder_id", None)
        )
        effective_include_subfolders: bool = (
            request.include_subfolders
            if request.include_subfolders is not None
            else bool(getattr(session, "include_subfolders", True))
        )
        folder_label: Optional[str] = None
        scope_kind: str = "folder" if effective_folder_id else "kb"
        scope_document_ids: List[str] = []

        if effective_folder_id:
            resolved = self._resolve_folder_document_ids(
                user_id=session.user_id,
                folder_id=effective_folder_id,
                include_subfolders=effective_include_subfolders,
                knowledge_base_ids=list(session.knowledge_base_ids or []),
            )
            scope_document_ids = resolved.get("document_ids", [])
            folder_label = resolved.get("folder_label")

        # 现在 scope 已解析完，构造 scope dict 给 system prompt
        scope_dict: Optional[Dict[str, Any]] = None
        if scope_kind == "folder":
            scope_dict = {
                "kind": "folder",
                "folder_id": effective_folder_id,
                "label": folder_label,
                "include_subfolders": effective_include_subfolders,
                "document_count": len(scope_document_ids),
                "knowledge_base_ids": list(session.knowledge_base_ids or []),
            }
        # ===== 技能索引注入（system prompt，稳定前缀）=====
        _skills_index: Optional[str] = None
        # Slash 显式召唤的技能正文：注入到当轮 user turn（见 chat_stream），
        # 而非 system prompt——避免污染稳定前缀、提升 KV cache 命中率。
        _forced_skills_block: str = ""
        _resolved_forced_skill_names: List[str] = []
        try:
            from src.service.skill.registry_singleton import get_registry
            _reg = get_registry()
            _forced_skills_block, _resolved_forced_skill_names = (
                self._build_forced_skills_block(
                    request.forced_skill_names, _reg,
                )
            )
            _exclude_skills = frozenset(_resolved_forced_skill_names)
            _skills_index = _reg.build_index(
                set(_enabled_tools),
                exclude_names=_exclude_skills if _exclude_skills else None,
            )
        except Exception as e:
            logger.warning(f"技能索引构建失败（不影响对话）: {e}")

        sys_prompt = (
            request.custom_system_prompt
            or session.system_prompt
            or build_chat_system_prompt(
                enabled_tools=_enabled_tools,
                scope=scope_dict,
                skills_index=_skills_index,
                explicit_skill_names=_resolved_forced_skill_names,
            )
        )

        # 思考强度档位优先级：request > session.thinking_level > cfg.default_thinking_level。
        # 最后按 effective_model 的 thinking_levels 钳位成可下发 / 可持久化的合法档位。
        if request.thinking_level is not None:
            _raw_thinking_level = request.thinking_level
        elif getattr(session, "thinking_level", None):
            _raw_thinking_level = session.thinking_level
        else:
            _raw_thinking_level = self._cfg.default_thinking_level
        try:
            from src.client.llm.registry import get_litellm_registry
            _clamp_model = effective_model or ""
            effective_thinking_level = (
                get_litellm_registry().clamp_thinking_level(
                    _clamp_model, _raw_thinking_level,
                )
                if _clamp_model
                else _raw_thinking_level
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"钳位 thinking_level 失败，沿用原值: {e}")
            effective_thinking_level = _raw_thinking_level

        return ChatTurnContext(
            session_id=session.session_id,
            user_id=session.user_id,
            query=request.query,
            mode=(
                request.mode
                if request.mode is not None
                else getattr(session, "mode", None) or self._cfg.default_mode
            ),
            thinking_level=effective_thinking_level,
            model_preset=(
                request.model_preset
                or session.model_preset
                or self._cfg.agent_model_preset
            ),
            model=effective_model,
            max_tool_rounds=(
                request.max_tool_rounds
                if request.max_tool_rounds is not None
                else session.max_tool_rounds
            ),
            retrieve_top_k=request.retrieve_top_k or self._cfg.retrieve_top_k,
            system_prompt=sys_prompt,
            knowledge_base_ids=list(session.knowledge_base_ids or []),
            skip_retrieval=request.skip_retrieval,
            enable_route_plan=(
                request.enable_route_plan
                if request.enable_route_plan is not None
                else bool(getattr(session, "enable_route_plan", False))
            ),
            scope_kind=scope_kind,
            folder_id=effective_folder_id,
            folder_label=folder_label,
            include_subfolders=effective_include_subfolders,
            scope_document_ids=scope_document_ids,
            skills_index=_skills_index or "",
            forced_skills_block=_forced_skills_block,
            forced_skill_names=_resolved_forced_skill_names,
        )

    def _resolve_folder_document_ids(
        self,
        *,
        user_id: str,
        folder_id: str,
        include_subfolders: bool,
        knowledge_base_ids: List[str],
    ) -> Dict[str, Any]:
        """folder scope → document_ids 解析（v0.8.0）

        每轮即时解析（不缓存到 session），保证文件夹增删能即时反映。

        Args:
            user_id: 当前用户
            folder_id: 起始文件夹 ID
            include_subfolders: 是否递归含子文件夹
            knowledge_base_ids: 用于附加 KB 过滤；空表示不限 KB

        Returns:
            ``{"document_ids": [...], "folder_label": "..."}``。
            folder 不存在 / 无权限 → ``document_ids = []`` + ``folder_label = None``
            （上游做软降级；提示词层会注入"空 folder"提示）。
        """
        from src.db.mysql.connection.factory import get_mysql_manager
        from src.db.mysql.repositories.business.workspace_file_system_repo import (
            WorkspaceFileSystemRepository,
        )
        from src.db.mysql.repositories.business.workspace_folder_repo import (
            workspace_folder_repo,
        )

        manager = get_mysql_manager()
        with manager.get_session() as db:
            folder = db.query(workspace_folder_repo.model).filter(
                workspace_folder_repo.model.folder_id == folder_id,
                workspace_folder_repo.model.deleted == 0,
            ).first()
            if not folder or (
                folder.knowledge_base_id not in (
                    knowledge_base_ids or [folder.knowledge_base_id]
                )
            ):
                logger.warning(
                    f"_resolve_folder_document_ids: folder_id={folder_id} "
                    f"不存在或不在会话范围内（user={user_id}）；返回空 scope（软降级）"
                )
                return {"document_ids": [], "folder_label": None}

            # 收集"目标文件夹 + 后代（如启用）"全部 folder_id
            # （kb 维度：共享库内后代文件夹可能由其他成员创建）
            folder_ids: List[str] = [folder_id]
            if include_subfolders:
                descendants = db.query(workspace_folder_repo.model).filter(
                    workspace_folder_repo.model.knowledge_base_id
                    == folder.knowledge_base_id,
                    workspace_folder_repo.model.full_path.like(
                        f"{folder.full_path}%"
                    ),
                    workspace_folder_repo.model.deleted == 0,
                ).all()
                folder_ids.extend(
                    f.folder_id for f in descendants
                    if f.folder_id != folder_id
                )

            file_repo = WorkspaceFileSystemRepository()
            files = db.query(file_repo.model).filter(
                file_repo.model.knowledge_base_id == folder.knowledge_base_id,
                file_repo.model.folder_id.in_(folder_ids),
                file_repo.model.deleted == 0,
            ).all()
            # WorkspaceFileSystem.document_id 可空（文件可能尚未完成索引）
            doc_ids = [f.document_id for f in files if f.document_id]
            # 去重保序
            seen: set = set()
            unique: List[str] = []
            for did in doc_ids:
                if did not in seen:
                    seen.add(did)
                    unique.append(did)
            # 给 LLM 看的可读 label：优先用 folder_name，其次取 full_path 末段
            label: Optional[str] = (
                getattr(folder, "folder_name", None)
                or (
                    folder.full_path.strip("/").split("/")[-1]
                    if getattr(folder, "full_path", None)
                    else None
                )
            )
            logger.debug(
                f"folder scope 解析: folder_id={folder_id}, "
                f"include_subfolders={include_subfolders}, "
                f"folders={len(folder_ids)}, files={len(files)}, "
                f"unique_documents={len(unique)}, label={label}"
            )
            return {"document_ids": unique, "folder_label": label}

    def _resolve_file_document_ids(
        self,
        *,
        user_id: str,
        file_id: str,
        knowledge_base_ids: List[str],
    ) -> Dict[str, Any]:
        """@ 文件 scope → document_ids 解析（v0.9.0）

        单文件提问：file_id -> workspace_file_system.document_id。
        与 folder scope 不同，这里最多返回 1 个 document_id（按内容去重后的 ID）。

        软降级：
        - 文件不存在 / 无权限 / 不属于 session 允许的 KB → document_ids=[] + label=None
        - 文件尚未索引完成（document_id is None）→ document_ids=[] + label=file_name
          （上游提示词层会注入"空范围"提示，告知模型该文件暂无可检索内容）

        Args:
            user_id: 当前用户
            file_id: @ 选中的文件 ID
            knowledge_base_ids: session 允许的 KB 列表；非空时用于校验 file 归属

        Returns:
            ``{"document_ids": [...], "file_label": "..."}``
        """
        from src.db.mysql.connection.factory import get_mysql_manager
        from src.db.mysql.repositories.business.workspace_file_system_repo import (
            WorkspaceFileSystemRepository,
        )

        manager = get_mysql_manager()
        with manager.get_session() as db:
            file_repo = WorkspaceFileSystemRepository()
            file = file_repo.get_by_file_id(db, file_id)
            if not file:
                logger.warning(
                    f"_resolve_file_document_ids: file_id={file_id} "
                    f"不存在或无权限（user={user_id}）；返回空 scope（软降级）"
                )
                return {"document_ids": [], "file_label": None}

            # 校验 file 所属 KB 在 session 允许范围内
            if knowledge_base_ids and file.knowledge_base_id not in knowledge_base_ids:
                logger.warning(
                    f"_resolve_file_document_ids: file_id={file_id} 所属 KB "
                    f"{file.knowledge_base_id} 不在 session.knowledge_base_ids "
                    f"{knowledge_base_ids}；返回空 scope（软降级）"
                )
                return {"document_ids": [], "file_label": file.file_name}

            doc_id = getattr(file, "document_id", None)
            label = file.file_name
            if not doc_id:
                logger.debug(
                    f"_resolve_file_document_ids: file_id={file_id} "
                    f"尚未完成索引（document_id 为空）；本轮无检索范围"
                )
                return {"document_ids": [], "file_label": label}

            logger.debug(
                f"file scope 解析: file_id={file_id}, document_id={doc_id}, "
                f"label={label}"
            )
            return {"document_ids": [doc_id], "file_label": label}

    def _resolve_mentions(
        self,
        *,
        user_id: str,
        mentions: Optional[List[Any]],
        knowledge_base_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Cursor 式 @ 内联引用解析（v0.10.0，软引用）

        把请求里的 ``mentions``（文件/目录，可多个）解析为可注入 user prompt 的
        引用条目。与 folder/document scope 不同，这里**不锁死 scope**，只是把
        document_id 暴露给模型，让它自行决定是否聚焦。

        软降级：
        - 文件/目录不存在或无权限 → 丢弃该条；
        - 文件未完成索引（无 document_id）→ 保留条目但标记 indexed=False；
        - 目录为空 → 保留条目但 document_ids=[]。

        Args:
            user_id: 当前用户
            mentions: ``ChatMention`` 列表（含 ``kind`` / ``id``）
            knowledge_base_ids: session 允许的 KB 列表，用于归属校验

        Returns:
            形如 ``[{"kind","id","label","document_ids","indexed"}, ...]`` 的有序去重列表
        """
        if not mentions:
            return []
        refs: List[Dict[str, Any]] = []
        seen: set = set()
        for m in mentions:
            kind = getattr(m, "kind", None)
            mid = getattr(m, "id", None)
            if not kind or not mid:
                continue
            key = (kind, mid)
            if key in seen:
                continue
            seen.add(key)

            if kind == "file":
                resolved = self._resolve_file_document_ids(
                    user_id=user_id,
                    file_id=mid,
                    knowledge_base_ids=knowledge_base_ids,
                )
                label = resolved.get("file_label")
                doc_ids = resolved.get("document_ids", [])
                if label is None and not doc_ids:
                    # 文件不存在 / 无权限 → 丢弃
                    continue
                refs.append({
                    "kind": "file",
                    "id": mid,
                    "label": label or mid,
                    "document_ids": doc_ids,
                    "indexed": bool(doc_ids),
                })
            elif kind == "folder":
                resolved = self._resolve_folder_document_ids(
                    user_id=user_id,
                    folder_id=mid,
                    include_subfolders=True,
                    knowledge_base_ids=knowledge_base_ids,
                )
                label = resolved.get("folder_label")
                doc_ids = resolved.get("document_ids", [])
                if label is None and not doc_ids:
                    # 目录不存在 / 无权限 → 丢弃
                    continue
                refs.append({
                    "kind": "folder",
                    "id": mid,
                    "label": label or mid,
                    "document_ids": doc_ids,
                    "indexed": True,
                })

        logger.debug(f"@ 内联引用解析: 输入 {len(mentions)} 项, 有效 {len(refs)} 项")
        return refs

    async def _load_document_full_text(
        self, document_id: str, *, max_chars: int,
    ) -> Optional[Dict[str, Any]]:
        """载入单篇文档的**有序全文**（方案A 全量注入用）。

        复用 ``DrillDown``（Document → Chunk，``include_content=True``）取回按
        ``(page_index, element_index, split_seq)`` 排好序的 chunk 正文并拼接。

        Args:
            document_id: 目标文档 ID
            max_chars: 全文字符上限；一旦累计超过即视为「大文件」，返回 ``None``
                （交由方案C 仅提示 document_id），避免把超大文档塞进上下文。

        Returns:
            ``{"text": str, "chunk_count": int}``；无内容 / 超阈值 / 失败 → ``None``。
        """
        if max_chars <= 0:
            return None
        try:
            from src.retrieve.capabilities.navigation.drill_down import DrillDown
            from src.retrieve.types.enums import GranularityLevel
            from src.retrieve.types.query import NavigationQuery

            cap = DrillDown()
            result = await cap.execute(
                query=NavigationQuery(
                    anchor_id=document_id,
                    anchor_type=GranularityLevel.DOCUMENT,
                    target_granularity=GranularityLevel.CHUNK,
                    include_content=True,
                ),
            )
        except Exception as e:  # noqa: BLE001 - 载入失败退化为方案C，不中断对话
            logger.warning(
                f"方案A 载入文档全文失败 document_id={document_id}: {e}",
            )
            return None

        parts: List[str] = []
        total = 0
        for item in result.items:
            text = getattr(item, "text", None)
            if not text:
                continue
            total += len(text)
            if total > max_chars:
                logger.debug(
                    f"方案A：document_id={document_id} 全文超阈值 "
                    f"(> {max_chars} 字)，退化为方案C 仅提示 document_id",
                )
                return None
            parts.append(text)

        if not parts:
            return None
        return {"text": "\n\n".join(parts), "chunk_count": len(result.items)}

    @staticmethod
    def _build_forced_skills_block(
        forced_skill_names: Optional[List[str]],
        registry: Any,
    ) -> tuple[str, List[str]]:
        """把 Slash 显式召唤的技能正文拼成注入当轮 user turn 的「显式技能」块。

        - 注入到 *给 LLM 的* user 消息**最末尾**（而非 system prompt），使 system prompt
          保持稳定前缀、提升 KV cache 命中率（Cursor 式 slash 指令即注入 user turn）。
        - 停用 / 不存在的技能静默跳过；无有效技能时返回 ``("", [])``。

        Args:
            forced_skill_names: 前端 `/` 选中的技能名列表。
            registry: SkillRegistry 单例（有 ``get(name) -> Skill | None``）。

        Returns:
            ``(block_text, resolved_names)`` — block 为注入 user 的文本；resolved_names
            为成功解析的技能名（供 build_index 剔除与 system override）。
        """
        if not forced_skill_names:
            return "", []

        parts: List[str] = []
        resolved: List[str] = []
        for name in forced_skill_names:
            try:
                skill = registry.get(name)
            except Exception:  # noqa: BLE001
                skill = None
            if skill is not None:
                resolved.append(name)
                parts.append(f"## 技能：{name}\n\n{skill.body}")
        if not parts:
            return "", []

        block = (
            "\n\n---\n"
            "【显式技能 / 最高优先级】\n"
            "本条消息通过 \"/\" 显式调用了以下技能。请把它们的完整 Procedure 与 "
            "Verification 作为本轮最高优先级要求，严格遵循：\n\n"
            "- 已注入的技能**无需**再 `skill_view` 加载同名技能；"
            "Level-2 模板（如 templates/report.html）按需 `skill_view(path=…)`。\n"
            "- 技能要求的结构化长文 / 完整 HTML 交付物不受 system「保持简洁」约束。\n"
            "- 取证阶段按技能 Procedure；证据满足技能要求后再停止工具并输出最终交付物。\n\n"
            + "\n\n".join(parts)
        )
        return block, resolved

    async def _build_references_block(
        self,
        references: List[Dict[str, Any]],
        *,
        alias_map: Optional[NavAliasMap] = None,
    ) -> str:
        """把解析后的 @ 引用拼成注入 user prompt 的「引用资料」块（方案A + 方案C）。

        - **方案A（小文件全量注入）**：已索引的单文档文件，若全文在
          ``mention_inject_max_chars`` 且不超过本轮合计预算
          ``mention_inject_total_budget``，则直接把全文拼进上下文，模型无需再调工具。
        - **方案C（大文件 / 目录 / 多文件超预算 → 仅提示）**：只给出 document_id
          （经 ``alias_map`` 转为 ``dN`` 短引用），
          让模型按需用 skeleton / drill_down / read_chunks 等工具自取。

        软引用语义：不锁死 scope，问题超出引用范围时模型仍可更大范围检索。
        无引用时返回空串（user 消息保持原样）。
        """
        if not references:
            return ""

        full_blocks: List[str] = []   # 方案A：全量注入
        hint_lines: List[str] = []    # 方案C：仅提示 document_id
        remaining_budget = self._cfg.mention_inject_total_budget

        for r in references:
            label = r.get("label", "")
            doc_ids = r.get("document_ids", []) or []
            kind = r.get("kind")

            if kind == "file":
                if not doc_ids:
                    hint_lines.append(
                        f"- 文件「{label}」：尚未完成索引，暂无可检索内容"
                    )
                    continue
                doc_id = doc_ids[0]
                doc_label = (
                    alias_map.alias_for_document(doc_id) if alias_map else doc_id
                )
                # 方案A：尝试全量注入（单文件、已索引、正文在预算内）
                loaded: Optional[Dict[str, Any]] = None
                if remaining_budget > 0:
                    loaded = await self._load_document_full_text(
                        doc_id,
                        max_chars=min(
                            self._cfg.mention_inject_max_chars, remaining_budget,
                        ),
                    )
                if loaded is not None:
                    text = loaded["text"]
                    remaining_budget -= len(text)
                    full_blocks.append(
                        f"### 文件「{label}」（document_id = {doc_label}，已全量载入，"
                        f"共 {len(text)} 字）\n{text}"
                    )
                else:
                    # 方案C：大文件 / 超预算 → 仅提示
                    hint_lines.append(
                        f"- 文件「{label}」：document_id = {doc_label}"
                        f"（已索引，正文较大未全量载入）"
                    )
            else:  # folder → 方案C
                n = len(doc_ids)
                if n:
                    if alias_map:
                        preview = ", ".join(
                            alias_map.alias_for_document(d) for d in doc_ids[:20]
                        )
                    else:
                        preview = ", ".join(doc_ids[:20])
                    more = " …" if n > 20 else ""
                    hint_lines.append(
                        f"- 目录「{label}」：含 {n} 篇文档"
                        f"（document_ids: {preview}{more}）"
                    )
                else:
                    hint_lines.append(f"- 目录「{label}」：该目录下暂无可检索文档")

        segments: List[str] = [
            "\n\n---\n本条消息通过 @ 引用了以下资料（请优先据此理解与回答）："
        ]
        if full_blocks:
            segments.append("\n\n".join(full_blocks))
        if hint_lines:
            segments.append(
                "以下资料仅提供 document_id（正文较大或为目录，未全量载入）：\n"
                + "\n".join(hint_lines)
                + "\n\n可用 skeleton / drill_down / grep_chunks / read_chunks / "
                "search_knowledge_base 传入上述 document_id 聚焦这些资料。"
            )
        segments.append(
            "若问题超出引用范围，可再做更大范围检索；"
            "未完成索引的资料暂无可检索内容，请如实告知用户。"
        )
        return "\n\n".join(segments)

    # ============================================================
    # 主入口：流式对话
    # ============================================================

    async def chat_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        """单轮对话事件流（async generator）

        事件顺序（典型路径）::

            SESSION_READY
            ( CONTENT_DELTA | THINKING_DELTA | TOOL_CALL_* ... )*
            MESSAGE_DONE             # 每轮 assistant 落库后
            ( RETRIEVAL_PROGRESS* TOOL_CALL_COMPLETED ... TOOL_ROUND_DONE )*   # Agent 才有
            TURN_DONE

        异常路径会在对应阶段透出 ``ERROR``，主流程继续直到 ``TURN_DONE``。
        """
        total_start = time.perf_counter()
        result = ChatTurnResult(
            session_id=request.session_id,
            user_message_id="",
        )

        # ---- 1) 权限 + 加载 session ----
        try:
            session = self._session_service.get_session(
                session_id=request.session_id, user_id=request.user_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"加载 session 异常: {e}")
            yield ChatEvent(
                ChatEventType.ERROR,
                {"phase": "load_session", "error": str(e)},
            )
            return
        if session is None:
            yield ChatEvent(
                ChatEventType.ERROR,
                {"phase": "load_session", "error": "session not found or forbidden"},
            )
            return

        ctx = self._resolve_turn_context(request, session)
        is_first_turn = (session.message_count or 0) == 0

        # turn 级 chunk 元数据缓存：同 chunk_id 在本 turn 内只查 4 张表 1 次。
        # 使用点：
        #   1) _build_citations_for_round（每轮 assistant 落库 + message.done 下发）
        #   2) tool_call.completed（如未来扩展到工具补充 chunks 也要 enrich）
        enrich_cache = TurnEnrichCache(
            user_id=ctx.user_id,
            knowledge_base_id=(
                ctx.knowledge_base_ids[0] if ctx.knowledge_base_ids else None
            ),
        )

        # session 级 chunk_id ↔ alias 映射：从历史 assistant.metadata.alias_additions
        # 累加重建。本 turn 新分配的 alias 在 _persist_assistant 里写回。
        # 占位创建，下面加载历史后再 rebuild。
        alias_map = NavAliasMap()

        # ---- 2) 持久化 user message ----
        user_msg_id = generate_message_id()
        result.user_message_id = user_msg_id
        try:
            await chat_message_repo.create(
                creator=ctx.user_id,
                _id=user_msg_id,
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                role=ChatRole.USER.value,
                content=ctx.query,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"写入 user 消息失败: {e}")
            yield ChatEvent(
                ChatEventType.ERROR,
                {"phase": "persist_user", "error": str(e)},
            )
            return

        # ---- 2a) 首条消息：锁定模式到 session ----
        if is_first_turn:
            try:
                self._session_service.update_session_mode(
                    session_id=ctx.session_id,
                    user_id=ctx.user_id,
                    mode=ctx.mode,
                    thinking_level=ctx.thinking_level,
                    max_tool_rounds=ctx.max_tool_rounds,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"锁定会话模式失败（忽略）: {e}")

        # ---- 2b) 每轮：把"轻偏好"回写到 session ----
        # 与首轮"锁模式"互补：``model`` / ``thinking_level`` / ``model_preset``
        # 是用户每轮可调整的项；只要本轮 ctx 与 session 当前值不同就 update，
        # 让用户下次进同一会话时 chip 默认选项跟随其上次选择。
        try:
            updates_payload: Dict[str, Any] = {}
            if (request.model is not None) and ((session.model or None) != request.model):
                updates_payload["model"] = request.model
            if (
                request.model_preset is not None
                and session.model_preset != request.model_preset
            ):
                updates_payload["model_preset"] = request.model_preset
            # 非首轮（首轮已被 update_session_mode 写过 thinking_level）才写
            if (
                not is_first_turn
                and request.thinking_level is not None
                and (getattr(session, "thinking_level", None) or "off") != ctx.thinking_level
            ):
                updates_payload["thinking_level"] = ctx.thinking_level
            if updates_payload:
                self._session_service.update_session_settings(
                    session_id=ctx.session_id,
                    user_id=ctx.user_id,
                    **updates_payload,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"写回会话偏好失败（忽略）: {e}")

        yield ChatEvent(
            ChatEventType.SESSION_READY,
            {
                "session_id": ctx.session_id,
                "user_message_id": user_msg_id,
                "mode": ctx.mode,
                "model_preset": ctx.model_preset,
                "model": ctx.model,
            },
        )

        # ---- 3) 加载历史（全量，按时间正序）----
        try:
            history_full = await self._session_service.load_full_history(
                session_id=ctx.session_id,
                limit=self._cfg.history_load_limit,
            )
            # 把刚写入的 user 消息排除（避免它出现在 history + user_message 两处）
            history_full = [m for m in history_full if m.id != user_msg_id]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"加载历史异常，按空历史继续: {e}")
            history_full = []

        # 硬约束：alias_map / turn_citations 必须用压缩前全量 history 重建。
        # 二者是纯服务端内存结构，不进 LLM 上下文；若在 _apply_context_compression
        # 之后重建，被 summarized 的 assistant.metadata.alias_additions / citations
        # 会丢失，导致压缩后新轮引用旧 [cN] 死链或错链。
        try:
            alias_map = rebuild_alias_map_from_history(history_full)
            logger.debug(
                f"alias_map 重建完成：chunk_size={alias_map.size}, "
                f"section_size={alias_map.section_size}, "
                f"document_size={alias_map.document_size}"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"alias_map 重建失败（按空 map 继续）: {e}")
            alias_map = NavAliasMap()

        turn_citations: Dict[str, Citation] = {}
        for m in history_full:
            if m.role == "assistant" and m.citations:
                for c in m.citations:
                    if c.chunk_id:
                        turn_citations[c.chunk_id] = c

        # 应用已持久化的摘要压缩：跳过 summarized=true 的旧消息，保留最新 summary
        history = self._apply_context_compression(history_full)

        # ---- 4) 收敛历史（ContextBudgeter + Cursor 式持久化摘要压缩）----
        client = self._get_llm_client(model=ctx.model, model_preset=ctx.model_preset)
        history = drop_assistant_tool_dangling(history)

        # Cursor 式 @ 内联引用：解析 mentions → 「引用资料」块（方案A 小文件全量注入 +
        # 方案C 大文件/目录仅提示 document_id），拼进 *给 LLM 的* user 副本尾部
        # （软引用：不锁死 scope）。持久化/显示用的 user 消息仍是 ctx.query 原文（见 §2）。
        # 必须在预算计量前定稿：该块最多可达 mention_inject_total_budget 字符，
        # 漏算会让压缩触发系统性偏晚。
        references = self._resolve_mentions(
            user_id=ctx.user_id,
            mentions=getattr(request, "mentions", None),
            knowledge_base_ids=ctx.knowledge_base_ids,
        )
        # 顺序：用户原文 → @ 引用资料块 → 显式技能块（/ 召唤，置于最末尾以强化 recency）。
        user_message_for_llm = (
            ctx.query
            + await self._build_references_block(references, alias_map=alias_map)
            + ctx.forced_skills_block
        )

        budgeter = self._get_context_budgeter()
        reserved = budgeter.resolve_reserved_output(
            client.model,
            preset_max_tokens=self._cfg.max_completion_tokens,
        )
        # 计入本轮完整请求：system（含技能索引）+ 工具 schema + 历史 +
        # 给 LLM 的 user 副本（含 @ 引用块与显式技能块）。
        budget_report = budgeter.evaluate(
            ContextBudgetInput(
                model=client.model,
                system_prompt=ctx.system_prompt or "",
                history=history,
                user_message=user_message_for_llm,
                tools_schema=agent_tools_schema(),
                reserved_output_tokens=reserved,
                skills_block=ctx.skills_index,
            )
        )
        est_tokens = budget_report.used_tokens

        if (
            self._cfg.summary_compress_threshold_ratio > 0
            and budget_report.over_soft_limit
        ):
            # Cursor 式持久化压缩：保留最近 1 轮（含完整工具链）+ 1 条摘要
            try:
                await self._session_service.compaction_keep_recent_turns(
                    session_id=ctx.session_id,
                    user_id=ctx.user_id,
                    keep_recent_turns=self._cfg.summary_keep_recent_turns,
                    summarize_fn=self._build_summarize_fn(model=client.model),
                    trigger="auto_threshold",
                    budget_snapshot=budget_report.to_dict(),
                )
                # 压缩后重新加载并应用压缩（alias_map / turn_citations 保持全量重建结果）
                history_full = await self._session_service.load_full_history(
                    session_id=ctx.session_id,
                    limit=self._cfg.history_load_limit,
                )
                history_full = [m for m in history_full if m.id != user_msg_id]
                history = self._apply_context_compression(history_full)
                history = drop_assistant_tool_dangling(history)
                logger.info(
                    f"上下文已压缩: used={est_tokens}, ratio={budget_report.ratio:.2f}, "
                    f"counting={budget_report.counting}, 压缩后 history={len(history)} 条"
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"上下文压缩失败，按未压缩继续: {e}")

        # token 窗口兜底（按模型 max_context；正常压缩后远不会触及）
        model_max_context = get_model_context_catalog().resolve(client.model).max_context
        history = apply_token_window(
            history,
            max_tokens=model_max_context or self._cfg.context_window_tokens,
            keep_system=False,
        )

        # ---- 6) 种子命中（固定初始检索已下线，恒空）----
        # Chat 模式去掉后不再做固定初始检索，统一由 Agent 工具循环按需检索；
        # seed_hits 保留为空列表占位，供下游 citations 构建函数签名兼容。
        seed_hits: List[ChunkItem] = []

        # ---- 7) 组装首轮 messages ----
        # user 副本（原文 + @ 引用块 + 显式技能块）已在 §4 预算计量前定稿。
        # 组装首轮 messages（seed_hits 现恒空，参考片段注入由 Agent 工具循环按需触发）
        # （compose_chat_messages 内部处理格式化 + alias 替换）
        messages: List[Dict[str, Any]] = compose_chat_messages(
            system_prompt=ctx.system_prompt,
            history=history,
            user_message=user_message_for_llm,
            retrieved_chunks=seed_hits,
            alias_map=alias_map,
        )

        # ---- 8) 进入执行路径（Agent 模式）----
        supplemented: List[ChunkItem] = []
        # 检索工具进度回调：通过 Queue 桥接到 async generator yield
        search_progress_queue: asyncio.Queue[
            Optional[Tuple[str, str, Optional[str], str]]
        ] = asyncio.Queue()

        async def _on_search_progress(
            tool_call_id: str,
            stage: str,
            *,
            model: Optional[str] = None,
            channel: str = "retrieval",
        ) -> None:
            await search_progress_queue.put(
                (tool_call_id, stage, model, channel),
            )

        kit = KnowledgeNavToolKit(
            supplemented_items=supplemented,
            alias_map=alias_map,
            retrieve_service=self._get_retrieve_service(),
            on_progress=_on_search_progress,
            user_id=ctx.user_id,
            knowledge_base_ids=ctx.knowledge_base_ids,
            scope_document_ids=ctx.scope_document_ids,
            scope_kind=ctx.scope_kind,
            scope_label=ctx.folder_id,
            enable_route_planner=ctx.enable_route_plan,
        )

        assistant_msg_ids: List[str] = []
        tool_msg_ids: List[str] = []

        tools_schema = kit.schemas() if kit else None
        # 思考强度：把 ctx.thinking_level 译成 reasoning_effort，再交给
        # LLMClient / ThinkingAdapter 生成 extra_body.thinking（及 DashScope
        # enable_thinking）。LiteLLM openai/ 路径（Model Lake）不会替我们翻译
        # DeepSeek / Qwen 思考字段；off 仍返回 None，避免 reasoning_effort='none'
        # 禁用工具。档案 miss 时 resolve 会透传用户档位，不能再把开关丢掉。
        try:
            from src.client.llm.registry import get_litellm_registry
            reasoning_effort = get_litellm_registry().resolve_reasoning_effort(
                client.model, ctx.thinking_level,
            )
        except Exception as e:  # noqa: BLE001
            raw_level = (ctx.thinking_level or "").strip()
            if raw_level.lower() in ("", "off", "none"):
                logger.debug(f"解析 reasoning_effort 失败，不下发: {e}")
                reasoning_effort = None
            else:
                logger.warning(
                    f"解析 reasoning_effort 失败，按用户档位透传: "
                    f"model={client.model} level={raw_level!r} err={e}"
                )
                reasoning_effort = raw_level

        llm_start = time.perf_counter()
        try:
            async for ev in self._run_loop_real(
                ctx=ctx,
                client=client,
                kit=kit,
                messages=messages,
                seed_hits=seed_hits,
                supplemented=supplemented,
                assistant_msg_ids=assistant_msg_ids,
                tool_msg_ids=tool_msg_ids,
                result=result,
                enrich_cache=enrich_cache,
                alias_map=alias_map,
                tools_schema=tools_schema,
                reasoning_effort=reasoning_effort,
                search_progress_queue=search_progress_queue,
                turn_citations=turn_citations,
            ):
                yield ev
        except asyncio.CancelledError:
            logger.info(f"会话 {ctx.session_id} 主循环被取消")
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"对话主循环异常: {e}")
            yield ChatEvent(
                ChatEventType.ERROR,
                {"phase": "llm_loop", "error": str(e)},
            )
            result.error = str(e)
        result.llm_time_ms = (time.perf_counter() - llm_start) * 1000 - result.tool_time_ms

        # ---- 8) 收尾：touch session + turn.done ----
        result.assistant_message_ids = assistant_msg_ids
        result.tool_message_ids = tool_msg_ids
        new_msg_count = 1 + len(assistant_msg_ids) + len(tool_msg_ids)
        try:
            self._session_service.touch_session(
                session_id=ctx.session_id, delta=new_msg_count,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"touch session 失败: {e}")

        # ---- 9) 后台异步起标题（仅首轮） ----
        if is_first_turn and assistant_msg_ids:
            try:
                # 取首条 assistant 文本作为助手回复输入
                first_asst_text = await self._fetch_first_assistant_text(
                    assistant_msg_ids[0],
                )
                self._title_service.schedule_in_background(
                    session_id=ctx.session_id,
                    user_id=ctx.user_id,
                    user_query=ctx.query,
                    assistant_reply=first_asst_text,
                    session_service=self._session_service,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"调度起标题任务失败（忽略）: {e}")

        result.total_time_ms = (time.perf_counter() - total_start) * 1000
        result.citations_count = self._merge_citations_count([], supplemented)
        yield ChatEvent(
            ChatEventType.TURN_DONE,
            {
                "session_id": ctx.session_id,
                "user_message_id": result.user_message_id,
                "assistant_message_ids": result.assistant_message_ids,
                "tool_message_ids": result.tool_message_ids,
                "rounds": result.rounds,
                "tool_rounds": result.tool_rounds,
                "tool_calls_count": result.tool_calls_count,
                "citations_count": result.citations_count,
                "context_shrinks": result.context_shrinks,
                "finish_reason": result.final_finish_reason,
                "total_time_ms": result.total_time_ms,
                "retrieval_time_ms": result.retrieval_time_ms,
                "llm_time_ms": result.llm_time_ms,
                "tool_time_ms": result.tool_time_ms,
                "error": result.error,
            },
        )

    # ============================================================
    # 主循环（Agent 工具循环）
    # ============================================================

    async def _run_loop_real(
        self,
        *,
        ctx: ChatTurnContext,
        client: LLMClient,
        kit: Optional[KnowledgeNavToolKit],
        messages: List[Dict[str, Any]],
        seed_hits: List[ChunkItem],
        supplemented: List[ChunkItem],
        assistant_msg_ids: List[str],
        tool_msg_ids: List[str],
        result: ChatTurnResult,
        enrich_cache: TurnEnrichCache,
        alias_map: NavAliasMap,
        tools_schema: Optional[List[Dict[str, Any]]],
        reasoning_effort: Optional[str],
        search_progress_queue: Optional[
            asyncio.Queue[Optional[Tuple[str, str, Optional[str], str]]]
        ] = None,
        turn_citations: Optional[Dict[str, Citation]] = None,
    ) -> AsyncIterator[ChatEvent]:
        """Agent 工具循环执行体

        循环**不设轮数上限**：每一轮流式拉取 LLM 输出，若模型返回
        ``tool_calls`` 则并行执行工具并把结果拼回 messages 进入下一轮；
        若模型返回纯文本（无 tool_calls）则持久化后退出。何时停止检索
        完全交给 LLM 自主决策（system prompt 里的"何时停止检索"约束负责
        引导其在拿到充分证据后立即作答）。
        """

        budgeter = self._get_context_budgeter()
        reserved = budgeter.resolve_reserved_output(
            client.model, preset_max_tokens=self._cfg.max_completion_tokens,
        )

        round_idx = 0
        while True:
            assistant_msg_id = generate_message_id()
            acc = StreamAccumulator(model=client.model)

            # ---- 每次 LLM 调用前重做预算 ----
            # 循环不设轮数上限，每轮追加的工具结果都会推高用量；单条 32K 硬顶
            # 只能约束"一条不太大"，管不住"多条加起来超窗"。这里按即将发出的
            # messages 重算，超线则由旧到新收缩工具结果（不删消息，配对不断）。
            if self._cfg.tool_loop_budget_check:
                try:
                    shrink = budgeter.shrink_messages_to_fit(
                        messages,
                        model=client.model,
                        tools_schema=tools_schema,
                        reserved_output_tokens=reserved,
                        floor_tokens=self._cfg.tool_output_floor_tokens,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"工具循环预算检查失败，按原样发送: {e}")
                else:
                    if shrink.applied:
                        result.context_shrinks += 1
                        logger.info(
                            f"工具循环预算收缩: round={round_idx}, "
                            f"释放 {shrink.freed_tokens} tokens "
                            f"(收紧 {shrink.shrunk_count} 条 / 省略 "
                            f"{shrink.elided_count} 条), "
                            f"收缩后 used={shrink.report.used_tokens}"
                        )
                    if not shrink.fits:
                        msg = (
                            f"上下文仍超出窗口: used={shrink.report.used_tokens}, "
                            f"soft_limit={shrink.report.soft_limit}"
                        )
                        logger.error(f"{msg}，中止工具循环")
                        yield ChatEvent(
                            ChatEventType.ERROR,
                            {
                                "phase": "context_budget",
                                "error": msg,
                                "round": round_idx,
                                "used_tokens": shrink.report.used_tokens,
                                "soft_limit": shrink.report.soft_limit,
                            },
                        )
                        result.error = msg
                        return

            # 流式拉一轮；思考墙钟从首个 thinking.delta 计到 content/tool 开始
            thinking_started_at: Optional[float] = None
            thinking_ms: Optional[float] = None
            try:
                stream = client.astream(
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto" if tools_schema else None,
                    reasoning_effort=reasoning_effort,
                    max_tokens=self._cfg.max_completion_tokens,
                )
                async for chunk in stream:
                    for sev in acc.feed(chunk):
                        thinking_started_at, thinking_ms = (
                            self._update_thinking_span(
                                thinking_started_at,
                                thinking_ms,
                                sev.type,
                            )
                        )
                        ev = self._stream_event_to_chat_event(sev)
                        if ev is not None:
                            yield ev
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error(f"LLM 流式调用失败: {e}")
                yield ChatEvent(
                    ChatEventType.ERROR,
                    {"phase": "llm_stream", "error": str(e), "round": round_idx},
                )
                result.error = str(e)
                return

            resp = acc.finalize()
            if thinking_started_at is not None and thinking_ms is None:
                thinking_ms = max(
                    0.0, (time.perf_counter() - thinking_started_at) * 1000,
                )
            result.rounds += 1

            # 把本轮 assistant 拼回 messages（继续下一轮 / 收尾用）
            messages.append(_assistant_message(resp))

            # ---- 分支 A: 无 tool_calls → 直接持久化 + message.done + 退出 ----
            if not resp.tool_calls:
                # 本 turn 内工具已累计补充的 chunk（多轮时后续轮仍可能引用 [c1]）
                citations_this_round = await self._build_citations_for_round(
                    seed_hits=seed_hits,
                    added_chunks=supplemented,
                    enrich_cache=enrich_cache,
                    alias_map=alias_map,
                    extra_citations=list((turn_citations or {}).values()),
                )
                # 累计跨轮引用（保留全量，供后续 turn 解析 [cN]）
                if turn_citations is not None:
                    for c in citations_this_round:
                        if c.chunk_id:
                            turn_citations[c.chunk_id] = c
                # 持久化全量 citations（含未引用的 chunk，供跨 turn 解析），
                # 仅 message.done 事件做 _filter 给前端渲染用。
                all_citations_for_persist = citations_this_round
                citations_for_display = self._filter_cited_only(
                    resp.content or "", citations_this_round,
                )
                await self._persist_assistant(
                    ctx=ctx,
                    message_id=assistant_msg_id,
                    resp=resp,
                    citations=all_citations_for_persist,
                    alias_map=alias_map,
                    kit=kit,
                    thinking_ms=thinking_ms,
                )
                assistant_msg_ids.append(assistant_msg_id)
                yield ChatEvent(
                    ChatEventType.MESSAGE_DONE,
                    {
                        "message_id": assistant_msg_id,
                        "role": "assistant",
                        "round": round_idx,
                        "finish_reason": resp.finish_reason,
                        "tool_calls_count": 0,
                        "citations_count": len(citations_for_display),
                        "citations": [c.model_dump() for c in citations_for_display],
                        "has_thinking": bool(resp.thinking),
                        "thinking_ms": thinking_ms,
                        "usage": {
                            "prompt_tokens": resp.usage.prompt_tokens,
                            "completion_tokens": resp.usage.completion_tokens,
                            "total_tokens": resp.usage.total_tokens,
                            "cached_tokens": getattr(resp.usage, "cached_tokens", 0) or 0,
                        },
                    },
                )
                result.final_finish_reason = resp.finish_reason
                return

            # ---- 分支 B: 有 tool_calls ----
            # 关键：先把工具跑完，把 result_brief / items_added 一并合并进 assistant 落库
            # ——这样历史回放时 UI 也能拿到工具结果（修复 "工具未返回摘要" / "新增 0 段"）。
            # 注意事件顺序仍然是：(content/tool_call stream) → message.done →
            #                    tool_call.completed × N → tool_round.done，
            # 前端 useKnowledgeChat 的 lastAssistantIdRef 逻辑依赖该顺序。
            result.tool_rounds += 1
            result.tool_calls_count += len(resp.tool_calls)
            tool_t0 = time.perf_counter()
            # 启动工具并行执行，同时消费检索进度队列
            tool_task = asyncio.create_task(
                self._exec_tools_parallel(kit, resp.tool_calls)
            )
            if search_progress_queue is not None:
                while not tool_task.done():
                    try:
                        item = await asyncio.wait_for(
                            search_progress_queue.get(), timeout=0.1,
                        )
                    except asyncio.TimeoutError:
                        continue
                    if item is None:
                        continue
                    tc_id, stage, model, channel = item
                    if channel == "tool":
                        payload: Dict[str, Any] = {
                            "stage": stage,
                            "tool_call_id": tc_id,
                        }
                        if model:
                            payload["model"] = model
                        yield ChatEvent(ChatEventType.TOOL_PROGRESS, payload)
                    else:
                        yield ChatEvent(
                            ChatEventType.RETRIEVAL_PROGRESS,
                            {"stage": stage, "tool_call_id": tc_id},
                        )
                # drain remaining items
                while not search_progress_queue.empty():
                    try:
                        item = search_progress_queue.get_nowait()
                        if item is not None:
                            tc_id, stage, model, channel = item
                            if channel == "tool":
                                payload = {
                                    "stage": stage,
                                    "tool_call_id": tc_id,
                                }
                                if model:
                                    payload["model"] = model
                                yield ChatEvent(
                                    ChatEventType.TOOL_PROGRESS,
                                    payload,
                                )
                            else:
                                yield ChatEvent(
                                    ChatEventType.RETRIEVAL_PROGRESS,
                                    {
                                        "stage": stage,
                                        "tool_call_id": tc_id,
                                    },
                                )
                    except asyncio.QueueEmpty:
                        break
            tool_results = await tool_task
            result.tool_time_ms += (time.perf_counter() - tool_t0) * 1000

            # 工具结果硬顶截断（避免单条 tool output 撑爆窗口）
            hard_cap = self._cfg.tool_output_hard_cap_tokens
            if hard_cap > 0:
                truncated_results = []
                for tc, content, items_added, time_ms in tool_results:
                    new_content = truncate_tool_output(
                        content or "",
                        max_tokens=hard_cap,
                        model=client.model,
                        catalog=get_model_context_catalog(),
                        heuristic_safety_factor=self._cfg.heuristic_safety_factor,
                    )
                    truncated_results.append((tc, new_content, items_added, time_ms))
                tool_results = truncated_results

            # enrich 检索工具的 chunks_brief（补充 file_name / section_title 等渲染元数据）
            if kit and kit.search_results:
                all_brief_chunks: List[ChunkItem] = []
                for cb_list, _, _ in kit.search_results.values():
                    for cb in cb_list:
                        all_brief_chunks.append(ChunkItem(
                            chunk_id=cb["chunk_id"],
                            score=cb.get("score", 0.0),
                            document_id=cb.get("document_id"),
                        ))
                if all_brief_chunks:
                    try:
                        await enrich_cache.ensure(all_brief_chunks)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"enrich search chunks 失败: {e}")
                # 把 enrich 后的元数据写回 chunks_brief
                for cb_list, _, _ in kit.search_results.values():
                    for cb in cb_list:
                        meta = enrich_cache.get(cb["chunk_id"])
                        if meta:
                            cb["section_title"] = meta.section_title
                            cb["page_index"] = meta.page_index
                            cb["chunk_type"] = meta.chunk_type
                            cb["file_id"] = meta.file_id
                            cb["file_name"] = meta.file_name

            citations_this_round = await self._build_citations_for_round(
                seed_hits=seed_hits,
                added_chunks=supplemented,
                enrich_cache=enrich_cache,
                alias_map=alias_map,
                extra_citations=list((turn_citations or {}).values()),
            )
            if turn_citations is not None:
                for c in citations_this_round:
                    if c.chunk_id:
                        turn_citations[c.chunk_id] = c
            all_citations_for_persist = citations_this_round
            citations_for_display = self._filter_cited_only(
                resp.content or "", citations_this_round,
            )
            await self._persist_assistant(
                ctx=ctx,
                message_id=assistant_msg_id,
                resp=resp,
                citations=all_citations_for_persist,
                tool_results=tool_results,
                alias_map=alias_map,
                kit=kit,
                thinking_ms=thinking_ms,
            )
            assistant_msg_ids.append(assistant_msg_id)

            yield ChatEvent(
                ChatEventType.MESSAGE_DONE,
                {
                    "message_id": assistant_msg_id,
                    "role": "assistant",
                    "round": round_idx,
                    "finish_reason": resp.finish_reason,
                    "tool_calls_count": len(resp.tool_calls),
                    "citations_count": len(citations_for_display),
                    "citations": [c.model_dump() for c in citations_for_display],
                    "has_thinking": bool(resp.thinking),
                    "thinking_ms": thinking_ms,
                    "usage": {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                        "cached_tokens": getattr(resp.usage, "cached_tokens", 0) or 0,
                    },
                },
            )

            tool_round_brief: List[Dict[str, Any]] = []
            for tc, content, items_added, time_ms in tool_results:
                tool_msg_id = generate_message_id()
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                })
                await self._persist_tool_message(
                    ctx=ctx,
                    message_id=tool_msg_id,
                    tc=tc,
                    content=content,
                )
                tool_msg_ids.append(tool_msg_id)

                brief = content or ""
                tool_round_brief.append({
                    "id": tc.id,
                    "name": tc.name,
                    "args": tc.arguments,
                    "result_brief": brief,
                    "items_added": items_added,
                    "time_ms": time_ms,
                })
                completed_event: Dict[str, Any] = {
                    "id": tc.id,
                    "name": tc.name,
                    "args": tc.arguments,
                    "result_brief": brief,
                    "items_added": items_added,
                    "time_ms": time_ms,
                    "tool_message_id": tool_msg_id,
                }
                # 检索工具：附加 chunks 数据供前端渲染"查看"按钮
                if kit and tc.name == "search_knowledge_base":
                    search_result = kit.search_results.get(tc.id)
                    if search_result:
                        completed_event["retrieval_chunks"] = search_result[0]
                        completed_event["retrieval_params"] = search_result[1]
                        if len(search_result) > 2 and search_result[2] is not None:
                            completed_event["recall_stats"] = search_result[2]
                if kit:
                    execution_model = kit.execution_model_for(tc.id)
                    if execution_model:
                        completed_event["execution_model"] = execution_model
                yield ChatEvent(
                    ChatEventType.TOOL_CALL_COMPLETED,
                    completed_event,
                )

            yield ChatEvent(
                ChatEventType.TOOL_ROUND_DONE,
                {"round": round_idx, "tool_calls": tool_round_brief},
            )
            round_idx += 1

    # ============================================================
    # 子功能：工具并行执行
    # ============================================================

    @staticmethod
    async def _exec_tools_parallel(
        kit: KnowledgeNavToolKit,
        tool_calls: List[ToolCall],
    ) -> List[Tuple[ToolCall, str, int, float]]:
        """并行执行多个 tool_calls；返回 ``[(tc, content, items_added, time_ms), ...]``"""

        async def _one(tc: ToolCall) -> Tuple[ToolCall, str, int, float]:
            t0 = time.perf_counter()
            if not kit.has(tc.name):
                text = f"未知工具或未启用: {tc.name}"
            else:
                text = await kit.call(tc.name, tc.arguments, tool_call_id=tc.id)
            elapsed = (time.perf_counter() - t0) * 1000
            added = kit.items_added_for(tc.id)  # noqa: SLF001
            return tc, text, added, elapsed

        return await asyncio.gather(*[_one(tc) for tc in tool_calls])

    # ============================================================
    # 子功能：StreamEvent → ChatEvent 翻译
    # ============================================================

    @staticmethod
    def _update_thinking_span(
        started_at: Optional[float],
        ended_ms: Optional[float],
        sev_type: StreamEventType,
    ) -> Tuple[Optional[float], Optional[float]]:
        """首个 thinking.delta 开表，首个 content / tool_call 停表。"""
        now = time.perf_counter()
        if sev_type == StreamEventType.THINKING_DELTA and started_at is None:
            return now, ended_ms
        if (
            started_at is not None
            and ended_ms is None
            and sev_type in (
                StreamEventType.CONTENT_DELTA,
                StreamEventType.TOOL_CALL_STARTED,
            )
        ):
            return started_at, max(0.0, (now - started_at) * 1000)
        return started_at, ended_ms

    @staticmethod
    def _stream_event_to_chat_event(sev: StreamEvent) -> Optional[ChatEvent]:
        if sev.type == StreamEventType.CONTENT_DELTA:
            return ChatEvent(ChatEventType.CONTENT_DELTA, {"text": sev.text})
        if sev.type == StreamEventType.THINKING_DELTA:
            return ChatEvent(ChatEventType.THINKING_DELTA, {"text": sev.text})
        if sev.type == StreamEventType.TOOL_CALL_STARTED:
            return ChatEvent(
                ChatEventType.TOOL_CALL_STARTED,
                {
                    "index": sev.tool_call_index,
                    "id": sev.tool_call_id,
                    "name": sev.tool_call_name,
                },
            )
        if sev.type == StreamEventType.TOOL_CALL_ARGS_DELTA:
            return ChatEvent(
                ChatEventType.TOOL_CALL_ARGS_DELTA,
                {"index": sev.tool_call_index, "text": sev.text},
            )
        # FINISH 不直接透出（用 MESSAGE_DONE 取代）
        return None

    # ============================================================
    # 子功能：持久化
    # ============================================================

    @staticmethod
    async def _persist_assistant(
        *,
        ctx: ChatTurnContext,
        message_id: str,
        resp: LLMResponse,
        citations: List[Citation],
        tool_results: Optional[List[Tuple[ToolCall, str, int, float]]] = None,
        alias_map: Optional[NavAliasMap] = None,
        kit: Optional[KnowledgeNavToolKit] = None,
        thinking_ms: Optional[float] = None,
    ) -> None:
        """落 assistant 消息到 MongoDB（thinking / thinking_ms / tool_calls / citations / usage）

        Args:
            tool_results: 本轮工具执行结果，``[(tc, content, items_added, time_ms), ...]``。
                传入时会按 ``tc.id`` 合并到 ``ToolCallRecord.result_brief`` /
                ``items_added``——这样历史回放时前端也能拿到工具结果。
                None 表示 LLM 这轮没返工具调用。
            alias_map: 若提供，会把"本轮新分配的 alias delta"写入
                ``metadata['alias_additions']``，便于下一 turn 重建累加。
                tool_calls 入参按模型原样（alias 口径 c1/s1/d1）落库——与流式
                帧、下一轮上下文装配、历史回放保持同一口径；真实 id 需要时
                由 metadata deltas 重建（rebuild_alias_map_from_history）。
            thinking_ms: 本轮思考墙钟耗时（毫秒）；无思考则为 None。
        """
        try:
            # tc.id → (result_brief, items_added, time_ms, execution_model)
            results_by_id: Dict[
                str, Tuple[str, int, Optional[float], Optional[str]]
            ] = {}
            for tr_tc, tr_content, tr_items_added, tr_time_ms in (tool_results or []):
                execution_model = (
                    kit._execution_models.get(tr_tc.id)
                    if kit is not None
                    else None
                )
                results_by_id[tr_tc.id] = (
                    tr_content or "",
                    tr_items_added,
                    tr_time_ms,
                    execution_model,
                )

            tool_call_records = []
            for tc in (resp.tool_calls or []):
                brief, items_added, tc_time_ms, execution_model = results_by_id.get(
                    tc.id, (None, 0, None, None),
                )
                # 入参按模型原样落库（模型写的是 alias 就存 alias）：
                # 下一轮上下文装配与本轮回放看到同一口径，别名的省 token /
                # 防泄漏目标不在多轮间被 unwrap 破坏；真实 id 由 metadata
                # deltas 重建 alias map 反查
                args = dict(tc.arguments or {})
                extra_kwargs: Dict[str, Any] = {}
                if kit and tc.name == "search_knowledge_base":
                    sr = kit.search_results.get(tc.id)
                    if sr:
                        extra_kwargs["retrieval_chunks"] = sr[0]
                        extra_kwargs["retrieval_params"] = sr[1]
                        if len(sr) > 2 and sr[2] is not None:
                            extra_kwargs["recall_stats"] = sr[2]
                if tc_time_ms is not None:
                    extra_kwargs["time_ms"] = tc_time_ms
                if execution_model:
                    extra_kwargs["execution_model"] = execution_model
                tool_call_records.append(
                    ToolCallRecord(
                        id=tc.id,
                        name=tc.name,
                        arguments=args,
                        result_brief=brief,
                        items_added=items_added,
                        **extra_kwargs,
                    )
                )
            usage = TokenUsageRecord(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                thinking_tokens=(
                    getattr(resp.usage, "thinking_tokens", None)
                ),
                cached_tokens=getattr(resp.usage, "cached_tokens", 0) or 0,
            )
            metadata: Dict[str, Any] = {"model": resp.model or ""}
            if alias_map is not None:
                delta = alias_map.consume_turn_delta()
                if delta:
                    metadata[METADATA_ALIAS_ADDITIONS_KEY] = delta
                section_delta = alias_map.consume_section_turn_delta()
                if section_delta:
                    metadata[METADATA_SECTION_ALIAS_ADDITIONS_KEY] = section_delta
                document_delta = alias_map.consume_document_turn_delta()
                if document_delta:
                    metadata[METADATA_DOCUMENT_ALIAS_ADDITIONS_KEY] = document_delta
            await chat_message_repo.create(
                creator=ctx.user_id,
                _id=message_id,
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                role=ChatRole.ASSISTANT.value,
                content=resp.content or "",
                thinking=(resp.thinking.reasoning if resp.thinking else None),
                thinking_ms=thinking_ms,
                tool_calls=tool_call_records,
                citations=citations,
                usage=usage,
                finish_reason=resp.finish_reason,
                metadata=metadata,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"持久化 assistant 失败: msg={message_id}, err={e}",
            )

    @staticmethod
    async def _persist_tool_message(
        *,
        ctx: ChatTurnContext,
        message_id: str,
        tc: ToolCall,
        content: str,
    ) -> None:
        """落 role=tool 消息"""
        try:
            await chat_message_repo.create(
                creator=ctx.user_id,
                _id=message_id,
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                role=ChatRole.TOOL.value,
                content=content,
                tool_call_id=tc.id,
                metadata={"tool_name": tc.name},
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"持久化 tool 消息失败: msg={message_id}, err={e}")

    # ============================================================
    # 子功能：citations 合并
    # ============================================================

    @staticmethod
    async def _build_citations_for_round(
        *,
        seed_hits: List[ChunkItem],
        added_chunks: List[ChunkItem],
        enrich_cache: TurnEnrichCache,
        alias_map: Optional[NavAliasMap] = None,
        extra_citations: Optional[List[Citation]] = None,
    ) -> List[Citation]:
        """构造 assistant 落库 / message.done 用的 citations，并把渲染元数据 enrich。

        调用方应传入本 user turn 内需要覆盖的 chunk 集合，典型为::

            seed_hits + supplemented   # 工具链路上累计补充（去重由本函数完成）

        这样任意一轮 assistant 的正文里引用 ``[c1]``（种子）或工具返回的
        ``[cN]``，同一条消息的 ``citations`` 都能带上 ``alias``，刷新后前端可解析。

        - 去重按 ``chunk_id``。
        - 元数据通过 ``enrich_cache`` 共享，相同 chunk_id 跨轮不会重复查表。
        - ``alias_map`` 若提供则给每条 Citation 填 ``alias``（``alias_for`` 须在
          更早阶段已为相关 chunk 分配过）。
        - ``extra_citations``：来自更早 assistant 段的已 enrich citations（跨 turn 引用），
          直接合并到结果中（不重新 enrich）。
        """
        seen: set = set()
        unique_chunks: List[ChunkItem] = []
        for c in list(seed_hits) + list(added_chunks):
            if not c.chunk_id or c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            unique_chunks.append(c)

        # extra_citations 已经是 enriched 的 Citation，无需再走 enrich 流程
        # 先收集它们的 chunk_id，后续去重合并
        extra_by_id: Dict[str, Citation] = {}
        for ec in (extra_citations or []):
            if ec.chunk_id and ec.chunk_id not in seen:
                extra_by_id[ec.chunk_id] = ec

        if not unique_chunks and not extra_by_id:
            return []

        # 触发本轮所需 chunks 的 enrich（命中缓存的部分零成本）
        if unique_chunks:
            try:
                await enrich_cache.ensure(unique_chunks)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"enrich citations 失败（按裸数据落库）: {e}")

        items: List[Citation] = []
        # 调试统计：本轮 chunks 的 file_id / file_name / section_title 命中率
        hit_file = 0
        hit_section = 0
        for c in unique_chunks:
            meta = enrich_cache.get(c.chunk_id)
            if meta and meta.file_id:
                hit_file += 1
            if meta and meta.section_title:
                hit_section += 1
            preview = (c.text or "").strip()
            items.append(Citation(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                knowledge_base_id=c.knowledge_base_id,
                score=c.score or 0.0,
                alias=alias_map.alias_of(c.chunk_id) if alias_map else None,
                chunk_type=meta.chunk_type if meta else None,
                page_index=meta.page_index if meta else None,
                section_title=meta.section_title if meta else None,
                file_id=meta.file_id if meta else None,
                file_name=meta.file_name if meta else None,
                preview=preview or None,
            ))
        total = len(unique_chunks)
        if total:
            logger.debug(
                f"[citations] 本轮 enrich 命中：file_id {hit_file}/{total}，"
                f"section_title {hit_section}/{total}（缓存 size={enrich_cache.size}）"
            )
        # 合并跨 turn 的已 enrich citations
        if extra_by_id:
            items.extend(extra_by_id.values())
            logger.debug(f"[citations] 合并跨 turn citations: {len(extra_by_id)} 条")
        if hit_file == 0 and total > 0:
            sample_cids = ", ".join(c.chunk_id for c in unique_chunks[:3])
            sample_dids = ", ".join((c.document_id or "?") for c in unique_chunks[:3])
            logger.warning(
                f"[citations] ⚠ 所有 chunks 都没拿到 file_id；样例 chunk_id=[{sample_cids}]，"
                f"document_id=[{sample_dids}]。请检查 workspace_file_system 表中 "
                f"(user_id, document_id) 是否存在记录。"
            )
        return items

    # 正则：提取 LLM 输出中实际引用的 alias，如 [c1]、[c12]
    _CITED_ALIAS_RE = re.compile(r"\[c(\d{1,6})\]", re.IGNORECASE)

    @staticmethod
    def _filter_cited_only(
        content: str,
        citations: List[Citation],
    ) -> List[Citation]:
        """仅保留 LLM 正文中实际引用（``[cN]``）过的 citations。

        - 扫描 ``content`` 里所有 ``[cN]`` 占位符，收集被引用的 alias 集合。
        - 从完整 ``citations`` 中筛选 ``alias in cited_aliases``。
        - 若 content 为空或没有任何引用，返回空列表（不再返回全部）。
        - 保持原始顺序（被引用的 citation 按在 citations 中的出现位置排列）。
        """
        if not content or not citations:
            return []
        cited_aliases: Set[str] = set()
        for m in ChatService._CITED_ALIAS_RE.finditer(content):
            cited_aliases.add(f"c{m.group(1)}")
        if not cited_aliases:
            logger.debug(
                "[citations] LLM 正文中未发现任何 [cN] 引用，返回空 citations"
            )
            return []
        filtered = [
            c for c in citations
            if c.alias and c.alias.lower() in cited_aliases
        ]
        logger.debug(
            f"[citations] 过滤：LLM 引用了 {len(cited_aliases)} 个 alias，"
            f"原始 {len(citations)} 条 → 保留 {len(filtered)} 条"
        )
        return filtered

    @staticmethod
    def _merge_citations_count(
        seed_hits: List[ChunkItem],
        supplemented: List[ChunkItem],
    ) -> int:
        seen: set = set()
        for c in list(seed_hits) + list(supplemented):
            if c.chunk_id:
                seen.add(c.chunk_id)
        return len(seen)

    @staticmethod
    def _chunk_brief(
        c: ChunkItem,
        *,
        meta: Optional[ChunkMeta] = None,
        alias: Optional[str] = None,
    ) -> Dict[str, Any]:
        """retrieval.done / tool_call.completed 帧里的 chunk 简要表示。

        在方案 B 下，前端拿到种子 chunks 时就直接能用 meta 字段渲染彩色 chip。
        Phase B 后多带一个 ``alias`` 字段（``c1`` / ``c2`` ...）；前端把 LLM
        输出里的 ``[c1]`` 解析为 chip 时按 alias 反查 citation。
        meta=None / alias=None 时降级输出（兼容路径，前端按 Optional 兜底）。
        """
        brief: Dict[str, Any] = {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "score": c.score,
            "preview": (c.text or ""),
        }
        if alias:
            brief["alias"] = alias
        if meta is not None:
            brief.update({
                "chunk_type": meta.chunk_type,
                "page_index": meta.page_index,
                "section_title": meta.section_title,
                "file_id": meta.file_id,
                "file_name": meta.file_name,
            })
            # 图片 chunk：传递存储路径，前端可按需请求 presigned URL
            if (
                meta.chunk_type == "image"
                and meta.image_file_path
                and meta.bucket_name
            ):
                brief["image_file_path"] = meta.image_file_path
                brief["bucket_name"] = meta.bucket_name
        return brief

    # ============================================================
    # 子功能：上下文压缩
    # ============================================================

    def _apply_context_compression(
        self, history: List[ChatMessage]
    ) -> List[ChatMessage]:
        """应用上下文压缩：如果有 summary 消息，跳过已总结的旧消息

        逻辑：
        1. 查找最新的 summary 消息
        2. 如果找到，过滤掉 metadata.summarized=true 的消息
        3. 保留 summary 消息本身 + summary 之后的新消息

        Args:
            history: 原始历史消息列表

        Returns:
            压缩后的历史消息列表
        """
        # 查找最新的 summary 消息
        latest_summary = None
        latest_summary_idx = -1
        for i, msg in enumerate(history):
            if msg.role == "summary":
                latest_summary = msg
                latest_summary_idx = i

        if latest_summary is None:
            # 没有 summary，返回原始历史
            return history

        # 过滤逻辑：
        # 1. 保留 summary 消息
        # 2. 跳过 summary 之前且被标记为 summarized 的消息
        # 3. 保留 summary 之后的所有消息（无论是否标记）
        compressed = []
        for i, msg in enumerate(history):
            if i == latest_summary_idx:
                # 保留 summary 消息
                compressed.append(msg)
            elif i < latest_summary_idx:
                # summary 之前的消息：跳过已总结的
                if not msg.metadata.get("summarized", False):
                    compressed.append(msg)
            else:
                # summary 之后的消息：全部保留
                compressed.append(msg)

        logger.debug(
            f"上下文压缩: 原始 {len(history)} 条 → 压缩后 {len(compressed)} 条"
        )
        return compressed

    # ============================================================
    # 子功能：摘要回调（fast preset）
    # ============================================================

    def _get_context_budgeter(self) -> ContextBudgeter:
        return ContextBudgeter(
            catalog=get_model_context_catalog(),
            threshold_ratio=self._cfg.summary_compress_threshold_ratio,
            reserved_output_fallback=self._cfg.reserved_output_fallback,
            heuristic_safety_factor=self._cfg.heuristic_safety_factor,
            tool_output_hard_cap_tokens=self._cfg.tool_output_hard_cap_tokens,
        )

    def _build_summarize_fn(self, *, model: Optional[str] = None):
        """返回 HierarchicalSummarizer 驱动的 ``summarize_fn``。

        签名兼容 ``Callable[[Sequence[Any]], Awaitable[str]]``，供
        ``compaction_keep_recent_turns`` / 手动摘要复用。
        """
        summarize_preset = self._cfg.agent_model_preset
        cfg = self._cfg

        async def _generate(messages: List[dict], max_tokens: int) -> str:
            try:
                client = self._get_llm_client(
                    model=model, model_preset=summarize_preset,
                )
            except Exception:  # noqa: BLE001
                from src.client.llm import create_llm_client_from_preset

                client = create_llm_client_from_preset(summarize_preset)
            resp = await client.agenerate(
                messages=messages, temperature=0.2, max_tokens=max_tokens,
            )
            return (resp.content or "").strip()

        summarizer = HierarchicalSummarizer(
            generate_fn=_generate,
            model=model or summarize_preset,
            catalog=get_model_context_catalog(),
            budgeter=self._get_context_budgeter(),
            chunk_budget_tokens=cfg.summary_chunk_tokens,
            summary_target_tokens=cfg.summary_target_tokens,
            map_concurrency=cfg.summary_map_concurrency,
            heuristic_safety_factor=cfg.heuristic_safety_factor,
        )

        async def _summarize(early_history) -> str:
            old_summary = None
            msgs = []
            for m in early_history:
                role = getattr(m, "role", None)
                if role == "summary":
                    content = (getattr(m, "content", None) or "").strip()
                    if content:
                        # 多条旧 summary 时取最后一条作为 old_summary，其余当普通消息
                        if old_summary:
                            msgs.append(m)
                        old_summary = content
                    continue
                msgs.append(m)
            result = await summarizer.summarize(
                msgs,
                old_summary=old_summary,
                target_tokens=cfg.summary_target_tokens,
            )
            # 把计量元数据挂到函数属性，供 session_service 写入 summary.metadata
            _summarize.last_result = result  # type: ignore[attr-defined]
            return result.summary_text

        _summarize.last_result = None  # type: ignore[attr-defined]
        return _summarize

    # ============================================================
    # 辅助：取首条 assistant 的 content
    # ============================================================

    @staticmethod
    async def _fetch_first_assistant_text(message_id: str) -> str:
        try:
            obj = await chat_message_repo.get_by_id(message_id)
            if obj and obj.content:
                return obj.content
        except Exception:  # noqa: BLE001
            pass
        return ""


# ============================================================
# 辅助：assistant message 序列化
# ============================================================


def _assistant_message(resp: LLMResponse) -> Dict[str, Any]:
    """把 LLMResponse 转成 OpenAI/LiteLLM 协议的 assistant message dict

    特殊处理
    --------
    - DeepSeek 在 thinking mode（含 ``deepseek-reasoner`` 和开启 ``reasoning_effort``
      的 ``deepseek-chat``）下，要求上一轮 assistant 的 ``reasoning_content``
      必须**原样**回传到下一轮 messages，否则 round 1+ 会被 DeepSeek 以
      ``"The reasoning_content in the thinking mode must be passed back to
      the API."`` 报 400。这里只要 ``resp.thinking`` 非空就附带 ``reasoning_content``
      字段；其他厂商（OpenAI / Anthropic / Qwen 等）会忽略未知字段，不影响兼容性。
    - 当包含 tool_calls 且无主文本内容时，content 设为 None（JSON null），
      符合 OpenAI 标准协议并避免上游网关反序列化报 `missing field text`。
    """
    content = resp.content if (resp.content or not resp.tool_calls) else None
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    if resp.thinking and resp.thinking.reasoning:
        msg["reasoning_content"] = resp.thinking.reasoning
    if resp.tool_calls:
        import json as _json

        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": (
                        _json.dumps(tc.arguments, ensure_ascii=False)
                        if isinstance(tc.arguments, dict)
                        else str(tc.arguments)
                    ),
                },
            }
            for tc in resp.tool_calls
        ]
    return msg


__all__ = [
    "ChatService",
    "ChatServiceConfig",
]
