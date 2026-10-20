#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : session_service.py
@Author  : caixiongjiang
@Date    : 2026/05/11
@Function:
    ChatSessionService ── 会话级业务封装

    职责定位
    --------
    把 ``ChatSessionRepository``（MySQL CRUD）和 ``ChatMessageRepository``
    （MongoDB CRUD）封装成"业务语义层"的会话管理 API，对外暴露：

    - ``create_session``：分配 UUID 风格 ``session_id`` 并落 MySQL。
    - ``get_session`` / ``list_sessions``：含 user_id 权限校验。
    - ``rename_session`` / ``soft_delete_session``：含权限校验；
      软删 session 时同步软删 MongoDB 端的消息（避免遗留孤儿）。
    - ``load_history``：从 MongoDB 拉取消息（按 create_time 正序），
      为 ChatService 主流程提供历史 ``ChatMessage`` 列表。

    Phase 3 设计取舍
    ----------------
    - 不接管 MySQL session 的生命周期（用 ``get_mysql_manager().get_session()``
      上下文管理器），保持与现有 business / extract 层一致；
    - 历史加载默认"全量按时间正序"，由 ChatService 自己决定是否做
      ``apply_token_window`` / ``compress_history_to_summary``；
    - 会话计数维护：``ChatService`` 在新消息落库后调用
      ``ChatSessionRepository.touch``，**不**在本服务集中维护——
      因为单轮可能落 N 条消息（user + 多轮 assistant/tool），
      把 touch 留给写消息的人控制更准确。
@Modify History:
    2026-05-11 - 首版（Phase 3）
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from loguru import logger

from src.db.mongodb.models.conversation.chat_message import ChatMessage
from src.db.mongodb.repositories.conversation import chat_message_repo
from src.db.mysql.connection.factory import get_mysql_manager
from src.db.mysql.models.conversation.chat_session import ChatSession
from src.db.mysql.repositories.conversation import chat_session_repo


def generate_session_id() -> str:
    """生成与项目约定一致的会话 ID：``sess_<uuid_hex16>``"""
    return f"sess_{uuid.uuid4().hex[:16]}"


def generate_message_id() -> str:
    """生成与项目约定一致的消息 ID：``chatmsg_<uuid_hex32>``"""
    return f"chatmsg_{uuid.uuid4().hex}"


class ChatSessionService:
    """会话 CRUD + 历史加载

    本类是无状态的薄业务包装；可直接 ``ChatSessionService()`` 创建实例。
    传入的 ``session_repo`` / ``message_repo`` 默认走全局单例，便于测试时
    注入 mock。
    """

    def __init__(
        self,
        *,
        session_repo=chat_session_repo,
        message_repo=chat_message_repo,
    ) -> None:
        self._session_repo = session_repo
        self._message_repo = message_repo

    # ============================================================
    # 创建
    # ============================================================

    def create_session(
        self,
        *,
        user_id: str,
        title: str = "新会话",
        knowledge_base_ids: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        include_subfolders: bool = True,
        model_preset: str = "fast",
        model: Optional[str] = None,
        mode: str = "agent",
        thinking_level: str = "off",
        enable_multimodal: bool = False,
        max_tool_rounds: int = 5,
        system_prompt: Optional[str] = None,
    ) -> Optional[ChatSession]:
        """创建新会话；返回完整记录对象，失败返回 None。

        Args:
            user_id: 创建者用户 ID
            title: 初始标题（一般用"新会话"占位；首条 user 消息后由
                ``TitleService`` 异步覆盖）
            knowledge_base_ids: 本会话允许检索的知识库 ID 列表；
                ``None`` / 空列表表示放开到用户全量 KB
            folder_id: 可选，会话绑定的文件夹 ID。传入后启用 folder scope，
                每轮检索范围限定在该文件夹下文档。会校验：
                - folder 必须属于该 ``user_id``；
                - folder 所属 KB 必须 ∈ ``knowledge_base_ids``（非空时）；
                - 若 ``knowledge_base_ids`` 为空，会自动用 folder 所属 KB 填上一份；
                校验失败抛 ``ValueError`` 让调用方转 422。
            include_subfolders: folder scope 是否含子文件夹，默认 True
            model_preset: ``[llm.presets.*]`` 名称（后台 agent 仍走 preset）
            model: LiteLLM 模型字符串；``None`` 表示由 ``model_preset`` 决定。
                与 ``model_preset`` 并存：``model`` 非空时优先用它选模型，
                ``model_preset`` 仍作为 temperature / max_tokens 等采样参数模板
                （详见 ``ChatService._get_llm_client``）；思考强度由前端 thinking_level 逐轮传入。
            thinking_level: 默认思考强度档位（off/minimal/low/medium/high/xhigh/max）
            enable_multimodal: 默认是否启用多模态读图
            max_tool_rounds: Agent 模式默认工具批次上限
            system_prompt: 用户自定义 system_prompt（``None`` 表示用模块默认）
        """
        kb_ids = list(knowledge_base_ids or [])

        # ===== folder_id 跨 KB 一致性校验 =====
        if folder_id:
            kb_ids = self._validate_folder_against_kb(
                user_id=user_id,
                folder_id=folder_id,
                knowledge_base_ids=kb_ids,
            )

        # ===== 知识库归属校验（H3）：每个 kb 必须对创建者可读 =====
        # 不校验时，任意登录用户可把他人 kb_id 塞进会话；
        # 个人库时代靠 Milvus user_id 过滤兜底，共享库上线后该兜底
        # 将按可见性分流，必须在会话创建口把住第一道门。
        if kb_ids:
            self._validate_kb_read(user_id=user_id, kb_ids=kb_ids)

        sess_id = generate_session_id()
        manager = get_mysql_manager()
        with manager.get_session() as db:
            obj = self._session_repo.create(
                db,
                session_id=sess_id,
                user_id=user_id,
                title=title,
                knowledge_base_ids=kb_ids,
                folder_id=folder_id,
                include_subfolders=include_subfolders,
                model_preset=model_preset,
                model=model,
                mode=mode,
                thinking_level=thinking_level,
                enable_multimodal=enable_multimodal,
                max_tool_rounds=max_tool_rounds,
                system_prompt=system_prompt,
                creator=user_id,
            )
            if obj is None:
                logger.warning(f"创建会话失败: user={user_id}")
                return None
            logger.info(
                f"创建会话: session_id={sess_id}, user={user_id}, "
                f"model_preset={model_preset}, model={model or '-'}, "
                f"scope={'folder=' + folder_id if folder_id else 'kb'}"
            )
            return obj

    # ============================================================
    # folder scope 校验辅助
    # ============================================================

    def _validate_folder_against_kb(
        self,
        *,
        user_id: str,
        folder_id: str,
        knowledge_base_ids: List[str],
    ) -> List[str]:
        """校验 folder_id 与 knowledge_base_ids 的一致性。

        校验规则（must_match）：

        1. folder_id 必须存在且属于 ``user_id``；
        2. folder 所属 KB 必须 ∈ ``knowledge_base_ids``（非空时）；
        3. 若 ``knowledge_base_ids`` 为空 → 自动用 folder 所属 KB 填上一份。

        Returns:
            校验后的 knowledge_base_ids（必要时已自动填充 folder 的 KB）

        Raises:
            ValueError: 校验失败时抛出，建议上游转 422
        """
        from src.db.mysql.repositories.business.workspace_folder_repo import (
            workspace_folder_repo,
        )

        manager = get_mysql_manager()
        with manager.get_session() as db:
            folder = db.query(workspace_folder_repo.model).filter(
                workspace_folder_repo.model.folder_id == folder_id,
                workspace_folder_repo.model.user_id == user_id,
                workspace_folder_repo.model.deleted == 0,
            ).first()
            if not folder:
                raise ValueError(
                    f"folder_id={folder_id} 不存在或不属于当前用户"
                )
            folder_kb = folder.knowledge_base_id
            if not folder_kb:
                # 数据脏：folder 没绑定 KB；保守起见允许通过但记录 warning
                logger.warning(
                    f"folder_id={folder_id} 没有绑定 knowledge_base_id；"
                    "跳过跨 KB 校验"
                )
                return knowledge_base_ids

            if not knowledge_base_ids:
                logger.info(
                    f"create_session 自动填充 KB：folder_id={folder_id} → "
                    f"knowledge_base_ids=[{folder_kb}]"
                )
                return [folder_kb]

            if folder_kb not in knowledge_base_ids:
                raise ValueError(
                    f"folder_id={folder_id} 所属 knowledge_base_id={folder_kb} "
                    f"不在请求的 knowledge_base_ids={knowledge_base_ids} 中"
                )
            return knowledge_base_ids

    def _validate_kb_read(
        self,
        *,
        user_id: str,
        kb_ids: List[str],
    ) -> None:
        """校验会话绑定的每个知识库对创建者可读（KBAccess 统一判定）。

        Raises:
            ValueError: 存在不可读的知识库（上游转 422）
        """
        from src.db.mysql.repositories.business.knowledge_base_repo import (
            knowledge_base_repo,
        )
        from src.service.knowledge.kb_access import resolve_kb_access

        manager = get_mysql_manager()
        with manager.get_session() as db:
            denied = []
            for kb_id in kb_ids:
                kb = knowledge_base_repo.get_by_id(db, kb_id)
                if kb is None or not resolve_kb_access(db, user_id, kb).can_read:
                    denied.append(kb_id)
        if denied:
            raise ValueError(
                f"无权访问以下知识库: {', '.join(denied)}"
            )

    # ============================================================
    # 查询
    # ============================================================

    def get_session(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> Optional[ChatSession]:
        """按会话 ID + user_id 加载（含权限校验，跨用户返回 None）"""
        manager = get_mysql_manager()
        with manager.get_session() as db:
            return self._session_repo.get_by_id_and_user(db, session_id, user_id)

    def list_sessions(
        self,
        *,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[ChatSession], int]:
        """分页拉用户的会话列表（按 ``last_message_at`` 倒序）"""
        manager = get_mysql_manager()
        with manager.get_session() as db:
            return self._session_repo.list_by_user(
                db, user_id, limit=limit, offset=offset,
            )

    # ============================================================
    # 修改
    # ============================================================

    def rename_session(
        self,
        *,
        session_id: str,
        user_id: str,
        title: str,
    ) -> Optional[ChatSession]:
        """重命名会话（含权限校验）"""
        manager = get_mysql_manager()
        with manager.get_session() as db:
            return self._session_repo.rename(
                db, session_id, user_id, title, updater=user_id,
            )

    def touch_session(
        self,
        *,
        session_id: str,
        delta: int = 1,
    ) -> bool:
        """新消息落库后刷新 ``message_count`` + ``last_message_at``

        典型在 ``ChatService`` 一轮结束、累计 ``delta=`` 实际新增条数后一次性
        调用，避免每条消息触发一次 MySQL update。
        """
        if delta <= 0:
            return True
        manager = get_mysql_manager()
        with manager.get_session() as db:
            return self._session_repo.touch(db, session_id, delta=delta)

    def update_session_mode(
        self,
        *,
        session_id: str,
        user_id: str,
        mode: Optional[str] = None,
        thinking_level: Optional[str] = None,
        max_tool_rounds: Optional[int] = None,
    ) -> bool:
        """首条消息发出后，把用户选择的运行参数回写到 session。"""
        manager = get_mysql_manager()
        with manager.get_session() as db:
            return self._session_repo.update_mode(
                db,
                session_id,
                mode=mode,
                thinking_level=thinking_level,
                max_tool_rounds=max_tool_rounds,
                updater=user_id,
            )

    def update_session_settings(
        self,
        *,
        session_id: str,
        user_id: str,
        model: Optional[str] = None,
        model_preset: Optional[str] = None,
        thinking_level: Optional[str] = None,
    ) -> bool:
        """把"会话级偏好"回写到 session（用户每轮可改的项）。

        与 ``update_session_mode`` 的分工：

        - ``update_session_mode``：首条消息后**锁定** ``mode`` /
          ``max_tool_rounds`` 等"会话定型"项；UI 上对应的 chip 在有消息后
          就是 disabled 的，不能再变。
        - ``update_session_settings``：随时可改的"轻偏好"——前端选了哪个
          ``model``、思考强度档位、用哪个 preset 模板，只要每轮请求带上
          就持久化，下次进同一会话时 UI 默认选项跟随。

        所有参数都是 ``Optional``：``None`` 表示"不动这一项"。
        """
        manager = get_mysql_manager()
        with manager.get_session() as db:
            return self._session_repo.update_settings(
                db,
                session_id,
                model=model,
                model_preset=model_preset,
                thinking_level=thinking_level,
                updater=user_id,
            )

    # ============================================================
    # 清空消息
    # ============================================================

    async def clear_messages(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> bool:
        """清空会话内的所有消息（保留会话本身）

        权限：仅 ``user_id`` 匹配的本人可操作。

        Args:
            session_id: 会话 ID
            user_id: 当前用户 ID（用于权限校验）

        Returns:
            True 表示清空成功，False 表示会话不存在或无权限
        """
        # 1. 权限校验：检查会话是否存在且属于当前用户
        manager = get_mysql_manager()
        with manager.get_session() as db:
            session = self._session_repo.get_by_id_and_user(db, session_id, user_id)
            if session is None:
                return False

        # 2. 软删除 MongoDB 中的所有消息
        try:
            cnt = await self._message_repo.soft_delete_by_session(
                session_id, updater=user_id,
            )
            logger.info(
                f"清空会话消息: session_id={session_id}, user={user_id}, "
                f"软删消息 {cnt} 条"
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"清空会话消息失败: session={session_id}, err={e}")
            return False

        # 3. 重置会话的 message_count 和 last_message_at
        with manager.get_session() as db:
            self._session_repo.reset_message_count(db, session_id)

        return True

    async def summarize_context(
        self,
        *,
        session_id: str,
        user_id: str,
        summarize_fn: Optional[Callable[..., Awaitable[str]]] = None,
    ) -> Optional[str]:
        """手动 /summary：压缩全部未总结消息（``keep_recent_turns=0``）。

        与自动 compaction 共用同一管线，仅保留轮数参数不同。
        """
        return await self.compaction_keep_recent_turns(
            session_id=session_id,
            user_id=user_id,
            keep_recent_turns=0,
            summarize_fn=summarize_fn,
            trigger="manual",
        )

    # ============================================================
    # Cursor 式持久化上下文压缩
    # ============================================================

    async def compaction_keep_recent_turns(
        self,
        *,
        session_id: str,
        user_id: str,
        keep_recent_turns: int = 1,
        summarize_fn: Optional[Callable[..., Awaitable[str]]] = None,
        trigger: str = "auto_threshold",
        budget_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Cursor 式持久化上下文压缩：保留最近 ``keep_recent_turns`` 轮 + 1 条摘要。

        - ``keep_recent_turns=0``：压缩全部消息（手动 /summary）
        - ``keep_recent_turns>=1``：保留最近 N 轮原始消息
        - 旧 ``role="summary"`` 内容并入新摘要（summary of summary）
        - summary.metadata 写入可观测字段（trigger / tokens / version 等）
        """
        # 1. 权限
        manager = get_mysql_manager()
        with manager.get_session() as db:
            session = self._session_repo.get_by_id_and_user(db, session_id, user_id)
            if session is None:
                return None

        # 2. 全量加载
        all_messages = await self._message_repo.list_by_session(
            session_id, limit=100000, ascending=True,
        )
        if not all_messages:
            return None

        # 3. 定位最近 N 轮起点
        keep_recent_turns = max(0, int(keep_recent_turns))
        user_indices = [i for i, m in enumerate(all_messages) if m.role == "user"]
        if keep_recent_turns == 0:
            recent_start = len(all_messages)
            early_part = list(all_messages)
        else:
            if len(user_indices) <= keep_recent_turns:
                return None
            recent_start = user_indices[-keep_recent_turns]
            early_part = all_messages[:recent_start]
        if not early_part:
            return None

        # 若 early 全空内容则退出
        if not any((m.content or "").strip() for m in early_part):
            return None

        # 统计旧 summary
        old_summary_ids = [m.id for m in early_part if m.role == "summary"]
        # summary_version：已有 summary 条数 + 1
        existing_summary_count = sum(1 for m in all_messages if m.role == "summary")

        # 4. 生成摘要
        summary_meta_extra: Dict[str, Any] = {}
        try:
            if summarize_fn is not None:
                summary_content = await summarize_fn(early_part)
                last = getattr(summarize_fn, "last_result", None)
                if last is not None:
                    summary_meta_extra = {
                        "input_tokens": getattr(last, "input_tokens", None),
                        "summary_tokens": getattr(last, "summary_tokens", None),
                        "counting": getattr(last, "counting", None),
                        "chunk_count": getattr(last, "chunk_count", None),
                    }
            else:
                from src.client.llm import (
                    create_llm_client_from_model,
                    create_llm_client_from_preset,
                )
                from src.service.chat.context import (
                    HierarchicalSummarizer,
                    get_model_context_catalog,
                )

                if session.model:
                    client = create_llm_client_from_model(
                        model=session.model,
                        chat_template_preset=session.model_preset or "fast",
                    )
                    model_name = session.model
                else:
                    client = create_llm_client_from_preset(session.model_preset or "fast")
                    model_name = getattr(client, "model", None) or (session.model_preset or "fast")

                async def _generate(messages: List[dict], max_tokens: int) -> str:
                    resp = await client.agenerate(
                        messages=messages, temperature=0.2, max_tokens=max_tokens,
                    )
                    return (resp.content or "").strip()

                summarizer = HierarchicalSummarizer(
                    generate_fn=_generate,
                    model=model_name,
                    catalog=get_model_context_catalog(),
                )
                old_summary = None
                msgs = []
                for msg in early_part:
                    if msg.role == "summary":
                        content = (msg.content or "").strip()
                        if content:
                            if old_summary:
                                msgs.append(msg)
                            old_summary = content
                        continue
                    msgs.append(msg)
                result = await summarizer.summarize(msgs, old_summary=old_summary)
                summary_content = result.summary_text
                summary_meta_extra = {
                    "input_tokens": result.input_tokens,
                    "summary_tokens": result.summary_tokens,
                    "counting": result.counting,
                    "chunk_count": result.chunk_count,
                }
            summary_content = (summary_content or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.error(f"compaction 生成摘要失败: {e}", exc_info=True)
            return None
        if not summary_content:
            return None

        # 5. 持久化新 summary 消息（带可观测 metadata）
        from datetime import datetime

        meta: Dict[str, Any] = {
            "summary_type": "context_compression",
            "trigger": trigger,
            "compressed_message_count": len(early_part),
            "compressed_range": {
                "first_message_id": early_part[0].id,
                "last_message_id": early_part[-1].id,
            },
            "keep_recent_turns": keep_recent_turns,
            "summary_version": existing_summary_count + 1,
            "merged_summary_ids": old_summary_ids,
            "model": getattr(session, "model", None) or getattr(session, "model_preset", None),
        }
        for k, v in summary_meta_extra.items():
            if v is not None:
                meta[k] = v
        if budget_snapshot:
            meta["budget_snapshot"] = {
                "used_tokens": budget_snapshot.get("used_tokens"),
                "soft_limit": budget_snapshot.get("soft_limit"),
                "ratio": budget_snapshot.get("ratio"),
                "counting": budget_snapshot.get("counting"),
            }

        summary_msg = ChatMessage(
            id=generate_message_id(),
            session_id=session_id,
            user_id=user_id,
            role="summary",
            content=summary_content,
            metadata=meta,
            create_time=datetime.now(),
            update_time=datetime.now(),
        )
        await summary_msg.insert()

        # 6. 标记 early_part 全部消息为已总结（含旧 summary）
        early_ids = [m.id for m in early_part]
        if early_ids:
            await self._message_repo.mark_as_summarized(early_ids, updater=user_id)

        logger.info(
            f"上下文压缩(compaction): session={session_id}, "
            f"trigger={trigger}, early={len(early_part)} 条已标记, "
            f"keep_recent_turns={keep_recent_turns}"
        )
        return summary_content

    # ============================================================
    # 删除
    # ============================================================

    async def soft_delete_session(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> bool:
        """软删除会话 + 级联软删该会话下所有消息

        权限：仅 ``user_id`` 匹配的本人可删。
        """
        manager = get_mysql_manager()
        with manager.get_session() as db:
            ok = self._session_repo.soft_delete_by_user(
                db, session_id, user_id, updater=user_id,
            )
        if not ok:
            return False
        # 级联软删 MongoDB 消息
        try:
            cnt = await self._message_repo.soft_delete_by_session(
                session_id, updater=user_id,
            )
            logger.info(
                f"软删会话: session_id={session_id}, user={user_id}, "
                f"级联软删消息 {cnt} 条"
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"级联软删消息失败但 session 已软删: session={session_id}, err={e}"
            )
        return True

    # ============================================================
    # 历史加载
    # ============================================================


    @staticmethod
    def _apply_context_compression(history: List[Any]) -> List[Any]:
        """与 ChatService._apply_context_compression 语义对齐的轻量实现。"""
        latest_summary_idx = -1
        for i, msg in enumerate(history):
            if getattr(msg, "role", None) == "summary":
                latest_summary_idx = i
        if latest_summary_idx < 0:
            return list(history)
        compressed = []
        for i, msg in enumerate(history):
            if i == latest_summary_idx:
                compressed.append(msg)
            elif i < latest_summary_idx:
                meta = getattr(msg, "metadata", None) or {}
                if not meta.get("summarized", False):
                    compressed.append(msg)
            else:
                compressed.append(msg)
        return compressed

    @staticmethod
    def _rebuild_system_prompt_for_status(session: Any) -> tuple[str, str]:
        """重建"若现在发一轮"的 system prompt，用于 context-status 计量。

        返回 ``(system_prompt, skills_index)``——技能索引已嵌在 prompt 内，
        单独回传一份供 Skills 分项扣除。

        与 ``ChatService._merge_turn_config`` 的构造对齐：会话自定义提示词优先，
        否则按 Agent 固定工具集 + folder scope + 技能索引渲染。folder 的
        ``document_count`` 不在此解析（仅影响一行文本），按 0 估。
        """
        custom = getattr(session, "system_prompt", None)
        if custom:
            return str(custom), ""

        from src.prompts.chat import build_chat_system_prompt
        from src.service.chat.tools.registry import AGENT_ENABLED_TOOLS

        scope: Optional[Dict[str, Any]] = None
        folder_id = getattr(session, "folder_id", None)
        if folder_id:
            scope = {
                "kind": "folder",
                "folder_id": folder_id,
                "label": folder_id,
                "include_subfolders": bool(
                    getattr(session, "include_subfolders", True),
                ),
                "document_count": 0,
                "knowledge_base_ids": list(
                    getattr(session, "knowledge_base_ids", None) or [],
                ),
            }

        skills_index = None
        try:
            from src.service.skill.registry_singleton import get_registry

            skills_index = get_registry().build_index(set(AGENT_ENABLED_TOOLS))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"context-status 技能索引构建失败（忽略）: {e}")

        try:
            prompt = build_chat_system_prompt(
                enabled_tools=AGENT_ENABLED_TOOLS,
                scope=scope,
                skills_index=skills_index,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"context-status system prompt 重建失败: {e}")
            return "", ""
        return prompt, (skills_index or "")

    async def get_context_status(
        self,
        *,
        session_id: str,
        user_id: str,
        system_prompt: str = "",
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        skills_block: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """计算会话当前上下文用量（供 REST context-status）。

        对"若现在发一轮空 query"做计量：history + system + tools schema；
        user 按 0 估。``system_prompt`` / ``tools_schema`` 未显式传入时，按下一轮
        真实请求重建，避免这两项恒为 0 导致用量系统性低估。
        会话不存在时返回 None。
        """
        from src.service.chat.chat_service import ChatServiceConfig
        from src.service.chat.context import (
            ContextBudgetInput,
            ContextBudgeter,
            get_model_context_catalog,
        )

        manager = get_mysql_manager()
        with manager.get_session() as db:
            session = self._session_repo.get_by_id_and_user(db, session_id, user_id)
            if session is None:
                return None

        cfg = ChatServiceConfig.from_config_manager()
        history_full = await self.load_full_history(
            session_id=session_id, limit=cfg.history_load_limit,
        )
        # 复用与 ChatService 相同的压缩语义：跳过 summarized，保留最新 summary
        history = self._apply_context_compression(history_full)
        model = session.model or None
        if not model:
            # 回落到 preset 对应模型字符串（若可得）
            try:
                from src.client.llm import create_llm_client_from_preset
                client = create_llm_client_from_preset(session.model_preset or cfg.agent_model_preset)
                model = getattr(client, "model", None) or cfg.agent_model_preset
            except Exception:  # noqa: BLE001
                model = session.model_preset or cfg.agent_model_preset

        rebuilt_skills = ""
        if not system_prompt:
            try:
                system_prompt, rebuilt_skills = (
                    self._rebuild_system_prompt_for_status(session)
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"context-status 重建 system prompt 失败: {e}")
                system_prompt = ""
        if skills_block is None:
            skills_block = rebuilt_skills
        if tools_schema is None:
            from src.service.chat.tools.registry import agent_tools_schema

            tools_schema = agent_tools_schema()

        budgeter = ContextBudgeter(
            catalog=get_model_context_catalog(),
            threshold_ratio=cfg.summary_compress_threshold_ratio,
            reserved_output_fallback=cfg.reserved_output_fallback,
            heuristic_safety_factor=cfg.heuristic_safety_factor,
            tool_output_hard_cap_tokens=cfg.tool_output_hard_cap_tokens,
        )
        reserved = budgeter.resolve_reserved_output(
            model, preset_max_tokens=cfg.max_completion_tokens,
        )
        report = budgeter.evaluate(
            ContextBudgetInput(
                model=model,
                system_prompt=system_prompt or "",
                history=history,
                user_message="",
                tools_schema=tools_schema,
                reserved_output_tokens=reserved,
                skills_block=skills_block or "",
            )
        )

        # last_compaction from latest summary metadata
        last_compaction = None
        summary_count = 0
        latest_summary = None
        for m in history_full:
            if m.role == "summary":
                summary_count += 1
                latest_summary = m
        if latest_summary is not None:
            md = latest_summary.metadata or {}
            at = None
            ct = getattr(latest_summary, "create_time", None)
            if ct is not None:
                try:
                    at = ct.isoformat()
                except Exception:  # noqa: BLE001
                    at = str(ct)
            last_compaction = {
                "at": at,
                "trigger": md.get("trigger"),
                "input_tokens": md.get("input_tokens"),
                "summary_tokens": md.get("summary_tokens"),
            }

        return {
            "session_id": session_id,
            "model": model,
            "max_context": report.max_context,
            "reserved_output": report.reserved_output,
            "used_tokens": report.used_tokens,
            "soft_limit": report.soft_limit,
            "ratio": round(report.ratio, 4),
            "threshold_ratio": cfg.summary_compress_threshold_ratio,
            "will_compact_at": report.will_compact_at,
            "counting": report.counting,
            "breakdown": {
                "system": report.breakdown.get("system", 0),
                "skills": report.breakdown.get("skills", 0),
                "tools_schema": report.breakdown.get("tools_schema", 0),
                "summary": report.breakdown.get("summary", 0),
                "history": report.breakdown.get("history", 0),
                "user": report.breakdown.get("user", 0),
                "reserved_output": report.breakdown.get("reserved_output", 0),
            },
            "last_compaction": last_compaction,
            "summary_count": summary_count,
        }

    async def load_full_history(
        self,
        *,
        session_id: str,
        limit: int = 100000,
    ) -> List[ChatMessage]:
        """加载会话全部历史消息（按 create_time 正序），供 ChatService 做压缩/装配。

        与 ``load_history``（取最近 N 条）不同：本方法从头拉全量，确保摘要压缩
        能看到所有未总结的早期消息。``limit`` 仅作防御性上限。

        Args:
            session_id: 会话 ID（外层应已通过 ``get_session`` 完成权限校验）
            limit: 防御性上限（默认 100000）

        Returns:
            按 create_time 正序的消息列表
        """
        return await self._message_repo.list_by_session(
            session_id, limit=limit, skip=0, ascending=True,
        )

    async def load_history(
        self,
        *,
        session_id: str,
        limit: int = 200,
        skip: int = 0,
    ) -> List[ChatMessage]:
        """加载会话历史消息（按 create_time 正序，便于直接拼回 messages）

        **仅用于 LLM 上下文装配**（``ChatService``），不是 UI 全量回放接口。
        Chat 主流程（``skip=0``）取**最近** ``limit`` 条，而非最早 ``limit`` 条。
        Agent 模式一轮常含多条 assistant/tool 消息；若从头部截断会在
        ``tool_calls`` 与 ``role=tool`` 之间切开，导致下一轮 LLM 400。

        UI 全量回放请走 REST ``GET /sessions/{id}/messages``（从头分页拼全量）。

        Args:
            session_id: 会话 ID（外层应已通过 ``get_session`` 完成权限校验）
            limit: 单次加载上限（默认 200；配置项 ``chat.history.max_messages``）
            skip: 跳过条数。``skip>0`` 时按时间正序从最早消息起算分页。
        """
        if skip > 0:
            return await self._message_repo.list_by_session(
                session_id, limit=limit, skip=skip, ascending=True,
            )
        return await self._message_repo.list_recent_by_session(
            session_id, limit=limit,
        )


__all__ = [
    "ChatSessionService",
    "generate_session_id",
    "generate_message_id",
]
