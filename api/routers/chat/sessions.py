#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : sessions.py
@Author  : caixiongjiang
@Date    : 2026/05/11
@Function:
    Chat 会话 REST 路由

    端点
    ----
      POST   /                        - 创建会话
      GET    /                        - 我的会话列表（分页，按活跃度倒序）
      GET    /{session_id}            - 单条会话详情
      PATCH  /{session_id}            - 重命名（rename）
      DELETE /{session_id}            - 软删除会话（级联软删消息）
      GET    /{session_id}/messages   - 历史消息（分页，按时间正序）

    所有端点都要求 ``X-User-Id`` 请求头（与 Knowledge / Memory 模块一致）；
    带权限校验：跨用户访问统一返回 404 不暴露存在性。
@Modify History:
    2026-05-11 - 首版（Phase 4）
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from api.dependencies.auth import get_current_user_id
from api.schemas.chat import (
    ChatMessageItem,
    ChatMessageListResponse,
    ChatSessionCreateRequest,
    ChatSessionInfo,
    ChatSessionListItem,
    ChatSessionListResponse,
    ChatSessionRenameRequest,
    ChatSessionUpdateResponse,
    CitationItem,
    TokenUsageItem,
    ToolCallItem,
    ContextStatusResponse,
)
from api.schemas.common import ApiResponse
from src.db.mongodb.models.conversation.chat_message import ChatMessage
from src.db.mongodb.repositories.conversation import chat_message_repo
from src.db.mysql.models.conversation.chat_session import ChatSession
from src.service.chat.session_service import ChatSessionService


router = APIRouter(tags=["Chat / Sessions"])


# ============================================================
# 单例服务（与 Knowledge retrieve 路由的做法一致：模块级懒加载）
# ============================================================

_session_service: ChatSessionService | None = None


def _get_service() -> ChatSessionService:
    global _session_service
    if _session_service is None:
        _session_service = ChatSessionService()
    return _session_service


# ============================================================
# DTO 转换
# ============================================================


def _to_session_info(s: ChatSession) -> ChatSessionInfo:
    return ChatSessionInfo(
        session_id=s.session_id,
        user_id=s.user_id,
        title=s.title,
        knowledge_base_ids=list(s.knowledge_base_ids or []),
        folder_id=getattr(s, "folder_id", None),
        include_subfolders=bool(getattr(s, "include_subfolders", True)),
        model_preset=s.model_preset,
        model=s.model,
        mode=getattr(s, "mode", None) or "agent",
        thinking_level=getattr(s, "thinking_level", None) or "off",
        enable_multimodal=bool(getattr(s, "enable_multimodal", False)),
        max_tool_rounds=int(s.max_tool_rounds),
        system_prompt=s.system_prompt,
        message_count=int(s.message_count or 0),
        last_message_at=s.last_message_at,
        create_time=s.create_time,
        update_time=s.update_time,
    )


def _to_session_list_item(s: ChatSession) -> ChatSessionListItem:
    return ChatSessionListItem(
        session_id=s.session_id,
        title=s.title,
        knowledge_base_ids=list(s.knowledge_base_ids or []),
        folder_id=getattr(s, "folder_id", None),
        include_subfolders=bool(getattr(s, "include_subfolders", True)),
        model_preset=s.model_preset,
        model=s.model,
        mode=getattr(s, "mode", None) or "agent",
        thinking_level=getattr(s, "thinking_level", None) or "off",
        message_count=int(s.message_count or 0),
        last_message_at=s.last_message_at,
        create_time=s.create_time,
    )


def _to_message_item(m: ChatMessage) -> ChatMessageItem:
    return ChatMessageItem(
        message_id=m.id,
        role=m.role,
        content=m.content or "",
        thinking=m.thinking,
        thinking_ms=getattr(m, "thinking_ms", None),
        tool_calls=[
            ToolCallItem(
                id=tc.id,
                name=tc.name,
                arguments=dict(tc.arguments or {}),
                result_brief=tc.result_brief,
                items_added=int(tc.items_added or 0),
                retrieval_chunks=tc.retrieval_chunks,
                retrieval_params=tc.retrieval_params,
                recall_stats=tc.recall_stats,
                time_ms=tc.time_ms,
                execution_model=tc.execution_model,
            )
            for tc in (m.tool_calls or [])
        ],
        tool_call_id=m.tool_call_id,
        citations=[
            CitationItem(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                knowledge_base_id=c.knowledge_base_id,
                score=float(c.score or 0.0),
                # Phase A：UI 渲染扩展字段（chip hover 预览所需）
                chunk_type=c.chunk_type,
                page_index=c.page_index,
                section_title=c.section_title,
                file_id=c.file_id,
                file_name=c.file_name,
                preview=c.preview,
                # Phase B：alias（cN 引用号），前端历史回放需靠它定位 chip
                alias=c.alias,
            )
            for c in (m.citations or [])
        ],
        usage=(
            TokenUsageItem(
                prompt_tokens=int(m.usage.prompt_tokens or 0),
                completion_tokens=int(m.usage.completion_tokens or 0),
                thinking_tokens=m.usage.thinking_tokens,
                total_tokens=int(m.usage.total_tokens or 0),
                cached_tokens=int(getattr(m.usage, "cached_tokens", 0) or 0),
            )
            if m.usage is not None
            else None
        ),
        finish_reason=m.finish_reason,
        metadata=dict(m.metadata or {}),
        create_time=m.create_time,
    )


# ============================================================
# 端点
# ============================================================


@router.post(
    "",
    response_model=ApiResponse[ChatSessionInfo],
    summary="创建会话",
)
async def create_session(
    body: ChatSessionCreateRequest,
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[ChatSessionInfo]:
    """创建一个新的对话会话；返回会话详情（含分配的 ``session_id``）"""
    service = _get_service()
    try:
        obj = service.create_session(
            user_id=user_id,
            title=body.title,
            knowledge_base_ids=body.knowledge_base_ids,
            folder_id=body.folder_id,
            include_subfolders=body.include_subfolders,
            model_preset=body.model_preset,
            model=body.model,
            mode=body.mode,
            thinking_level=body.thinking_level,
            enable_multimodal=body.enable_multimodal,
            max_tool_rounds=body.max_tool_rounds,
            system_prompt=body.system_prompt,
        )
    except ValueError as e:
        # folder_id 跨 KB 校验等失败：转 422 给前端，文案保留原始 detail
        raise HTTPException(status_code=422, detail=str(e)) from e
    if obj is None:
        raise HTTPException(status_code=500, detail="创建会话失败")
    logger.info(f"REST create_session: session_id={obj.session_id}, user={user_id}")
    return ApiResponse.success(data=_to_session_info(obj))


@router.get(
    "",
    response_model=ApiResponse[ChatSessionListResponse],
    summary="我的会话列表",
)
async def list_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    knowledge_base_id: Optional[str] = Query(
        None,
        description="按知识库 ID 过滤；前端按 KB 维度组织面板时使用，"
                    "服务端过滤可避免拉全量后再丢弃",
    ),
    scope: Optional[str] = Query(
        None,
        description=(
            "检索范围维度：'kb'=仅 KB 级会话（folder_id IS NULL）；"
            "'folder'=仅指定 folder 的会话（须配合 folder_id）。"
            "不传则返回该 KB 下全部会话（兼容旧客户端）。"
        ),
    ),
    folder_id: Optional[str] = Query(
        None,
        description="scope='folder' 时必填，匹配 chat_session.folder_id",
    ),
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[ChatSessionListResponse]:
    """按 ``last_message_at`` 倒序分页拉取当前用户的会话

    若提供 ``knowledge_base_id``，先过滤出 ``knowledge_base_ids`` 包含该 KB 的
    会话，再分页（``total`` 与 ``items`` 都基于过滤后的集合）。

    v0.8.0：配合 ``scope`` / ``folder_id`` 把 KB 级与 folder 级会话列表拆开，
    前端切换文件夹 / 知识库时各自展示独立历史。
    """
    if scope == "folder" and not folder_id:
        raise HTTPException(
            status_code=422,
            detail="scope=folder 时必须提供 folder_id",
        )
    if scope and scope not in ("kb", "folder"):
        raise HTTPException(
            status_code=422,
            detail="scope 仅支持 'kb' 或 'folder'",
        )

    service = _get_service()
    if knowledge_base_id or scope:
        # 服务端过滤：拉一页足够大的窗口（用户级会话量级有限），
        # 应用层做包含判断后再分页。这里走 limit=1000 一次拉到底以避免
        # 在 ORM 层引入 JSON 包含查询的方言耦合。
        all_items, _ = service.list_sessions(
            user_id=user_id, limit=1000, offset=0,
        )
        filtered = list(all_items)
        if knowledge_base_id:
            filtered = [
                s for s in filtered
                if knowledge_base_id in (s.knowledge_base_ids or [])
            ]
        if scope == "kb":
            filtered = [s for s in filtered if not getattr(s, "folder_id", None)]
        elif scope == "folder":
            filtered = [
                s for s in filtered
                if getattr(s, "folder_id", None) == folder_id
            ]
        total = len(filtered)
        start = (page - 1) * page_size
        items = filtered[start:start + page_size]
    else:
        items, total = service.list_sessions(
            user_id=user_id,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
    payload = ChatSessionListResponse(
        items=[_to_session_list_item(s) for s in items],
        total=total,
        page=page,
        page_size=page_size,
    )
    return ApiResponse.success(data=payload)


@router.get(
    "/{session_id}",
    response_model=ApiResponse[ChatSessionInfo],
    summary="会话详情",
)
async def get_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[ChatSessionInfo]:
    """按 ``session_id`` 查询单条会话（权限校验：仅本人可见）"""
    obj = _get_service().get_session(session_id=session_id, user_id=user_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="会话不存在或无权限")
    return ApiResponse.success(data=_to_session_info(obj))


@router.patch(
    "/{session_id}",
    response_model=ApiResponse[ChatSessionInfo],
    summary="重命名会话",
)
async def rename_session(
    session_id: str,
    body: ChatSessionRenameRequest,
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[ChatSessionInfo]:
    obj = _get_service().rename_session(
        session_id=session_id, user_id=user_id, title=body.title,
    )
    if obj is None:
        raise HTTPException(status_code=404, detail="会话不存在或无权限")
    logger.info(f"REST rename_session: session_id={session_id}, title={body.title}")
    return ApiResponse.success(data=_to_session_info(obj))


@router.delete(
    "/{session_id}",
    response_model=ApiResponse[ChatSessionUpdateResponse],
    summary="软删除会话（级联消息）",
)
async def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[ChatSessionUpdateResponse]:
    ok = await _get_service().soft_delete_session(
        session_id=session_id, user_id=user_id,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在或无权限")
    logger.info(f"REST delete_session: session_id={session_id}")
    return ApiResponse.success(
        data=ChatSessionUpdateResponse(session_id=session_id, success=True, message="deleted"),
    )


@router.get(
    "/{session_id}/messages",
    response_model=ApiResponse[ChatMessageListResponse],
    summary="会话历史消息",
)
async def list_messages(
    session_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[ChatMessageListResponse]:
    """拉取会话历史消息（窗口内按 ``create_time`` 正序）

    用途：前端 UI **全量回放**。与 Chat 主流程上下文装配分离：
    - 本接口按时间从头分页（page=1 = 最早），前端应翻页拼完整会话
    - LLM 上下文装配走 ``ChatSessionService.load_history`` → 最近 N 条
      （再叠加轮数/token/摘要滑窗），不会用本接口截断结果
    """
    service = _get_service()
    obj = service.get_session(session_id=session_id, user_id=user_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="会话不存在或无权限")

    total = int(await chat_message_repo.count_by_session(session_id))
    # 从头分页：page=1 → [0, page_size)；page=2 → 下一段，前端循环拉完即全量回放
    start = (page - 1) * page_size
    if start >= total or total <= 0:
        messages = []
    else:
        messages = await chat_message_repo.list_by_session(
            session_id,
            limit=page_size,
            skip=start,
            ascending=True,
        )

    payload = ChatMessageListResponse(
        session_id=session_id,
        items=[_to_message_item(m) for m in messages],
        total=total,
        page=page,
        page_size=page_size,
    )
    return ApiResponse.success(data=payload)


@router.delete(
    "/{session_id}/messages",
    response_model=ApiResponse[ChatSessionUpdateResponse],
    summary="清空会话消息",
)
async def clear_messages(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[ChatSessionUpdateResponse]:
    """清空会话内的所有消息（保留会话本身）

    用于「清空上下文」功能：清空消息历史，但保留会话配置
    （知识库、文件夹、模型设置等）。

    权限：仅会话所有者可操作。
    """
    ok = await _get_service().clear_messages(
        session_id=session_id,
        user_id=user_id,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在或无权限")
    logger.info(f"REST clear_messages: session_id={session_id}")
    return ApiResponse.success(
        data=ChatSessionUpdateResponse(
            session_id=session_id,
            success=True,
            message="messages_cleared",
        ),
    )


# ---- context-status 进程内短缓存 ----
_CONTEXT_STATUS_CACHE: dict = {}


@router.get(
    "/{session_id}/context-status",
    response_model=ApiResponse[ContextStatusResponse],
    summary="会话上下文用量状态",
)
async def get_context_status(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[ContextStatusResponse]:
    """返回当前会话的上下文 token 用量（Cursor 式 Context 指示器数据源）。

    - 权限不匹配 / 会话不存在 → 404
    - 结果带进程内短缓存（默认 15s，按 session 最新消息 id 作为 key 的一部分）
    """
    service = _get_service()

    # 用最新消息 id 做缓存戳，避免压缩后仍返回旧用量
    try:
        latest = await chat_message_repo.list_by_session(
            session_id, limit=1, ascending=False,
        )
        stamp = latest[0].id if latest else "empty"
    except Exception:  # noqa: BLE001
        stamp = "na"

    cache_key = f"{session_id}:{user_id}:{stamp}"
    now = time.time()
    cached = _CONTEXT_STATUS_CACHE.get(cache_key)
    if cached and (now - cached[0]) < 15:
        data = cached[1]
        return ApiResponse.success(data=ContextStatusResponse(**data))

    data = await service.get_context_status(session_id=session_id, user_id=user_id)
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    _CONTEXT_STATUS_CACHE[cache_key] = (now, data)
    # 简单限幅，防止缓存无限涨
    if len(_CONTEXT_STATUS_CACHE) > 256:
        oldest = sorted(_CONTEXT_STATUS_CACHE.items(), key=lambda kv: kv[1][0])[:64]
        for k, _ in oldest:
            _CONTEXT_STATUS_CACHE.pop(k, None)

    return ApiResponse.success(data=ContextStatusResponse(**data))


@router.post(
    "/{session_id}/summarize",
    response_model=ApiResponse[ChatSessionUpdateResponse],
    summary="总结对话上下文",
)
async def summarize_context(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[ChatSessionUpdateResponse]:
    """总结当前对话上下文，生成摘要并标记旧消息

    用于「总结对话」功能：
    1. 调用 LLM 生成对话摘要
    2. 存储为 role="summary" 的消息
    3. 标记之前所有消息为已总结

    后续构建 LLM 上下文时，会跳过已总结的旧消息，只发送
    摘要 + 新消息，节省 token。

    权限：仅会话所有者可操作。
    """
    service = _get_service()
    # 404 只表达会话不存在或无权限；不能把模型调用失败误报为会话丢失。
    if service.get_session(session_id=session_id, user_id=user_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在、无权限或已删除")

    summary = await service.summarize_context(
        session_id=session_id,
        user_id=user_id,
    )
    if summary is None:
        raise HTTPException(
            status_code=422,
            detail="当前会话没有可总结的对话消息，或摘要模型调用失败，请查看服务端日志",
        )
    logger.info(f"REST summarize_context: session_id={session_id}")
    return ApiResponse.success(
        data=ChatSessionUpdateResponse(
            session_id=session_id,
            success=True,
            message="context_summarized",
        ),
    )
