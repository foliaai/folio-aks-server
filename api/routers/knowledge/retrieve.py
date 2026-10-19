#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : retrieve.py
@Author  : caixiongjiang
@Date    : 2026/01/21 10:00
@Function:
    Knowledge 检索路由
    提供 Knowledge 检索相关的 API 端点：
      POST /retrieve        — 智能检索（完整 Pipeline: LLM₁ → 多路 → Rerank）
      POST /retrieve/custom — 自定义路由组合检索
      POST /retrieve/single — 单原子能力直调
      GET  /retrieve/routes — 列出所有可用路由
@Modify History:
    2026/04/03 - 实现完整检索路由
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.orm import Session

from api.dependencies.auth import get_current_user_id
from api.dependencies.database import get_db_session
from api.schemas.common import ApiResponse
from api.schemas.knowledge.retrieve import (
    ChunkItemSchema,
    CustomRetrieveRequestSchema,
    PhaseTimingsSchema,
    RetrieveRequestSchema,
    RetrieveResponseSchema,
    RouteInfoSchema,
    SingleRetrieveRequestSchema,
)
from src.db.mysql.models.business.workspace_file_system import WorkspaceFileSystem
from src.retrieve.pipeline.types import (
    RetrieveRequest,
    RetrieveResponse,
    RouteConfig,
)
from src.retrieve.types.query import MetadataFilter
from src.service.knowledge.kb_access import (
    build_kb_scoped_filter,
    has_read_any_kb,
    require_kb_access,
)
from src.service.knowledge.retrieve_service import RetrieveService
from src.service.knowledge.tombstone import exclude_deleted


router = APIRouter(tags=["Retrieve"])

_service: RetrieveService | None = None


def _get_service() -> RetrieveService:
    global _service
    if _service is None:
        _service = RetrieveService()
    return _service


def _response_to_schema(resp: RetrieveResponse) -> RetrieveResponseSchema:
    items = [
        ChunkItemSchema(
            chunk_id=item.chunk_id,
            score=item.score,
            document_id=item.document_id,
            section_id=item.section_id,
            knowledge_base_id=item.knowledge_base_id,
            text=item.text,
            metadata=item.metadata,
        )
        for item in resp.items
    ]
    timings = PhaseTimingsSchema(
        planning_ms=resp.phase_timings.planning_ms,
        recall_ms=resp.phase_timings.recall_ms,
        alignment_ms=resp.phase_timings.alignment_ms,
        fusion_ms=resp.phase_timings.fusion_ms,
        rerank_ms=resp.phase_timings.rerank_ms,
    )
    return RetrieveResponseSchema(
        items=items,
        total_count=resp.total_count,
        execution_time_ms=resp.execution_time_ms,
        phase_timings=timings,
    )


# ==================== 智能检索 ====================


@router.post("/retrieve", summary="智能检索")
async def smart_retrieve(
    body: RetrieveRequestSchema,
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[RetrieveResponseSchema]:
    """完整 Pipeline 检索: LLM₁ 路由规划 → 多路召回 → Rerank"""
    try:
        service = _get_service()

        request = RetrieveRequest(
            query_text=body.query_text,
            filters=exclude_deleted(
                build_kb_scoped_filter(
                    user_id,
                    body.knowledge_base_id,
                    document_id=body.document_id,
                )
            ),
            top_k=body.top_k,
            enable_rerank=body.enable_rerank,
            enable_route_planner=body.enable_route_planner,
            route_hints=body.route_hints,
        )

        response = await service.retrieve(request)
        schema = _response_to_schema(response)

        logger.info(
            f"智能检索完成: query='{body.query_text[:50]}', "
            f"results={schema.total_count}, "
            f"time={schema.execution_time_ms:.0f}ms"
        )

        return ApiResponse.success(data=schema)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"智能检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"检索失败: {e}") from e


# ==================== 自定义路由检索 ====================


@router.post("/retrieve/custom", summary="自定义路由组合检索")
async def custom_retrieve(
    body: CustomRetrieveRequestSchema,
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[RetrieveResponseSchema]:
    """指定路由组合执行检索，跳过 LLM₁ 路由规划"""
    try:
        service = _get_service()

        routes = [
            RouteConfig(
                route=r.get("route", ""),
                top_k=r.get("top_k", 20),
                params=r.get("params", {}),
            )
            for r in body.routes
        ]

        response = await service.retrieve_custom(
            routes=routes,
            query_text=body.query_text,
            filters=exclude_deleted(
                build_kb_scoped_filter(
                    user_id,
                    body.knowledge_base_id,
                    document_id=body.document_id,
                )
            ),
            top_k=body.top_k,
            enable_rerank=body.enable_rerank,
        )

        schema = _response_to_schema(response)
        return ApiResponse.success(data=schema)

    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"无效路由: {e}") from e
    except Exception as e:
        logger.error(f"自定义检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"检索失败: {e}") from e


# ==================== 单能力直调 ====================


@router.post("/retrieve/single", summary="单原子能力直调")
async def single_retrieve(
    body: SingleRetrieveRequestSchema,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse:
    """直接调用指定的原子检索能力（目标范围必须对当前用户可读）"""
    try:
        # ===== 越权修复（H1）：目标 kb/document 必须可读，且必须提供其一 =====
        if body.knowledge_base_id:
            require_kb_access(
                session, user_id, body.knowledge_base_id, level="read"
            )
        elif body.document_id:
            # SHA256 去重使同一 document 可出现在多个库：任一所在库可读即放行
            # （同文档内容相同，跨库命中不产生额外泄露面）
            doc_kb_ids = [
                row[0]
                for row in session.query(
                    WorkspaceFileSystem.knowledge_base_id
                )
                .filter(
                    WorkspaceFileSystem.document_id == body.document_id,
                    WorkspaceFileSystem.deleted == 0,
                )
                .distinct()
                .all()
            ]
            if not doc_kb_ids:
                raise HTTPException(status_code=404, detail="文档不存在")
            if not has_read_any_kb(session, user_id, doc_kb_ids):
                raise HTTPException(
                    status_code=403, detail="无权读取该文档所在的知识库"
                )
        else:
            # 无界直调等价于跨用户全库检索，直接拒绝
            raise HTTPException(
                status_code=400,
                detail="必须提供 knowledge_base_id 或 document_id 以限定检索范围",
            )

        service = _get_service()
        result = await service.retrieve_single(
            capability_name=body.capability_name,
            query_text=body.query_text,
            top_k=body.top_k,
            knowledge_base_id=body.knowledge_base_id,
            document_id=body.document_id,
            **body.params,
        )

        items = [item.model_dump() for item in result.items]
        return ApiResponse.success(data={
            "items": items,
            "total_count": result.total_count,
        })

    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"无效能力: {e}") from e
    except Exception as e:
        logger.error(f"单能力检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"检索失败: {e}") from e


# ==================== 路由列表 ====================


@router.get("/retrieve/routes", summary="列出所有可用路由")
async def list_routes(
    user_id: str = Depends(get_current_user_id),
) -> ApiResponse[List[RouteInfoSchema]]:
    """列出所有已注册的检索路由及其描述"""
    service = _get_service()
    descriptors = service._registry.list_descriptors()

    routes = [
        RouteInfoSchema(
            name=d.name,
            display_name=d.display_name,
            description=d.description,
        )
        for d in descriptors
    ]
    return ApiResponse.success(data=routes)
