#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : chunk.py
@Author  : caixiongjiang
@Date    : 2026/06/09
@Function:
    Chunk 操作路由
    提供 chunk 级别的 API 端点：
      GET /{chunk_id}/image-preview - 获取图片 chunk 的预览 URL（MinIO 预签名URL）
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.orm import Session
from urllib.parse import quote

from api.dependencies.auth import get_current_user_id, get_current_user_id_from_token
from api.dependencies.database import get_db_session, get_storage_manager
from api.schemas.common import ApiResponse
from api.schemas.knowledge.file import ChunkImagePreviewResponse, ChunkPositionResponse, ElementPosition
from src.db.mysql.repositories.base.chunk_meta_info_repo import chunk_meta_info_repo
from src.db.mysql.repositories.base.element_meta_info_repo import element_meta_info_repo
from src.db.storage.factory import StorageFactory
from src.db.storage.manager import StorageManager
from src.db.storage.range_utils import is_range_satisfiable, parse_range_header
from src.service.knowledge.kb_access import require_kb_access

router = APIRouter(tags=["Chunk"])


def _require_chunk_kb_read(session: Session, meta, user_id: str) -> None:
    """chunk 归属校验（H2 修复）：chunk → 所属知识库 → can_read。

    无法定位所属知识库的 chunk 一律拒绝（404），避免无主数据成为旁路。
    历史行的 chunk_meta_info.knowledge_base_id 为 NULL（写入侧 2026-10-26
    才补齐），此处回退到 chunk_section_document 权威映射表取 kb。
    """
    kb_id = getattr(meta, "knowledge_base_id", None)
    if not kb_id:
        from src.db.mysql.models.base.chunk_section_document import (
            ChunkSectionDocument,
        )

        row = (
            session.query(ChunkSectionDocument.knowledge_base_id)
            .filter(
                ChunkSectionDocument.chunk_id == getattr(meta, "chunk_id", None),
                ChunkSectionDocument.deleted == 0,
            )
            .first()
        )
        kb_id = row[0] if row else None
    if not kb_id:
        raise HTTPException(status_code=404, detail="Chunk 不存在或已删除")
    require_kb_access(session, user_id, kb_id, level="read")

# MinerU pipeline content_list bbox：左上角原点，[x0,y0,x1,y1]，各轴映射到 0~1000
MINERU_COORD_SPACE = "mineru-normalized-1000"
MINERU_COORD_RANGE = 1000


# ==================== Chunk 图片预览 ====================


@router.get(
    "/{chunk_id}/image-preview",
    response_model=ApiResponse[ChunkImagePreviewResponse],
    summary="获取图片 chunk 的预览 URL",
    description=(
        "根据 chunk_id 生成图片 chunk 的 MinIO 预签名 URL。"
        "仅支持 chunk_type=image 且具有有效存储路径的 chunk。"
        "URL 默认有效期 1 小时，可通过 expires 参数调整（60 ~ 86400 秒）。"
    ),
)
async def get_chunk_image_preview(
    chunk_id: str,
    expires: int = Query(
        default=3600, ge=60, le=86400,
        description="URL 过期时间（秒），最小 60，最大 86400，默认 3600",
    ),
    _user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
    storage: StorageManager = Depends(get_storage_manager),
) -> ApiResponse[ChunkImagePreviewResponse]:
    # 1. 查询 ChunkMetaInfo
    meta = chunk_meta_info_repo.get_by_id(session, chunk_id)
    if not meta or getattr(meta, "deleted", 0) != 0:
        raise HTTPException(status_code=404, detail="Chunk 不存在或已删除")

    # 2. 归属校验：所属知识库必须可读
    _require_chunk_kb_read(session, meta, _user_id)

    if meta.chunk_type != "image":
        raise HTTPException(
            status_code=400, detail="非图片类型 chunk，无法生成图片预览"
        )

    if not meta.bucket_name or not meta.image_file_path:
        raise HTTPException(
            status_code=404, detail="图片存储路径缺失，无法生成预览链接"
        )

    # 2. 生成 presigned URL
    storage_path = f"{meta.bucket_name}/{meta.image_file_path}"
    try:
        preview_url = await storage.get_preview_url(storage_path, expires)
    except Exception as e:
        logger.error(
            f"生成 chunk 图片预览URL失败: chunk_id={chunk_id}, "
            f"storage_path={storage_path}, error={e}"
        )
        raise HTTPException(status_code=500, detail="生成预览URL失败")

    return ApiResponse.success(
        data=ChunkImagePreviewResponse(
            chunk_id=chunk_id,
            preview_url=preview_url,
            expires_in=expires,
            file_name=getattr(meta, "image_file_name", None),
        ),
        message="图片预览URL生成成功",
    )


# ==================== Chunk 图片原始内容（流式返回，供前端预览） ====================

# image_file_type（png/jpg/jpeg/svg/...）-> MIME 映射
_IMAGE_MIME_MAP = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "svg": "image/svg+xml",
    "tiff": "image/tiff",
    "tif": "image/tiff",
}


def _resolve_image_mime(meta) -> str:
    """从 ChunkMetaInfo 推断图片 MIME，缺失时回退到 image/png。"""
    for attr in ("image_file_type", "image_file_suffix"):
        val = getattr(meta, attr, None)
        if val:
            key = val.lower().lstrip(".")
            if key in _IMAGE_MIME_MAP:
                return _IMAGE_MIME_MAP[key]
    return "image/png"


@router.get(
    "/{chunk_id}/raw-image",
    summary="获取图片 chunk 的原始内容（内联返回，供前端预览）",
    description=(
        "服务端从对象存储读取图片字节并以 Content-Disposition: inline 返回，"
        "避免把 MinIO 内部预签名 URL（http + 内网域名）直接暴露给浏览器。"
        "鉴权通过 query 参数 token（与 WebSocket 鉴权通道一致），"
        "因为 <img> 等浏览器原生资源加载无法自定义请求头。"
    ),
)
async def get_chunk_raw_image(
    chunk_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id_from_token),
    session: Session = Depends(get_db_session),
) -> Response:
    # 1. 查询 ChunkMetaInfo
    meta = chunk_meta_info_repo.get_by_id(session, chunk_id)
    if not meta or getattr(meta, "deleted", 0) != 0:
        raise HTTPException(status_code=404, detail="Chunk 不存在或已删除")

    # 2. 归属校验：所属知识库必须可读
    _require_chunk_kb_read(session, meta, user_id)

    if meta.chunk_type != "image":
        raise HTTPException(
            status_code=400, detail="非图片类型 chunk，无法返回图片内容"
        )

    if not meta.bucket_name or not meta.image_file_path:
        raise HTTPException(
            status_code=404, detail="图片存储路径缺失，无法读取原始内容"
        )

    storage_path = f"{meta.bucket_name}/{meta.image_file_path}"
    media_type = _resolve_image_mime(meta)
    filename = getattr(meta, "image_file_name", None) or chunk_id

    # 流式端点不复用 get_storage_manager 依赖：该依赖会在响应体流式发送前被
    # FastAPI 清理（关闭 urllib3 连接池），导致流式中断。这里独立创建适配器，
    # 由生成器在结束时自行释放底层响应与连接。
    try:
        adapter = StorageFactory.create_adapter()
        size, etag = adapter.stat_file(storage_path)
    except Exception as e:
        logger.error(
            f"stat chunk 图片原始内容失败: chunk_id={chunk_id}, "
            f"storage_path={storage_path}, error={e}"
        )
        raise HTTPException(status_code=500, detail="读取图片原始内容失败")

    total = size or 0
    range_header = request.headers.get("range")

    # 公共响应头：图片内容按 chunk_id 不可变，长期缓存 + immutable 让浏览器
    # 命中本地缓存，重复预览同一图片即时显示，不再每次都走 frp 隧道重传。
    # X-Accel-Buffering: no 让 Nginx 关闭对该响应的代理缓冲，直接透传字节流。
    base_headers = {
        "Content-Disposition": f'inline; filename="{quote(filename)}"',
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=86400, immutable",
        "X-Accel-Buffering": "no",
    }
    if etag:
        base_headers["ETag"] = etag

    # ---- Range 请求 ----
    if range_header:
        parsed = parse_range_header(range_header, total)
        if parsed is None:
            if not is_range_satisfiable(range_header, total):
                return Response(
                    status_code=416,
                    headers={
                        **base_headers,
                        "Content-Range": f"bytes */{total}" if total else "bytes */*",
                    },
                )
            # 多区间等不支持的形式：退回整文件 200
        else:
            offset, end, length = parsed
            range_headers = {
                **base_headers,
                "Content-Range": f"bytes {offset}-{end}/{total}",
                "Content-Length": str(length),
            }
            return StreamingResponse(
                adapter.download_file_range_stream(storage_path, offset, length),
                status_code=206,
                media_type=media_type,
                headers=range_headers,
            )

    # ---- 整文件 200 ----
    full_headers = {**base_headers}
    if total:
        full_headers["Content-Length"] = str(total)
    return StreamingResponse(
        adapter.download_file_stream(storage_path),
        media_type=media_type,
        headers=full_headers,
    )


# ==================== Chunk 定位信息 ====================


@router.get(
    "/{chunk_id}/position",
    response_model=ApiResponse[ChunkPositionResponse],
    summary="获取 chunk 的定位信息",
    description=(
        "根据 chunk_id 返回其关联元素的位置坐标（MinerU bbox），"
        "用于文件预览中的定位高亮。"
    ),
)
async def get_chunk_position(
    chunk_id: str,
    _user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[ChunkPositionResponse]:
    # 1. 查询 ChunkMetaInfo
    meta = chunk_meta_info_repo.get_by_id(session, chunk_id)
    if not meta or getattr(meta, "deleted", 0) != 0:
        raise HTTPException(status_code=404, detail="Chunk 不存在或已删除")

    # 2. 归属校验：所属知识库必须可读
    _require_chunk_kb_read(session, meta, _user_id)

    # 3. 查询关联的 Element 位置信息
    #    文本 chunk 的 chunk 级 page_index 取自 buffer 首个元素，可能与本 chunk
    #    实际元素所在页不一致（文本跨页时）。因此这里逐元素带上各自的 page_index，
    #    并用「首个有效元素的页码」作为定位跳转的目标页，使文本与图片/表格一致。
    elements: list[ElementPosition] = []
    raw_element_ids = getattr(meta, "element_ids", None) or []
    if raw_element_ids:
        for eid in raw_element_ids:
            el = element_meta_info_repo.get_by_id(session, eid)
            if not el or getattr(el, "deleted", 0) != 0:
                continue
            # 解析 page_position JSON 字符串
            pos = getattr(el, "page_position", None)
            parsed_pos = None
            if pos:
                try:
                    import json as _json
                    parsed_pos = _json.loads(pos)
                    if not isinstance(parsed_pos, list):
                        parsed_pos = None
                except Exception:
                    parsed_pos = None
            elements.append(
                ElementPosition(
                    element_id=getattr(el, "element_id", eid),
                    element_type=getattr(el, "element_type", "unknown"),
                    page_index=getattr(el, "page_index", None),
                    page_position=parsed_pos,
                )
            )

    # 定位目标页：优先取首个有效元素的页码（与实际高亮元素所在页对齐）；
    # 缺失时回退到 chunk 级 page_index。
    resolved_page_index = getattr(meta, "page_index", None)
    for e in elements:
        if e.page_index is not None:
            resolved_page_index = e.page_index
            break

    return ApiResponse.success(
        data=ChunkPositionResponse(
            chunk_id=chunk_id,
            chunk_type=getattr(meta, "chunk_type", None),
            page_index=resolved_page_index,
            coord_space=MINERU_COORD_SPACE,
            coord_range=MINERU_COORD_RANGE,
            elements=elements,
        ),
    )
