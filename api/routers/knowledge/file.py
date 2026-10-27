#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : file.py
@Author  : caixiongjiang
@Date    : 2026/03/16
@Function: 
    文件操作路由
    提供文件级别的 API 端点：
      PUT    /{file_id}/move     - 移动文件到指定文件夹
      POST   /batch-move         - 批量移动文件
      GET    /{file_id}/preview  - 获取文件预览URL（MinIO 预签名URL）
      DELETE /{file_id}          - 删除单个文件（不可恢复）
      POST   /batch-delete       - 批量删除文件（不可恢复）
@Modify History:
    2026/03/17 - 从 folder.py 迁入文件软删除接口
    2026/03/18 - 文件移动逻辑抽取到 move_service，增加知识库一致性校验
    2026/09/01 - 下线回收站，删除改为标记墓碑 + 投递 Kafka 清理任务

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote
import json
from pathlib import Path
from typing import Optional

from api.dependencies.auth import get_current_user_id, get_current_user_id_from_token
from api.dependencies.database import (
    get_db_session,
    get_kafka_producer,
    get_storage_manager,
)
from api.schemas.common import ApiResponse
from api.schemas.knowledge.file import (
    BatchFileDeleteRequest,
    BatchFileDeleteResponse,
    BatchFileMoveRequest,
    BatchFileMoveResponse,
    FileDeleteResponse,
    FileMoveRequest,
    FileMoveResponse,
    FilePreviewResponse,
    SkippedFileDetail,
)
from api.schemas.knowledge.folder import FileInfo, FileListResponse
from src.db.mysql.repositories.business.workspace_file_system_repo import (
    workspace_file_system_repo,
)
from src.db.kafka.producer import KafkaProducer
from src.db.storage.factory import StorageFactory
from src.db.storage.manager import StorageManager
from src.db.storage.range_utils import is_range_satisfiable, parse_range_header
from src.service.knowledge import tombstone
from src.service.knowledge.delete_service import knowledge_delete_service
from src.service.knowledge.kb_access import require_kb_access
from src.service.knowledge.move_service import knowledge_move_service

router = APIRouter(tags=["File"])


def _require_file_access(
    session: Session, user_id: str, file_id: str, level: str = "read"
):
    """取文件并经其所属知识库判权（file_id 为 uuid 全局唯一；共享库内
    可见/可管理其他成员上传的文件）。无 kb 归属的脏数据按属主校验。"""
    file_record = workspace_file_system_repo.get_by_file_id(session, file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="文件不存在或无权限")
    if file_record.knowledge_base_id:
        require_kb_access(
            session, user_id, file_record.knowledge_base_id, level=level
        )
    elif file_record.user_id != user_id:
        raise HTTPException(status_code=404, detail="文件不存在或无权限")
    return file_record


# ==================== 文件搜索（供前端 @ 文件选择器） ====================


@router.get(
    "/search",
    response_model=ApiResponse[FileListResponse],
    summary="按文件名搜索知识库内文件",
    description=(
        "在指定知识库内按文件名模糊搜索文件，供前端 @ 文件选择器使用。"
        "q 为空时返回该知识库下的前 limit 个文件。"
    ),
)
async def search_files(
    knowledge_base_id: str = Query(..., description="知识库ID（限定搜索范围）"),
    q: str = Query(default="", description="文件名关键字，空串返回前 limit 个"),
    limit: int = Query(default=20, ge=1, le=50, description="返回条数上限"),
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[FileListResponse]:
    require_kb_access(session, user_id, knowledge_base_id, level="read")
    try:
        records = workspace_file_system_repo.search_by_name_in_kb(
            session, knowledge_base_id, q, limit
        )
    except SQLAlchemyError as e:
        logger.error(f"搜索文件失败: {e}")
        raise HTTPException(status_code=500, detail="搜索文件失败")

    files = [
        FileInfo(
            file_id=r.file_id,
            file_name=r.file_name,
            folder_id=r.folder_id,
            file_size=r.file_size,
            mime_type=r.mime_type,
            status=r.status,
            knowledge_base_id=r.knowledge_base_id or "",
            description=r.description,
        )
        for r in records
    ]

    return ApiResponse.success(
        data=FileListResponse(files=files, total=len(files)),
        message="搜索成功",
    )


# ==================== 文件移动 ====================


@router.put(
    "/{file_id}/move",
    response_model=ApiResponse[FileMoveResponse],
    summary="移动文件",
    description=(
        "将文件移动到指定文件夹。target_folder_id 为 null 时移动到知识库根目录。"
        "只能在同一知识库内移动文件。"
    ),
)
async def move_file(
    file_id: str,
    request: FileMoveRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[FileMoveResponse]:
    try:
        file_name = knowledge_move_service.move_file(
            session, user_id, file_id, request.target_folder_id
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="文件不存在或无权限")
    except PermissionError:
        raise HTTPException(status_code=403, detail="无权操作该文件")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SQLAlchemyError as e:
        logger.error(f"移动文件失败: {e}")
        raise HTTPException(status_code=500, detail="移动文件失败")

    return ApiResponse.success(
        data=FileMoveResponse(
            file_id=file_id,
            file_name=file_name,
            folder_id=request.target_folder_id,
            knowledge_base_id="",
        ),
        message="文件移动成功",
    )


@router.post(
    "/batch-move",
    response_model=ApiResponse[BatchFileMoveResponse],
    summary="批量移动文件",
    description="将多个文件移动到指定文件夹。target_folder_id 为 null 时移动到知识库根目录。",
)
async def batch_move_files(
    request: BatchFileMoveRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[BatchFileMoveResponse]:
    try:
        move_result = knowledge_move_service.batch_move_files(
            session, user_id, request.file_ids, request.target_folder_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SQLAlchemyError as e:
        logger.error(f"批量移动文件失败: {e}")
        raise HTTPException(status_code=500, detail="批量移动文件失败")

    return ApiResponse.success(
        data=BatchFileMoveResponse(
            moved_count=move_result.moved_count,
            total_requested=move_result.total_requested,
            skipped_files=[
                SkippedFileDetail(file_id=sf.file_id, reason=sf.reason)
                for sf in move_result.skipped_files
            ],
        ),
        message=f"成功移动 {move_result.moved_count}/{move_result.total_requested} 个文件",
    )


# ==================== 文件预览 ====================



_OFFICE_SUFFIXES_FOR_PDF_PREVIEW = {".doc", ".docx", ".ppt", ".pptx"}


def _parse_ext_attributes(raw: Optional[str]) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_converted_pdf_storage_path(file_record) -> Optional[str]:
    attrs = _parse_ext_attributes(getattr(file_record, "ext_attributes", None))
    path = attrs.get("converted_pdf_storage_path")
    return path if isinstance(path, str) and path.strip() else None


def _is_office_pdf_previewable(file_record) -> bool:
    suffix = (getattr(file_record, "file_suffix", None) or "").lower()
    if suffix in _OFFICE_SUFFIXES_FOR_PDF_PREVIEW:
        return True
    name = (getattr(file_record, "file_name", None) or "").lower()
    return any(name.endswith(ext) for ext in _OFFICE_SUFFIXES_FOR_PDF_PREVIEW)


def _resolve_preview_storage(
    file_record,
    *,
    prefer_converted: bool = True,
) -> tuple[str, str, bool]:
    """
    返回 (storage_path, media_type, has_converted_pdf)

    Word/PPT：默认优先返回转换 PDF（bbox 与之对齐）；
    prefer_converted=False 时强制原始文件。
    """
    converted = _get_converted_pdf_storage_path(file_record)
    has_converted = bool(converted)
    original_path = file_record.storage_path
    original_mime = file_record.mime_type or "application/octet-stream"

    if prefer_converted and has_converted and _is_office_pdf_previewable(file_record):
        return converted, "application/pdf", True

    # 原生 PDF
    suffix = (getattr(file_record, "file_suffix", None) or "").lower()
    name = (getattr(file_record, "file_name", None) or "").lower()
    if suffix == ".pdf" or name.endswith(".pdf") or "pdf" in (original_mime or ""):
        return original_path, "application/pdf", has_converted

    return original_path, original_mime, has_converted

@router.get(
    "/{file_id}/preview",
    response_model=ApiResponse[FilePreviewResponse],
    summary="获取文件预览URL",
    description=(
        "根据 file_id 生成 MinIO 预签名URL 用于文件预览。"
        "URL 默认有效期 1 小时，可通过 expires 参数调整（60 ~ 86400 秒）。"
        "前端页面 /knowledge/file/{file_id} 可调用此接口获取预览地址。"
    ),
)
async def get_file_preview(
    file_id: str,
    expires: int = Query(
        default=3600, ge=60, le=86400,
        description="URL 过期时间（秒），最小 60，最大 86400，默认 3600",
    ),
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
    storage: StorageManager = Depends(get_storage_manager),
) -> ApiResponse[FilePreviewResponse]:
    file_record = _require_file_access(session, user_id, file_id, "read")

    if not file_record.file_path:
        raise HTTPException(
            status_code=400, detail="文件存储路径缺失，无法生成预览链接"
        )

    preview_storage_path, preview_mime, has_converted = _resolve_preview_storage(
        file_record, prefer_converted=True
    )

    try:
        preview_url = await storage.get_preview_url(
            preview_storage_path, expires
        )
    except Exception as e:
        logger.error(
            f"生成预览URL失败: file_id={file_id}, "
            f"storage_path={preview_storage_path}, error={e}"
        )
        raise HTTPException(status_code=500, detail="生成预览URL失败")

    render_as = "pdf" if preview_mime == "application/pdf" else "original"

    return ApiResponse.success(
        data=FilePreviewResponse(
            file_id=file_id,
            file_name=file_record.file_name,
            mime_type=file_record.mime_type,
            file_size=file_record.file_size,
            preview_url=preview_url,
            expires_in=expires,
            render_as=render_as,
            preview_mime_type=preview_mime,
            has_converted_pdf=has_converted,
        ),
        message="预览URL生成成功",
    )


# ==================== Excel 内容视图（元素流驱动预览） ====================


@router.get(
    "/{file_id}/excel-content",
    summary="获取 Excel 文件内容视图（元素流：工作表分组 + 表格/图片卡片）",
    description=(
        "xlsx/xls 的预览数据源：MySQL element_meta_info（顺序/类型）+ MongoDB "
        "element_data（内容）按 page_index=工作表 分组组装；图片元素签发"
        "MinIO 预签名 URL。卡片带 element_id，前端框选直接按 id 匹配。"
    ),
)
async def get_excel_content(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
    storage: StorageManager = Depends(get_storage_manager),
) -> ApiResponse:
    from sqlalchemy import text as _text

    from src.db.mongodb.mongodb_manager import get_mongodb_manager

    file_record = _require_file_access(session, user_id, file_id, "read")

    suffix = (getattr(file_record, "file_suffix", None) or "").lower()
    if suffix not in (".xlsx", ".xls"):
        raise HTTPException(status_code=400, detail="仅支持 xlsx/xls 文件")

    document_id = getattr(file_record, "document_id", None)
    if not document_id:
        raise HTTPException(status_code=400, detail="文件尚未完成解析，无法预览")

    rows = session.execute(
        _text(
            "SELECT element_id, page_index, element_type, text_level, "
            "bucket_name, image_file_path "
            "FROM element_meta_info "
            "WHERE document_id = :d AND deleted = 0 "
            "ORDER BY page_index, element_index"
        ),
        {"d": document_id},
    ).mappings().all()
    if not rows:
        raise HTTPException(status_code=404, detail="文件没有可预览的元素")

    ids = [r["element_id"] for r in rows]
    mongo = await get_mongodb_manager()
    db = await mongo.get_database()
    contents: dict = {}
    async for doc in db.element_data.find({"_id": {"$in": ids}}, {"content": 1}):
        contents[doc["_id"]] = doc.get("content") or {}

    def join_ann(v) -> Optional[str]:
        if isinstance(v, list):
            s = ", ".join(x for x in v if x)
            return s or None
        return v or None

    # page_index 分组；组内首个 text 标题元素即工作表名
    sheets: dict = {}
    sheet_titles: dict = {}
    for r in rows:
        page = r["page_index"] or 0
        content = contents.get(r["element_id"], {})
        if r["element_type"] == "text" and r["text_level"]:
            if page not in sheet_titles:
                sheet_titles[page] = str(content.get("text") or f"Sheet {page + 1}")
            continue  # 标题元素只作标签，不渲染卡片
        card = {
            "element_id": r["element_id"],
            "type": r["element_type"],
        }
        if r["element_type"] == "table":
            card["table_body"] = content.get("table_body") or ""
            card["caption"] = join_ann(content.get("table_caption"))
            card["footnote"] = join_ann(content.get("table_footnote"))
        elif r["element_type"] == "image":
            card["caption"] = join_ann(content.get("image_caption"))
            bucket, path = r["bucket_name"], r["image_file_path"]
            if bucket and path:
                try:
                    card["image_url"] = await storage.get_preview_url(
                        f"{bucket}/{path}", 3600
                    )
                except Exception as e:
                    logger.warning(f"excel 视图图片预签名失败: {r['element_id']}: {e}")
        elif content.get("text"):
            card["text"] = str(content["text"])
        else:
            continue  # 无内容的元素不渲染
        sheets.setdefault(page, []).append(card)

    sheet_views = [
        {
            "page_index": page,
            "title": sheet_titles.get(page, f"Sheet {page + 1}"),
            "elements": cards,
        }
        for page, cards in sorted(sheets.items())
    ]

    from api.schemas.knowledge.file import ExcelContentResponse

    data = ExcelContentResponse(
        file_id=file_id,
        file_name=file_record.file_name,
        total_sheets=len(sheet_views),
        sheets=sheet_views,
    )
    return ApiResponse.success(data=data, message="Excel 内容视图获取成功")


# ==================== 文件原始内容（流式返回，供前端 PDF 预览） ====================


@router.get(
    "/{file_id}/raw",
    summary="获取文件原始内容（内联返回，供前端 PDF 预览）",
    description=(
        "服务端从对象存储读取文件字节并以 Content-Disposition: inline 返回，"
        "避免把 MinIO 内部预签名 URL（http + 内网域名）直接暴露给浏览器。"
        "鉴权通过 query 参数 token（与 WebSocket 鉴权通道一致），"
        "因为 react-pdf / <img> 等浏览器原生资源加载无法自定义请求头。"
    ),
)
async def get_file_raw(
    file_id: str,
    request: Request,
    source: str = Query(
        default="auto",
        description=(
            "读取源：auto=Office 优先转换 PDF；converted=强制转换 PDF；"
            "original=强制原始文件"
        ),
    ),
    user_id: str = Depends(get_current_user_id_from_token),
    session: Session = Depends(get_db_session),
) -> Response:
    file_record = _require_file_access(session, user_id, file_id, "read")

    if not file_record.file_path:
        raise HTTPException(
            status_code=400, detail="文件存储路径缺失，无法读取原始内容"
        )

    prefer_converted = source != "original"
    if source == "converted":
        converted = _get_converted_pdf_storage_path(file_record)
        if not converted:
            raise HTTPException(status_code=404, detail="转换 PDF 不存在")
        storage_path, media_type = converted, "application/pdf"
        has_converted = True
    else:
        storage_path, media_type, has_converted = _resolve_preview_storage(
            file_record, prefer_converted=prefer_converted
        )

    filename = file_record.file_name or file_id
    # 转换 PDF 预览时，让浏览器/PDF.js 按 .pdf 处理
    if media_type == "application/pdf" and not filename.lower().endswith(".pdf"):
        filename = f"{Path(filename).stem}.pdf"

    # 流式端点不复用 get_storage_manager 依赖：该依赖会在响应体流式发送前被
    # FastAPI 清理（关闭 urllib3 连接池），导致流式中断。这里独立创建适配器，
    # 由生成器在结束时自行释放底层响应与连接。
    try:
        adapter = StorageFactory.create_adapter()
        size, etag = adapter.stat_file(storage_path)
    except Exception as e:
        logger.error(
            f"stat 文件原始内容失败: file_id={file_id}, "
            f"storage_path={storage_path}, error={e}"
        )
        raise HTTPException(status_code=500, detail="读取文件原始内容失败")

    total = size or 0
    range_header = request.headers.get("range")

    # 公共响应头：内容按 file_id 不可变，长期缓存 + immutable 让浏览器命中
    # 本地缓存，重复打开同一文件即时显示，不再每次都走 frp 隧道重传。
    # X-Accel-Buffering: no 让 Nginx 关闭对该响应的代理缓冲，直接透传字节流，
    # 否则 Nginx 会先把整个上游响应缓冲到磁盘/内存再发给客户端，抵消流式收益。
    base_headers = {
        "Content-Disposition": f'inline; filename="{quote(filename)}"',
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=86400, immutable",
        "X-Accel-Buffering": "no",
    }
    if etag:
        base_headers["ETag"] = etag

    # ---- Range 请求：PDF.js 渐进式加载会发 bytes=0-N 等区间 ----
    if range_header:
        parsed = parse_range_header(range_header, total)
        if parsed is None:
            # 区间不可满足 → 416
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


# ==================== 文件删除（不可恢复） ====================


@router.delete(
    "/{file_id}",
    response_model=ApiResponse[FileDeleteResponse],
    summary="删除单个文件",
    description=(
        "删除文件，不可恢复。请求内只标记 deleted=1 并投递清理任务，O(1) 返回；"
        "向量 / 文档 / 元数据 / 对象存储由 CleanupWorker 异步清理。"
        "文件立即从列表和问答检索中消失。"
    ),
)
async def delete_file(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
    producer: KafkaProducer = Depends(get_kafka_producer),
) -> ApiResponse[FileDeleteResponse]:
    try:
        ticket = knowledge_delete_service.mark_file_deleted(
            session, user_id, file_id
        )
        if ticket is None:
            session.rollback()
            raise HTTPException(status_code=404, detail="文件不存在或已删除")
        session.commit()
    except PermissionError:
        session.rollback()
        raise HTTPException(status_code=403, detail="无权操作该文件")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"删除文件失败: {e}")
        raise HTTPException(status_code=500, detail="删除失败")

    tombstone.invalidate(ticket.user_id)
    if ticket.knowledge_base_id:
        tombstone.invalidate_kb(ticket.knowledge_base_id)
    await knowledge_delete_service.publish_cleanup(producer, [ticket])

    return ApiResponse.success(
        data=FileDeleteResponse(file_id=file_id, success=True),
        message="文件已删除",
    )


@router.post(
    "/batch-delete",
    response_model=ApiResponse[BatchFileDeleteResponse],
    summary="批量删除文件",
    description="批量删除文件，不可恢复。语义与单个删除一致。",
)
async def batch_delete_files(
    request: BatchFileDeleteRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
    producer: KafkaProducer = Depends(get_kafka_producer),
) -> ApiResponse[BatchFileDeleteResponse]:
    try:
        tickets = knowledge_delete_service.mark_files_deleted(
            session, user_id, request.file_ids
        )
        session.commit()
    except PermissionError:
        session.rollback()
        raise HTTPException(status_code=403, detail="无权操作其中的文件")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"批量删除失败: {e}")
        raise HTTPException(status_code=500, detail="批量删除失败")

    for t in tickets:
        tombstone.invalidate(t.user_id)
        if t.knowledge_base_id:
            tombstone.invalidate_kb(t.knowledge_base_id)
    await knowledge_delete_service.publish_cleanup(producer, tickets)

    return ApiResponse.success(
        data=BatchFileDeleteResponse(
            deleted_count=len(tickets),
            total_requested=len(request.file_ids),
        ),
        message=f"成功删除 {len(tickets)}/{len(request.file_ids)} 个文件",
    )
