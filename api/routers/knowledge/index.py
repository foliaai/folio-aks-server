#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : index.py
@Author  : caixiongjiang
@Date    : 2026/01/21 10:00
@Function: 
    Knowledge 索引路由
    提供 Knowledge 索引相关的 API 端点：
    
    文件上传（独立操作，文件直接上传到 MinIO/S3）：
      POST /upload        - 上传单个文件
      POST /upload/batch  - 批量上传文件
    
    索引构建（独立操作，向 Kafka 发送索引消息）：
      POST /build         - 触发索引构建
    
    进度查询：
      GET /progress/{file_id}  - 查询单个文件索引进度
      POST /progress/batch     - 批量查询文件索引进度
@Modify History:
    2026/02/18 - 实现文件上传、索引构建、进度查询 API
    2026/03/17 - 移除直接删除接口（删除流程统一走 file / folder 路由）
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import hashlib
import uuid
import asyncio
import mimetypes
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger
from sqlalchemy.orm import Session

from api.dependencies.auth import get_current_user_id
from api.dependencies.database import (
    get_db_session,
    get_file_progress_manager,
    get_kafka_producer,
    get_storage_manager,
)
from src.states.state_manager import FileProgressManager
from api.schemas.common import ApiResponse
from api.schemas.knowledge.index import (
    BatchFileUploadResponse,
    BatchProgressResponse,
    FileProgressResponse,
    FileUploadResponse,
    IndexBuildFileResult,
    IndexBuildRequest,
    IndexBuildResponse,
)
from src.db.kafka.producer import KafkaProducer
from src.db.kafka.topics import KafkaTopics
from src.db.mysql.models.business.workspace_file_system import WorkspaceFileSystem
from src.db.mysql.repositories.business.workspace_file_system_repo import (
    workspace_file_system_repo,
)
from src.db.mysql.repositories.business.knowledge_base_repo import (
    knowledge_base_repo,
)
from src.db.mysql.repositories.business.workspace_folder_repo import (
    workspace_folder_repo,
)
from src.db.storage.manager import StorageManager
from src.service.knowledge.kb_access import require_kb_access
from src.types.messages.index import IndexStartMessage
from src.utils.config_manager import get_config_manager
from src.utils.pdf_linearize import maybe_linearize

router = APIRouter(tags=["Knowledge Index"])

_config = get_config_manager()
_SUPPORTED_FORMATS: list[str] = _config.get(
    "file_upload.supported_formats",
    [
        "pdf", "docx", "pptx", "xlsx", "txt", "md", "json",
        "png", "jpg", "jpeg", "gif", "webp", "svg", "bmp",
        "doc", "xls", "ppt", "csv",
    ],
)
_MAX_FILE_SIZE_MB: int = _config.get("file_upload.max_file_size", 100)
_MAX_FILE_SIZE_BYTES: int = _MAX_FILE_SIZE_MB * 1024 * 1024

_FILE_MAGIC_BYTES: dict[str, list[bytes]] = {
    "pdf": [b"%PDF-"],
    "docx": [b"PK\x03\x04"],
    "pptx": [b"PK\x03\x04"],
    "xlsx": [b"PK\x03\x04"],
    "json": [b"{", b"["],
    "png": [b"\x89PNG"],
    "jpg": [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
    "gif": [b"GIF87a", b"GIF89a"],
    "bmp": [b"BM"],
    "webp": [b"RIFF"],
}

_FILE_FORMAT_MAP: dict[str, str] = {
    "pdf": "Portable Document Format",
    "docx": "Microsoft Word (OpenXML)",
    "pptx": "Microsoft PowerPoint (OpenXML)",
    "xlsx": "Microsoft Excel (OpenXML)",
    "txt": "Plain Text",
    "md": "Markdown",
    "json": "JSON",
    "png": "PNG Image",
    "jpg": "JPEG Image",
    "jpeg": "JPEG Image",
    "gif": "GIF Image",
    "webp": "WebP Image",
    "svg": "SVG Image",
    "bmp": "BMP Image",
    "doc": "Microsoft Word (Legacy)",
    "xls": "Microsoft Excel (Legacy)",
    "ppt": "Microsoft PowerPoint (Legacy)",
    "csv": "Comma-Separated Values",
}


# ==================== 辅助函数 ====================


def _generate_file_id() -> str:
    return f"file-{uuid.uuid4()}"


def _generate_document_id() -> str:
    return f"document-{uuid.uuid4()}"


def _generate_session_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"session_{ts}_{short_id}"


def _guess_mime_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def _compute_sha256(file_bytes: bytes) -> bytes:
    """计算文件 SHA256 哈希值，返回 32 字节二进制"""
    return hashlib.sha256(file_bytes).digest()


def _validate_file_extension(filename: str) -> str:
    """校验并返回文件扩展名（不含点号），不合法则抛出异常"""
    ext = PurePosixPath(filename).suffix.lstrip(".").lower()
    if not ext or ext not in _SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不支持的文件类型: .{ext}，"
                f"支持的类型: {', '.join('.' + f for f in _SUPPORTED_FORMATS)}"
            ),
        )
    return ext


def _validate_file_magic(file_bytes: bytes, ext: str) -> None:
    """通过文件头魔数校验文件内容是否与声明的后缀匹配"""
    magic_list = _FILE_MAGIC_BYTES.get(ext)
    if magic_list is None:
        return
    stripped = file_bytes.lstrip()
    if not any(stripped.startswith(magic) for magic in magic_list):
        raise HTTPException(
            status_code=400,
            detail=f"文件内容与声明的类型 .{ext} 不匹配，请检查文件是否损坏或后缀是否正确",
        )


_MYSQL_STATUS_MAP: dict[int, tuple[float, str, Optional[str]]] = {
    0: (0.0, "pending", None),
    1: (0.1, "processing", "index_start"),
    2: (1.0, "success", "split_end"),
    3: (0.0, "failed", None),
}


def _resolve_folder(
    session: Session,
    folder_id: Optional[str],
    user_id: str,
    knowledge_base_id: str,
) -> tuple[Optional[str], str]:
    """
    解析 folder_id 和 folder_path（kb 维度：共享库内可引用其他成员创建的文件夹）。

    传入 folder_id 时查询对应文件夹；为空时文件放在顶层（根目录）。

    Returns:
        (folder_id, full_path) 元组，folder_id 为 None 表示顶层

    Raises:
        HTTPException: 404 文件夹不存在；400 跨知识库引用
    """
    if folder_id:
        from src.db.mysql.models.business.workspace_folder import WorkspaceFolder
        folder = session.query(WorkspaceFolder).filter(
            WorkspaceFolder.folder_id == folder_id,
            WorkspaceFolder.deleted == 0,
        ).first()
        if not folder:
            raise HTTPException(status_code=404, detail="目标文件夹不存在")
        if folder.knowledge_base_id != knowledge_base_id:
            raise HTTPException(
                status_code=400, detail="目标文件夹不属于该知识库"
            )
        return folder.folder_id, folder.full_path

    return None, "/"


def _fallback_progress_from_mysql(
    file_record: WorkspaceFileSystem,
) -> FileProgressResponse:
    """MySQL fallback：当 Redis 中无数据时，从 MySQL status 字段生成粗粒度进度"""
    progress, status_str, stage = _MYSQL_STATUS_MAP.get(
        file_record.status, (0.0, "pending", None)
    )
    return FileProgressResponse(
        file_id=file_record.file_id,
        file_name=file_record.file_name,
        progress=progress,
        status=status_str,
        stage=stage,
        message=file_record.message,
    )


# ==================== 文件上传 API ====================


@router.post(
    "/upload",
    response_model=ApiResponse[FileUploadResponse],
    summary="上传单个文件",
    description="上传文件到 MinIO/S3 对象存储，创建文件元数据记录。上传与索引构建解耦，可先批量上传再统一触发索引。",
)
async def upload_file(
    file: UploadFile = File(..., description="待上传的文件"),
    knowledge_base_id: str = Form(..., description="目标知识库ID"),
    folder_id: Optional[str] = Form(default=None, description="目标文件夹ID"),
    description: Optional[str] = Form(default=None, description="文件描述"),
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
    storage: StorageManager = Depends(get_storage_manager),
) -> ApiResponse[FileUploadResponse]:
    """上传单个文件到对象存储并创建 MySQL 元数据记录"""

    kb = require_kb_access(session, user_id, knowledge_base_id, level="write")

    filename = file.filename or "unknown"
    ext = _validate_file_extension(filename)
    file_suffix = f".{ext}"
    mime_type = file.content_type or _guess_mime_type(filename)

    file_bytes = await file.read()
    file_size = len(file_bytes)
    if file_size == 0:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if file_size > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小 {file_size / 1024 / 1024:.1f}MB 超过限制 {_MAX_FILE_SIZE_MB}MB",
        )

    _validate_file_magic(file_bytes, ext)

    # PDF 线性化（fast web view）：让存入 MinIO 的就是线性化版本，
    # 配合 /raw 端点的 Range 支持，PDF.js 首屏只取首页字节、滚动按页取。
    # 失败回退原字节，不阻断上传。在 SHA256 之前做，保证哈希与存储内容一致。
    if ext.lower() == "pdf":
        file_bytes = await asyncio.to_thread(maybe_linearize, file_bytes, ext)
        file_size = len(file_bytes)

    file_sha256 = _compute_sha256(file_bytes)

    # 基于 SHA256 去重：相同内容的文件共享同一 document_id
    existing = workspace_file_system_repo.get_by_sha256(session, file_sha256)
    document_id = existing.document_id if existing and existing.document_id else _generate_document_id()

    file_id = _generate_file_id()
    session_id = _generate_session_id()

    resolved_folder_id, folder_path = _resolve_folder(
        session, folder_id, user_id, knowledge_base_id,
    )

    try:
        storage_path, bucket_name, object_path = await storage.upload_raw_file(
            file_bytes=file_bytes,
            user_id=user_id,
            session_id=session_id,
            file_id=file_id,
            file_suffix=file_suffix,
            folder_path=folder_path,
        )
    except Exception as e:
        logger.error(f"文件上传到对象存储失败: user_id={user_id}, filename={filename}, error={e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {e}") from e

    file_record = workspace_file_system_repo.create(
        session,
        user_id=user_id,
        file_id=file_id,
        file_name=filename,
        folder_id=resolved_folder_id,
        file_size=file_size,
        file_type=ext,
        file_format=_FILE_FORMAT_MAP.get(ext),
        file_suffix=file_suffix,
        mime_type=mime_type,
        file_sha256=file_sha256,
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_base_name=kb.knowledge_base_name,
        description=description,
        status=0,
        creator=user_id,
        bucket_name=bucket_name,
        file_path=object_path,
        session_id=session_id,
    )

    if not file_record:
        logger.error(f"创建文件元数据失败: user_id={user_id}, file_id={file_id}")
        raise HTTPException(status_code=500, detail="创建文件元数据失败")

    logger.info(
        f"文件上传成功: user_id={user_id}, file_id={file_id}, "
        f"filename={filename}, size={file_size}, sha256={file_sha256.hex()}"
    )

    return ApiResponse.success(
        data=FileUploadResponse(
            file_id=file_id,
            document_id=document_id,
            file_name=filename,
            session_id=session_id,
            file_size=file_size,
            mime_type=mime_type,
            file_sha256=file_sha256.hex(),
        ),
        message="文件上传成功",
    )


@router.post(
    "/upload/batch",
    response_model=ApiResponse[BatchFileUploadResponse],
    summary="批量上传文件",
    description="批量上传多个文件到 MinIO/S3 对象存储，同一批次共享 session_id。",
)
async def upload_files_batch(
    files: List[UploadFile] = File(..., description="待上传的文件列表"),
    knowledge_base_id: str = Form(..., description="目标知识库ID"),
    folder_id: Optional[str] = Form(default=None, description="目标文件夹ID"),
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
    storage: StorageManager = Depends(get_storage_manager),
) -> ApiResponse[BatchFileUploadResponse]:
    """批量上传文件，共享同一 session_id"""

    if not files:
        raise HTTPException(status_code=400, detail="文件列表不能为空")

    kb = require_kb_access(session, user_id, knowledge_base_id, level="write")

    resolved_folder_id, folder_path = _resolve_folder(
        session, folder_id, user_id, knowledge_base_id,
    )

    session_id = _generate_session_id()
    uploaded: list[FileUploadResponse] = []
    failed: list[dict[str, str]] = []

    for upload_file_item in files:
        filename = upload_file_item.filename or "unknown"
        try:
            ext = _validate_file_extension(filename)
            file_suffix = f".{ext}"
            mime_type = upload_file_item.content_type or _guess_mime_type(filename)

            file_bytes = await upload_file_item.read()
            file_size = len(file_bytes)
            if file_size == 0:
                raise ValueError("文件内容为空")
            if file_size > _MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"文件大小 {file_size / 1024 / 1024:.1f}MB 超过限制 {_MAX_FILE_SIZE_MB}MB"
                )

            _validate_file_magic(file_bytes, ext)

            # PDF 线性化（fast web view）：在 SHA256 之前做，保证哈希与存储内容一致。
            if ext.lower() == "pdf":
                file_bytes = await asyncio.to_thread(maybe_linearize, file_bytes, ext)
                file_size = len(file_bytes)

            file_sha256 = _compute_sha256(file_bytes)

            existing = workspace_file_system_repo.get_by_sha256(session, file_sha256)
            document_id = existing.document_id if existing and existing.document_id else _generate_document_id()

            file_id = _generate_file_id()

            storage_path, bucket_name, object_path = await storage.upload_raw_file(
                file_bytes=file_bytes,
                user_id=user_id,
                session_id=session_id,
                file_id=file_id,
                file_suffix=file_suffix,
                folder_path=folder_path,
            )

            file_record = workspace_file_system_repo.create(
                session,
                user_id=user_id,
                file_id=file_id,
                file_name=filename,
                folder_id=resolved_folder_id,
                file_size=file_size,
                file_type=ext,
                file_format=_FILE_FORMAT_MAP.get(ext),
                file_suffix=file_suffix,
                mime_type=mime_type,
                file_sha256=file_sha256,
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=kb.knowledge_base_name,
                status=0,
                creator=user_id,
                bucket_name=bucket_name,
                file_path=object_path,
                session_id=session_id,
            )

            if not file_record:
                raise RuntimeError("创建文件元数据失败")

            uploaded.append(
                FileUploadResponse(
                    file_id=file_id,
                    document_id=document_id,
                    file_name=filename,
                    session_id=session_id,
                    file_size=file_size,
                    mime_type=mime_type,
                    file_sha256=file_sha256.hex(),
                )
            )
            logger.info(f"批量上传 - 文件成功: {filename}, file_id={file_id}")

        except HTTPException as he:
            failed.append({"filename": filename, "error": he.detail})
            logger.warning(f"批量上传 - 文件失败: {filename}, error={he.detail}")
        except Exception as e:
            failed.append({"filename": filename, "error": str(e)})
            logger.warning(f"批量上传 - 文件失败: {filename}, error={e}")

    total = len(files)
    return ApiResponse.success(
        data=BatchFileUploadResponse(
            session_id=session_id,
            uploaded_files=uploaded,
            failed_files=failed,
            total=total,
            success_count=len(uploaded),
            fail_count=len(failed),
        ),
        message=f"批量上传完成: {len(uploaded)}/{total} 成功",
    )


# ==================== 索引构建 API ====================


@router.post(
    "/build",
    response_model=ApiResponse[IndexBuildResponse],
    summary="触发索引构建",
    description=(
        "对已上传的文件触发索引构建流程。"
        "API 向 Kafka knowledge_base.index.start Topic 发送消息，"
        "由下游 Pipeline Worker 异步处理。"
    ),
)
async def build_index(
    request: IndexBuildRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
    producer: KafkaProducer = Depends(get_kafka_producer),
    progress_mgr: FileProgressManager = Depends(get_file_progress_manager),
) -> ApiResponse[IndexBuildResponse]:
    """触发索引构建，向 Kafka 发送 IndexStartMessage"""

    kb = require_kb_access(
        session, user_id, request.knowledge_base_id, level="write"
    )

    parent_kb_name: Optional[str] = None
    if kb.parent_knowledge_base_id:
        parent_kb = knowledge_base_repo.get_by_id(
            session, kb.parent_knowledge_base_id
        )
        if parent_kb:
            parent_kb_name = parent_kb.knowledge_base_name

    results: list[IndexBuildFileResult] = []
    submitted = 0
    failed = 0

    for file_id in request.file_ids:
        # 查询文件记录（kb 维度：共享库内团队成员可触发其他成员上传文件的索引）
        file_record: Optional[WorkspaceFileSystem] = (
            workspace_file_system_repo.get_by_file_id(session, file_id)
        )

        if not file_record or (
            file_record.knowledge_base_id != request.knowledge_base_id
        ):
            results.append(
                IndexBuildFileResult(
                    file_id=file_id,
                    file_name="",
                    status="failed",
                    message="文件不存在或无权限",
                )
            )
            failed += 1
            continue

        if not file_record.file_path:
            results.append(
                IndexBuildFileResult(
                    file_id=file_id,
                    file_name=file_record.file_name,
                    status="failed",
                    message="文件存储路径缺失，请重新上传",
                )
            )
            failed += 1
            continue

        if not file_record.document_id:
            results.append(
                IndexBuildFileResult(
                    file_id=file_id,
                    file_name=file_record.file_name,
                    status="failed",
                    message="文件缺少 document_id，请重新上传",
                )
            )
            failed += 1
            continue

        message = IndexStartMessage(
            user_id=user_id,
            file_id=file_id,
            storage_path=file_record.storage_path,
            filename=file_record.file_name,
            knowledge_base_id=request.knowledge_base_id,
            knowledge_base_name=kb.knowledge_base_name,
            document_id=file_record.document_id,
            session_id=file_record.session_id,
            parent_knowledge_base_id=kb.parent_knowledge_base_id,
            parent_knowledge_base_name=parent_kb_name,
            knowledge_type=kb.knowledge_type,
            file_size=file_record.file_size,
            mime_type=file_record.mime_type,
            file_extension=file_record.file_suffix,
            parse_options=request.parse_options,
        )

        # 发送 Kafka 消息
        try:
            await producer.send_and_flush(
                topic=KafkaTopics.INDEX_START,
                message=message,
            )
        except Exception as e:
            logger.error(f"发送索引消息失败: file_id={file_id}, error={e}")
            results.append(
                IndexBuildFileResult(
                    file_id=file_id,
                    file_name=file_record.file_name,
                    status="failed",
                    message=f"发送索引消息失败: {e}",
                )
            )
            failed += 1
            continue

        # 写入 Redis 初始进度
        try:
            await progress_mgr.init_progress(
                file_id=file_id,
                user_id=user_id,
                file_name=file_record.file_name,
            )
        except Exception as e:
            logger.warning(f"写入 Redis 初始进度失败（不阻塞流程）: file_id={file_id}, error={e}")

        file_record.status = 1
        file_record.message = "索引构建已提交"
        file_record.knowledge_base_id = request.knowledge_base_id
        file_record.knowledge_base_name = kb.knowledge_base_name
        file_record.parent_knowledge_base_id = kb.parent_knowledge_base_id
        file_record.parent_knowledge_base_name = parent_kb_name
        file_record.knowledge_type = kb.knowledge_type
        file_record.updater = user_id
        try:
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"更新文件状态失败: file_id={file_id}, error={e}")

        results.append(
            IndexBuildFileResult(
                file_id=file_id,
                file_name=file_record.file_name,
                status="submitted",
                message="索引构建已提交",
            )
        )
        submitted += 1

        logger.info(
            f"索引构建已提交: user_id={user_id}, file_id={file_id}, "
            f"kb_id={request.knowledge_base_id}"
        )

    return ApiResponse.success(
        data=IndexBuildResponse(
            knowledge_base_id=request.knowledge_base_id,
            results=results,
            submitted_count=submitted,
            failed_count=failed,
        ),
        message=f"索引构建完成: {submitted} 已提交, {failed} 失败",
    )


# ==================== 进度查询 API ====================


@router.get(
    "/progress/{file_id}",
    response_model=ApiResponse[FileProgressResponse],
    summary="查询文件索引进度",
    description="查询单个文件的索引构建进度。优先从 Redis 读取细粒度进度，Redis 无数据时 fallback 到 MySQL。",
)
async def get_file_progress(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
    progress_mgr: FileProgressManager = Depends(get_file_progress_manager),
) -> ApiResponse[FileProgressResponse]:
    """查询单个文件的索引进度，优先 Redis → fallback MySQL"""

    # 归属校验先行：Redis 进度按 file_id 全局存储，不校验会泄露他人进度
    from api.routers.knowledge.file import _require_file_access

    file_record = _require_file_access(session, user_id, file_id, "read")

    redis_progress = await progress_mgr.get_progress(file_id)
    if redis_progress:
        return ApiResponse.success(
            data=FileProgressResponse(
                file_id=redis_progress.file_id,
                file_name=redis_progress.file_name,
                progress=redis_progress.progress,
                status=redis_progress.status.value,
                stage=redis_progress.stage,
                message=redis_progress.message,
            )
        )

    return ApiResponse.success(data=_fallback_progress_from_mysql(file_record))


@router.post(
    "/progress/batch",
    response_model=ApiResponse[BatchProgressResponse],
    summary="批量查询文件索引进度",
    description="批量查询多个文件的索引构建进度。优先从 Redis 批量读取，缺失的 fallback 到 MySQL。",
)
async def get_batch_progress(
    file_ids: List[str],
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
    progress_mgr: FileProgressManager = Depends(get_file_progress_manager),
) -> ApiResponse[BatchProgressResponse]:
    """批量查询文件索引进度，优先 Redis → fallback MySQL"""

    if not file_ids:
        raise HTTPException(status_code=400, detail="file_ids 不能为空")

    # 归属校验先行（与单个进度一致），无权访问的文件统一按 not_found 口径返回
    from api.routers.knowledge.file import _require_file_access

    accessible: dict[str, WorkspaceFileSystem] = {}
    for fid in file_ids:
        try:
            accessible[fid] = _require_file_access(session, user_id, fid, "read")
        except HTTPException:
            accessible[fid] = None

    redis_results = await progress_mgr.get_batch_progress(file_ids)

    file_progresses: list[FileProgressResponse] = []
    completed = 0
    processing = 0
    failed_count = 0

    for fid, redis_prog in zip(file_ids, redis_results):
        if redis_prog and accessible.get(fid) is not None:
            resp = FileProgressResponse(
                file_id=redis_prog.file_id,
                file_name=redis_prog.file_name,
                progress=redis_prog.progress,
                status=redis_prog.status.value,
                stage=redis_prog.stage,
                message=redis_prog.message,
            )
        else:
            file_record = accessible.get(fid)
            if not file_record:
                resp = FileProgressResponse(
                    file_id=fid,
                    file_name="",
                    progress=0.0,
                    status="not_found",
                    message="文件不存在或无权限",
                )
                failed_count += 1
                file_progresses.append(resp)
                continue
            resp = _fallback_progress_from_mysql(file_record)

        status_val = resp.status
        if status_val == "success":
            completed += 1
        elif status_val == "processing":
            processing += 1
        elif status_val == "failed":
            failed_count += 1

        file_progresses.append(resp)

    return ApiResponse.success(
        data=BatchProgressResponse(
            files=file_progresses,
            total=len(file_ids),
            completed_count=completed,
            processing_count=processing,
            failed_count=failed_count,
        )
    )
