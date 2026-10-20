#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : folder.py
@Author  : caixiongjiang
@Date    : 2026/02/18
@Function:
    文件夹管理路由
    提供文件夹相关的 API 端点：
      POST   /create            - 创建文件夹
      GET    /list              - 获取知识库下的文件夹列表（平铺）
      GET    /children          - 获取指定文件夹的直接子文件夹
      GET    /root-files        - 获取知识库根目录文件
      GET    /{folder_id}       - 获取单个文件夹详情
      GET    /{folder_id}/files - 获取文件夹内的文件列表
      PUT    /{folder_id}/rename - 重命名文件夹
      PUT    /{folder_id}/move   - 移动文件夹
      DELETE /{folder_id}       - 删除文件夹（含后代）

    共享库改造（设计文档 §5.2.3）：入口经 KBAccess 判权后一律按 kb 维度
    查询与管理——团队成员可见并管理其他成员创建的文件夹；private 库
    判权后 kb 维度与 user 维度等价，行为不变。
@Modify History:
    2026/03/17 - 文件软删除迁移到 file.py 模块
    2026/10/18 - KBAccess 判权 + kb 维度查询（团队/公共库支持）

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from api.dependencies.auth import get_current_user_id
from api.dependencies.database import get_db_session, get_kafka_producer
from api.schemas.common import ApiResponse
from api.schemas.knowledge.folder import (
    FileInfo,
    FileListResponse,
    FolderCreateRequest,
    FolderCreateResponse,
    FolderDeleteResponse,
    FolderInfo,
    FolderListResponse,
    FolderMoveRequest,
    FolderRenameRequest,
)
from src.db.mysql.models.business.workspace_file_system import WorkspaceFileSystem
from src.db.mysql.repositories.business.workspace_file_system_repo import (
    workspace_file_system_repo,
)
from src.db.mysql.repositories.business.workspace_folder_repo import (
    workspace_folder_repo,
)
from src.db.kafka.producer import KafkaProducer
from src.service.knowledge import tombstone
from src.service.knowledge.delete_service import (
    CleanupTicket,
    knowledge_delete_service,
)
from src.service.knowledge.kb_access import require_kb_access

router = APIRouter(tags=["Folder"])


def _to_file_info(file: WorkspaceFileSystem) -> FileInfo:
    """将文件 ORM 模型转为 Pydantic 响应模型"""
    return FileInfo(
        file_id=file.file_id,
        file_name=file.file_name,
        folder_id=file.folder_id,
        file_size=file.file_size,
        mime_type=file.mime_type,
        status=file.status,
        knowledge_base_id=file.knowledge_base_id or "",
        description=file.description,
    )


def _to_folder_info(folder) -> FolderInfo:
    """将 ORM 模型转为 Pydantic 响应模型"""
    return FolderInfo(
        folder_id=folder.folder_id,
        folder_name=folder.folder_name,
        full_path=folder.full_path,
        parent_folder_id=folder.parent_folder_id,
        depth=folder.depth,
        is_default=folder.is_default,
        knowledge_base_id=folder.knowledge_base_id or "",
        description=folder.description,
    )


def _require_folder_access(
    session: Session, user_id: str, folder_id: str, level: str = "read"
):
    """取文件夹并经其所属知识库判权；不存在/无权 404（不泄露存在性）"""
    folder = workspace_folder_repo.get_by_folder_id(session, folder_id)
    if not folder or not folder.knowledge_base_id:
        raise HTTPException(status_code=404, detail="文件夹不存在或无权限")
    require_kb_access(session, user_id, folder.knowledge_base_id, level=level)
    return folder


# ==================== 创建 ====================


@router.post(
    "/create",
    response_model=ApiResponse[FolderCreateResponse],
    summary="创建文件夹",
    description="在指定知识库下创建文件夹。可通过 parent_folder_id 指定父文件夹以实现嵌套（父文件夹须属于同一知识库）。",
)
async def create_folder(
    request: FolderCreateRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[FolderCreateResponse]:
    kb = require_kb_access(
        session, user_id, request.knowledge_base_id, level="write"
    )

    parent_path = "/"
    depth = 0

    if request.parent_folder_id:
        parent_folder = workspace_folder_repo.get_by_folder_id(
            session, request.parent_folder_id
        )
        if (
            not parent_folder
            or parent_folder.knowledge_base_id != request.knowledge_base_id
        ):
            raise HTTPException(status_code=404, detail="父文件夹不存在或无权限")
        parent_path = parent_folder.full_path
        depth = parent_folder.depth + 1

    full_path = f"{parent_path}{request.folder_name}/"

    existing = workspace_folder_repo.get_by_full_path_in_kb(
        session, request.knowledge_base_id, full_path
    )
    if existing:
        raise HTTPException(status_code=409, detail=f"同路径文件夹已存在: {full_path}")

    folder_id = f"folder-{uuid.uuid4()}"
    folder = workspace_folder_repo.create(
        session,
        folder_id=folder_id,
        user_id=user_id,
        folder_name=request.folder_name,
        parent_folder_id=request.parent_folder_id,
        full_path=full_path,
        depth=depth,
        sort_order=0,
        is_default=0,
        knowledge_base_id=request.knowledge_base_id,
        knowledge_base_name=kb.knowledge_base_name,
        description=request.description,
        creator=user_id,
    )
    if not folder:
        raise HTTPException(status_code=500, detail="创建文件夹失败")

    logger.info(
        f"创建文件夹: user_id={user_id}, folder_id={folder_id}, "
        f"path={full_path}, kb_id={request.knowledge_base_id}"
    )

    return ApiResponse.success(
        data=FolderCreateResponse(
            folder_id=folder_id,
            folder_name=request.folder_name,
            full_path=full_path,
            parent_folder_id=request.parent_folder_id,
            depth=depth,
            knowledge_base_id=request.knowledge_base_id,
        ),
        message="文件夹创建成功",
    )


# ==================== 查询 ====================


@router.get(
    "/list",
    response_model=ApiResponse[FolderListResponse],
    summary="获取知识库完整文件夹树（平铺，kb 维度）",
    description=(
        "一次性返回指定知识库下的所有层级文件夹（平铺列表，按 depth + sort_order 排序）。"
        "共享库返回全部成员创建的文件夹。"
    ),
)
async def list_folders(
    knowledge_base_id: str = Query(..., description="知识库ID"),
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[FolderListResponse]:
    require_kb_access(session, user_id, knowledge_base_id, level="read")
    folders = workspace_folder_repo.get_by_kb(session, knowledge_base_id)
    items = [_to_folder_info(f) for f in folders]
    return ApiResponse.success(
        data=FolderListResponse(folders=items, total=len(items))
    )


@router.get(
    "/children",
    response_model=ApiResponse[FolderListResponse],
    summary="获取当前目录的子文件夹",
    description=(
        "获取指定父文件夹下的直接子文件夹（仅一层，不递归）。"
        "parent_folder_id 不传则返回根目录下的文件夹（当前用户自建，兼容旧行为）。"
    ),
)
async def list_children(
    parent_folder_id: Optional[str] = Query(
        default=None, description="父文件夹ID（不传则查询根目录）"
    ),
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[FolderListResponse]:
    if parent_folder_id is None:
        # 兼容旧行为：根目录文件夹按用户返回（顶级知识库导航走 /base/list）
        children = workspace_folder_repo.get_children(session, user_id, None)
    else:
        folder = _require_folder_access(session, user_id, parent_folder_id, "read")
        children = workspace_folder_repo.get_children_in_kb(
            session, folder.knowledge_base_id, parent_folder_id
        )
    items = [_to_folder_info(f) for f in children]
    return ApiResponse.success(
        data=FolderListResponse(folders=items, total=len(items))
    )


@router.get(
    "/root-files",
    response_model=ApiResponse[FileListResponse],
    summary="获取知识库根目录文件（kb 维度）",
    description=(
        "获取指定知识库根目录下的文件（即 folder_id 为空的文件）。"
        "共享库返回全部成员上传的文件。"
    ),
)
async def list_root_files(
    knowledge_base_id: str = Query(..., description="知识库ID"),
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[FileListResponse]:
    require_kb_access(session, user_id, knowledge_base_id, level="read")
    files = workspace_file_system_repo.get_by_folder_id_in_kb(
        session, knowledge_base_id, None
    )
    items = [_to_file_info(f) for f in files]
    return ApiResponse.success(
        data=FileListResponse(files=items, total=len(items))
    )


@router.get(
    "/{folder_id}",
    response_model=ApiResponse[FolderInfo],
    summary="获取文件夹详情",
    description="根据 folder_id 获取单个文件夹的详细信息。",
)
async def get_folder(
    folder_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[FolderInfo]:
    folder = _require_folder_access(session, user_id, folder_id, "read")
    return ApiResponse.success(data=_to_folder_info(folder))


@router.get(
    "/{folder_id}/files",
    response_model=ApiResponse[FileListResponse],
    summary="获取文件夹内的文件列表（kb 维度）",
    description="获取指定文件夹下的所有文件（不含子文件夹中的文件）。共享库含全部成员的文件。",
)
async def list_folder_files(
    folder_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[FileListResponse]:
    folder = _require_folder_access(session, user_id, folder_id, "read")
    files = workspace_file_system_repo.get_by_folder_id_in_kb(
        session, folder.knowledge_base_id, folder_id
    )
    items = [_to_file_info(f) for f in files]
    return ApiResponse.success(
        data=FileListResponse(files=items, total=len(items))
    )


# ==================== 修改 ====================


@router.put(
    "/{folder_id}/rename",
    response_model=ApiResponse[FolderInfo],
    summary="重命名文件夹",
    description="重命名文件夹，会级联更新所有后代文件夹的 full_path（kb 维度）。默认文件夹不允许重命名。",
)
async def rename_folder(
    folder_id: str,
    request: FolderRenameRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[FolderInfo]:
    folder = _require_folder_access(session, user_id, folder_id, "write")
    if folder.is_default == 1:
        raise HTTPException(status_code=400, detail="默认文件夹不允许重命名")

    try:
        updated = workspace_folder_repo.rename_in_kb(
            session, folder_id, request.folder_name, updater=user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if not updated:
        raise HTTPException(status_code=500, detail="重命名失败")

    return ApiResponse.success(
        data=_to_folder_info(updated), message="重命名成功"
    )


@router.put(
    "/{folder_id}/move",
    response_model=ApiResponse[FolderInfo],
    summary="移动文件夹",
    description=(
        "将文件夹移动到新的父文件夹下，级联更新所有后代的 full_path 和 depth（kb 维度）。"
        "默认文件夹不允许移动；不能移动到自身的子目录下；不能跨知识库。"
    ),
)
async def move_folder(
    folder_id: str,
    request: FolderMoveRequest,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
) -> ApiResponse[FolderInfo]:
    folder = _require_folder_access(session, user_id, folder_id, "write")
    if folder.is_default == 1:
        raise HTTPException(status_code=400, detail="默认文件夹不允许移动")

    try:
        updated = workspace_folder_repo.move_in_kb(
            session,
            folder_id,
            request.target_parent_folder_id,
            updater=user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not updated:
        raise HTTPException(status_code=500, detail="移动失败")

    return ApiResponse.success(
        data=_to_folder_info(updated), message="移动成功"
    )


# ==================== 删除 ====================


@router.delete(
    "/{folder_id}",
    response_model=ApiResponse[FolderDeleteResponse],
    summary="删除文件夹",
    description=(
        "删除文件夹及其所有后代文件夹，不可恢复。"
        "文件夹行直接物理删除；树内的文件标记 deleted=1 并投递清理任务，"
        "向量 / 文档 / 元数据 / 对象存储由 CleanupWorker 异步清理。"
        "文件立即从列表和问答检索中消失。默认文件夹不允许删除。"
    ),
)
async def delete_folder(
    folder_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_db_session),
    producer: KafkaProducer = Depends(get_kafka_producer),
) -> ApiResponse[FolderDeleteResponse]:
    folder = _require_folder_access(session, user_id, folder_id, "write")
    if folder.is_default == 1:
        raise HTTPException(status_code=400, detail="默认文件夹不允许删除")

    full_path = folder.full_path
    kb_id = folder.knowledge_base_id

    try:
        # 先标记文件：文件夹行删掉之后就查不到树内还有哪些文件了
        deleted_folder_ids = workspace_folder_repo.get_subtree_ids_in_kb(
            session, folder_id=folder_id, full_path_prefix=full_path
        )
        marked_files = workspace_file_system_repo.cascade_mark_deleted_by_folder_ids_in_kb(
            session,
            folder_ids=deleted_folder_ids,
            updater=user_id,
        )
        # 清理任务的 user_id 必须是文件行的属主（联合主键定位 + worker 物理删除依据），
        # 共享库中操作者可能不是上传者
        cleanup_tickets = [
            CleanupTicket(
                user_id=f.user_id,
                file_id=f.file_id,
                document_id=f.document_id,
                storage_path=f.storage_path,
                knowledge_base_id=f.knowledge_base_id,
            )
            for f in marked_files
        ]

        workspace_folder_repo.hard_delete_by_ids(session, deleted_folder_ids)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"删除文件夹失败: {e}")
        raise HTTPException(status_code=500, detail="删除失败")

    # 墓碑缓存失效：文件属主 + 知识库双口径（共享库删除对全员生效）
    for ticket in cleanup_tickets:
        tombstone.invalidate(ticket.user_id)
    if kb_id:
        tombstone.invalidate_kb(kb_id)
    await knowledge_delete_service.publish_cleanup(producer, cleanup_tickets)

    logger.info(
        f"删除文件夹: operator={user_id}, folder_id={folder_id}, "
        f"folders={len(deleted_folder_ids)}, files={len(cleanup_tickets)}, "
        f"kb_id={kb_id}"
    )

    return ApiResponse.success(
        data=FolderDeleteResponse(
            folder_id=folder_id,
            deleted_folder_count=len(deleted_folder_ids),
            deleted_file_count=len(cleanup_tickets),
        ),
        message=f"已删除 {len(deleted_folder_ids)} 个文件夹、{len(cleanup_tickets)} 个文件",
    )
