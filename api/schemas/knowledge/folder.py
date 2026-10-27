#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : folder.py
@Author  : caixiongjiang
@Date    : 2026/02/18
@Function: 
    文件夹管理 API 模型
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import List, Optional
from pydantic import BaseModel, Field


# ==================== 通用文件夹信息 ====================


class FolderInfo(BaseModel):
    """文件夹信息（列表/详情共用）"""
    folder_id: str = Field(..., description="文件夹ID")
    folder_name: str = Field(..., description="文件夹名称")
    full_path: str = Field(..., description="完整路径")
    parent_folder_id: Optional[str] = Field(default=None, description="父文件夹ID")
    depth: int = Field(..., description="目录层级深度")
    is_default: int = Field(default=0, description="是否为默认文件夹")
    knowledge_base_id: str = Field(default="", description="所属知识库ID")
    description: Optional[str] = Field(default=None, description="文件夹描述")


# ==================== 创建 ====================


class FolderCreateRequest(BaseModel):
    """创建文件夹请求"""
    folder_name: str = Field(
        ..., min_length=1, max_length=255, description="文件夹名称"
    )
    knowledge_base_id: str = Field(..., description="所属知识库ID")
    parent_folder_id: Optional[str] = Field(
        default=None, description="父文件夹ID（不传则在根目录创建）"
    )
    description: Optional[str] = Field(default=None, description="文件夹描述")


class FolderCreateResponse(FolderInfo):
    """创建文件夹响应"""
    pass


# ==================== 查询 ====================


class FolderListResponse(BaseModel):
    """文件夹列表响应"""
    folders: List[FolderInfo] = Field(default_factory=list, description="文件夹列表")
    total: int = Field(default=0, description="总数")


# ==================== 重命名 ====================


class FolderRenameRequest(BaseModel):
    """重命名文件夹请求"""
    folder_name: str = Field(
        ..., min_length=1, max_length=255, description="新文件夹名称"
    )


# ==================== 移动 ====================


class FolderMoveRequest(BaseModel):
    """移动文件夹请求"""
    target_parent_folder_id: Optional[str] = Field(
        default=None, description="目标父文件夹ID（None 表示移到根目录）"
    )


# ==================== 文件夹内文件查询 ====================


class FileInfo(BaseModel):
    """文件夹内的文件信息"""
    file_id: str = Field(..., description="文件ID")
    file_name: str = Field(..., description="文件名")
    folder_id: Optional[str] = Field(default=None, description="所属文件夹ID")
    file_size: Optional[int] = Field(default=None, description="文件大小（字节）")
    mime_type: Optional[str] = Field(default=None, description="MIME 类型")
    status: Optional[int] = Field(default=None, description="处理状态")
    knowledge_base_id: str = Field(default="", description="所属知识库ID")
    description: Optional[str] = Field(default=None, description="文件描述")
    user_id: Optional[str] = Field(
        default=None,
        description="上传者用户ID（共享库内即上传成员，个人库为库主人）",
    )
    create_time: Optional[str] = Field(
        default=None, description="上传时间（ISO 格式）"
    )


class FileListResponse(BaseModel):
    """文件列表响应"""
    files: List[FileInfo] = Field(default_factory=list, description="文件列表")
    total: int = Field(default=0, description="总数")


# ==================== 删除 ====================


class FolderDeleteResponse(BaseModel):
    """删除文件夹响应"""
    folder_id: str = Field(..., description="被删除的文件夹ID")
    deleted_folder_count: int = Field(..., description="删除的文件夹数量（含后代）")
    deleted_file_count: int = Field(..., description="级联删除的文件数量")
