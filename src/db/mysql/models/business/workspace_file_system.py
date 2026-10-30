#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : workspace_file_system.py
@Author  : caixiongjiang
@Date    : 2026/01/06
@Function: 
    WorkspaceFileSystem Schema 定义
@Modify History:
    2026/02/16 - 移除 AgentMixin，用 folder_id 替代 folder_path/folder_parent_path，
                 修复 is_text_readable 类型，添加索引
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from sqlalchemy import Column, String, Integer, BigInteger, LargeBinary, Text, Index
from src.db.mysql.models.base_model import BaseModel, KnowledgeMixin


class WorkspaceFileSystem(BaseModel, KnowledgeMixin):
    """
    工作空间文件系统表
    
    存储用户工作空间中的文件信息，包括文件基本信息、存储位置、
    文档元数据、Git信息、处理状态等。
    通过 folder_id 关联 WorkspaceFolder 表实现目录管理。
    
    本表同时承担原 DocumentMetaInfo 的职责，
    存储文件路径、文件类型、SHA256 哈希值等文档元数据。
    
    主键：(user_id, file_id) 联合主键
    """
    __tablename__ = "workspace_file_system"
    
    __table_args__ = (
        Index("idx_user_kb", "user_id", "knowledge_base_id"),
        Index("idx_user_folder", "user_id", "folder_id"),
        Index("idx_user_status", "user_id", "status"),
    )
    
    # ==================== 主键（联合主键） ====================
    
    user_id = Column(
        String(64), 
        primary_key=True,
        comment="用户ID（主键之一）"
    )
    
    file_id = Column(
        String(255), 
        primary_key=True,
        nullable=False,
        comment="文件ID（主键之一，格式: file-{uuid}，业务层唯一标识）"
    )
    
    # ==================== 文件基本信息 ====================
    
    file_name = Column(
        String(255), 
        nullable=False,
        comment="用户上传时的原始文件名"
    )
    
    folder_id = Column(
        String(64),
        nullable=True,
        index=True,
        comment="所属文件夹ID（关联 workspace_folder.folder_id，NULL 表示根目录）"
    )
    
    file_size = Column(
        BigInteger, 
        nullable=True,
        comment="文件大小（字节）"
    )
    
    file_type = Column(
        String(50),
        nullable=True,
        comment="文件类型：pdf, docx, txt 等"
    )
    
    file_format = Column(
        String(255),
        nullable=True,
        comment="文件格式详细信息"
    )
    
    file_suffix = Column(
        String(50),
        nullable=True,
        comment="文件后缀名（含点号，如 .pdf, .docx）"
    )
    
    mime_type = Column(
        String(255), 
        nullable=True,
        index=True,
        comment="文件MIME类型（如 text/plain, image/png）"
    )
    
    file_sha256 = Column(
        LargeBinary(32),
        nullable=True,
        comment="文件SHA256哈希值（32字节二进制）"
    )
    
    document_id = Column(
        String(255), 
        index=True, 
        nullable=True,
        comment="关联的Document ID（格式: document-{uuid}，基于file_sha256的后台唯一标识，相同内容的文件共享同一document_id）"
    )
    
    # ==================== 存储信息 ====================
    
    storage_id = Column(
        Integer,
        default=-1,
        nullable=False,
        comment="存储系统ID：-1=本地存储，其他值对应具体存储系统"
    )
    
    bucket_name = Column(
        String(255),
        nullable=True,
        comment="对象存储桶名称"
    )
    
    file_path = Column(
        String(1024),
        nullable=True,
        comment="文件在对象存储中的路径"
    )
    
    session_id = Column(
        String(128),
        nullable=True,
        comment="上传会话ID"
    )
    
    # ==================== Git 相关 ====================
    
    git_path = Column(
        String(500), 
        nullable=True,
        comment="Git仓库中的物理路径"
    )
    
    git_commit_hash = Column(
        String(40), 
        nullable=True,
        comment="Git commit hash（40位十六进制）"
    )
    
    # ==================== 文件特征 ====================
    
    is_text_readable = Column(
        Integer,
        nullable=True,
        comment="文本可读性标识：0=不可读，1=可读，2=未知"
    )
    
    # ==================== 处理状态 ====================

    sort_order = Column(
        Integer,
        nullable=True,
        comment="文件夹内手动排序权重（0..n-1 固化写入）；NULL=未排序，按 create_time 沉底",
    )

    queue_position = Column(
        Integer, 
        nullable=True,
        comment="队列位置（用于排队处理）"
    )
    
    message = Column(
        String(1024), 
        nullable=True,
        comment="处理消息或错误信息"
    )
    
    is_duplication = Column(
        Integer, 
        nullable=True,
        comment="是否重复文件：0=否，1=是"
    )
    
    # ==================== 扩展信息 ====================
    
    ext_attributes = Column(
        String(4096), 
        nullable=True,
        comment="扩展属性（JSON格式）"
    )
    
    description = Column(
        Text, 
        nullable=True,
        comment="文件描述"
    )

    @property
    def storage_path(self) -> str:
        """重建完整存储路径：bucket_name/file_path"""
        if self.bucket_name and self.file_path:
            return f"{self.bucket_name}/{self.file_path}"
        return self.file_path or ""
