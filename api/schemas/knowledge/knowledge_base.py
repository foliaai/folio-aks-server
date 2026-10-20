#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : knowledge_base.py
@Author  : caixiongjiang
@Date    : 2026/02/19
@Function: 
    知识库管理 API 模型
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import List, Optional

from pydantic import BaseModel, Field


# ==================== 通用信息 ====================


class KnowledgeBaseInfo(BaseModel):
    """知识库信息"""
    knowledge_base_id: str = Field(..., description="知识库ID")
    knowledge_base_name: str = Field(..., description="知识库名称")
    parent_knowledge_base_id: Optional[str] = Field(
        default=None, description="父知识库ID"
    )
    knowledge_type: str = Field(default="common_file", description="知识库类型")
    description: Optional[str] = Field(default=None, description="知识库描述")
    visibility: str = Field(
        default="private",
        description="可见性：private=个人 / team=团队 / public=公共",
    )
    team_id: Optional[str] = Field(
        default=None, description="visibility=team 时绑定的团队ID"
    )
    team_name: Optional[str] = Field(
        default=None, description="团队名称（visibility=team 时返回）"
    )
    owner_user_id: str = Field(
        default="", description="归属用户ID（private 的 owner / 共享库的创建者）"
    )


# ==================== 创建 ====================


class KnowledgeBaseCreateRequest(BaseModel):
    """创建知识库请求"""
    knowledge_base_name: str = Field(
        ..., min_length=1, max_length=255, description="知识库名称"
    )
    parent_knowledge_base_id: Optional[str] = Field(
        default=None, description="父知识库ID（不传则为顶级知识库）"
    )
    knowledge_type: str = Field(
        default="common_file", description="知识库类型"
    )
    description: Optional[str] = Field(default=None, description="知识库描述")
    visibility: str = Field(
        default="private",
        description="可见性：private=个人（默认）/ team=团队 / public=公共",
    )
    team_id: Optional[str] = Field(
        default=None,
        description="visibility=team 时必填：绑定的团队ID（创建者须为该团队成员）",
    )


# ==================== 查询 ====================


class KnowledgeBaseListPermissions(BaseModel):
    """列表附带的当前用户能力位（前端据此显隐创建入口）"""
    can_create_public: bool = Field(
        default=False, description="是否允许创建公共知识库"
    )


class KnowledgeBaseListResponse(BaseModel):
    """知识库列表响应

    返回当前用户可读的全量知识库：自己的 private + 所在团队的 team + 全部 public。
    """
    knowledge_bases: List[KnowledgeBaseInfo] = Field(
        default_factory=list, description="知识库列表"
    )
    total: int = Field(default=0, description="总数")
    permissions: KnowledgeBaseListPermissions = Field(
        default_factory=KnowledgeBaseListPermissions,
        description="当前用户能力位",
    )


# ==================== 更新 ====================


class KnowledgeBaseUpdateRequest(BaseModel):
    """更新知识库请求"""
    knowledge_base_name: Optional[str] = Field(
        default=None, min_length=1, max_length=255, description="新名称"
    )
    description: Optional[str] = Field(default=None, description="新描述")


# ==================== 删除 ====================


class KnowledgeBaseDeleteResponse(BaseModel):
    """删除知识库响应"""
    knowledge_base_id: str = Field(..., description="被删除的知识库ID")
    deleted_folder_count: int = Field(
        default=0, description="同时清理的空文件夹数量"
    )
    deleted_kb_count: int = Field(
        default=1, description="被删除的知识库数量（含子知识库）"
    )
