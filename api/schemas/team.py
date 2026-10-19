#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : team.py
@Author  : caixiongjiang
@Date    : 2026/10/18
@Function:
    团队管理 API 模型（授权客体，见设计文档 §5.2.2）

    团队是知识库共享（visibility='team'）的授权客体：
    - owner = 创建者，可管理团队信息与成员、解散团队
    - member = 读写同权（对绑定团队的知识库）
    - JP 侧"从部门导入"不是专属端点：前端先调 auth 的部门成员
      API 拿 user_id 列表，随 POST /api/team 的 member_user_ids 带入
      （快照语义由此天然成立，AKS 对 OA 无感知）。
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import List, Optional

from pydantic import BaseModel, Field

# 成员上限（含 owner）：防误操作与查询性能兜底
MAX_TEAM_MEMBERS = 500


# ==================== 创建 ====================


class TeamCreateRequest(BaseModel):
    """创建团队请求"""
    name: str = Field(..., min_length=1, max_length=255, description="团队名称")
    description: Optional[str] = Field(
        default=None, max_length=2000, description="团队描述"
    )
    member_user_ids: List[str] = Field(
        default_factory=list,
        description=(
            "初始成员 user_id 列表（可选；JP 从部门导入时由前端把部门成员"
            " user_id 填入此处，创建即快照）"
        ),
    )


# ==================== 查询 ====================


class TeamMemberInfo(BaseModel):
    """团队成员信息"""
    user_id: str = Field(..., description="成员用户ID")
    role: str = Field(..., description="角色：owner=创建者 / member=成员")
    joined_time: Optional[str] = Field(
        default=None, description="加入时间（ISO 格式）"
    )


class TeamInfo(BaseModel):
    """团队信息（列表项）"""
    team_id: str = Field(..., description="团队ID")
    name: str = Field(..., description="团队名称")
    description: Optional[str] = Field(default=None, description="团队描述")
    my_role: str = Field(..., description="当前用户在团队中的角色")
    member_count: int = Field(..., description="成员数量")
    create_time: Optional[str] = Field(
        default=None, description="创建时间（ISO 格式）"
    )


class TeamDetailResponse(BaseModel):
    """团队详情（含成员列表）"""
    team_id: str = Field(..., description="团队ID")
    name: str = Field(..., description="团队名称")
    description: Optional[str] = Field(default=None, description="团队描述")
    my_role: str = Field(..., description="当前用户在团队中的角色")
    members: List[TeamMemberInfo] = Field(
        default_factory=list, description="成员列表（owner 在前）"
    )


class TeamListResponse(BaseModel):
    """团队列表响应"""
    teams: List[TeamInfo] = Field(default_factory=list, description="团队列表")
    total: int = Field(default=0, description="总数")


# ==================== 更新 ====================


class TeamUpdateRequest(BaseModel):
    """更新团队请求"""
    name: Optional[str] = Field(
        default=None, min_length=1, max_length=255, description="新名称"
    )
    description: Optional[str] = Field(
        default=None, max_length=2000, description="新描述"
    )


class TeamMemberAddRequest(BaseModel):
    """批量添加成员请求"""
    user_ids: List[str] = Field(
        ..., min_length=1, description="成员 user_id 列表"
    )
