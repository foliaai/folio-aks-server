#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : team_member.py
@Author  : caixiongjiang
@Date    : 2026/10/18
@Function:
    TeamMember Schema 定义（团队成员，纯关系表）

    设计要点（docs/特殊功能设计/团队知识库与公共知识库设计.md §4）：
    - 联合主键 (team_id, user_id)，不继承 BaseModel——纯关系表无审计列、
      不软删（物理删行），因此也不能走 BaseRepository（其 get_by_id 依赖
      单主键与 deleted 列）；
    - idx_tm_user 支撑"我加入的团队"反查。
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from datetime import datetime

from sqlalchemy import Column, String, DateTime, Index

from src.db.mysql.models.base_model import Base


class TeamMember(Base):
    """团队成员表（快照语义：导入后不受 OA 同步影响）"""
    __tablename__ = "team_member"

    __table_args__ = (
        Index("idx_tm_user", "user_id"),
    )

    team_id = Column(
        String(64),
        primary_key=True,
        comment="团队ID（→ team.team_id）",
    )

    user_id = Column(
        String(64),
        primary_key=True,
        comment="成员用户ID（OA 工号 / Logto sub，与 auth user_profile 对齐）",
    )

    role = Column(
        String(16),
        default="member",
        nullable=False,
        comment="角色：owner=创建者（可管理团队与成员）/ member=成员（读写同权）",
    )

    joined_time = Column(
        DateTime,
        default=datetime.now,
        nullable=False,
        comment="加入时间",
    )
