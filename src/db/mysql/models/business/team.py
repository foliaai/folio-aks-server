#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : team.py
@Author  : caixiongjiang
@Date    : 2026/10/18
@Function:
    Team Schema 定义（团队知识库的授权客体）

    设计要点（docs/特殊功能设计/团队知识库与公共知识库设计.md §4）：
    - 独立于 OA 组织（organization/user_organization 在 auth 库），
      不受 OA 登录同步"整体替换"语义影响；
    - JP 侧建团队时可从部门一次性导入成员快照，此后独立维护；
    - owner 不在表上冗余，由 team_member.role='owner' 表达（一人一档，
      避免双写不一致）。
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from sqlalchemy import Column, String, Text, Index

from src.db.mysql.models.base_model import BaseModel


class Team(BaseModel):
    """
    团队表

    团队知识库（knowledge_base.visibility='team'）的授权客体，
    成员关系见 team_member 表。

    删除规则：
    - 仍有 visibility='team' 且 team_id 指向本团队的知识库时禁止解散
      （由业务层校验，409）
    """
    __tablename__ = "team"

    __table_args__ = (
        Index("idx_team_creator", "creator"),
    )

    team_id = Column(
        String(64),
        primary_key=True,
        comment="团队ID（team-{uuid}）",
    )

    name = Column(
        String(255),
        nullable=False,
        comment="团队名称（不做全局唯一，不同 owner 可同名）",
    )

    description = Column(
        Text,
        nullable=True,
        comment="团队描述",
    )

    # BaseModel 字段会自动继承：
    # - status, creator, create_time, updater, update_time, deleted
