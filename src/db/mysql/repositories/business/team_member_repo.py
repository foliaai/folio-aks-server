#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : team_member_repo.py
@Author  : caixiongjiang
@Date    : 2026/10/18
@Function:
    TeamMember Repository

    纯关系表（联合主键 team_id+user_id、无 deleted 列），
    不继承 BaseRepository（其 get_by_id 依赖单主键与软删列），
    查询手写。
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import List, Optional

from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.db.mysql.models.business.team_member import TeamMember


class TeamMemberRepository:
    """TeamMember Repository（物理删行，无软删）"""

    def get_member_role(
        self,
        session: Session,
        team_id: str,
        user_id: str,
    ) -> Optional[str]:
        """点查成员角色；非成员返回 None（判权热路径，联合主键点查）"""
        try:
            row = (
                session.query(TeamMember.role)
                .filter(
                    TeamMember.team_id == team_id,
                    TeamMember.user_id == user_id,
                )
                .first()
            )
            return row[0] if row else None
        except SQLAlchemyError as e:
            logger.error(f"查询团队成员角色失败: {e}")
            return None

    def list_by_team(
        self,
        session: Session,
        team_id: str,
    ) -> List[TeamMember]:
        """列出团队成员（owner 在前，其余按加入时间）"""
        try:
            return (
                session.query(TeamMember)
                .filter(TeamMember.team_id == team_id)
                .order_by(
                    TeamMember.role.desc(),  # owner > member（字母序恰好 owner 在前）
                    TeamMember.joined_time.asc(),
                )
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"查询团队成员列表失败: {e}")
            return []

    def list_team_ids_by_user(
        self,
        session: Session,
        user_id: str,
    ) -> List[str]:
        """我加入的团队 ID 列表"""
        try:
            rows = (
                session.query(TeamMember.team_id)
                .filter(TeamMember.user_id == user_id)
                .all()
            )
            return [r[0] for r in rows]
        except SQLAlchemyError as e:
            logger.error(f"查询用户团队列表失败: {e}")
            return []

    def count_by_team(self, session: Session, team_id: str) -> int:
        """团队成员数量"""
        try:
            return (
                session.query(TeamMember)
                .filter(TeamMember.team_id == team_id)
                .count()
            )
        except SQLAlchemyError as e:
            logger.error(f"统计团队成员数量失败: {e}")
            return 0

    def add_members(
        self,
        session: Session,
        team_id: str,
        user_ids: List[str],
        role: str = "member",
    ) -> List[TeamMember]:
        """批量添加成员（跳过已存在的 user_id）

        **不会 commit**，由调用方统一提交事务。
        """
        existing = {m.user_id for m in self.list_by_team(session, team_id)}
        new_members = [
            TeamMember(
                team_id=team_id,
                user_id=uid,
                role=role,
                joined_time=datetime.now(),
            )
            for uid in dict.fromkeys(user_ids)  # 去重保序
            if uid and uid not in existing
        ]
        if new_members:
            session.add_all(new_members)
        return new_members

    def remove_member(
        self,
        session: Session,
        team_id: str,
        user_id: str,
    ) -> int:
        """移除成员（物理删行）

        **不会 commit**，由调用方统一提交事务。
        """
        return (
            session.query(TeamMember)
            .filter(
                TeamMember.team_id == team_id,
                TeamMember.user_id == user_id,
            )
            .delete(synchronize_session="fetch")
        )

    def remove_all_by_team(self, session: Session, team_id: str) -> int:
        """解散团队时清空成员关系（物理删行）

        **不会 commit**，由调用方统一提交事务。
        """
        return (
            session.query(TeamMember)
            .filter(TeamMember.team_id == team_id)
            .delete(synchronize_session="fetch")
        )


# 全局实例
team_member_repo = TeamMemberRepository()
