#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : knowledge_base_repo.py
@Author  : caixiongjiang
@Date    : 2026/02/19
@Function: 
    KnowledgeBase Repository
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import List, Optional

from loguru import logger
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.db.mysql.models.business.knowledge_base import KnowledgeBase
from src.db.mysql.models.business.workspace_file_system import WorkspaceFileSystem
from src.db.mysql.models.business.workspace_folder import WorkspaceFolder
from src.db.mysql.repositories.base_repository import BaseRepository


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    """KnowledgeBase Repository"""

    def __init__(self) -> None:
        super().__init__(KnowledgeBase)

    def get_by_user_id(
        self,
        session: Session,
        user_id: str,
    ) -> List[KnowledgeBase]:
        """获取用户的所有知识库"""
        try:
            return (
                session.query(self.model)
                .filter(
                    self.model.user_id == user_id,
                    self.model.deleted == 0,
                )
                .order_by(self.model.create_time.desc())
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"查询用户知识库列表失败: {e}")
            return []

    def get_by_id_and_user(
        self,
        session: Session,
        knowledge_base_id: str,
        user_id: str,
    ) -> Optional[KnowledgeBase]:
        """按 ID + user_id 查询（含权限校验）"""
        try:
            return (
                session.query(self.model)
                .filter(
                    self.model.knowledge_base_id == knowledge_base_id,
                    self.model.user_id == user_id,
                    self.model.deleted == 0,
                )
                .first()
            )
        except SQLAlchemyError as e:
            logger.error(f"查询知识库失败: {e}")
            return None

    def get_children(
        self,
        session: Session,
        user_id: str,
        parent_knowledge_base_id: Optional[str],
    ) -> List[KnowledgeBase]:
        """获取子知识库列表"""
        try:
            query = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.deleted == 0,
            )
            if parent_knowledge_base_id is None:
                query = query.filter(
                    self.model.parent_knowledge_base_id.is_(None)
                )
            else:
                query = query.filter(
                    self.model.parent_knowledge_base_id
                    == parent_knowledge_base_id
                )
            return query.order_by(self.model.create_time.desc()).all()
        except SQLAlchemyError as e:
            logger.error(f"查询子知识库失败: {e}")
            return []

    def get_children_by_parent(
        self,
        session: Session,
        parent_knowledge_base_id: Optional[str],
    ) -> List[KnowledgeBase]:
        """按父 ID 查子知识库（kb 维度，不限创建者——共享库的子树
        可能由不同成员创建）"""
        try:
            query = session.query(self.model).filter(
                self.model.deleted == 0,
            )
            if parent_knowledge_base_id is None:
                query = query.filter(
                    self.model.parent_knowledge_base_id.is_(None)
                )
            else:
                query = query.filter(
                    self.model.parent_knowledge_base_id
                    == parent_knowledge_base_id
                )
            return query.order_by(self.model.create_time.desc()).all()
        except SQLAlchemyError as e:
            logger.error(f"查询子知识库失败: {e}")
            return []

    def get_all_descendants(
        self,
        session: Session,
        user_id: str,
        knowledge_base_id: str,
    ) -> List[str]:
        """BFS 收集所有后代知识库 ID（不含自身）"""
        result: List[str] = []
        queue = [knowledge_base_id]
        while queue:
            parent_id = queue.pop(0)
            children = self.get_children(session, user_id, parent_id)
            for child in children:
                result.append(child.knowledge_base_id)
                queue.append(child.knowledge_base_id)
        return result

    def check_tree_has_files(
        self,
        session: Session,
        user_id: str,
        kb_ids: List[str],
    ) -> int:
        """检查一组知识库下的总文件数

        deleted=1 的待清理文件也算占用：此时向量等关联数据还没清干净，
        先把知识库删掉会留下没人认领的孤儿数据。
        """
        if not kb_ids:
            return 0
        try:
            return (
                session.query(WorkspaceFileSystem)
                .filter(
                    WorkspaceFileSystem.user_id == user_id,
                    WorkspaceFileSystem.knowledge_base_id.in_(kb_ids),
                    WorkspaceFileSystem.deleted.in_([0, 1]),
                )
                .count()
            )
        except SQLAlchemyError as e:
            logger.error(f"批量检查知识库文件数量失败: {e}")
            return -1

    def hard_delete(
        self,
        session: Session,
        user_id: str,
        knowledge_base_id: str,
    ) -> int:
        """
        物理删除知识库，同时清理其下的所有文件夹。
        **不会 commit**，由调用方统一提交事务。

        前置条件：知识库下不存在任何文件（由调用方保证）。

        Returns:
            被删除的文件夹数量
        """
        session.query(self.model).filter(
            self.model.knowledge_base_id == knowledge_base_id,
            self.model.user_id == user_id,
        ).delete(synchronize_session='fetch')

        folder_count = session.query(WorkspaceFolder).filter(
            WorkspaceFolder.user_id == user_id,
            WorkspaceFolder.knowledge_base_id == knowledge_base_id,
        ).delete(synchronize_session='fetch')

        logger.info(
            f"物理删除知识库: user_id={user_id}, "
            f"kb_id={knowledge_base_id}, 清理{folder_count}个文件夹"
        )
        return folder_count

    def name_exists(
        self,
        session: Session,
        user_id: str,
        name: str,
        exclude_id: Optional[str] = None,
    ) -> bool:
        """检查同一用户下是否存在同名知识库"""
        try:
            query = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.knowledge_base_name == name,
                self.model.deleted == 0,
            )
            if exclude_id:
                query = query.filter(
                    self.model.knowledge_base_id != exclude_id
                )
            return query.first() is not None
        except SQLAlchemyError as e:
            logger.error(f"检查知识库名称重复失败: {e}")
            return False

    # ==================== 共享库（team/public）支持 ====================

    def get_accessible(
        self,
        session: Session,
        user_id: str,
        team_ids: Optional[List[str]] = None,
    ) -> List[KnowledgeBase]:
        """三路 union：自己的 private + 所在 team 的 + 全部 public

        team_ids 为空列表/None 时（未加入任何团队）退化为 private + public。
        """
        try:
            conditions = [
                and_(
                    self.model.visibility == "private",
                    self.model.user_id == user_id,
                ),
                self.model.visibility == "public",
            ]
            if team_ids:
                conditions.append(
                    and_(
                        self.model.visibility == "team",
                        self.model.team_id.in_(team_ids),
                    )
                )
            return (
                session.query(self.model)
                .filter(
                    or_(*conditions),
                    self.model.deleted == 0,
                )
                .order_by(self.model.create_time.desc())
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"查询可访问知识库列表失败: {e}")
            return []

    def count_active_by_team(
        self,
        session: Session,
        team_id: str,
    ) -> int:
        """绑定在某团队上的存活知识库数量（团队解散保护用）"""
        try:
            return (
                session.query(self.model)
                .filter(
                    self.model.team_id == team_id,
                    self.model.visibility == "team",
                    self.model.deleted == 0,
                )
                .count()
            )
        except SQLAlchemyError as e:
            logger.error(f"统计团队知识库数量失败: {e}")
            return 0

    def get_all_descendants_by_kb(
        self,
        session: Session,
        knowledge_base_id: str,
    ) -> List[str]:
        """BFS 收集所有后代知识库 ID（kb 维度，不限创建者——共享库子树
        可能由不同成员创建，user 维度 BFS 会漏）"""
        result: List[str] = []
        queue = [knowledge_base_id]
        while queue:
            parent_id = queue.pop(0)
            children = (
                session.query(self.model.knowledge_base_id)
                .filter(
                    self.model.parent_knowledge_base_id == parent_id,
                    self.model.deleted == 0,
                )
                .all()
            )
            for (child_id,) in children:
                result.append(child_id)
                queue.append(child_id)
        return result

    def check_kb_tree_has_files(
        self,
        session: Session,
        kb_ids: List[str],
    ) -> int:
        """检查一组知识库下的总文件数（kb 维度，成员上传的文件同样计入；
        deleted=1 的待清理文件也算占用）"""
        if not kb_ids:
            return 0
        try:
            return (
                session.query(WorkspaceFileSystem)
                .filter(
                    WorkspaceFileSystem.knowledge_base_id.in_(kb_ids),
                    WorkspaceFileSystem.deleted.in_([0, 1]),
                )
                .count()
            )
        except SQLAlchemyError as e:
            logger.error(f"批量检查知识库文件数量失败: {e}")
            return -1

    def hard_delete_by_kb(
        self,
        session: Session,
        knowledge_base_id: str,
    ) -> int:
        """物理删除知识库及其下所有文件夹（kb 维度，不限创建者）。

        **不会 commit**，由调用方统一提交事务。
        前置条件：知识库下不存在任何文件（由调用方保证）。
        """
        session.query(self.model).filter(
            self.model.knowledge_base_id == knowledge_base_id,
        ).delete(synchronize_session="fetch")

        folder_count = session.query(WorkspaceFolder).filter(
            WorkspaceFolder.knowledge_base_id == knowledge_base_id,
        ).delete(synchronize_session="fetch")

        logger.info(
            f"物理删除知识库(kb维度): kb_id={knowledge_base_id}, "
            f"清理{folder_count}个文件夹"
        )
        return folder_count


# 全局实例
knowledge_base_repo = KnowledgeBaseRepository()
