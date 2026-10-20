#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : workspace_folder_repo.py
@Author  : caixiongjiang
@Date    : 2026/02/16
@Function: 
    WorkspaceFolder Repository
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import uuid
from typing import List, Optional

from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.db.mysql.models.business.workspace_folder import WorkspaceFolder
from src.db.mysql.repositories.base_repository import BaseRepository

_DEFAULT_FOLDER_NAME = "user_uploads"
_DEFAULT_FOLDER_PATH = f"/{_DEFAULT_FOLDER_NAME}/"


class WorkspaceFolderRepository(BaseRepository[WorkspaceFolder]):
    """WorkspaceFolder Repository"""
    
    def __init__(self):
        super().__init__(WorkspaceFolder)
    
    def get_or_create_default(
        self,
        session: Session,
        user_id: str,
        knowledge_base_id: str,
    ) -> WorkspaceFolder:
        """
        获取或创建用户在指定知识库下的默认文件夹

        同一 (user_id, knowledge_base_id) 下最多一个 is_default=1 的文件夹。
        首次调用时自动创建，后续直接返回已有记录。
        知识库名称从 knowledge_base 表自动查询，无需外部传入。

        Args:
            session: 数据库会话
            user_id: 用户 ID
            knowledge_base_id: 知识库 ID

        Returns:
            默认文件夹的 WorkspaceFolder 实例

        Raises:
            RuntimeError: 创建默认文件夹失败时抛出
        """
        try:
            existing = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.knowledge_base_id == knowledge_base_id,
                self.model.is_default == 1,
                self.model.deleted == 0,
            ).first()

            if existing:
                return existing

            from src.db.mysql.models.business.knowledge_base import KnowledgeBase
            kb = session.query(KnowledgeBase).filter(
                KnowledgeBase.knowledge_base_id == knowledge_base_id,
            ).first()
            kb_name = kb.knowledge_base_name if kb else ""

            folder = self.model(
                folder_id=str(uuid.uuid4()),
                user_id=user_id,
                folder_name=_DEFAULT_FOLDER_NAME,
                parent_folder_id=None,
                full_path=_DEFAULT_FOLDER_PATH,
                depth=0,
                sort_order=0,
                is_default=1,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=kb_name,
                creator=user_id,
            )
            session.add(folder)
            session.flush()

            logger.info(
                f"创建默认文件夹: user_id={user_id}, "
                f"kb_id={knowledge_base_id}, folder_id={folder.folder_id}"
            )
            return folder

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"获取/创建默认文件夹失败: {e}")
            raise RuntimeError(f"获取默认文件夹失败: {e}") from e

    def get_by_user_id(
        self,
        session: Session,
        user_id: str
    ) -> List[WorkspaceFolder]:
        """
        根据 user_id 查询所有文件夹
        
        Args:
            session: 数据库会话
            user_id: 用户ID
        
        Returns:
            WorkspaceFolder 列表
        """
        try:
            results = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.deleted == 0
            ).all()
            
            logger.debug(
                f"查询到{len(results)}个WorkspaceFolder: user_id={user_id}"
            )
            return results
        except SQLAlchemyError as e:
            logger.error(f"根据user_id查询WorkspaceFolder失败: {e}")
            return []
    
    def get_by_user_and_knowledge_base(
        self,
        session: Session,
        user_id: str,
        knowledge_base_id: str
    ) -> List[WorkspaceFolder]:
        """
        根据用户ID和知识库ID查询文件夹列表
        
        Args:
            session: 数据库会话
            user_id: 用户ID
            knowledge_base_id: 知识库ID
        
        Returns:
            WorkspaceFolder 列表
        """
        try:
            results = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.knowledge_base_id == knowledge_base_id,
                self.model.deleted == 0
            ).order_by(self.model.depth, self.model.sort_order).all()
            
            logger.debug(
                f"查询到{len(results)}个WorkspaceFolder: "
                f"user_id={user_id}, knowledge_base_id={knowledge_base_id}"
            )
            return results
        except SQLAlchemyError as e:
            logger.error(
                f"根据user_id和knowledge_base_id查询WorkspaceFolder失败: {e}"
            )
            return []
    
    def get_children(
        self,
        session: Session,
        user_id: str,
        parent_folder_id: Optional[str]
    ) -> List[WorkspaceFolder]:
        """
        获取指定文件夹的直接子文件夹
        
        Args:
            session: 数据库会话
            user_id: 用户ID
            parent_folder_id: 父文件夹ID（None 表示查询根目录下的文件夹）
        
        Returns:
            WorkspaceFolder 列表
        """
        try:
            query = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.deleted == 0
            )
            
            if parent_folder_id is None:
                query = query.filter(self.model.parent_folder_id.is_(None))
            else:
                query = query.filter(
                    self.model.parent_folder_id == parent_folder_id
                )
            
            results = query.order_by(self.model.sort_order).all()
            
            logger.debug(
                f"查询到{len(results)}个子文件夹: "
                f"user_id={user_id}, parent_folder_id={parent_folder_id}"
            )
            return results
        except SQLAlchemyError as e:
            logger.error(f"查询子文件夹失败: {e}")
            return []
    
    def get_by_full_path(
        self,
        session: Session,
        user_id: str,
        full_path: str
    ) -> Optional[WorkspaceFolder]:
        """
        根据完整路径查询文件夹
        
        Args:
            session: 数据库会话
            user_id: 用户ID
            full_path: 完整路径
        
        Returns:
            WorkspaceFolder 实例，未找到返回 None
        """
        try:
            result = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.full_path == full_path,
                self.model.deleted == 0
            ).first()
            
            if not result:
                logger.debug(
                    f"未找到WorkspaceFolder: "
                    f"user_id={user_id}, full_path={full_path}"
                )
            
            return result
        except SQLAlchemyError as e:
            logger.error(f"根据full_path查询WorkspaceFolder失败: {e}")
            return None
    
    def get_descendants(
        self,
        session: Session,
        user_id: str,
        full_path_prefix: str
    ) -> List[WorkspaceFolder]:
        """
        获取指定路径前缀下的所有后代文件夹
        
        通过 full_path 的 LIKE 前缀匹配实现。
        
        Args:
            session: 数据库会话
            user_id: 用户ID
            full_path_prefix: 路径前缀（如 /项目A/文档/）
        
        Returns:
            WorkspaceFolder 列表
        """
        try:
            results = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.full_path.like(f"{full_path_prefix}%"),
                self.model.deleted == 0
            ).order_by(self.model.depth).all()
            
            logger.debug(
                f"查询到{len(results)}个后代文件夹: "
                f"user_id={user_id}, prefix={full_path_prefix}"
            )
            return results
        except SQLAlchemyError as e:
            logger.error(f"查询后代文件夹失败: {e}")
            return []
    
    def get_subtree_ids(
        self,
        session: Session,
        user_id: str,
        folder_id: str,
        full_path_prefix: str,
    ) -> List[str]:
        """
        获取文件夹自身及其所有后代文件夹的 ID。

        Args:
            session: 数据库会话
            user_id: 用户ID
            folder_id: 文件夹ID
            full_path_prefix: 该文件夹的完整路径前缀

        Returns:
            folder_id 列表（含目标自身）
        """
        descendant_rows = session.query(self.model.folder_id).filter(
            self.model.user_id == user_id,
            self.model.full_path.like(f"{full_path_prefix}%"),
            self.model.folder_id != folder_id,
        ).all()
        return [folder_id] + [row[0] for row in descendant_rows]

    def hard_delete_by_ids(
        self,
        session: Session,
        folder_ids: List[str],
    ) -> int:
        """
        物理删除指定的文件夹行。

        文件夹只是 MySQL 里的一行，没有向量 / 文档 / 对象存储数据，
        所以不需要走异步清理，直接删掉即可。树内的文件由调用方单独标记删除并投递清理任务。
        **不会 commit**，由调用方统一提交事务。

        Returns:
            被删除的行数
        """
        if not folder_ids:
            return 0
        count = session.query(self.model).filter(
            self.model.folder_id.in_(folder_ids),
        ).delete(synchronize_session='fetch')
        logger.debug(f"物理删除{count}个文件夹")
        return count


    def rename(
        self,
        session: Session,
        user_id: str,
        folder_id: str,
        new_name: str,
        updater: str = "",
    ) -> Optional[WorkspaceFolder]:
        """
        重命名文件夹，并级联更新所有后代的 full_path

        Args:
            session: 数据库会话
            user_id: 用户ID
            folder_id: 文件夹ID
            new_name: 新名称
            updater: 更新者

        Returns:
            重命名后的 WorkspaceFolder，失败返回 None
        """
        try:
            folder = session.query(self.model).filter(
                self.model.folder_id == folder_id,
                self.model.user_id == user_id,
                self.model.deleted == 0,
            ).first()
            if not folder:
                return None

            old_path: str = folder.full_path
            parent_path = old_path.rsplit(folder.folder_name + "/", 1)[0]
            new_path = f"{parent_path}{new_name}/"

            dup = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.full_path == new_path,
                self.model.deleted == 0,
                self.model.folder_id != folder_id,
            ).first()
            if dup:
                raise ValueError(f"同级已存在同名文件夹: {new_path}")

            folder.folder_name = new_name
            folder.full_path = new_path
            folder.updater = updater

            descendants = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.full_path.like(f"{old_path}%"),
                self.model.folder_id != folder_id,
                self.model.deleted == 0,
            ).all()
            for d in descendants:
                d.full_path = new_path + d.full_path[len(old_path):]
                d.updater = updater

            session.commit()
            session.refresh(folder)
            logger.info(
                f"重命名文件夹: folder_id={folder_id}, "
                f"{old_path} -> {new_path}, 级联更新{len(descendants)}个后代"
            )
            return folder

        except ValueError:
            session.rollback()
            raise
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"重命名文件夹失败: {e}")
            return None

    def move(
        self,
        session: Session,
        user_id: str,
        folder_id: str,
        target_parent_folder_id: Optional[str],
        updater: str = "",
    ) -> Optional[WorkspaceFolder]:
        """
        移动文件夹到新的父文件夹，级联更新后代的 full_path 和 depth

        Args:
            session: 数据库会话
            user_id: 用户ID
            folder_id: 文件夹ID
            target_parent_folder_id: 目标父文件夹ID，None 表示移到根目录
            updater: 更新者

        Returns:
            移动后的 WorkspaceFolder，失败返回 None
        """
        try:
            folder = session.query(self.model).filter(
                self.model.folder_id == folder_id,
                self.model.user_id == user_id,
                self.model.deleted == 0,
            ).first()
            if not folder:
                return None

            old_path: str = folder.full_path
            old_depth: int = folder.depth

            new_parent_path = "/"
            new_depth = 0

            if target_parent_folder_id is not None:
                parent = session.query(self.model).filter(
                    self.model.folder_id == target_parent_folder_id,
                    self.model.user_id == user_id,
                    self.model.deleted == 0,
                ).first()
                if not parent:
                    raise ValueError("目标父文件夹不存在或无权限")

                if parent.full_path.startswith(old_path):
                    raise ValueError("不能将文件夹移动到自身的子目录下")

                new_parent_path = parent.full_path
                new_depth = parent.depth + 1

            new_path = f"{new_parent_path}{folder.folder_name}/"

            dup = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.full_path == new_path,
                self.model.deleted == 0,
                self.model.folder_id != folder_id,
            ).first()
            if dup:
                raise ValueError(f"目标位置已存在同名文件夹: {new_path}")

            depth_delta = new_depth - old_depth
            folder.parent_folder_id = target_parent_folder_id
            folder.full_path = new_path
            folder.depth = new_depth
            folder.updater = updater

            descendants = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.full_path.like(f"{old_path}%"),
                self.model.folder_id != folder_id,
                self.model.deleted == 0,
            ).all()
            for d in descendants:
                d.full_path = new_path + d.full_path[len(old_path):]
                d.depth = d.depth + depth_delta
                d.updater = updater

            session.commit()
            session.refresh(folder)
            logger.info(
                f"移动文件夹: folder_id={folder_id}, "
                f"{old_path} -> {new_path}, 级联更新{len(descendants)}个后代"
            )
            return folder

        except ValueError:
            session.rollback()
            raise
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"移动文件夹失败: {e}")
            return None


    # ==================== kb 维度（共享库支持） ====================
    #
    # 共享库（team/public）内文件夹由多成员创建，user_id 只是"创建者"，
    # 查询与管理一律按 knowledge_base_id 聚合（入口已过 KBAccess 判权）。
    # folder_id 为 uuid 全局唯一，单查不再叠加 user 条件。

    def get_by_folder_id(
        self,
        session: Session,
        folder_id: str,
    ) -> Optional[WorkspaceFolder]:
        """按 folder_id 单查（判权由调用方经其所属 kb 完成）"""
        try:
            return session.query(self.model).filter(
                self.model.folder_id == folder_id,
                self.model.deleted == 0,
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"根据folder_id单查失败: {e}")
            return None

    def get_by_kb(
        self,
        session: Session,
        knowledge_base_id: str,
    ) -> List[WorkspaceFolder]:
        """知识库下全部存活文件夹（平铺，按 depth + sort_order）"""
        try:
            return (
                session.query(self.model)
                .filter(
                    self.model.knowledge_base_id == knowledge_base_id,
                    self.model.deleted == 0,
                )
                .order_by(self.model.depth, self.model.sort_order)
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"kb维度查询文件夹失败: {e}")
            return []

    def get_children_in_kb(
        self,
        session: Session,
        knowledge_base_id: str,
        parent_folder_id: Optional[str],
    ) -> List[WorkspaceFolder]:
        """kb 内某父文件夹的直接子文件夹（parent None = 根）"""
        try:
            query = session.query(self.model).filter(
                self.model.knowledge_base_id == knowledge_base_id,
                self.model.deleted == 0,
            )
            if parent_folder_id is None:
                query = query.filter(self.model.parent_folder_id.is_(None))
            else:
                query = query.filter(
                    self.model.parent_folder_id == parent_folder_id
                )
            return query.order_by(self.model.sort_order).all()
        except SQLAlchemyError as e:
            logger.error(f"kb维度查询子文件夹失败: {e}")
            return []

    def get_by_full_path_in_kb(
        self,
        session: Session,
        knowledge_base_id: str,
        full_path: str,
    ) -> Optional[WorkspaceFolder]:
        """按 kb + full_path 查文件夹（创建去重检查用）"""
        try:
            return session.query(self.model).filter(
                self.model.knowledge_base_id == knowledge_base_id,
                self.model.full_path == full_path,
                self.model.deleted == 0,
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"kb维度根据full_path查询失败: {e}")
            return None

    def get_subtree_ids_in_kb(
        self,
        session: Session,
        folder_id: str,
        full_path_prefix: str,
    ) -> List[str]:
        """文件夹自身及所有后代 ID（kb 维度，不限创建者）"""
        descendant_rows = session.query(self.model.folder_id).filter(
            self.model.full_path.like(f"{full_path_prefix}%"),
            self.model.folder_id != folder_id,
            self.model.deleted == 0,
        ).all()
        return [folder_id] + [row[0] for row in descendant_rows]

    def rename_in_kb(
        self,
        session: Session,
        folder_id: str,
        new_name: str,
        updater: str = "",
    ) -> Optional[WorkspaceFolder]:
        """重命名文件夹并级联更新后代 full_path（kb 维度）"""
        try:
            folder = session.query(self.model).filter(
                self.model.folder_id == folder_id,
                self.model.deleted == 0,
            ).first()
            if not folder:
                return None

            old_path: str = folder.full_path
            kb_id = folder.knowledge_base_id
            parent_path = old_path.rsplit(folder.folder_name + "/", 1)[0]
            new_path = f"{parent_path}{new_name}/"

            dup = session.query(self.model).filter(
                self.model.knowledge_base_id == kb_id,
                self.model.full_path == new_path,
                self.model.deleted == 0,
                self.model.folder_id != folder_id,
            ).first()
            if dup:
                raise ValueError(f"同级已存在同名文件夹: {new_path}")

            folder.folder_name = new_name
            folder.full_path = new_path
            folder.updater = updater

            descendants = session.query(self.model).filter(
                self.model.knowledge_base_id == kb_id,
                self.model.full_path.like(f"{old_path}%"),
                self.model.folder_id != folder_id,
                self.model.deleted == 0,
            ).all()
            for d in descendants:
                d.full_path = new_path + d.full_path[len(old_path):]
                d.updater = updater

            session.commit()
            session.refresh(folder)
            logger.info(
                f"重命名文件夹(kb维度): folder_id={folder_id}, "
                f"{old_path} -> {new_path}, 级联更新{len(descendants)}个后代"
            )
            return folder

        except ValueError:
            session.rollback()
            raise
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"重命名文件夹失败(kb维度): {e}")
            return None

    def move_in_kb(
        self,
        session: Session,
        folder_id: str,
        target_parent_folder_id: Optional[str],
        updater: str = "",
    ) -> Optional[WorkspaceFolder]:
        """移动文件夹并级联更新后代 full_path/depth（kb 维度）"""
        try:
            folder = session.query(self.model).filter(
                self.model.folder_id == folder_id,
                self.model.deleted == 0,
            ).first()
            if not folder:
                return None

            old_path: str = folder.full_path
            old_depth: int = folder.depth
            kb_id = folder.knowledge_base_id

            new_parent_path = "/"
            new_depth = 0

            if target_parent_folder_id is not None:
                parent = session.query(self.model).filter(
                    self.model.folder_id == target_parent_folder_id,
                    self.model.deleted == 0,
                ).first()
                if not parent:
                    raise ValueError("目标父文件夹不存在或无权限")
                if parent.knowledge_base_id != kb_id:
                    raise ValueError("不允许跨知识库移动文件夹")
                if parent.full_path.startswith(old_path):
                    raise ValueError("不能将文件夹移动到自身的子目录下")
                new_parent_path = parent.full_path
                new_depth = parent.depth + 1

            new_path = f"{new_parent_path}{folder.folder_name}/"

            dup = session.query(self.model).filter(
                self.model.knowledge_base_id == kb_id,
                self.model.full_path == new_path,
                self.model.deleted == 0,
                self.model.folder_id != folder_id,
            ).first()
            if dup:
                raise ValueError(f"目标位置已存在同名文件夹: {new_path}")

            depth_delta = new_depth - old_depth
            folder.parent_folder_id = target_parent_folder_id
            folder.full_path = new_path
            folder.depth = new_depth
            folder.updater = updater

            descendants = session.query(self.model).filter(
                self.model.knowledge_base_id == kb_id,
                self.model.full_path.like(f"{old_path}%"),
                self.model.folder_id != folder_id,
                self.model.deleted == 0,
            ).all()
            for d in descendants:
                d.full_path = new_path + d.full_path[len(old_path):]
                d.depth = d.depth + depth_delta
                d.updater = updater

            session.commit()
            session.refresh(folder)
            logger.info(
                f"移动文件夹(kb维度): folder_id={folder_id}, "
                f"{old_path} -> {new_path}, 级联更新{len(descendants)}个后代"
            )
            return folder

        except ValueError:
            session.rollback()
            raise
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"移动文件夹失败(kb维度): {e}")
            return None


# 全局实例
workspace_folder_repo = WorkspaceFolderRepository()