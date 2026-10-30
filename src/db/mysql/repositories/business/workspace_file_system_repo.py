#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : workspace_file_system_repo.py
@Author  : caixiongjiang
@Date    : 2026/01/06
@Function: 
    WorkspaceFileSystem Repository
@Modify History:
    2026/02/16 - 适配新结构：新增按 folder_id、knowledge_base_id 查询方法
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import List, Optional
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from src.db.mysql.models.business.workspace_file_system import WorkspaceFileSystem
from src.db.mysql.repositories.base_repository import BaseRepository


class WorkspaceFileSystemRepository(BaseRepository[WorkspaceFileSystem]):
    """WorkspaceFileSystem Repository（联合主键表）"""
    
    def __init__(self):
        super().__init__(WorkspaceFileSystem)
    
    def get_by_user_and_file(
        self,
        session: Session,
        user_id: str,
        file_id: str
    ) -> Optional[WorkspaceFileSystem]:
        """
        根据联合主键查询
        
        Args:
            session: 数据库会话
            user_id: 用户ID
            file_id: 文件ID
        
        Returns:
            WorkspaceFileSystem 实例，未找到返回 None
        """
        try:
            result = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.file_id == file_id,
                self.model.deleted == 0
            ).first()
            
            if not result:
                logger.debug(
                    f"未找到WorkspaceFileSystem: user_id={user_id}, file_id={file_id}"
                )
            
            return result
        except SQLAlchemyError as e:
            logger.error(f"查询WorkspaceFileSystem失败: {e}")
            return None
    
    def get_by_user_id(
        self,
        session: Session,
        user_id: str
    ) -> List[WorkspaceFileSystem]:
        """
        根据 user_id 查询所有文件
        
        Args:
            session: 数据库会话
            user_id: 用户ID
        
        Returns:
            WorkspaceFileSystem 列表
        """
        try:
            results = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.deleted == 0
            ).all()
            
            logger.debug(
                f"查询到{len(results)}个WorkspaceFileSystem: user_id={user_id}"
            )
            return results
        except SQLAlchemyError as e:
            logger.error(f"根据user_id查询失败: {e}")
            return []
    
    def get_by_folder_id(
        self,
        session: Session,
        user_id: str,
        folder_id: Optional[str],
        knowledge_base_id: Optional[str] = None,
    ) -> List[WorkspaceFileSystem]:
        """
        根据文件夹ID查询该文件夹下的所有文件
        
        Args:
            session: 数据库会话
            user_id: 用户ID
            folder_id: 文件夹ID（None 表示根目录下的文件）
            knowledge_base_id: 可选，按知识库ID筛选（folder_id=None 时建议传入）
        
        Returns:
            WorkspaceFileSystem 列表
        """
        try:
            query = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.deleted == 0
            )
            
            if folder_id is None:
                query = query.filter(self.model.folder_id.is_(None))
            else:
                query = query.filter(self.model.folder_id == folder_id)

            if knowledge_base_id:
                query = query.filter(
                    self.model.knowledge_base_id == knowledge_base_id
                )
            
            results = query.all()
            
            logger.debug(
                f"查询到{len(results)}个WorkspaceFileSystem: "
                f"user_id={user_id}, folder_id={folder_id}"
            )
            return results
        except SQLAlchemyError as e:
            logger.error(f"根据folder_id查询失败: {e}")
            return []
    
    def get_by_folder_ids(
        self,
        session: Session,
        user_id: str,
        folder_ids: List[str],
        knowledge_base_id: Optional[str] = None,
    ) -> List[WorkspaceFileSystem]:
        """根据多个文件夹 ID 批量查询文件（folder scope 解析的核心查询）

        与 ``get_by_folder_id`` 区别：本方法专为 ChatService 解析
        ``scope_document_ids`` 设计——给定一组 folder_id（含 + 子文件夹后代），
        一次性查出所有文件，避免 N+1 查询。

        Args:
            session: 数据库会话
            user_id: 用户ID
            folder_ids: 文件夹 ID 列表；空列表直接返回 ``[]``
            knowledge_base_id: 可选，按知识库 ID 进一步筛选

        Returns:
            WorkspaceFileSystem 列表（已去重；deleted=0）
        """
        if not folder_ids:
            return []
        try:
            query = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.folder_id.in_(folder_ids),
                self.model.deleted == 0,
            )
            if knowledge_base_id:
                query = query.filter(
                    self.model.knowledge_base_id == knowledge_base_id
                )
            results = query.all()
            logger.debug(
                f"批量按 folder_ids 查询文件: user_id={user_id}, "
                f"folders={len(folder_ids)}, files={len(results)}"
            )
            return results
        except SQLAlchemyError as e:
            logger.error(f"批量按 folder_ids 查询失败: {e}")
            return []

    def get_by_knowledge_base_id(
        self,
        session: Session,
        user_id: str,
        knowledge_base_id: str
    ) -> List[WorkspaceFileSystem]:
        """
        根据知识库ID查询所有文件
        
        Args:
            session: 数据库会话
            user_id: 用户ID
            knowledge_base_id: 知识库ID
        
        Returns:
            WorkspaceFileSystem 列表
        """
        try:
            results = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.knowledge_base_id == knowledge_base_id,
                self.model.deleted == 0
            ).all()
            
            logger.debug(
                f"查询到{len(results)}个WorkspaceFileSystem: "
                f"user_id={user_id}, knowledge_base_id={knowledge_base_id}"
            )
            return results
        except SQLAlchemyError as e:
            logger.error(f"根据knowledge_base_id查询失败: {e}")
            return []
    
    def search_by_name(
        self,
        session: Session,
        user_id: str,
        knowledge_base_id: str,
        q: str,
        limit: int = 20,
    ) -> List[WorkspaceFileSystem]:
        """按文件名模糊搜索某知识库内的文件（供前端 @ 文件选择器使用）。

        Args:
            session: 数据库会话
            user_id: 用户ID
            knowledge_base_id: 知识库ID（必填，限定搜索范围）
            q: 文件名关键字（空串返回该 KB 下前 limit 个文件）
            limit: 返回条数上限（封顶 50）

        Returns:
            WorkspaceFileSystem 列表（deleted=0），按文件名排序
        """
        try:
            capped = max(1, min(limit, 50))
            query = session.query(self.model).filter(
                self.model.user_id == user_id,
                self.model.knowledge_base_id == knowledge_base_id,
                self.model.deleted == 0,
            )
            keyword = (q or "").strip()
            if keyword:
                # 转义 LIKE 通配符，避免用户输入 % / _ 造成意外匹配
                escaped = (
                    keyword.replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                query = query.filter(
                    self.model.file_name.like(f"%{escaped}%", escape="\\")
                )
            results = (
                query.order_by(self.model.file_name)
                .limit(capped)
                .all()
            )
            logger.debug(
                f"按文件名搜索: user_id={user_id}, kb={knowledge_base_id}, "
                f"q={keyword!r}, hits={len(results)}"
            )
            return results
        except SQLAlchemyError as e:
            logger.error(f"按文件名搜索失败: {e}")
            return []

    def get_by_sha256(
        self,
        session: Session,
        file_sha256: bytes
    ) -> Optional[WorkspaceFileSystem]:
        """
        根据 SHA256 哈希值查询已有文件（用于内容去重，复用 document_id）
        
        Args:
            session: 数据库会话
            file_sha256: 文件 SHA256 哈希值（32字节二进制）
        
        Returns:
            第一个匹配的 WorkspaceFileSystem 实例，未找到返回 None
        """
        try:
            result = session.query(self.model).filter(
                self.model.file_sha256 == file_sha256,
                self.model.document_id.isnot(None),
                self.model.deleted == 0
            ).first()
            return result
        except SQLAlchemyError as e:
            logger.error(f"根据SHA256查询失败: {e}")
            return None
    
    def get_by_document_id(
        self,
        session: Session,
        document_id: str
    ) -> List[WorkspaceFileSystem]:
        """
        根据 document_id 查询所有文件
        
        Args:
            session: 数据库会话
            document_id: Document ID
        
        Returns:
            WorkspaceFileSystem 列表
        """
        try:
            results = session.query(self.model).filter(
                self.model.document_id == document_id,
                self.model.deleted == 0
            ).all()
            
            logger.debug(
                f"查询到{len(results)}个WorkspaceFileSystem: document_id={document_id}"
            )
            return results
        except SQLAlchemyError as e:
            logger.error(f"根据document_id查询失败: {e}")
            return []
    
    def delete_by_user_and_file(
        self,
        session: Session,
        user_id: str,
        file_id: str,
        updater: str = ""
    ) -> bool:
        """
        根据联合主键软删除
        
        Args:
            session: 数据库会话
            user_id: 用户ID
            file_id: 文件ID
            updater: 更新者
        
        Returns:
            删除成功返回 True，否则返回 False
        """
        try:
            obj = self.get_by_user_and_file(session, user_id, file_id)
            if obj:
                obj.deleted = 1
                obj.updater = updater
                session.commit()
                logger.debug(
                    f"成功删除WorkspaceFileSystem: user_id={user_id}, file_id={file_id}"
                )
                return True
            
            logger.debug(
                f"未找到要删除的WorkspaceFileSystem: user_id={user_id}, file_id={file_id}"
            )
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"删除WorkspaceFileSystem失败: {e}")
            return False
    
    # ==================== 墓碑（待异步清理） ====================
    #
    # deleted=1 表示"已删除，关联数据待 CleanupWorker 清理"，是一个以秒计的中间态：
    # 用户看不到、检索排除、清理完成后整行物理删除。没有恢复入口。

    def cascade_mark_deleted_by_folder_ids(
        self,
        session: Session,
        user_id: str,
        folder_ids: List[str],
        updater: str = "",
    ) -> List[WorkspaceFileSystem]:
        """
        按文件夹ID列表级联标记删除文件（deleted=1）。
        **不会 commit**，由调用方统一提交事务。

        Args:
            session: 数据库会话
            user_id: 用户ID
            folder_ids: 需要级联删除文件的文件夹ID列表
            updater: 更新者

        Returns:
            被标记的文件对象列表（调用方据此投递清理任务）
        """
        if not folder_ids:
            return []

        files = session.query(self.model).filter(
            self.model.user_id == user_id,
            self.model.folder_id.in_(folder_ids),
            self.model.deleted == 0,
        ).all()
        for f in files:
            f.deleted = 1
            f.updater = updater

        logger.debug(f"级联标记删除{len(files)}个文件: folder_ids count={len(folder_ids)}")
        return files

    def get_tombstoned_document_ids(
        self,
        session: Session,
        user_id: str,
        knowledge_base_id: Optional[str] = None,
    ) -> List[str]:
        """
        获取待清理文件的 document_id 列表，供检索侧排除。

        清理完成前，这些文件的向量仍在 Milvus 里；Milvus 默认 Bounded 一致性，
        即便已经删除并 flush，短窗口内搜索仍可能命中，所以必须显式排除。

        Args:
            session: 数据库会话
            user_id: 用户ID
            knowledge_base_id: 可选，按知识库缩小范围

        Returns:
            去重后的 document_id 列表
        """
        try:
            query = session.query(self.model.document_id).filter(
                self.model.user_id == user_id,
                self.model.deleted != 0,
                self.model.document_id.isnot(None),
            )
            if knowledge_base_id:
                query = query.filter(
                    self.model.knowledge_base_id == knowledge_base_id
                )
            return list({row[0] for row in query.all() if row[0]})
        except SQLAlchemyError as e:
            logger.error(f"查询待清理 document_id 失败: {e}")
            return []

    def hard_delete_by_user_and_file(
        self,
        session: Session,
        user_id: str,
        file_id: str,
    ) -> bool:
        """
        物理删除文件行（不区分 deleted 状态），清理流程的最后一步。
        **不会 commit**。

        Returns:
            是否找到并删除
        """
        count = session.query(self.model).filter(
            self.model.user_id == user_id,
            self.model.file_id == file_id,
        ).delete(synchronize_session='fetch')
        return count > 0

    # ==================== kb 维度（共享库支持） ====================
    #
    # 共享库（team/public）内文件由多成员上传，user_id 只是"上传者"，
    # 查询与管理一律按 knowledge_base_id 聚合（入口已过 KBAccess 判权）。

    def get_by_file_id(
        self,
        session: Session,
        file_id: str,
    ) -> Optional[WorkspaceFileSystem]:
        """按 file_id 单查（file_id 为 uuid 全局唯一；判权由调用方经其所属 kb 完成）"""
        try:
            return session.query(self.model).filter(
                self.model.file_id == file_id,
                self.model.deleted == 0,
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"根据file_id查询失败: {e}")
            return None

    def get_by_folder_id_in_kb(
        self,
        session: Session,
        knowledge_base_id: str,
        folder_id: Optional[str],
    ) -> List[WorkspaceFileSystem]:
        """知识库内某文件夹下的文件（folder_id=None 表示根目录）"""
        try:
            query = session.query(self.model).filter(
                self.model.knowledge_base_id == knowledge_base_id,
                self.model.deleted == 0,
            )
            if folder_id is None:
                query = query.filter(self.model.folder_id.is_(None))
            else:
                query = query.filter(self.model.folder_id == folder_id)
            # 手动排序在前（0..n-1），未排序的按上传时间沉底
            return query.order_by(
                self.model.sort_order.is_(None),
                self.model.sort_order,
                self.model.create_time,
            ).all()
        except SQLAlchemyError as e:
            logger.error(f"kb维度查询文件夹文件失败: {e}")
            return []

    def reorder_files_in_kb(
        self,
        session: Session,
        knowledge_base_id: str,
        folder_id: Optional[str],
        ordered_file_ids: List[str],
    ) -> int:
        """按给定顺序固化文件夹内文件的 sort_order（0..n-1）。

        未列出的存活行不动（仍 NULL 沉底）。不提交，由调用方事务控制。
        返回更新的行数。
        """
        updated = 0
        try:
            for index, file_id in enumerate(ordered_file_ids):
                result = (
                    session.query(self.model)
                    .filter(
                        self.model.knowledge_base_id == knowledge_base_id,
                        self.model.file_id == file_id,
                        self.model.deleted == 0,
                    )
                    .update({"sort_order": index}, synchronize_session=False)
                )
                updated += result
            return updated
        except SQLAlchemyError as e:
            logger.error(f"kb维度重排文件夹文件失败: {e}")
            raise

    def search_by_name_in_kb(
        self,
        session: Session,
        knowledge_base_id: str,
        q: str,
        limit: int = 20,
    ) -> List[WorkspaceFileSystem]:
        """按文件名模糊搜索知识库内文件（kb 维度，供 @ 文件选择器）"""
        try:
            capped = max(1, min(limit, 50))
            query = session.query(self.model).filter(
                self.model.knowledge_base_id == knowledge_base_id,
                self.model.deleted == 0,
            )
            keyword = (q or "").strip()
            if keyword:
                escaped = (
                    keyword.replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                query = query.filter(
                    self.model.file_name.like(f"%{escaped}%", escape="\\")
                )
            return query.order_by(self.model.file_name).limit(capped).all()
        except SQLAlchemyError as e:
            logger.error(f"kb维度按文件名搜索失败: {e}")
            return []

    def cascade_mark_deleted_by_folder_ids_in_kb(
        self,
        session: Session,
        folder_ids: List[str],
        updater: str = "",
    ) -> List[WorkspaceFileSystem]:
        """按文件夹ID列表级联标记删除文件（不限上传者）。

        **不会 commit**，由调用方统一提交事务。
        """
        if not folder_ids:
            return []
        files = session.query(self.model).filter(
            self.model.folder_id.in_(folder_ids),
            self.model.deleted == 0,
        ).all()
        for f in files:
            f.deleted = 1
            f.updater = updater
        logger.debug(
            f"级联标记删除{len(files)}个文件(kb维度): "
            f"folder_ids count={len(folder_ids)}"
        )
        return files

    def get_tombstoned_document_ids_by_kb(
        self,
        session: Session,
        knowledge_base_id: str,
    ) -> List[str]:
        """知识库维度获取待清理 document_id（共享库：任一成员删除即对全员生效）"""
        try:
            rows = session.query(self.model.document_id).filter(
                self.model.knowledge_base_id == knowledge_base_id,
                self.model.deleted != 0,
                self.model.document_id.isnot(None),
            ).all()
            return list({row[0] for row in rows if row[0]})
        except SQLAlchemyError as e:
            logger.error(f"kb维度查询待清理 document_id 失败: {e}")
            return []


# 全局实例
workspace_file_system_repo = WorkspaceFileSystemRepository()
