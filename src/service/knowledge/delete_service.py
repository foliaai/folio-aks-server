#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : delete_service.py
@Author  : caixiongjiang
@Date    : 2026/01/21 10:00
@Function: 
    Knowledge 删除服务
    删除拆成两段：受理（同步标记墓碑 + 投递清理任务）与清理（异步跨数据库级联删除）
@Modify History:
    2026/03/09 - 实现完整删除服务：软删除、永久删除、跨数据库级联删除
    2026/09/01 - 下线回收站，改为墓碑 + Kafka 异步清理
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.db.mysql.repositories.base.chunk_meta_info_repo import chunk_meta_info_repo
from src.db.mysql.repositories.base.chunk_section_document_repo import chunk_section_document_repo
from src.db.mysql.repositories.base.element_meta_info_repo import element_meta_info_repo
from src.db.mysql.repositories.base.section_document_repo import section_document_repo
from src.db.mysql.repositories.base.section_meta_info_repo import section_meta_info_repo
from src.db.mysql.repositories.extract.section_atomic_qa_repo import section_atomic_qa_repo
from src.db.mysql.repositories.extract.chunk_summary_repo import chunk_summary_repo
from src.db.mysql.repositories.extract.document_summary_repo import document_summary_repo
from src.db.mysql.repositories.extract.section_summary_repo import section_summary_repo
from src.db.mysql.repositories.business.workspace_file_system_repo import workspace_file_system_repo

from src.db.mongodb.repositories import (
    document_data_repository,
    section_data_repository,
    chunk_data_repository,
    element_data_repository,
)

from src.db.milvus.repositories.base import ChunkRepository, SectionRepository
from src.db.milvus.repositories.enhanced import EnhancedChunkRepository
from src.db.milvus.repositories.extract import (
    AtomicQARepository,
    FileSummaryRepository,
    SectionSummaryRepository,
)
from src.db.milvus.repositories.kg import SPORepository, TagRepository

from src.db.storage.manager import StorageManager

from src.db.kafka.producer import KafkaProducer
from src.db.kafka.topics import KafkaTopics
from src.types.messages.cleanup import CleanupMessage


@dataclass
class DeleteResult:
    """删除操作的结果统计"""
    mysql_deleted: int = 0
    mongodb_deleted: int = 0
    milvus_deleted: int = 0
    storage_deleted: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def total_deleted(self) -> int:
        return self.mysql_deleted + self.mongodb_deleted + self.milvus_deleted + self.storage_deleted

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


@dataclass
class CleanupTicket:
    """一次删除受理产生的清理任务

    标记删除时就把 worker 需要的字段全部带上，这样 worker 不必再查 MySQL，
    墓碑行被清掉之后消息重投也仍然能定位到要清理的数据。
    """
    user_id: str
    file_id: str
    document_id: Optional[str] = None
    storage_path: Optional[str] = None
    knowledge_base_id: Optional[str] = None


class KnowledgeDeleteService:
    """Knowledge 删除服务

    删除分两段：
    - **受理**（同步，O(1)）：MySQL 标记 deleted=1 写下墓碑，立即返回。
      墓碑对用户不可见，检索侧按 document_id 排除，所以删除对问答是即时生效的。
    - **清理**（异步，CleanupWorker）：按 document_id 级联清掉向量 / 文档 / 元数据 /
      对象存储，最后物理删除墓碑行。

    之所以不能把清理放进 HTTP 请求：一次清理要打 8 个 Milvus collection
    （每个都 load + delete + flush）、4 个 Mongo 集合、9 张 MySQL 表和一次对象存储删除，
    全程串行，而 API 只跑单个 uvicorn worker，会把整个事件循环占住。

    数据库覆盖范围：
    - MySQL: 元数据表（chunk_meta_info, section_document, element_meta_info 等）
    - MongoDB: 数据表（document_data, section_data, chunk_data, element_data）
    - Milvus: 向量表（chunk, section, enhanced_chunk, summary, atomic_qa, tag, spo）
    - Storage: 对象存储（原始文件、解析产物）
    """

    # ==================== 受理（同步） ====================

    @staticmethod
    def _require_file_write(session: Session, user_id: str, file_obj) -> None:
        """删除/操作文件的判权（经其所属知识库；无 kb 归属的脏数据按属主校验）

        Raises:
            PermissionError: 无权操作
        """
        from src.db.mysql.repositories.business.knowledge_base_repo import (
            knowledge_base_repo,
        )
        from src.service.knowledge.kb_access import resolve_kb_access

        kb_id = file_obj.knowledge_base_id
        if kb_id:
            kb = knowledge_base_repo.get_by_id(session, kb_id)
            if kb is None or not resolve_kb_access(session, user_id, kb).can_write:
                raise PermissionError(f"无权操作该文件: file_id={file_obj.file_id}")
        elif file_obj.user_id != user_id:
            raise PermissionError(f"无权操作该文件: file_id={file_obj.file_id}")

    def mark_file_deleted(
        self,
        session: Session,
        user_id: str,
        file_id: str,
    ) -> Optional[CleanupTicket]:
        """标记删除单个文件，返回待投递的清理任务

        **不会 commit**，由调用方统一提交事务。

        共享库（team/public）内团队成员可删除其他成员上传的文件；
        清理任务的 user_id 取文件行属主（联合主键定位 + CleanupWorker
        物理删除依据），操作者只记入 updater / 日志。

        Args:
            session: MySQL 数据库会话
            user_id: 操作者用户ID
            file_id: 文件ID（uuid 全局唯一）

        Returns:
            CleanupTicket；文件不存在或已被标记时返回 None

        Raises:
            PermissionError: 操作者对文件所属知识库无写权限
        """
        file_obj = workspace_file_system_repo.get_by_file_id(session, file_id)
        if not file_obj:
            logger.warning(f"文件不存在或已删除: file_id={file_id}")
            return None

        self._require_file_write(session, user_id, file_obj)

        file_obj.deleted = 1
        file_obj.updater = user_id

        return CleanupTicket(
            user_id=file_obj.user_id,
            file_id=file_id,
            document_id=file_obj.document_id,
            storage_path=file_obj.storage_path,
            knowledge_base_id=file_obj.knowledge_base_id,
        )

    def mark_files_deleted(
        self,
        session: Session,
        user_id: str,
        file_ids: List[str],
    ) -> List[CleanupTicket]:
        """批量标记删除文件

        **不会 commit**，由调用方统一提交事务。

        Returns:
            成功标记的清理任务列表（不存在的文件被跳过）
        """
        tickets = []
        for file_id in file_ids:
            ticket = self.mark_file_deleted(session, user_id, file_id)
            if ticket:
                tickets.append(ticket)
        return tickets

    @staticmethod
    async def publish_cleanup(
        producer: KafkaProducer,
        tickets: List[CleanupTicket],
    ) -> int:
        """投递清理任务到 Kafka

        投递失败不抛异常：墓碑已经落库，文件对用户和检索都已消失，
        漏投只是让清理延后，由兜底扫描重新捡起来，不该让删除请求整体失败。

        Returns:
            成功投递的任务数
        """
        sent = 0
        for ticket in tickets:
            try:
                await producer.send_and_flush(
                    topic=KafkaTopics.CLEANUP_START,
                    message=CleanupMessage(
                        user_id=ticket.user_id,
                        file_id=ticket.file_id,
                        document_id=ticket.document_id,
                        storage_path=ticket.storage_path,
                        knowledge_base_id=ticket.knowledge_base_id,
                    ),
                )
                sent += 1
            except Exception as e:
                logger.error(
                    f"投递清理任务失败（墓碑已生效，等待兜底重投）: "
                    f"file_id={ticket.file_id}, error={e}"
                )
        return sent

    # ==================== 清理（异步 Worker 调用） ====================

    async def purge_file(
        self,
        session: Session,
        user_id: str,
        file_id: str,
        document_id: Optional[str] = None,
        storage_path: Optional[str] = None,
        storage_manager: Optional[StorageManager] = None,
    ) -> DeleteResult:
        """清理文件的所有关联数据并物理删除墓碑行

        流程：
        1. 检查是否有其他文件引用同一 document_id（内容去重）
        2. 若无其他引用，级联删除该 document 在所有数据库中的数据
        3. 删除对象存储中的文件
        4. 物理删除文件记录

        幂等：墓碑行可能已被上一次投递清理掉，此时 document_id / storage_path
        由消息带入，仍然可以完成剩余清理。

        Args:
            session: MySQL 数据库会话
            user_id: 用户ID
            file_id: 文件ID
            document_id: 文档ID，缺省时从墓碑行读取
            storage_path: 对象存储路径，缺省时从墓碑行读取
            storage_manager: 对象存储管理器（可选）

        Returns:
            DeleteResult 包含各数据库删除统计和错误信息
        """
        result = DeleteResult()

        file_obj = session.query(
            workspace_file_system_repo.model
        ).filter(
            workspace_file_system_repo.model.user_id == user_id,
            workspace_file_system_repo.model.file_id == file_id,
        ).first()

        if file_obj:
            document_id = document_id or file_obj.document_id
            storage_path = storage_path or file_obj.storage_path
        elif not document_id and not storage_path:
            logger.debug(f"文件已清理完成，跳过: file_id={file_id}")
            return result

        if document_id:
            # 引用计数只看存活文件：同内容的另一份也在待清理队列里时，
            # 谁先跑到这里谁清，共享数据不会被漏掉。
            other_refs = workspace_file_system_repo.get_by_document_id(session, document_id)
            other_refs = [
                f for f in other_refs
                if not (f.user_id == user_id and f.file_id == file_id)
            ]

            if not other_refs:
                cascade_result = await self._cascade_delete_document(
                    session, document_id, user_id
                )
                result.mysql_deleted += cascade_result.mysql_deleted
                result.mongodb_deleted += cascade_result.mongodb_deleted
                result.milvus_deleted += cascade_result.milvus_deleted
                result.errors.extend(cascade_result.errors)
            else:
                logger.debug(
                    f"document_id={document_id} 仍被 {len(other_refs)} 个文件引用，跳过级联删除"
                )

        if storage_manager and storage_path:
            try:
                deleted = await storage_manager.delete_file(storage_path)
                if deleted:
                    result.storage_deleted += 1
                    logger.debug(f"已删除存储文件: {storage_path}")
            except Exception as e:
                error_msg = f"删除存储文件失败: {storage_path}, {e}"
                result.errors.append(error_msg)
                logger.error(error_msg)

        if result.errors:
            # 有清理失败项时保留墓碑行：它既是检索排除的依据，也是兜底扫描重投的线索。
            # 一旦提前删掉，残留的向量/对象就再也没人认领了。
            logger.warning(
                f"清理未全部完成，保留墓碑等待重投: file_id={file_id}, "
                f"errors={result.errors}"
            )
            return result

        try:
            if workspace_file_system_repo.hard_delete_by_user_and_file(
                session, user_id, file_id
            ):
                result.mysql_deleted += 1
        except SQLAlchemyError as e:
            error_msg = f"物理删除文件记录失败: {e}"
            result.errors.append(error_msg)
            logger.error(error_msg)

        logger.info(
            f"清理文件完成: file_id={file_id}, "
            f"mysql={result.mysql_deleted}, mongodb={result.mongodb_deleted}, "
            f"milvus={result.milvus_deleted}, storage={result.storage_deleted}"
        )
        return result

    async def _cascade_delete_document(
        self,
        session: Session,
        document_id: str,
        updater: str = "",
    ) -> DeleteResult:
        """级联删除文档的所有关联数据（跨所有数据库）

        删除顺序：Milvus（向量） → MongoDB（数据） → MySQL（元数据）

        Args:
            session: MySQL 数据库会话
            document_id: 文档ID
            updater: 操作者ID

        Returns:
            DeleteResult 删除统计
        """
        result = DeleteResult()

        sections = section_document_repo.get_by_document_id(session, document_id)
        section_ids = [s.section_id for s in sections]

        chunks = chunk_section_document_repo.get_by_document_id(session, document_id)
        chunk_ids = [c.chunk_id for c in chunks]

        elements = element_meta_info_repo.get_by_document_id(session, document_id)
        element_ids = [e.element_id for e in elements]

        logger.debug(
            f"级联删除 document_id={document_id}: "
            f"sections={len(section_ids)}, chunks={len(chunk_ids)}, elements={len(element_ids)}"
        )

        milvus_count = self._delete_milvus_data(document_id, result)
        result.milvus_deleted += milvus_count

        mongodb_count = await self._delete_mongodb_data(
            document_id, section_ids, chunk_ids, element_ids, updater, result
        )
        result.mongodb_deleted += mongodb_count

        mysql_count = self._delete_mysql_metadata(
            session, document_id, section_ids, chunk_ids, element_ids, updater, result
        )
        result.mysql_deleted += mysql_count

        return result

    def _delete_milvus_data(self, document_id: str, result: DeleteResult) -> int:
        """删除 Milvus 中的向量数据（硬删除）

        Args:
            document_id: 文档ID
            result: 用于记录错误的 DeleteResult

        Returns:
            成功删除的 collection 数量
        """
        deleted_count = 0
        milvus_deletions = [
            ("chunk", lambda: ChunkRepository().delete_by_document(document_id)),
            ("section", lambda: SectionRepository().delete_by_document(document_id)),
            ("enhanced_chunk", lambda: EnhancedChunkRepository().delete_by_document(document_id)),
            ("file_summary", lambda: FileSummaryRepository().delete_by_document(document_id)),
            ("section_summary", lambda: SectionSummaryRepository().delete_by_document(document_id)),
            ("atomic_qa", lambda: AtomicQARepository().delete_by_document(document_id)),
            ("tag", lambda: TagRepository().delete_by_document(document_id)),
            ("spo", lambda: SPORepository().delete_by_document(document_id)),
        ]

        for name, delete_fn in milvus_deletions:
            try:
                delete_fn()
                deleted_count += 1
                logger.debug(f"Milvus {name} 删除成功: document_id={document_id}")
            except Exception as e:
                error_msg = f"Milvus {name} 删除失败: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg)

        return deleted_count

    async def _delete_mongodb_data(
        self,
        document_id: str,
        section_ids: List[str],
        chunk_ids: List[str],
        element_ids: List[str],
        updater: str,
        result: DeleteResult,
    ) -> int:
        """删除 MongoDB 中的文档数据（软删除）

        Args:
            document_id: 文档ID
            section_ids: 章节ID列表
            chunk_ids: 分块ID列表
            element_ids: 元素ID列表
            updater: 操作者
            result: 用于记录错误的 DeleteResult

        Returns:
            成功删除的记录数量
        """
        deleted_count = 0

        try:
            if await document_data_repository.delete(document_id, updater=updater):
                deleted_count += 1
        except Exception as e:
            result.errors.append(f"MongoDB document_data 删除失败: {e}")
            logger.error(f"MongoDB document_data 删除失败: {e}")

        if section_ids:
            try:
                count = await section_data_repository.bulk_delete_by_ids(section_ids, updater=updater)
                deleted_count += count
            except Exception as e:
                result.errors.append(f"MongoDB section_data 删除失败: {e}")
                logger.error(f"MongoDB section_data 删除失败: {e}")

        if chunk_ids:
            try:
                count = await chunk_data_repository.bulk_delete_by_ids(chunk_ids, updater=updater)
                deleted_count += count
            except Exception as e:
                result.errors.append(f"MongoDB chunk_data 删除失败: {e}")
                logger.error(f"MongoDB chunk_data 删除失败: {e}")

        if element_ids:
            try:
                count = await element_data_repository.delete_elements_by_ids(element_ids, updater=updater)
                deleted_count += count
            except Exception as e:
                result.errors.append(f"MongoDB element_data 删除失败: {e}")
                logger.error(f"MongoDB element_data 删除失败: {e}")

        return deleted_count

    def _delete_mysql_metadata(
        self,
        session: Session,
        document_id: str,
        section_ids: List[str],
        chunk_ids: List[str],
        element_ids: List[str],
        updater: str,
        result: DeleteResult,
    ) -> int:
        """删除 MySQL 中的元数据（软删除）

        Args:
            session: 数据库会话
            document_id: 文档ID
            section_ids: 章节ID列表
            chunk_ids: 分块ID列表
            element_ids: 元素ID列表
            updater: 操作者
            result: 用于记录错误的 DeleteResult

        Returns:
            成功执行的软删除操作数量
        """
        deleted_count = 0

        mysql_operations = [
            (
                "chunk_section_document",
                lambda: chunk_section_document_repo.bulk_delete_by_ids(session, chunk_ids, updater=updater),
                bool(chunk_ids),
            ),
            (
                "chunk_meta_info",
                lambda: chunk_meta_info_repo.bulk_delete_by_ids(session, chunk_ids, updater=updater),
                bool(chunk_ids),
            ),
            (
                "section_document",
                lambda: section_document_repo.bulk_delete_by_ids(session, section_ids, updater=updater),
                bool(section_ids),
            ),
            (
                "section_meta_info",
                lambda: section_meta_info_repo.bulk_delete_by_ids(session, section_ids, updater=updater),
                bool(section_ids),
            ),
            (
                "element_meta_info",
                lambda: element_meta_info_repo.bulk_delete_by_ids(session, element_ids, updater=updater),
                bool(element_ids),
            ),
            (
                "section_atomic_qa",
                lambda: section_atomic_qa_repo.delete_by_document_id(session, document_id, updater=updater),
                True,
            ),
            (
                "section_summary",
                lambda: section_summary_repo.bulk_delete_by_ids(session, section_ids, updater=updater),
                bool(section_ids),
            ),
            (
                "chunk_summary",
                lambda: chunk_summary_repo.bulk_delete_by_ids(session, chunk_ids, updater=updater),
                bool(chunk_ids),
            ),
            (
                "document_summary",
                lambda: document_summary_repo.delete(session, document_id, updater=updater),
                True,
            ),
        ]

        for name, delete_fn, should_execute in mysql_operations:
            if not should_execute:
                continue
            try:
                delete_fn()
                deleted_count += 1
                logger.debug(f"MySQL {name} 软删除成功: document_id={document_id}")
            except SQLAlchemyError as e:
                error_msg = f"MySQL {name} 软删除失败: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg)

        return deleted_count


knowledge_delete_service = KnowledgeDeleteService()
