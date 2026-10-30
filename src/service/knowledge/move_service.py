#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : move_service.py
@Author  : caixiongjiang
@Date    : 2026/03/18
@Function: 
    Knowledge 文件移动服务
    提供文件移动相关的业务逻辑：单文件移动、批量移动、目标文件夹校验、知识库一致性校验
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.db.mysql.models.business.workspace_folder import WorkspaceFolder
from src.db.mysql.repositories.business.workspace_file_system_repo import (
    workspace_file_system_repo,
)


@dataclass
class SkippedFile:
    """被跳过的文件信息"""
    file_id: str
    reason: str


@dataclass
class MoveResult:
    """文件移动操作的结果"""
    moved_count: int = 0
    total_requested: int = 0
    skipped_files: List[SkippedFile] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return self.moved_count == self.total_requested


class KnowledgeMoveService:
    """Knowledge 文件移动服务

    提供文件级别的移动能力，包含：
    - 目标文件夹存在性与权限校验
    - 同一知识库内移动校验
    - 单文件移动与批量移动
    """

    def validate_target_folder(
        self,
        session: Session,
        user_id: str,
        target_folder_id: str | None,
        expected_knowledge_base_id: str | None = None,
    ) -> Optional[WorkspaceFolder]:
        """校验目标文件夹，返回文件夹对象

        校验：文件夹存在（kb 维度）且与 expected_knowledge_base_id 同库。
        可见性/写权限由文件的所属知识库 KBAccess 把关（见 _require_file_write）。

        Raises:
            ValueError: 文件夹不存在或跨知识库
        """
        if target_folder_id is None:
            return None

        folder = session.query(WorkspaceFolder).filter(
            WorkspaceFolder.folder_id == target_folder_id,
            WorkspaceFolder.deleted == 0,
        ).first()

        if not folder:
            raise ValueError("目标文件夹不存在或无权限")

        if (
            expected_knowledge_base_id is not None
            and folder.knowledge_base_id != expected_knowledge_base_id
        ):
            raise ValueError(
                f"不允许跨知识库移动文件: "
                f"文件属于知识库 {expected_knowledge_base_id}, "
                f"目标文件夹属于知识库 {folder.knowledge_base_id}"
            )

        return folder

    @staticmethod
    def _require_file_write(session: Session, user_id: str, file_obj) -> None:
        """移动文件的判权（经其所属知识库；无 kb 归属的脏数据按属主校验）

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
                raise PermissionError(
                    f"无权操作该文件: file_id={file_obj.file_id}"
                )
        elif file_obj.user_id != user_id:
            raise PermissionError(
                f"无权操作该文件: file_id={file_obj.file_id}"
            )

    def move_file(
        self,
        session: Session,
        user_id: str,
        file_id: str,
        target_folder_id: str | None,
        before_file_id: str | None = None,
    ) -> str:
        """移动单个文件到目标文件夹（可选落位），幂等。

        file_id 为 uuid 全局唯一；判权经其所属知识库（共享库团队成员
        可移动其他成员上传的文件）。同文件夹移动=排序（不再报错），
        位置不变时重排结果相同、无副作用。

        Raises:
            FileNotFoundError: 文件不存在
            PermissionError: 无权操作该文件
            ValueError: 目标文件夹无效、跨知识库、或落位文件无效
            SQLAlchemyError: 数据库操作失败
        """
        file_record = workspace_file_system_repo.get_by_file_id(
            session, file_id
        )
        if not file_record:
            raise FileNotFoundError(f"文件不存在或无权限: file_id={file_id}")

        self._require_file_write(session, user_id, file_record)

        same_folder = file_record.folder_id == target_folder_id
        if not same_folder:
            self.validate_target_folder(
                session, user_id, target_folder_id,
                expected_knowledge_base_id=file_record.knowledge_base_id,
            )

        if before_file_id is not None:
            before_file = workspace_file_system_repo.get_by_file_id(
                session, before_file_id
            )
            if (
                not before_file
                or before_file.file_id == file_id
                or before_file.knowledge_base_id
                != file_record.knowledge_base_id
                or before_file.folder_id != target_folder_id
            ):
                raise ValueError("插入位置文件无效")

        kb_id = file_record.knowledge_base_id
        if not same_folder:
            file_record.folder_id = target_folder_id
            file_record.updater = user_id

        # 落位：目标文件夹当前序 → 移除自己 → before 前插入/末尾 → 全量固化
        ordered_ids = [
            f.file_id
            for f in workspace_file_system_repo.get_by_folder_id_in_kb(
                session, kb_id, target_folder_id
            )
        ]
        if file_id in ordered_ids:
            ordered_ids.remove(file_id)
        if before_file_id and before_file_id in ordered_ids:
            ordered_ids.insert(ordered_ids.index(before_file_id), file_id)
        else:
            ordered_ids.append(file_id)
        workspace_file_system_repo.reorder_files_in_kb(
            session, kb_id, target_folder_id, ordered_ids
        )

        session.commit()

        logger.info(
            f"文件移动成功: operator={user_id}, owner={file_record.user_id}, "
            f"file_id={file_id}, target_folder_id={target_folder_id}"
        )
        return file_record.file_name

    def batch_move_files(
        self,
        session: Session,
        user_id: str,
        file_ids: List[str],
        target_folder_id: str | None,
    ) -> MoveResult:
        """批量移动文件到目标文件夹

        对每个文件独立校验，跳过不满足条件的文件并记录原因，
        成功的文件在同一事务中提交。无权操作的文件按"跳过"处理
        （与批量语义一致，恶意混合批次不会静默获得部分成功之外的收益）。

        Raises:
            ValueError: 目标文件夹不存在
            SQLAlchemyError: 数据库操作失败
        """
        result = MoveResult(total_requested=len(file_ids))

        target_folder = self.validate_target_folder(
            session, user_id, target_folder_id
        )
        target_kb_id = target_folder.knowledge_base_id if target_folder else None
        moved_ids: List[str] = []
        moved_kb_id: str | None = None

        try:
            for file_id in file_ids:
                file_record = workspace_file_system_repo.get_by_file_id(
                    session, file_id
                )
                if not file_record:
                    result.skipped_files.append(
                        SkippedFile(file_id=file_id, reason="文件不存在或无权限")
                    )
                    continue

                try:
                    self._require_file_write(session, user_id, file_record)
                except PermissionError:
                    result.skipped_files.append(
                        SkippedFile(file_id=file_id, reason="无权操作该文件")
                    )
                    continue

                if file_record.folder_id == target_folder_id:
                    result.skipped_files.append(
                        SkippedFile(file_id=file_id, reason="文件已在目标文件夹中")
                    )
                    continue

                if (
                    target_kb_id is not None
                    and file_record.knowledge_base_id != target_kb_id
                ):
                    result.skipped_files.append(
                        SkippedFile(
                            file_id=file_id,
                            reason=f"不允许跨知识库移动: 文件属于 {file_record.knowledge_base_id}",
                        )
                    )
                    continue

                file_record.folder_id = target_folder_id
                file_record.updater = user_id
                moved_ids.append(file_id)
                moved_kb_id = file_record.knowledge_base_id
                result.moved_count += 1

            # 批量移动的文件统一落到目标文件夹末尾（固化目标文件夹全序）
            if moved_ids:
                kb_for_order = (
                    target_folder.knowledge_base_id
                    if target_folder
                    else moved_kb_id
                )
                if kb_for_order:
                    ordered_ids = [
                        f.file_id
                        for f in workspace_file_system_repo
                        .get_by_folder_id_in_kb(
                            session, kb_for_order, target_folder_id
                        )
                    ]
                    ordered_ids = [
                        fid for fid in ordered_ids if fid not in moved_ids
                    ] + moved_ids
                    workspace_file_system_repo.reorder_files_in_kb(
                        session, kb_for_order, target_folder_id, ordered_ids
                    )

            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"批量移动文件失败: {e}")
            raise

        logger.info(
            f"批量移动文件完成: operator={user_id}, "
            f"moved={result.moved_count}/{result.total_requested}, "
            f"skipped={len(result.skipped_files)}, "
            f"target_folder_id={target_folder_id}"
        )
        return result


knowledge_move_service = KnowledgeMoveService()
