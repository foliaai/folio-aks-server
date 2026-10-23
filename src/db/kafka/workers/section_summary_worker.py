#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
SectionSummary Worker

监听: knowledge_base.split.end
功能: 对文档内每个 section 生成 section 级摘要，写入 MySQL/MongoDB/Milvus
输出:
  - db_write.meta.start   (MySQL section_summary 关联表 + section_document 拓扑回写)
  - db_write.mongo.start  (MongoDB section_data.summary 字段，局部 $set)
  - db_write.embedding.start (Milvus summary collection, role=section_summary)
  - knowledge_base.section_summary.end (触发下游 FileSummaryWorker)
    [当前禁用] 触发 file_summary 的 SectionSummaryEndMessage 发送已注释，
    保留 _send_section_summary_end_message 方法与 FileSummaryWorker 代码不动，
    后续需要接通时取消注释即可。

设计要点：
- 后台阶段，对前台进度完全无感知：不调用 _update_file_progress()，
  并覆盖 _fail_file_progress() 为空实现，避免 BaseWorker 兜底把 Redis
  status 写成 failed 污染用户可见状态（失败隔离）。
- 组件开关：config/components.json 中 section_summary.enabled=false 时
  空跑（消费但跳过），避免 Kafka lag 堆积。
- **不读数据库**：section/chunk 输入完全来自 SplitEndMessage 消息体
  （sections + chunks 字段），避免 split 阶段写库异步导致的竞态。
- Worker 层只做编排：Kafka 消息构造/分发；业务逻辑全部在 SectionSummaryService 中。
"""

from typing import Optional, List, Dict, Any
from loguru import logger

from src.db.kafka.workers.base_worker import BaseWorker
from src.db.kafka.topics import KafkaTopics
from src.states.states import IndexStage
from src.types.messages.index import SplitEndMessage
from src.types.messages.extract import SectionSummaryEndMessage
from src.types.messages.db_write import (
    EmbeddingWriteMessage,
    MetaWriteMessage,
    MongoWriteMessage,
    MySQLTable,
    MongoCollection,
    MilvusCollection,
    WriteOperation,
)
from src.service.knowledge.components.section_summary_service import (
    SectionSummaryService,
)
from src.types.models.section_summary_result import SectionSummaryResult
from src.utils.component_config_manager import get_component_config_manager


class SectionSummaryWorker(BaseWorker):
    """
    SectionSummary Worker

    职责:
    - 消费 Kafka 消息 (knowledge_base.split.end)
    - 校验组件开关 (section_summary.enabled)
    - 从 SplitEndMessage 的 sections + chunks 字段构造上下文（不读库）
    - 调用 SectionSummaryService 生成各 section 摘要
    - 将结果分发到 db_write.meta / db_write.mongo / db_write.embedding
    - 发送 SectionSummaryEndMessage (触发下游 file_summary)

    输入消息: SplitEndMessage
    输出消息:
      - MetaWriteMessage (db_write.meta.start) → section_summary 表
      - MongoWriteMessage (db_write.mongo.start) → section_data.summary
      - EmbeddingWriteMessage (db_write.embedding.start) → summary collection
      - SectionSummaryEndMessage (knowledge_base.section_summary.end)

    配置要求:
    - 资源: 4 CPU, 8GB RAM（LLM 调用为主，IO 次之）
    - 扩容触发: Kafka lag > 50
    """

    COMPONENT_NAME = "section_summary"

    def __init__(
        self,
        *args,
        section_summary_service: Optional[SectionSummaryService] = None,
        **kwargs,
    ):
        """
        初始化 SectionSummary Worker

        Args:
            section_summary_service: SectionSummaryService 实例（可选，懒加载）
        """
        super().__init__(*args, **kwargs)
        self._summary_service = section_summary_service
        self._config_manager = get_component_config_manager()

    # ========== 抽象方法实现 ==========

    def get_original_topic(self) -> str:
        return KafkaTopics.SPLIT_END

    def _get_failure_stage(self) -> str:
        # 后台阶段，仅用于日志标识；_fail_file_progress 已覆盖为空实现
        return IndexStage.SECTION_SUMMARY_END

    # ========== 失败隔离：后台阶段不污染前台进度 ==========

    async def _fail_file_progress(
        self, file_id: str, stage: str, error_message: str
    ) -> None:
        """覆盖基类行为：后台阶段失败不写 Redis 进度，仅记录日志。"""
        logger.warning(
            f"SectionSummary 后台阶段失败（不污染前台进度）: "
            f"file_id={file_id}, stage={stage}, error={error_message}"
        )

    # ========== 懒加载依赖 ==========

    def _get_summary_service(self) -> SectionSummaryService:
        if self._summary_service is None:
            self._summary_service = SectionSummaryService()
            logger.info("SectionSummaryService 已创建")
        return self._summary_service

    # ========== 主流程 ==========

    async def process_message_impl(self, message: SplitEndMessage) -> bool:
        """
        处理切分完成消息，生成各 section 摘要并分发下游。

        Args:
            message: SplitEndMessage

        Returns:
            bool: 是否处理成功
        """
        logger.info(
            f"开始 Section 摘要抽取: user_id={message.user_id}, "
            f"file_id={message.file_id}, document_id={message.document_id}"
        )

        # 1. 组件开关守卫：禁用时空跑（消费但不落库，避免 lag 堆积）
        if not self._config_manager.is_component_enabled(self.COMPONENT_NAME):
            logger.debug(
                f"SectionSummary 组件已禁用，跳过处理: file_id={message.file_id}"
            )
            return True

        # 2. 校验溯源字段
        document_id = message.document_id
        if not document_id:
            logger.error(
                f"SectionSummary: SplitEndMessage 缺少 document_id，跳过: "
                f"file_id={message.file_id}"
            )
            # 不重试（溯源信息缺失重试也无解），ack 掉
            return True

        try:
            summary_service = self._get_summary_service()

            # 3. 调用 Service 生成摘要
            #    输入完全来自 SplitEndMessage 消息体（sections + chunks），
            #    不读数据库——避免 split 写库异步导致的竞态。
            result: SectionSummaryResult = await summary_service.summarize_document_sections(
                document_id=document_id,
                sections_data=message.sections,
                chunks_data=message.chunks,
                language=message.language,
                knowledge_base_id=message.knowledge_base_id,
                knowledge_base_name=message.knowledge_base_name,
                # 解析来源（markdown → heading_level 建树，其余编号推断）
                parse_tool=message.parse_tool,
            )

            if not result.is_success():
                logger.warning(
                    f"SectionSummary: 文档无 section 摘要产出（可能无 section 或全部失败）: "
                    f"file_id={message.file_id}, document_id={document_id}"
                )
                # 无 section 摘要产出时仍发送 end 消息，让下游 FileSummary 知晓并跳过
                await self._send_section_summary_end_message(message, result)
                return True

            # 4. 分发到 db_write.*
            await self._send_mysql_messages(message, result)
            await self._send_mongodb_messages(message, result)
            await self._send_embedding_messages(message, result)

            # 5. 发送 SectionSummaryEndMessage（触发下游 file_summary）
            await self._send_section_summary_end_message(message, result)

            logger.info(
                f"Section 摘要处理完成: file_id={message.file_id}, "
                f"document_id={document_id}, sections={result.total_sections}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Section 摘要处理失败: file_id={message.file_id}, error={e}",
                exc_info=True,
            )
            return False

    # ========== 下游消息发送 ==========

    async def _send_mysql_messages(
        self,
        message: SplitEndMessage,
        result: SectionSummaryResult,
    ) -> None:
        """发送 MySQL 写入消息：section_summary 关联表 + section_document 拓扑回写。"""
        if not self._producer:
            logger.warning("Producer 未配置，无法发送 MySQL 消息")
            return

        mysql_data = result.get_mysql_data()
        meta_msgs: List[MetaWriteMessage] = []

        # 1) section_summary 关联表（UPSERT，按 section_id 主键）
        for record in mysql_data.get("section_summary", []):
            record_id = record.get("section_id") or message.file_id
            meta_msgs.append(MetaWriteMessage(
                user_id=message.user_id,
                file_id=message.file_id,
                table_name=MySQLTable.SECTION_SUMMARY,
                record_data=record,
                operation=WriteOperation.UPSERT,
                record_id=record_id,
            ))

        # 2) section_document 拓扑回写（parent_section_id / is_leaf）。
        #    v1.1：拓扑从 Mongo section_data 迁到 MySQL section_document，骨架树重建与
        #    TextAnalyzer 叶子过滤都在 MySQL 完成。UPSERT 走 ON DUPLICATE KEY UPDATE，
        #    与 split 阶段的 section_document 写入互不覆盖（各自只更新自己写入的列），
        #    消除「split 写库异步 → section_summary 读库竞态」与跨消费者乱序问题。
        for record in mysql_data.get("section_document", []):
            record_id = record.get("section_id") or message.file_id
            meta_msgs.append(MetaWriteMessage(
                user_id=message.user_id,
                file_id=message.file_id,
                table_name=MySQLTable.SECTION_DOCUMENT,
                record_data=record,
                operation=WriteOperation.UPSERT,
                record_id=record_id,
            ))

        if meta_msgs:
            await self._producer.send_messages(
                topic=KafkaTopics.DB_WRITE_META,
                messages=meta_msgs,
            )
        logger.debug(
            f"SectionSummary MySQL 消息发送完成: {len(meta_msgs)} 条 "
            f"(section_summary + section_document 拓扑)"
        )

    async def _send_mongodb_messages(
        self,
        message: SplitEndMessage,
        result: SectionSummaryResult,
    ) -> None:
        """发送 MongoDB section_data.summary 局部更新消息。"""
        if not self._producer:
            logger.warning("Producer 未配置，无法发送 MongoDB 消息")
            return

        mongo_data = result.get_mongodb_data()
        documents = mongo_data.get("section_data", [])
        if not documents:
            return

        mongo_msgs: List[MongoWriteMessage] = []
        for doc in documents:
            document_id = doc.get("_id", message.file_id)
            mongo_msgs.append(MongoWriteMessage(
                user_id=message.user_id,
                file_id=message.file_id,
                collection_name=MongoCollection.SECTION_DATA,
                document_data=doc,
                operation=WriteOperation.UPSERT,
                document_id=str(document_id),
            ))

        if mongo_msgs:
            await self._producer.send_messages(
                topic=KafkaTopics.DB_WRITE_MONGO,
                messages=mongo_msgs,
            )
        logger.debug(
            f"SectionSummary MongoDB 消息发送完成: {len(mongo_msgs)} 条 (section_data.summary)"
        )

    async def _send_embedding_messages(
        self,
        message: SplitEndMessage,
        result: SectionSummaryResult,
    ) -> None:
        """发送 Milvus summary collection 向量化消息（role=section_summary）。"""
        if not self._producer:
            logger.warning("Producer 未配置，无法发送 Embedding 消息")
            return

        items = result.get_embedding_messages()
        if not items:
            return

        emb_msg = EmbeddingWriteMessage(
            user_id=message.user_id,
            file_id=message.file_id,
            collection_type=MilvusCollection.SECTION_SUMMARY,
            items=items,
            source_stage="section_summary",
            document_id=message.document_id or "",
            knowledge_base_id=message.knowledge_base_id,
            knowledge_base_name=message.knowledge_base_name,
            language=message.language,
        )

        await self._producer.send_message(
            topic=KafkaTopics.DB_WRITE_EMBEDDING,
            message=emb_msg,
        )
        logger.debug(
            f"SectionSummary Embedding 消息发送完成: {len(items)} 条 (summary collection)"
        )

    async def _send_section_summary_end_message(
        self,
        message: SplitEndMessage,
        result: SectionSummaryResult,
    ) -> None:
        """发送 SectionSummaryEndMessage，触发下游 FileSummaryWorker。"""
        if not self._producer:
            logger.warning("Producer 未配置，无法发送 SectionSummaryEndMessage")
            return

        end_msg = SectionSummaryEndMessage(
            user_id=message.user_id,
            file_id=message.file_id,
            document_id=message.document_id or "",
            # 轻量统计（给状态管理器/日志）
            section_summaries=result.get_section_summaries_stats(),
            # 完整 payload（含正文，给 FileSummaryWorker 自包含消费）
            section_summaries_payload=result.get_section_summaries_payload(),
            # 透传字段（供 FileSummary 的 LLM prompt 与语言检测使用）
            document_title=(message.filename or ""),
            language=message.language or "unknown",
            knowledge_base_id=message.knowledge_base_id,
            knowledge_base_name=message.knowledge_base_name,
            total_sections=result.total_sections,
            successful_sections=result.successful_sections,
            llm_model=result.llm_model,
            token_usage=result.token_usage,
        )

        await self._producer.send_message(
            topic=KafkaTopics.SECTION_SUMMARY_END,
            message=end_msg,
        )
        logger.debug(
            f"发送 SectionSummaryEndMessage: file_id={message.file_id}, "
            f"sections={result.total_sections}, "
            f"payload_items={len(end_msg.section_summaries_payload)}"
        )
