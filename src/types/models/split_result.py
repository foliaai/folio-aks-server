#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
SplitResult 数据模型

文本切分结果的统一数据模型，用于 TextSplitterWorker 返回切分结果。
提供数据库转换方法，方便后续存储。

设计原则：
- 类似 ParseResult 的设计风格
- 提供 to_mysql_dict()、to_mongodb_dict() 等转换方法
- Section 和 Chunk 分离，形成两层结构
- 从 MySQL 和 MongoDB 加载 ParseResult 数据

设计决策：
- ❌ 移除全局索引（index字段）：
  - 旧策略使用 index 记录 Section 和 Chunk 的出现顺序
  - 新策略直接使用 chunk_id / section_id 唯一标识
  - 顺序关系通过 page_index 和 section_id 关联即可
  - 简化数据模型，减少冗余字段
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid

from src.types.utils.image_chunk_text import format_image_chunk_embed_text


class ChunkType(str, Enum):
    """Chunk类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CODE_BLOCK = "code_block"


class SplitStatus(str, Enum):
    """切分状态枚举"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    PENDING = "pending"


class ChunkInfo(BaseModel):
    """
    Chunk信息模型
    
    单个Chunk的完整信息，包含：
    - 文本内容
    - 元数据信息
    - 页面索引（page_index）
    - 关联关系
    
    存储映射：
    - MySQL (chunk_meta_info): 元信息（chunk_id, chunk_type, page_index等）
    - MongoDB (chunk_data): 内容数据（text, translations, summary等）
    - Milvus: 向量数据（由 EmbeddingMilvusWriter 处理）
    
    注意：
    - 不再使用全局索引（index字段），通过 chunk_id 唯一标识
    - 通过 section_id 关联章节结构
    - 通过 page_index 定位所在页面（分块后位置信息已丢失，不再保留 page_position）
    """
    
    # ========== 基础字段 ==========
    chunk_id: str = Field(
        default_factory=lambda: f"chunk-{uuid.uuid4()}",
        description="Chunk唯一ID（UUID格式）"
    )
    
    chunk_type: ChunkType = Field(
        ...,
        description="Chunk类型（text/image/table/code_block）"
    )
    
    # ========== 层级关系 ==========
    document_id: Optional[str] = Field(
        default=None,
        description="所属的Document ID（文档级关联）"
    )
    
    section_id: Optional[str] = Field(
        default=None,
        description="所属的Section ID"
    )
    
    parent_chunk_id: Optional[str] = Field(
        default=None,
        description="父Chunk ID（用于嵌套结构，暂未使用）"
    )
    
    # ========== 文档溯源 ==========
    element_ids: List[str] = Field(
        default_factory=list,
        description="关联的Element ID列表（用于文档溯源，追踪Chunk包含的元素）"
    )
    
    split_seq: int = Field(
        default=0,
        description="分块序号：同一组 element 被切分为多个 chunk 时的局部顺序（从0开始）"
    )
    
    # ========== 内容字段 ==========
    content: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk内容（包含original和translations）"
    )
    
    # ========== 位置信息 ==========
    page_index: Optional[int] = Field(
        default=None,
        description="Chunk所在页码（从0开始）"
    )
    
    # ========== 语言信息 ==========
    language: str = Field(
        default="unknown",
        description="Chunk的语言（zh, en, etc.）"
    )
    
    # ========== 图片特定字段（从 ElementInfo 继承） ==========
    image_url: Optional[str] = Field(
        default=None,
        description="图片URL（image类型使用）"
    )
    
    image_caption: Optional[str] = Field(
        default=None,
        description="图片说明（image类型使用）"
    )
    
    image_footnote: Optional[str] = Field(
        default=None,
        description="图片脚注（image类型使用）"
    )
    
    # 对象存储信息（从 ParseResult 的 ElementInfo 继承）
    bucket_name: Optional[str] = Field(
        default=None,
        description="对象存储桶名称（image类型）"
    )
    
    image_file_path: Optional[str] = Field(
        default=None,
        description="图片文件路径（image类型）"
    )
    
    image_file_name: Optional[str] = Field(
        default=None,
        description="图片文件名（image类型）"
    )
    
    image_file_type: Optional[str] = Field(
        default=None,
        description="图片文件类型（image类型）"
    )
    
    image_file_format: Optional[str] = Field(
        default=None,
        description="图片格式详细信息（image类型）"
    )
    
    image_file_suffix: Optional[str] = Field(
        default=None,
        description="图片文件后缀名（image类型）"
    )
    
    # ========== 表格特定字段 ==========
    table_body: Optional[str] = Field(
        default=None,
        description="表格主体内容（table类型，Markdown或HTML格式）"
    )
    
    table_caption: Optional[str] = Field(
        default=None,
        description="表格标题（table类型）"
    )
    
    table_footnote: Optional[str] = Field(
        default=None,
        description="表格脚注（table类型）"
    )
    
    # ========== 检索 / 展示文本（split 阶段产出） ==========
    vector_text: Optional[str] = Field(
        default=None,
        description=(
            "检索文本：Milvus chunk_store 向量化源，对应 MongoDB search_text。"
            "image/table 为去包装语义文本；text/code 为原文。"
        )
    )

    display_text: Optional[str] = Field(
        default=None,
        description=(
            "展示文本：MongoDB text 字段，供 read_chunks / LLM 预览。"
            "image/table 保留结构化包装。"
        )
    )

    enhanced_vector_text: Optional[str] = Field(
        default=None,
        description=(
            "增强检索文本（Section 标题 + vector_text）；"
            "Milvus enhanced_chunk_store 向量化源。"
        )
    )

    enhanced_display_text: Optional[str] = Field(
        default=None,
        description="增强展示文本（Section 标题 + display_text），对应 MongoDB enhanced_text。",
    )
    
    # ========== 后续处理字段（后续填充） ==========
    summary: Optional[str] = Field(
        default=None,
        description="Chunk摘要（后续由 TextAnalyzer 填充）"
    )
    
    atomic_qa: List[Dict[str, str]] = Field(
        default_factory=list,
        description="原子问答对（后续由 TextAnalyzer 填充）"
    )
    
    # ========== 标签和元数据 ==========
    tags: List[str] = Field(
        default_factory=list,
        description="Chunk标签"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="额外元数据"
    )
    
    class Config:
        use_enum_values = True
    
    # ========== 类型判断方法 ==========
    
    def is_text(self) -> bool:
        """判断是否为文本Chunk"""
        return self.chunk_type == ChunkType.TEXT
    
    def is_image(self) -> bool:
        """判断是否为图片Chunk"""
        return self.chunk_type == ChunkType.IMAGE
    
    def is_table(self) -> bool:
        """判断是否为表格Chunk"""
        return self.chunk_type == ChunkType.TABLE
    
    def is_code_block(self) -> bool:
        """判断是否为代码块Chunk"""
        return self.chunk_type == ChunkType.CODE_BLOCK
    
    # ========== 数据转换方法 ==========
    
    def to_mysql_dict(self) -> Dict[str, Any]:
        """
        转换为 MySQL chunk_meta_info 表的字典格式
        
        注意：document_id 不在此表中，而是在 chunk_section_document 关系表中
        
        Returns:
            MySQL 表字段字典
        """
        data = {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "page_index": self.page_index,
            "split_seq": self.split_seq,
            "element_ids": self.element_ids,
        }
        
        # 图片特定字段
        if self.is_image():
            data.update({
                "bucket_name": self.bucket_name,
                "image_file_path": self.image_file_path,
                "image_file_name": self.image_file_name,
                "image_file_type": self.image_file_type,
                "image_file_format": self.image_file_format,
                "image_file_suffix": self.image_file_suffix,
            })
        
        return data
    
    def to_relation_dict(self) -> Dict[str, Any]:
        """
        转换为 MySQL chunk_section_document 关系表的字典格式
        
        Returns:
            关系表字段字典
        """
        return {
            "chunk_id": self.chunk_id,
            "parent_chunk_id": self.parent_chunk_id,
            "section_id": self.section_id,
            "document_id": self.document_id,
        }
    
    def to_mongodb_dict(self) -> Dict[str, Any]:
        """
        转换为 MongoDB chunk_data 表的字典格式

        - search_text   ↔ chunk_store 向量化源（vector_text）
        - text_meta     结构化内容元数据（JSON），按 chunk_type 存储不同字段
        - enhanced_text ↔ 增强展示（enhanced_display_text）

        Returns:
            MongoDB 表字段字典
        """
        enhanced_display = self.enhanced_display_text or self.enhanced_vector_text
        data: Dict[str, Any] = {
            "_id": self.chunk_id,
            "type": self.chunk_type,
            "search_text": self.vector_text,
            "text_meta": self._build_text_meta(),
            "enhanced_text": enhanced_display,
            "translation": self.content.get("translations", []),
            "atomic_qa": [],  # 后续填充
            "language": self.language,
        }
        return data

    def _build_text_meta(self) -> Dict[str, Any]:
        """
        按 chunk_type 构建 text_meta JSON 结构。

        Returns:
            text_meta 字典
        """
        if self.is_text() or self.is_code_block():
            return {"text": self.get_text_content() or ""}

        if self.is_image():
            meta: Dict[str, Any] = {}
            if self.image_caption:
                meta["image_caption"] = self.image_caption
            if self.image_footnote:
                meta["image_footnote"] = self.image_footnote
            # section_title 从 section_id 关联的 section content 获取
            # 这里先留空，由上层调用时补充
            return meta

        if self.is_table():
            meta: Dict[str, Any] = {}
            if self.table_caption:
                meta["table_caption"] = self.table_caption
            if self.table_body:
                meta["table_body"] = self.table_body
            if self.table_footnote:
                meta["table_footnote"] = self.table_footnote
            return meta

        # 兜底：当作 text 处理
        return {"text": self.get_text_content() or ""}

    def to_embedding_message_dict(self) -> Optional[Dict[str, Any]]:
        """
        转换为发送到 db_write.embedding.start 的消息格式（chunk_store）。

        统一从 ``vector_text`` 取向量化源文本；为空时返回 None。
        """
        if not self.vector_text:
            return None
        return {
            "chunk_id": self.chunk_id,
            "text": self.vector_text,
            "language": self.language,
            "collection_type": "chunk",  # chunk_store
            "metadata": {
                "chunk_type": self.chunk_type,
                "page_index": self.page_index,
            }
        }

    def to_enhanced_embedding_message_dict(self) -> Optional[Dict[str, Any]]:
        """
        转换为 enhanced_chunk_store 的 Embedding 消息格式。

        使用 ``enhanced_vector_text``（Section 标题 + 正文/占位符）作为向量化源文本；
        为空时返回 None。
        """
        if not self.enhanced_vector_text:
            return None
        return {
            "chunk_id": self.chunk_id,
            "text": self.enhanced_vector_text,
            "language": self.language,
            "collection_type": "enhanced_chunk",
            "metadata": {
                "chunk_type": self.chunk_type,
                "page_index": self.page_index,
            }
        }

    def get_text_content(self) -> Optional[str]:
        """
        获取 split 解析出的「原始正文」（text/table/code 的 original.content）。

        仅作为 split 阶段构造 ``vector_text`` 的输入；图片 chunk 无正文，返回 None。
        消费端请统一使用 ``vector_text`` / MongoDB.text。
        """
        if self.is_image():
            return None
        if "original" in self.content and "content" in self.content["original"]:
            return self.content["original"]["content"]
        return None

    def build_image_embedding_text(
        self,
        section_title: Optional[str] = None,
    ) -> str:
        """
        组装图片 Chunk 检索文本（写入 vector_text / MongoDB.search_text）。

        Args:
            section_title: 所属 Section 标题（无 caption 时的 fallback）

        Returns:
            去包装的向量化源文本
        """
        return format_image_chunk_embed_text(
            image_caption=self.image_caption,
            image_footnote=self.image_footnote,
            section_title=section_title,
            page_index=self.page_index,
        )

    def build_image_display_text(
        self,
        section_title: Optional[str] = None,
    ) -> str:
        """组装图片 Chunk 展示文本（写入 display_text / MongoDB.text）。"""
        from src.types.utils.chunk_search_text import format_image_display_text

        return format_image_display_text(
            image_caption=self.image_caption,
            image_footnote=self.image_footnote,
            section_title=section_title,
            page_index=self.page_index,
        )


class SectionInfo(BaseModel):
    """
    Section信息模型
    
    文档章节/标题的信息，形成文档的层级结构。
    
    存储映射：
    - MySQL (section_meta_info): 元信息（section_id, level, page_index等）
    - MongoDB (section_data): 内容数据（text, translations）
    
    注意：
    - 不再使用全局索引（index字段），通过 section_id 唯一标识
    - 通过 level 表示层级关系
    - 通过 page_index 定位页面位置
    """
    
    # ========== 基础字段 ==========
    section_id: str = Field(
        default_factory=lambda: f"section-{uuid.uuid4()}",
        description="Section唯一ID（UUID格式）"
    )
    
    level: int = Field(
        ...,
        ge=1,
        le=6,
        description="标题层级（1=一级标题，2=二级标题，最大6级）"
    )
    
    # ========== 层级关系 ==========
    document_id: Optional[str] = Field(
        default=None,
        description="所属的Document ID（文档级关联）"
    )
    
    # ========== 文档溯源 ==========
    element_id: Optional[str] = Field(
        default=None,
        description="关联的Element ID（用于文档溯源，追踪Section对应的元素）"
    )
    
    # ========== 内容字段 ==========
    content: str = Field(
        ...,
        description="Section标题文本"
    )
    
    # ========== 位置信息 ==========
    page_index: Optional[int] = Field(
        default=None,
        description="Section所在页码（从0开始）"
    )
    
    page_position: Optional[List[float]] = Field(
        default=None,
        description="MinerU bbox [x0, y0, x1, y1]，0~1000 归一化，左上角原点",
    )
    
    # ========== 子Chunk列表 ==========
    chunk_id_list: List[str] = Field(
        default_factory=list,
        description="该Section下的子Chunk ID列表"
    )
    
    # ========== 向量化源文本字段 ==========
    vector_text: Optional[str] = Field(
        default=None,
        description="用于向量化的源文本（Section标题，发送到Kafka，由Embedding服务处理）"
    )
    
    # ========== 元数据 ==========
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="额外元数据"
    )
    
    # ========== 数据转换方法 ==========
    
    def to_mysql_dict(self) -> Dict[str, Any]:
        """
        转换为 MySQL section_meta_info 表的字典格式
        
        注意：
        - MySQL 表使用 text_level 字段名，不是 level
        - document_id 不在此表中，而是通过 chunk_section_document 关系表关联
        
        Returns:
            MySQL 表字段字典
        """
        return {
            "section_id": self.section_id,
            "text_level": self.level,
            "start_page_index": self.page_index,
            "end_page_index": self.page_index,
            "element_id": self.element_id,
        }
    
    def to_relation_dict(self) -> Dict[str, Any]:
        """
        转换为 MySQL section_document 关系表的字典格式
        
        注意：Section 与 Document 的关系也可以通过 chunk_section_document 表建立
        或者单独建立 section_document 关系表
        
        Returns:
            关系表字段字典
        """
        return {
            "section_id": self.section_id,
            "document_id": self.document_id,
        }
    
    def to_mongodb_dict(self) -> Dict[str, Any]:
        """
        转换为 MongoDB section_data 表的字典格式
        
        Returns:
            MongoDB 表字段字典
        """
        return {
            "_id": self.section_id,
            "text": self.content,
            "translation": [],  # 后续填充
        }
    
    def to_embedding_message_dict(self) -> Optional[Dict[str, Any]]:
        """
        转换为发送到 db_write.embedding.start 的消息格式
        
        使用 vector_text（Section 标题文本）作为向量化源文本。
        如果 vector_text 为空则返回 None（不需要向量化）。
        
        Returns:
            Embedding 消息字典，或 None
        """
        text = self.vector_text or self.content
        if not text:
            return None
        
        return {
            "id": self.section_id,
            "text": text,
            "collection_type": "section",
            "metadata": {
                "level": self.level,
                "page_index": self.page_index,
            }
        }


class SplitResult(BaseModel):
    """
    文本切分结果统一模型
    
    这是 TextSplitterWorker 的核心返回类型，包含了文本切分的所有信息：
    - 切分状态和错误信息
    - Section列表（章节结构）
    - Chunk列表（实际内容块）
    - 统计信息
    
    设计原则：
    - Section-Chunk 两层结构
    - 所有Chunk和Section共享一个全局索引序列
    - 提供三个存储方法：get_mysql_data、get_mongodb_data、get_embedding_messages
    """
    
    # ========== 基础信息 ==========
    user_id: str = Field(
        ...,
        description="用户ID"
    )
    
    file_id: str = Field(
        ...,
        description="文件ID"
    )
    
    filename: str = Field(
        ...,
        description="文件名"
    )
    
    # ========== 切分状态 ==========
    status: SplitStatus = Field(
        default=SplitStatus.PENDING,
        description="切分状态"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息（切分失败时）"
    )
    
    # ========== Section和Chunk列表 ==========
    sections: List[SectionInfo] = Field(
        default_factory=list,
        description="文档的Section列表（章节结构）"
    )
    
    chunks: List[ChunkInfo] = Field(
        default_factory=list,
        description="文档的Chunk列表（实际内容块）"
    )
    
    # ========== 切分配置 ==========
    split_method: str = Field(
        default="structure_first",
        description="使用的切分方法"
    )
    
    chunk_size: int = Field(
        default=1000,
        description="Chunk大小"
    )
    
    chunk_overlap: int = Field(
        default=0,
        description="Chunk重叠大小"
    )
    
    # ========== 统计信息 ==========
    total_sections: int = Field(
        default=0,
        description="Section总数"
    )
    
    total_chunks: int = Field(
        default=0,
        description="Chunk总数"
    )
    
    total_chars: int = Field(
        default=0,
        description="文档总字符数"
    )
    
    # ========== 语言信息 ==========
    document_language: str = Field(
        default="unknown",
        description="文档主语言"
    )
    
    # ========== 知识库信息 ==========
    knowledge_base_id: Optional[str] = Field(
        default=None,
        description="关联的知识库ID"
    )
    
    knowledge_base_name: Optional[str] = Field(
        default=None,
        description="关联的知识库名称"
    )
    
    class Config:
        use_enum_values = True
    
    # ========== 状态判断方法 ==========
    
    def is_success(self) -> bool:
        """判断切分是否成功"""
        return self.status in [SplitStatus.SUCCESS, SplitStatus.PARTIAL_SUCCESS]
    
    # ========== 统计方法 ==========
    
    def get_chunks_by_type(self, chunk_type: ChunkType) -> List[ChunkInfo]:
        """根据类型获取Chunk列表"""
        return [c for c in self.chunks if c.chunk_type == chunk_type]
    
    @property
    def text_chunks(self) -> List[ChunkInfo]:
        """获取所有文本Chunk"""
        return self.get_chunks_by_type(ChunkType.TEXT)
    
    @property
    def image_chunks(self) -> List[ChunkInfo]:
        """获取所有图片Chunk"""
        return self.get_chunks_by_type(ChunkType.IMAGE)
    
    @property
    def table_chunks(self) -> List[ChunkInfo]:
        """获取所有表格Chunk"""
        return self.get_chunks_by_type(ChunkType.TABLE)
    
    @property
    def code_chunks(self) -> List[ChunkInfo]:
        """获取所有代码块Chunk"""
        return self.get_chunks_by_type(ChunkType.CODE_BLOCK)
    
    # ========== 数据转换方法 ==========
    
    def get_mysql_data(self, document_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取用于 MySQL 的所有数据
        
        Args:
            document_id: 文档ID
        
        Returns:
            包含多个表数据的字典：
            - section_document
            - section_meta_info
            - chunk_section_document
            - chunk_meta_info
        """
        return {
            "section_document": [
                {
                    "section_id": s.section_id,
                    "document_id": document_id,
                    "knowledge_base_id": self.knowledge_base_id or "",
                    "knowledge_base_name": self.knowledge_base_name or "",
                }
                for s in self.sections
            ],
            "section_meta_info": [s.to_mysql_dict() for s in self.sections],
            "chunk_section_document": [
                {
                    "chunk_id": c.chunk_id,
                    "section_id": c.section_id,
                    "parent_chunk_id": c.parent_chunk_id or "",
                    "document_id": document_id,
                    "knowledge_base_id": self.knowledge_base_id or "",
                    "knowledge_base_name": self.knowledge_base_name or "",
                }
                for c in self.chunks
            ],
            # chunk_meta_info 的行字典（to_mysql_dict）不含 kb，这里补上：
            # chunk 三端点判权（raw-image 等）依赖该列，缺省会全量 404
            "chunk_meta_info": [
                {
                    **c.to_mysql_dict(),
                    "knowledge_base_id": self.knowledge_base_id or "",
                    "knowledge_base_name": self.knowledge_base_name or "",
                }
                for c in self.chunks
            ],
        }
    
    def get_mongodb_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取用于 MongoDB 的所有数据
        
        Returns:
            包含多个集合数据的字典：
            - section_data
            - chunk_data
        """
        return {
            "section_data": [s.to_mongodb_dict() for s in self.sections],
            "chunk_data": [c.to_mongodb_dict() for c in self.chunks],
        }
    
    def get_embedding_messages(self) -> List[Dict[str, Any]]:
        """
        获取用于发送到 db_write.embedding.start 的消息列表
        
        Returns:
            Embedding消息列表（原始文本，不包含向量）
        """
        messages = []
        
        for chunk in self.chunks:
            if chunk.is_text() or chunk.is_table() or chunk.is_code_block() or chunk.is_image():
                msg = chunk.to_embedding_message_dict()
                if msg is not None:
                    messages.append(msg)
        
        return messages
    
    def get_section_embedding_messages(self) -> List[Dict[str, Any]]:
        """
        获取 Section 的 Embedding 消息列表
        
        将有 vector_text 的 Section 标题发送到 db_write.embedding.start，
        目标 collection 为 section_store。
        
        Returns:
            Section Embedding 消息列表
        """
        messages = []
        for section in self.sections:
            msg = section.to_embedding_message_dict()
            if msg is not None:
                messages.append(msg)
        return messages
    
    def get_enhanced_chunk_embedding_messages(self) -> List[Dict[str, Any]]:
        """
        获取 Enhanced Chunk 的 Embedding 消息列表
        
        使用 enhanced_vector_text（Section标题 + Chunk文本）作为源文本，
        目标 collection 为 enhanced_chunk_store。
        仅当 Chunk 有 enhanced_vector_text 时才生成消息。
        
        Returns:
            Enhanced Chunk Embedding 消息列表
        """
        messages = []
        for chunk in self.chunks:
            if (
                chunk.is_text()
                or chunk.is_table()
                or chunk.is_code_block()
                or chunk.is_image()
            ):
                msg = chunk.to_enhanced_embedding_message_dict()
                if msg is not None:
                    messages.append(msg)
        return messages
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取切分结果摘要
        
        Returns:
            切分结果摘要字典
        """
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "status": self.status,
            "total_sections": len(self.sections),
            "total_chunks": len(self.chunks),
            "text_chunks": len(self.text_chunks),
            "image_chunks": len(self.image_chunks),
            "table_chunks": len(self.table_chunks),
            "code_chunks": len(self.code_chunks),
            "total_chars": self.total_chars,
            "language": self.document_language,
            "split_method": self.split_method,
        }
