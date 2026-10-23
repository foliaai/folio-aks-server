#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
索引相关消息模型

定义文件索引流程的各阶段消息：
- IndexStartMessage: 索引开始
- ParseEndMessage: 解析完成
- SplitEndMessage: 分割完成
"""

from typing import Dict, List, Optional, Any
from pydantic import Field

from src.types.messages.base import BaseMessage


class IndexStartMessage(BaseMessage):
    """
    索引开始消息
    
    用户上传文件后，存储完成，触发索引构建流程。
    发送到: knowledge_base.index.start
    消费者: FileParser
    """
    
    # 存储路径（MinIO/S3 路径）
    storage_path: str = Field(
        ...,
        description="文件在对象存储中的路径"
    )
    
    # 文件名
    filename: str = Field(
        ...,
        description="原始文件名"
    )
    
    # 知识库ID
    knowledge_base_id: str = Field(
        ...,
        description="知识库ID"
    )
    
    # 知识库名称
    knowledge_base_name: str = Field(
        ...,
        description="知识库名称"
    )
    
    # Document ID（基于 file_sha256 的后台唯一标识，格式: document-{uuid}）
    document_id: str = Field(
        ...,
        description="Document ID（基于file_sha256的后台唯一标识，格式: document-{uuid}）"
    )
    
    # 会话ID（可选）
    session_id: Optional[str] = Field(
        default=None,
        description="会话ID"
    )
    
    # 父知识库ID（可选）
    parent_knowledge_base_id: Optional[str] = Field(
        default=None,
        description="父知识库ID"
    )
    
    # 父知识库名称（可选）
    parent_knowledge_base_name: Optional[str] = Field(
        default=None,
        description="父知识库名称"
    )
    
    # 知识类型（可选）
    knowledge_type: Optional[str] = Field(
        default=None,
        description="知识类型"
    )
    
    # 上传时间（可选）
    upload_time: Optional[str] = Field(
        default=None,
        description="文件上传时间"
    )
    
    # 文件大小（字节，可选）
    file_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="文件大小（字节）"
    )
    
    # MIME 类型（可选）
    mime_type: Optional[str] = Field(
        default=None,
        description="文件 MIME 类型"
    )
    
    # 文件扩展名（可选）
    file_extension: Optional[str] = Field(
        default=None,
        description="文件扩展名（如 .pdf, .docx）"
    )
    
    # 用户指定的解析选项（可选）
    parse_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="用户指定的解析选项"
    )


class ParseEndMessage(BaseMessage):
    """
    解析完成消息
    
    文件解析完成，提取出结构化信息。
    发送到: knowledge_base.parse.end
    消费者: 下游处理组件
    """
    
    # 文件名
    filename: str = Field(
        ...,
        description="文件名"
    )
    
    # 解析状态
    status: str = Field(
        ...,
        description="解析状态（success, partial_success, failed）"
    )
    
    # 总页数
    total_pages: int = Field(
        default=0,
        ge=0,
        description="文档总页数"
    )
    
    # 总字符数
    total_chars: int = Field(
        default=0,
        ge=0,
        description="文档总字符数"
    )
    
    # 解析使用的工具
    parse_tool: Optional[str] = Field(
        default="mineru",
        description="使用的解析工具名称（如 mineru, pypdf）"
    )
    
    # 解析质量评分（0-1）
    parse_quality: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="解析质量评分"
    )
    
    # 文档语言
    document_language: Optional[str] = Field(
        default=None,
        description="文档语言（如 zh, en）"
    )
    
    # Document ID（从 IndexStartMessage 透传）
    document_id: str = Field(
        ...,
        description="Document ID（基于file_sha256的后台唯一标识，格式: document-{uuid}）"
    )
    
    # 知识库信息（从 IndexStartMessage 透传，供下游 Worker 使用）
    knowledge_base_id: Optional[str] = Field(
        default=None,
        description="知识库ID"
    )
    
    knowledge_base_name: Optional[str] = Field(
        default=None,
        description="知识库名称"
    )
    
    # 错误信息（如果失败）
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息（解析失败时）"
    )
    
    # 是否包含图片（保留兼容性）
    has_images: bool = Field(
        default=False,
        description="文档是否包含图片"
    )
    
    # 图片信息（如果有）
    images: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="文档中的图片信息列表"
    )
    
    # 文档语言
    language: str = Field(
        default="unknown",
        description="检测到的文档语言"
    )
    
    # 是否包含表格
    has_tables: bool = Field(
        default=False,
        description="文档是否包含表格"
    )
    
    # 表格信息（如果有）
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="文档中的表格信息列表"
    )

    # Elements（自包含，供下游 TextSplitterWorker 直接消费，避免 parse→split 读库竞态）
    elements: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "文档全部元素（自包含）。每项字段：element_id, element_index, page_index, "
            "element_type, page_position, text_level, text, image_caption, image_footnote, "
            "table_body, table_caption, table_footnote, bucket_name, image_file_path, "
            "image_file_name, image_file_type, image_file_format, image_file_suffix。"
            "split 阶段从此字段构造 ParseResult，不回读 MySQL/MongoDB。"
        )
    )


class SplitEndMessage(BaseMessage):
    """
    分割完成消息
    
    文本分割完成，生成多个 Chunk。
    发送到: knowledge_base.split.end
    消费者:
    - SectionSummary（后台串行，基础抽取第一步）
    - FileSummary（历史，已重新布线到 section_summary.end）
    - EmbeddingMilvusWriter（数据库写入）
    """
    
    # 文本 Chunks
    chunks: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "分割后的文本 Chunks（自包含，供下游后台 Worker 直接消费，避免读库竞态）。"
            "每项字段：chunk_id, section_id, chunk_type, text, image_caption, "
            "image_footnote, page_index, language。"
        )
    )

    # Sections（v1.1 新增，自包含，供 SectionSummaryWorker 直接消费）
    sections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "文档 section 树（自包含）。每项字段：section_id, title, level, "
            "page_index, chunk_id_list。后台阶段从此字段取数，不回读 MySQL/MongoDB。"
        )
    )
    
    # 分割策略
    split_strategy: str = Field(
        ...,
        description="使用的分割策略（如 semantic, fixed_size）"
    )
    
    # Chunk 统计信息
    chunk_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk 统计信息（总数、平均长度等）"
    )
    
    # 文档总长度（字符数）
    total_length: int = Field(
        ...,
        ge=0,
        description="文档总长度（字符数）"
    )
    
    # 文档语言
    language: str = Field(
        default="unknown",
        description="检测到的文档语言"
    )

    # 解析来源（从 ParseEndMessage.parse_tool 透传）
    parse_tool: Optional[str] = Field(
        default=None,
        description=(
            "解析来源（markdown / mineru）。SectionSummary 建树据此分派策略："
            "markdown → heading_level（标题层级即真值），其余 → 标题编号推断"
        )
    )

    # 前台进度完成标志（用户可见进度到此为 100%）
    frontend_complete: bool = Field(
        default=True,
        description="前台进度是否完成"
    )
    
    # 文档摘要（简短）
    brief_summary: Optional[str] = Field(
        default=None,
        description="文档简短摘要（前台显示用）"
    )

    # ========== 后台抽取链路所需溯源字段（v1.1 新增，可选）==========
    # 供下游 SectionSummaryWorker / FileSummaryWorker 加载数据用；
    # 留空时下游需自行按 file_id 反查 workspace_file_system。
    filename: Optional[str] = Field(
        default=None,
        description="文件名（从 ParseEndMessage 透传，供 FileSummary 的 document_title 使用）"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="文档 ID（document-{uuid}，基于 file_sha256 的后台唯一标识）"
    )
    knowledge_base_id: Optional[str] = Field(
        default=None,
        description="知识库 ID"
    )
    knowledge_base_name: Optional[str] = Field(
        default=None,
        description="知识库名称"
    )
