#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
ParseResult 数据模型

文件解析结果的统一数据模型，用于 FileParserService 返回解析结果。
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ParseStatus(str, Enum):
    """解析状态枚举"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    PENDING = "pending"


class ElementType(str, Enum):
    """元素类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    CODE_BLOCK = "code_block"
    DISCARDED = "discarded"


class ElementInfo(BaseModel):
    """
    统一的元素信息模型
    
    包含文本、图片、表格三种类型元素的完整信息。
    根据 element_type 字段区分不同类型，不同类型使用不同的字段。
    
    存储映射：
    - MySQL (element_meta_info): 元信息（element_id, element_index, page_index, element_type等）
    - MongoDB (element_data): 内容数据（text, table_data等）
    - MinIO: 图片文件（仅 image 类型）
    """
    
    # ========== 通用字段（所有类型都需要） ==========
    element_id: str = Field(
        ...,
        description="元素唯一ID（格式: element-{uuid}）"
    )
    
    document_id: str = Field(
        ...,
        description="所属Document的ID（格式: document-{uuid}，基于file_sha256的后台唯一标识）"
    )
    
    element_index: int = Field(
        ...,
        ge=0,
        description="元素在当前页内的顺序（每页从0开始），需配合 page_index 确定文档全局顺序"
    )
    
    element_type: ElementType = Field(
        ...,
        description="元素类型（text/image/table/discarded）"
    )
    
    page_index: Optional[int] = Field(
        default=None,
        description="元素所在页码（从0开始）"
    )
    
    page_position: Optional[List[float]] = Field(
        default=None,
        description="MinerU bbox [x0, y0, x1, y1]，0~1000 归一化，左上角原点",
    )
    
    # ========== 文本元素字段（element_type=text） ==========
    text: Optional[str] = Field(
        default=None,
        description="文本内容（text类型使用）"
    )
    
    text_level: Optional[int] = Field(
        default=None,
        description="文本层级深度（1=一级标题，2=二级标题，仅text类型）"
    )
    
    # ========== 图片元素字段（element_type=image） ==========
    bucket_name: Optional[str] = Field(
        default=None,
        description="对象存储桶名称（image类型使用）"
    )
    
    image_file_path: Optional[str] = Field(
        default=None,
        description="图片文件路径（image类型使用）"
    )
    
    image_file_name: Optional[str] = Field(
        default=None,
        description="图片文件名（image类型使用）"
    )
    
    image_file_type: Optional[str] = Field(
        default=None,
        description="图片文件类型（png, jpg等，image类型使用）"
    )
    
    image_file_format: Optional[str] = Field(
        default=None,
        description="图片格式详细信息（image类型使用）"
    )
    
    image_file_suffix: Optional[str] = Field(
        default=None,
        description="图片文件后缀名（含.，image类型使用）"
    )
    
    image_caption: Optional[str] = Field(
        default=None,
        description="图片标题/说明（image类型使用）"
    )
    
    image_footnote: Optional[str] = Field(
        default=None,
        description="图片脚注（image类型使用）"
    )
    
    # ========== 表格元素字段（element_type=table） ==========
    table_footnote: Optional[str] = Field(
        default=None,
        description="表格脚注（table类型使用）"
    )
    
    table_body: Optional[str] = Field(
        default=None,
        description="表格主体内容（table类型使用）"
    )
    
    table_caption: Optional[str] = Field(
        default=None,
        description="表格标题/说明（table类型使用）"
    )
    
    # ========== 通用扩展字段 ==========
    extra_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="额外的元数据"
    )
    
    class Config:
        use_enum_values = True
    
    def is_text(self) -> bool:
        """判断是否为文本元素"""
        return self.element_type == ElementType.TEXT
    
    def is_image(self) -> bool:
        """判断是否为图片元素"""
        return self.element_type == ElementType.IMAGE
    
    def is_table(self) -> bool:
        """判断是否为表格元素"""
        return self.element_type == ElementType.TABLE

    def is_equation(self) -> bool:
        """判断是否为公式元素"""
        return self.element_type == ElementType.EQUATION

    def is_code_block(self) -> bool:
        """判断是否为代码块元素"""
        return self.element_type == ElementType.CODE_BLOCK
    
    def to_mysql_dict(self) -> Dict[str, Any]:
        """
        转换为 MySQL element_meta_info 表的字典格式
        
        Returns:
            MySQL 表字段字典
        """
        data = {
            "element_id": self.element_id,
            "document_id": self.document_id,
            "element_index": self.element_index,
            "page_index": self.page_index,
            "element_type": self.element_type,
            "page_position": str(self.page_position) if self.page_position else None,
        }
        
        # 文本特定字段
        if self.is_text():
            data["text_level"] = self.text_level
        
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
    
    def to_mongodb_dict(self) -> Dict[str, Any]:
        """
        转换为 MongoDB element_data 表的字典格式
        
        Returns:
            MongoDB 表字段字典
        """
        data = {
            "_id": self.element_id,
            "type": self.element_type,
            "content": {}
        }
        
        # 文本内容
        if self.is_text() and self.text:
            data["content"]["text"] = self.text
        
        # 图片内容（caption 和 footnote）
        if self.is_image():
            if self.image_caption:
                data["content"]["image_caption"] = self.image_caption
            if self.image_footnote:
                data["content"]["image_footnote"] = self.image_footnote
        
        # 表格内容
        if self.is_table():
            if self.table_body:
                data["content"]["table_body"] = self.table_body
            if self.table_caption:
                data["content"]["table_caption"] = self.table_caption
            if self.table_footnote:
                data["content"]["table_footnote"] = self.table_footnote
        
        # 额外数据
        if self.extra_data:
            data["content"].update(self.extra_data)
        
        return data


class ParseResult(BaseModel):
    """
    文件解析结果统一模型
    
    这是 FileParserService 的核心返回类型，包含了文件解析的所有信息：
    - 解析状态和错误信息
    - 统一的元素列表（文本、图片、表格）
    - 文档元数据
    - 文档语言和页数等统计信息
    
    设计原则：
    - 所有元素（文本、图片、表格）统一存储在 elements 列表中
    - 根据 element_type 区分不同类型
    - 提供三个存储方法：save_to_mysql、save_to_mongodb、save_to_minio
    """
    
    # ========== 基础信息 ==========
    user_id: str = Field(
        ...,
        description="用户ID"
    )
    
    file_id: str = Field(
        ...,
        description="文件ID（格式: file-{uuid}，业务层唯一标识）"
    )
    
    document_id: str = Field(
        ...,
        description="Document ID（格式: document-{uuid}，基于file_sha256的后台唯一标识）"
    )
    
    filename: str = Field(
        ...,
        description="文件名"
    )
    
    # ========== 解析状态 ==========
    status: ParseStatus = Field(
        default=ParseStatus.PENDING,
        description="解析状态"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息（解析失败时）"
    )
    
    # ========== 统一的元素列表 ==========
    elements: List[ElementInfo] = Field(
        default_factory=list,
        description="文档中的所有元素（文本、图片、表格），按 (page_index, element_index) 排序"
    )
    
    # ========== 文档元数据 ==========
    document_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="文档元数据（作者、创建时间等）"
    )
    
    # ========== 解析工具信息 ==========
    parse_tool: str = Field(
        default="unknown",
        description="使用的解析工具（如 mineru, pypdf）"
    )
    
    parse_quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="解析质量评分（0-1）"
    )
    
    # ========== 文档统计信息 ==========
    document_language: str = Field(
        default="unknown",
        description="文档语言（zh, en, etc.）"
    )
    
    total_pages: Optional[int] = Field(
        default=None,
        ge=0,
        description="文档总页数"
    )
    
    total_chars: int = Field(
        default=0,
        ge=0,
        description="文档总字符数"
    )
    
    # ========== 存储路径 ==========
    storage_path: Optional[str] = Field(
        default=None,
        description="文件在对象存储中的路径"
    )
    
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
        """
        判断解析是否成功
        
        Returns:
            是否成功
        """
        return self.status in [ParseStatus.SUCCESS, ParseStatus.PARTIAL_SUCCESS]
    
    # ========== 元素统计方法 ==========
    
    @property
    def total_elements(self) -> int:
        """文档总元素数量"""
        return len(self.elements)
    
    def get_elements_by_type(self, element_type: ElementType) -> List[ElementInfo]:
        """
        根据类型获取元素列表
        
        Args:
            element_type: 元素类型
        
        Returns:
            指定类型的元素列表
        """
        return [e for e in self.elements if e.element_type == element_type]
    
    @property
    def text_elements(self) -> List[ElementInfo]:
        """获取所有文本元素"""
        return self.get_elements_by_type(ElementType.TEXT)
    
    @property
    def image_elements(self) -> List[ElementInfo]:
        """获取所有图片元素"""
        return self.get_elements_by_type(ElementType.IMAGE)
    
    @property
    def table_elements(self) -> List[ElementInfo]:
        """获取所有表格元素"""
        return self.get_elements_by_type(ElementType.TABLE)
    
    def has_images(self) -> bool:
        """判断文档是否包含图片"""
        return len(self.image_elements) > 0
    
    def has_tables(self) -> bool:
        """判断文档是否包含表格"""
        return len(self.table_elements) > 0
    
    # ========== 摘要方法 ==========
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取解析结果摘要
        
        Returns:
            解析结果摘要字典
        """
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "status": self.status,
            "total_pages": self.total_pages,
            "total_elements": self.total_elements,
            "total_chars": self.total_chars,
            "text_count": len(self.text_elements),
            "image_count": len(self.image_elements),
            "table_count": len(self.table_elements),
            "language": self.document_language,
            "parse_tool": self.parse_tool,
            "parse_quality": self.parse_quality
        }
    
    def get_element_stats(self) -> Dict[str, int]:
        """
        获取元素统计信息
        
        Returns:
            元素类型统计字典
        """
        stats = {
            "text": len(self.text_elements),
            "image": len(self.image_elements),
            "table": len(self.table_elements),
            "total": self.total_elements
        }
        return stats
    
    # ========== 数据转换方法 ==========
    
    def get_mysql_data(self) -> List[Dict[str, Any]]:
        """
        获取用于 MySQL element_meta_info 表的数据列表
        
        Returns:
            MySQL 数据字典列表
        """
        return [element.to_mysql_dict() for element in self.elements]
    
    def get_mongodb_data(self) -> List[Dict[str, Any]]:
        """
        获取用于 MongoDB element_data 表的数据列表
        
        Returns:
            MongoDB 数据字典列表
        """
        # 所有类型的元素都需要存储到 MongoDB
        # 文本：text
        # 图片：image_caption, image_footnote
        # 表格：table_body, table_caption, table_footnote
        return [element.to_mongodb_dict() for element in self.elements]
    
