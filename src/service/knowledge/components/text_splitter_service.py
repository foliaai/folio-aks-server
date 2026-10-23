#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : text_splitter_service.py
@Author  : caixiongjiang
@Date    : 2026/02/07
@Function: 
    TextSplitter Service - 文本切分服务
    
    核心职责:
    - 从 ParseEndMessage 自包含 payload 构造 ParseResult（不读数据库）
    - 执行文本切分
    - 生成 SplitResult
    
    架构说明:
    TextSplitterService (本类) → 完整流程编排
      1. 从 ParseEndMessage.elements 构造 ParseResult（自包含，不读库）
      2. 调用切分器处理文本、表格、图片、代码
      3. 生成 Section 和 Chunk
      4. 返回 SplitResult
    
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import List, Optional, Dict, Any, Tuple
from loguru import logger

from src.types.models.parse_result import ParseResult, ParseStatus, ElementInfo, ElementType
from src.types.models.split_result import (
    SplitResult,
    SplitStatus,
    SectionInfo,
    ChunkInfo
)
from src.utils.component_config_manager import get_component_config_manager
from src.utils.language_detector import detect_language
from src.index.common_file_extract.splitter.models import SplitConfig
from src.index.common_file_extract.splitter.text_splitter import TextSplitter
from src.index.common_file_extract.splitter.table_splitter import TableSplitter
from src.index.common_file_extract.splitter.element_processor import ElementProcessor
from src.index.common_file_extract.splitter.text_cleaner import TextCleaner


class TextSplitterService:
    """
    文本切分服务
    
    核心功能：
    - 从 ParseEndMessage 自包含 payload 构造 ParseResult（不读数据库）
    - 执行文本切分
    - 生成 SplitResult
    
    设计原则：
    - **不读数据库**：parse 阶段的 element 写库走 db_write.* 异步 Consumer，
      与 parse.end 的消费无顺序保证；故 split 的输入必须完全来自
      ParseEndMessage 消息体本身，避免「读库时数据尚未落盘」的竞态。
    """
    
    def __init__(self, config: Optional[SplitConfig] = None):
        """
        初始化切分服务
        
        Args:
            config: 切分配置（如果为 None，则从 config/components.json 加载）
        """
        if config is None:
            config_manager = get_component_config_manager()
            config_dict = config_manager.get_text_splitter_config()
            # 过滤掉不属于 SplitConfig 的字段（如 enabled, comment）
            valid_fields = SplitConfig.model_fields.keys()
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
            config = SplitConfig(**filtered_config)
        
        self.config = config
        self.text_splitter = TextSplitter(self.config)
        self.table_splitter = TableSplitter()
        self.element_processor = ElementProcessor()
        self.text_cleaner = TextCleaner()
    
    @staticmethod
    def _list_to_str(value: Any) -> Optional[str]:
        """将 list 转为 str（取首个元素），兼容已是 str / None 的情况。

        MinerU 的 image_caption/image_footnote/table_caption/table_footnote 在 content_list
        里是 list 形态，ElementInfo 需要 str。与历史 load 逻辑保持一致。
        """
        if isinstance(value, list) and len(value) > 0:
            return value[0]
        return value if isinstance(value, str) else None

    @staticmethod
    def build_parse_result_from_payload(message: "ParseEndMessage") -> ParseResult:
        """
        从 ParseEndMessage 的自包含 elements payload 构造 ParseResult。

        不访问任何数据库；输入完全来自消息体，消除 parse 写库异步导致的
        parse→split 读库竞态（历史问题：MySQL element_meta 已写、MongoDB
        element_data 未写时 split 读到空 text，产生空标题 section / 空 chunk）。

        Args:
            message: ParseEndMessage（含 elements 自包含字段）

        Returns:
            ParseResult（elements 已填充）
        """
        elements_data = message.elements or []
        document_id = message.document_id
        list_to_str = TextSplitterService._list_to_str

        elements: List[ElementInfo] = []
        for e in elements_data:
            element_id = e.get("element_id")
            if not element_id:
                continue
            try:
                element_type = ElementType(e.get("element_type"))
            except Exception:
                logger.warning(
                    f"build_parse_result_from_payload: 未知 element_type={e.get('element_type')}，"
                    f"按 text 兜底"
                )
                element_type = ElementType.TEXT

            elements.append(ElementInfo(
                element_id=element_id,
                document_id=document_id,
                element_index=int(e.get("element_index") or 0),
                element_type=element_type,
                page_index=e.get("page_index"),
                page_position=e.get("page_position") or None,
                # 文本特定
                text=e.get("text"),
                text_level=e.get("text_level"),
                # 图片特定
                bucket_name=e.get("bucket_name"),
                image_file_path=e.get("image_file_path"),
                image_file_name=e.get("image_file_name"),
                image_file_type=e.get("image_file_type"),
                image_file_format=e.get("image_file_format"),
                image_file_suffix=e.get("image_file_suffix"),
                image_caption=list_to_str(e.get("image_caption")),
                image_footnote=list_to_str(e.get("image_footnote")),
                # 表格特定
                table_body=e.get("table_body"),
                table_caption=list_to_str(e.get("table_caption")),
                table_footnote=list_to_str(e.get("table_footnote")),
            ))

        total_pages = message.total_pages or (
            max((el.page_index or 0) for el in elements) + 1 if elements else 0
        )
        status = ParseStatus.SUCCESS if message.status == "success" else ParseStatus.FAILED

        parse_result = ParseResult(
            user_id=message.user_id,
            file_id=message.file_id,
            document_id=document_id,
            filename=message.filename or "unknown",
            status=status,
            elements=elements,
            parse_tool=message.parse_tool or "unknown",
            total_pages=total_pages,
            total_chars=message.total_chars or 0,
            document_language=message.document_language or "unknown",
            knowledge_base_id=message.knowledge_base_id,
            knowledge_base_name=message.knowledge_base_name,
        )

        logger.info(
            f"ParseResult 从消息 payload 构造完成: elements={len(elements)}, "
            f"pages={parse_result.total_pages}"
        )
        return parse_result

    def _map_chunks_to_elements(
        self,
        merged_text: str,
        split_texts: List[str],
        element_char_ranges: List[Tuple[int, int, str]],
    ) -> List[List[str]]:
        """将每个分块精确映射到其实际包含的 element_id 列表

        通过字符位置匹配：先确定每个 split_text 在 merged_text 中的位置，
        再找出与该位置范围有交集的 element。

        Args:
            merged_text: 由 text_buffer 拼接的完整文本
            split_texts: text_splitter 切分后的文本列表
            element_char_ranges: 每个 element 在 merged_text 中的字符范围
                [(start, end, element_id), ...]

        Returns:
            per_chunk_element_ids: 每个 chunk 对应的 element_id 列表
        """
        all_element_ids = [r[2] for r in element_char_ranges]
        result: List[List[str]] = []
        cursor = 0

        for split_text in split_texts:
            search_start = max(0, cursor - self.config.chunk_overlap - 50)
            idx = merged_text.find(split_text, search_start)

            if idx == -1:
                idx = merged_text.find(split_text)

            if idx >= 0:
                chunk_start = idx
                chunk_end = idx + len(split_text)
                cursor = idx + 1

                chunk_element_ids: List[str] = []
                for elem_start, elem_end, elem_id in element_char_ranges:
                    if elem_start < chunk_end and elem_end > chunk_start:
                        chunk_element_ids.append(elem_id)

                result.append(chunk_element_ids if chunk_element_ids else list(all_element_ids))
            else:
                logger.warning(
                    f"无法在 merged_text 中定位分块文本(len={len(split_text)})，"
                    f"回退至关联所有 {len(all_element_ids)} 个 element"
                )
                result.append(list(all_element_ids))

        return result

    def _flush_text_buffer(
        self,
        text_buffer: List[str],
        buffer_element_ids: List[str],
        buffer_page_index: Optional[int],
        section_id: Optional[str],
        document_id: str,
        language: str
    ) -> List[ChunkInfo]:
        """
        将累积的文本 buffer 合并、切分并生成 Chunk

        精确溯源策略：
        1. 计算每个 element 在 merged_text 中的字符范围
        2. 切分后，根据每个 chunk 文本在 merged_text 中的位置
           匹配与之有交集的 element，得到精确的 per-chunk element_ids

        Args:
            text_buffer: 累积的已清洗文本列表
            buffer_element_ids: 累积的 Element ID 列表
            buffer_page_index: 首个文本元素的页码
            section_id: 当前 Section ID
            document_id: 文档 ID
            language: 文档语言

        Returns:
            生成的 ChunkInfo 列表（buffer 为空时返回空列表）
        """
        if not text_buffer:
            return []

        separator = "\n\n"

        element_char_ranges: List[Tuple[int, int, str]] = []
        offset = 0
        for i, text in enumerate(text_buffer):
            start = offset
            end = offset + len(text)
            element_char_ranges.append((start, end, buffer_element_ids[i]))
            offset = end + len(separator)

        merged_text = separator.join(text_buffer)
        split_texts = self.text_splitter.split_text(merged_text)

        per_chunk_element_ids = self._map_chunks_to_elements(
            merged_text, split_texts, element_char_ranges,
        )

        text_chunks = self.element_processor.create_text_chunks(
            per_chunk_element_ids=per_chunk_element_ids,
            page_index=buffer_page_index,
            section_id=section_id,
            split_texts=split_texts,
            document_id=document_id,
            language=language
        )

        logger.debug(
            f"flush文本buffer: {len(buffer_element_ids)}个元素, "
            f"合并{len(merged_text)}字符, 切分为{len(text_chunks)}个chunk"
        )

        return text_chunks
    
    async def split_document(
        self,
        parse_result: ParseResult,
        document_id: str
    ) -> SplitResult:
        """
        执行文档切分
        
        采用"文本累积 buffer"策略：连续的文本 Element 先累积到 buffer 中，
        遇到非文本元素（图片/表格/标题）或循环结束时再统一 flush（合并 + 切分）。
        这避免了连续短文本 Element 各自产生碎片 chunk 的问题。
        
        Args:
            parse_result: 解析结果
            document_id: 文档ID
        
        Returns:
            切分结果
        """
        logger.info(f"开始文档切分: file_id={parse_result.file_id}, 元素数={len(parse_result.elements)}")
        
        sections: List[SectionInfo] = []
        chunks: List[ChunkInfo] = []
        current_section_id: Optional[str] = None
        
        # ===== 文本累积 buffer 状态 =====
        text_buffer: List[str] = []
        buffer_element_ids: List[str] = []
        buffer_page_index: Optional[int] = None
        
        # ===== 遍历所有元素 =====
        for element in parse_result.elements:
            # 处理 Section（标题）—— 先 flush buffer，再创建 Section
            if element.is_text() and element.text_level and element.text_level > 0:
                chunks.extend(self._flush_text_buffer(
                    text_buffer, buffer_element_ids, buffer_page_index,
                    current_section_id, document_id, parse_result.document_language
                ))
                text_buffer.clear()
                buffer_element_ids.clear()
                buffer_page_index = None
                
                section = self.element_processor.create_section_from_element(
                    element=element,
                    document_id=document_id
                )
                sections.append(section)
                current_section_id = section.section_id
                logger.debug(f"创建Section: level={section.level}, content={section.content[:50]}...")
            
            # 处理普通文本 —— 清洗后追加到 buffer
            elif element.is_text() and element.text:
                text = element.text
                if self.config.enable_text_clean:
                    text = self.text_cleaner.clean_all(text)

                if not text:
                    continue

                text_buffer.append(text)
                buffer_element_ids.append(element.element_id)
                if buffer_page_index is None:
                    buffer_page_index = element.page_index

            # 处理公式 —— 追加到文本 buffer（LaTeX 不做文本清洗）
            elif element.is_equation() and element.text:
                text_buffer.append(element.text)
                buffer_element_ids.append(element.element_id)
                if buffer_page_index is None:
                    buffer_page_index = element.page_index

            # 处理代码块 —— flush buffer 后独立成 chunk（围栏原文，不过文本清洗，
            # 缩进结构得以保留；chunk 层 CODE_BLOCK 类型走独立检索路由）
            elif element.is_code_block() and element.text:
                chunks.extend(self._flush_text_buffer(
                    text_buffer, buffer_element_ids, buffer_page_index,
                    current_section_id, document_id, parse_result.document_language
                ))
                text_buffer.clear()
                buffer_element_ids.clear()
                buffer_page_index = None

                code_chunk = self.element_processor.create_code_block_chunk(
                    element=element,
                    section_id=current_section_id,
                    document_id=document_id,
                    language=parse_result.document_language
                )
                chunks.append(code_chunk)

                logger.debug(f"创建代码块Chunk: element_index={element.element_index}")

            # 处理图片 —— 先 flush buffer，再创建 Image Chunk
            elif element.is_image():
                chunks.extend(self._flush_text_buffer(
                    text_buffer, buffer_element_ids, buffer_page_index,
                    current_section_id, document_id, parse_result.document_language
                ))
                text_buffer.clear()
                buffer_element_ids.clear()
                buffer_page_index = None
                
                image_chunk = self.element_processor.create_image_chunk(
                    element=element,
                    section_id=current_section_id,
                    document_id=document_id,
                    language=parse_result.document_language
                )
                chunks.append(image_chunk)
                
                logger.debug(f"创建图片Chunk: image_file_name={element.image_file_name}")
            
            # 处理表格 —— 先 flush buffer，再处理表格
            elif element.is_table():
                chunks.extend(self._flush_text_buffer(
                    text_buffer, buffer_element_ids, buffer_page_index,
                    current_section_id, document_id, parse_result.document_language
                ))
                text_buffer.clear()
                buffer_element_ids.clear()
                buffer_page_index = None
                
                table_texts = self.table_splitter.assemble_and_split_table(
                    table_body=element.table_body or "",
                    table_caption=element.table_caption,
                    table_footnote=element.table_footnote,
                    chunk_size=4000
                )
                
                table_chunks = self.element_processor.create_table_chunks(
                    element=element,
                    section_id=current_section_id,
                    assembled_table_texts=table_texts,
                    document_id=document_id,
                    language=parse_result.document_language
                )
                chunks.extend(table_chunks)
                
                logger.debug(f"切分表格: 切分后={len(table_chunks)}个chunk")
        
        # ===== 循环结束，flush 剩余 buffer =====
        chunks.extend(self._flush_text_buffer(
            text_buffer, buffer_element_ids, buffer_page_index,
            current_section_id, document_id, parse_result.document_language
        ))
        
        # 更新 Section 的 chunk_id_list
        for section in sections:
            section.chunk_id_list = [
                chunk.chunk_id
                for chunk in chunks
                if chunk.section_id == section.section_id
            ]
        
        # 填充检索 / 展示双轨文本
        # - vector_text / enhanced_vector_text          → Milvus 向量化 / MongoDB.search_text
        # - display_text / enhanced_display_text        → MongoDB.text / enhanced_text
        from src.types.utils.chunk_search_text import (
            format_table_search_text_from_display,
        )

        section_content_map = {s.section_id: s.content for s in sections}
        vector_count = 0
        enhanced_count = 0
        for chunk in chunks:
            section_title = (
                section_content_map.get(chunk.section_id)
                if chunk.section_id
                else None
            )

            if chunk.is_image():
                chunk.vector_text = chunk.build_image_embedding_text(
                    section_title=section_title,
                )
                chunk.display_text = chunk.build_image_display_text(
                    section_title=section_title,
                )
            elif chunk.is_table():
                chunk.display_text = chunk.get_text_content()
                if chunk.display_text:
                    chunk.vector_text = format_table_search_text_from_display(
                        chunk.display_text,
                    )
            else:
                chunk_text = chunk.get_text_content()
                if chunk_text:
                    chunk.vector_text = chunk_text
                    chunk.display_text = chunk_text

            if chunk.vector_text:
                vector_count += 1

            if section_title and chunk.vector_text:
                chunk.enhanced_vector_text = f"{section_title}\n{chunk.vector_text}"
                enhanced_count += 1
            if section_title and chunk.display_text:
                chunk.enhanced_display_text = (
                    f"{section_title}\n{chunk.display_text}"
                )

        if vector_count > 0:
            logger.debug(
                f"填充 vector_text: {vector_count}/{len(chunks)}，"
                f"enhanced_vector_text: {enhanced_count}/{len(chunks)}"
            )

        # chunk 级语言检测：对每个 chunk 的 vector_text 跑 detect_language（Unicode 脚本
        # 统计，无第三方依赖），覆盖前面 create_*_chunk 传入的文档级语言初值。
        # - text/table/code chunk：按实际正文检测，混合语言文档可区分到 chunk 级
        # - image chunk：按 caption/footnote 拼出的 vector_text 检测
        # - vector_text 为空（理论上不应出现）：回退文档级 document_language
        # 与 SectionSummaryService 的 per-section 检测策略一致。
        detected_count = 0
        doc_lang = parse_result.document_language or "unknown"
        for chunk in chunks:
            chunk_language = detect_language(chunk.vector_text, fallback=doc_lang)
            chunk.language = chunk_language
            if chunk_language != "unknown":
                detected_count += 1
        logger.debug(
            f"chunk 语言检测完成: {detected_count}/{len(chunks)} 识别成功，"
            f"document_language={doc_lang}"
        )
        
        # 计算总字符数
        total_chars = sum(
            len(chunk.get_text_content() or "")
            for chunk in chunks
            if chunk.is_text() or chunk.is_table()
        )
        
        # 构建 SplitResult
        split_result = SplitResult(
            user_id=parse_result.user_id,
            file_id=parse_result.file_id,
            filename=parse_result.filename,
            status=SplitStatus.SUCCESS,
            sections=sections,
            chunks=chunks,
            split_method=self.config.split_method,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            total_sections=len(sections),
            total_chunks=len(chunks),
            total_chars=total_chars,
            document_language=parse_result.document_language,
            knowledge_base_id=parse_result.knowledge_base_id,
            knowledge_base_name=parse_result.knowledge_base_name
        )
        
        logger.info(
            f"文档切分完成: sections={len(sections)}, chunks={len(chunks)}, "
            f"text={len(split_result.text_chunks)}, image={len(split_result.image_chunks)}, "
            f"table={len(split_result.table_chunks)}"
        )
        
        return split_result
