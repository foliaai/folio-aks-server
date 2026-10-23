#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : file_parser_service.py
@Author  : caixiongjiang
@Date    : 2026/02/04
@Function: 
    FileParser Service - 文件解析服务（完整流程）
    
    核心职责:
    - 从对象存储下载文件
    - 调用 FileParser 进行解析（纯解析）
    - 上传图片到 MinIO
    - 向 Kafka 发送数据库写入消息
    - 清理临时文件
    - 返回标准化的 ParseResult
    
    架构说明:
    FileParserService (本类) → 完整流程编排
      1. 下载文件
      2. 保存临时文件
      3. 调用 FileParser.parse() → 纯解析，返回解析结果
      4. 上传图片到 MinIO
      5. 构建 MySQL/MongoDB 数据
      6. 向 Kafka 发送消息（不直接插入数据库）
      7. 清理临时文件
      8. 返回 ParseResult
    
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import tempfile
import uuid
import json
import base64
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from loguru import logger

from src.db.storage.manager import StorageManager
from src.index.common_file_extract.parser.file_parser import FileParser
from src.types.models.parse_result import ParseResult, ParseStatus


class FileParserService:
    """
    文件解析服务（完整流程）
    
    核心功能:
    1. 从对象存储下载文件
    2. 保存到临时文件
    3. 调用 FileParser.parse() 进行解析（纯解析，不存储）
    4. 上传图片到 MinIO
    5. 构建 MySQL/MongoDB 数据
    6. 向 Kafka 发送数据库写入消息
    7. 清理临时文件
    8. 返回标准化的 ParseResult
    
    使用方式:
        ```python
        async with StorageManager() as storage:
            service = FileParserService(storage_manager=storage)
            result = await service.parse_file(
                user_id="user1",
                file_id="file1",
                filename="doc.pdf",
                storage_path="bucket/path/to/file.pdf",
                knowledge_base_id="kb1",
                knowledge_base_name="我的知识库"
            )
            
            # 获取构建好的数据库消息
            mysql_data = result.mysql_messages
            mongodb_data = result.mongodb_messages
        ```
    """
    
    def __init__(self, storage_manager: StorageManager):
        """
        初始化文件解析服务
        
        Args:
            storage_manager: 对象存储管理器(必须)
        """
        self.storage_manager = storage_manager
        logger.info("FileParserService 初始化完成")
    
    # ========== 核心接口 ==========
    
    async def parse_file(
        self,
        user_id: str,
        file_id: str,
        filename: str,
        storage_path: str,
        knowledge_base_id: str,
        knowledge_base_name: str,
        document_id: str,
        session_id: Optional[str] = None,
        parent_knowledge_base_id: Optional[str] = None,
        parent_knowledge_base_name: Optional[str] = None,
        knowledge_type: Optional[str] = None,
        creator: str = "system",
        store_images: bool = True
    ) -> Tuple[ParseResult, List[Dict], List[Dict], List[Dict]]:
        """
        解析文件(核心方法)
        
        完整流程:
        1. 从对象存储下载文件
        2. 保存到临时文件
        3. 调用 FileParser.parse() 进行解析
        4. 上传图片到 MinIO
        5. 构建 MySQL/MongoDB 数据 + 自包含 element payload
        6. 清理临时文件
        7. 返回 ParseResult、数据库消息、elements_payload
        
        Args:
            user_id: 用户ID
            file_id: 文件ID (格式: file-{uuid}，业务层唯一标识)
            filename: 文件名
            storage_path: 文件在对象存储中的路径(格式: bucket/path/to/file)
            knowledge_base_id: 知识库ID
            knowledge_base_name: 知识库名称
            document_id: Document ID (格式: document-{uuid}，基于file_sha256的后台唯一标识)
            session_id: 会话ID(可选,用于图片上传路径)
            parent_knowledge_base_id: 父知识库ID(可选)
            parent_knowledge_base_name: 父知识库名称(可选)
            knowledge_type: 知识类型(可选)
            creator: 创建者
            store_images: 是否上传图片到 MinIO
            
        Returns:
            Tuple[ParseResult, List[Dict], List[Dict], List[Dict]]:
                - ParseResult: 解析结果
                - MySQL消息列表
                - MongoDB消息列表
                - elements_payload（供 ParseEndMessage 自包含携带，消除 parse→split 读库竞态）
            
        Raises:
            Exception: 下载、解析或处理失败
        """
        logger.info(f"开始解析文件: user_id={user_id}, file_id={file_id}, filename={filename}")
        
        # 初始化结果对象
        result = ParseResult(
            user_id=user_id,
            file_id=file_id,
            document_id=document_id,
            filename=filename,
            status=ParseStatus.PENDING,
            storage_path=storage_path,
            knowledge_base_id=knowledge_base_id,
            knowledge_base_name=knowledge_base_name
        )
        
        temp_file_path = None
        mysql_messages = []
        mongodb_messages = []
        elements_payload: List[Dict[str, Any]] = []
        
        try:
            # 1. 下载文件
            logger.debug(f"下载文件: {storage_path}")
            file_bytes = await self._download_file(storage_path)
            logger.debug(f"文件下载成功: {len(file_bytes)} bytes")

            # 2. 保存到临时文件
            temp_file_path = self._save_to_temp_file(file_bytes, filename)
            logger.debug(f"临时文件创建: {temp_file_path}")

            # 3. 调用 FileParser 进行解析（纯解析，不存储）
            logger.debug("开始解析文件...")
            parse_result = await FileParser.parse(
                file_path=temp_file_path,
                file_name=filename
            )
            logger.debug("文件解析完成")

            # 3.1 Word/PPT 等 Office 转换 PDF：上传 MinIO，供前端溯源预览
            await self._persist_converted_pdf_if_needed(
                parse_result=parse_result,
                user_id=user_id,
                file_id=file_id,
                session_id=session_id,
                source_storage_path=storage_path,
            )
            
            # 4. 构建知识库信息
            knowledge_base_info = {
                "knowledge_base_id": knowledge_base_id,
                "knowledge_base_name": knowledge_base_name,
                "parent_knowledge_base_id": parent_knowledge_base_id,
                "parent_knowledge_base_name": parent_knowledge_base_name,
                "knowledge_type": knowledge_type
            }
            
            # 5. 处理解析结果：上传图片 + 构建数据库消息 + 构建 element payload
            logger.debug("处理解析结果...")
            mysql_messages, mongodb_messages, elements_payload = await self._process_parse_result(
                parse_result=parse_result,
                user_id=user_id,
                file_id=file_id,
                document_id=document_id,
                session_id=session_id,
                knowledge_base_info=knowledge_base_info,
                creator=creator,
                store_images=store_images
            )
            logger.debug(
                f"消息构建完成: MySQL={len(mysql_messages)}, MongoDB={len(mongodb_messages)}, "
                f"elements_payload={len(elements_payload)}"
            )
            
            # 6. 转换为标准 ParseResult
            result = self._convert_to_parse_result(
                parse_result=parse_result,
                mysql_count=len(mysql_messages),
                mongodb_count=len(mongodb_messages),
                user_id=user_id,
                file_id=file_id,
                document_id=document_id,
                filename=filename,
                storage_path=storage_path,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name
            )
            
            logger.info(f"文件解析成功: {result.get_summary()}")
            return result, mysql_messages, mongodb_messages, elements_payload
            
        except Exception as e:
            # 错误处理
            error_msg = f"解析文件失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.status = ParseStatus.FAILED
            result.error_message = error_msg
            return result, mysql_messages, mongodb_messages, elements_payload
            
        finally:
            # 7. 清理临时文件
            if temp_file_path:
                self._cleanup_temp_file(temp_file_path)
    

    async def _persist_converted_pdf_if_needed(
        self,
        parse_result: Dict,
        user_id: str,
        file_id: str,
        session_id: Optional[str],
        source_storage_path: str,
    ) -> None:
        """
        若解析结果携带 Office 转换 PDF 临时路径，则上传到 MinIO 并写回 parse_result。

        路径与原生 PDF 上传对齐：{user}/{session}/{folder}/{file_id}.pdf
        folder / session 优先从原始 storage_path 解析，保证与原件同目录。

        成功后设置：
        - converted_pdf_storage_path: MinIO 完整路径
        同时清理本地临时 PDF。
        """
        temp_path = parse_result.pop("converted_pdf_temp_path", None)
        if not temp_path:
            return

        pdf_path = Path(temp_path)
        try:
            if not pdf_path.exists():
                logger.warning(f"转换 PDF 临时文件不存在，跳过持久化: {temp_path}")
                return

            pdf_bytes = pdf_path.read_bytes()
            if not pdf_bytes:
                logger.warning(f"转换 PDF 为空，跳过持久化: {temp_path}")
                return

            # 线性化后更利于 /raw 渐进加载
            try:
                from src.utils.pdf_linearize import maybe_linearize
                import asyncio
                pdf_bytes = await asyncio.to_thread(maybe_linearize, pdf_bytes, "pdf")
            except Exception as e:
                logger.warning(f"转换 PDF 线性化失败，使用原字节: {e}")

            effective_session_id, folder_path = self._resolve_converted_pdf_location(
                source_storage_path=source_storage_path,
                user_id=user_id,
                session_id=session_id,
                file_id=file_id,
            )
            storage_path = await self.storage_manager.upload_converted_pdf(
                pdf_bytes=pdf_bytes,
                user_id=user_id,
                session_id=effective_session_id,
                file_id=file_id,
                folder_path=folder_path,
            )
            parse_result["converted_pdf_storage_path"] = storage_path
            logger.info(
                f"转换 PDF 已持久化: file_id={file_id}, path={storage_path}"
            )
        except Exception as e:
            # 持久化失败不阻断解析主流程；前端将无法对 Office 做 PDF 预览高亮
            logger.error(
                f"转换 PDF 持久化失败（不阻断解析）: file_id={file_id}, error={e}",
                exc_info=True,
            )
        finally:
            self._cleanup_converted_pdf_temp(pdf_path)

    @staticmethod
    def _resolve_converted_pdf_location(
        source_storage_path: str,
        user_id: str,
        session_id: Optional[str],
        file_id: str,
    ) -> tuple[str, str]:
        """
        从原始文件 storage_path 解析 session_id 与 folder_path。

        原始路径格式：bucket/{user_id}/{session_id}/[folder/]{file_id}{suffix}
        解析失败时回退 session_id or "default"，folder_path="/"。
        """
        fallback_session = session_id or "default"
        try:
            parts = source_storage_path.split("/", 1)
            object_path = parts[1] if len(parts) > 1 else parts[0]
            prefix = f"{user_id}/"
            if not object_path.startswith(prefix):
                return fallback_session, "/"

            rest = object_path[len(prefix):]  # {session}/[folder/]{file_id}.ext
            segments = rest.split("/")
            if len(segments) < 2:
                return fallback_session, "/"

            resolved_session = segments[0] or fallback_session
            middle = segments[1:-1]
            if middle:
                return resolved_session, "/" + "/".join(middle) + "/"
            return resolved_session, "/"
        except Exception:
            return fallback_session, "/"

    def _cleanup_converted_pdf_temp(self, pdf_path: Path) -> None:
        """清理 Office→PDF 转换产生的临时文件及其空目录。"""
        try:
            out_dir = pdf_path.parent
            if pdf_path.exists():
                pdf_path.unlink()
            if out_dir.exists() and out_dir.name.startswith(
                ("ppt_convert_", "word_convert_", "office_convert_")
            ) and not any(out_dir.iterdir()):
                out_dir.rmdir()
        except Exception as e:
            logger.debug(f"清理转换 PDF 临时文件失败（可忽略）: {e}")

    # ========== 私有方法: 文件下载 ==========
    
    async def _download_file(self, storage_path: str) -> bytes:
        """
        从对象存储下载文件
        
        Args:
            storage_path: 存储路径(格式: bucket/path/to/file)
            
        Returns:
            bytes: 文件字节内容
            
        Raises:
            Exception: 下载失败
        """
        try:
            return await self.storage_manager.download_file(storage_path)
        except Exception as e:
            raise Exception(f"下载文件失败: {e}")
    
    def _save_to_temp_file(self, file_bytes: bytes, filename: str) -> str:
        """
        保存字节内容到临时文件
        
        Args:
            file_bytes: 文件字节内容
            filename: 原始文件名(用于保留扩展名)
            
        Returns:
            str: 临时文件路径
            
        Raises:
            Exception: 保存失败
        """
        try:
            # 获取文件扩展名
            suffix = Path(filename).suffix
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                prefix="file_parser_"
            ) as tmp_file:
                tmp_file.write(file_bytes)
                temp_path = tmp_file.name
            
            return temp_path
            
        except Exception as e:
            raise Exception(f"保存临时文件失败: {e}")
    
    def _cleanup_temp_file(self, temp_file_path: str):
        """
        清理临时文件
        
        Args:
            temp_file_path: 临时文件路径
        """
        try:
            Path(temp_file_path).unlink(missing_ok=True)
            logger.debug(f"临时文件已清理: {temp_file_path}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")
    
    # ========== 私有方法: 处理解析结果 ==========
    
    async def _process_parse_result(
        self,
        parse_result: Dict,
        user_id: str,
        file_id: str,
        document_id: str,
        session_id: Optional[str],
        knowledge_base_info: Dict[str, Any],
        creator: str,
        store_images: bool
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        处理解析结果：上传图片 + 构建数据库消息 + 构建自包含 element payload
        
        Args:
            parse_result: FileParser.parse() 的返回结果
            user_id: 用户ID
            file_id: 文件ID (格式: file-{uuid})
            document_id: Document ID (格式: document-{uuid}，基于file_sha256)
            session_id: 会话ID
            knowledge_base_info: 知识库信息
            creator: 创建者
            store_images: 是否上传图片
            
        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]:
                MySQL消息列表, MongoDB消息列表, elements_payload（供 ParseEndMessage 自包含携带）
        """
        struct_content = parse_result.get("struct_content", {})
        root_pages = struct_content.get("root", [])
        
        mysql_messages = []
        mongodb_messages = []
        elements_payload: List[Dict[str, Any]] = []
        
        # 遍历每一页
        for page_data in root_pages:
            page_idx = page_data.get("page_idx")
            page_info_list = page_data.get("page_info", [])
            
            # 遍历每个元素
            for element in page_info_list:
                element_id = "element-" + str(uuid.uuid4())
                element_type = element.get("type")
                bbox = element.get("bbox", [])
                element_index = element.get("element_index", 0)
                
                # 处理图片上传（如果需要）
                bucket_name = None
                image_file_path = None
                if store_images and element_type == "image" and session_id:
                    bucket_name, image_file_path = await self._upload_image(
                        element=element,
                        user_id=user_id,
                        file_id=file_id,
                        session_id=session_id,
                        element_id=element_id
                    )
                
                # 构建 MySQL 消息
                mysql_message = self._build_mysql_message(
                    element_id=element_id,
                    element_type=element_type,
                    element_index=element_index,
                    page_idx=page_idx,
                    bbox=bbox,
                    element=element,
                    knowledge_base_info=knowledge_base_info,
                    creator=creator,
                    bucket_name=bucket_name,
                    image_file_path=image_file_path,
                    document_id=document_id
                )
                mysql_messages.append(mysql_message)
                
                # 构建 MongoDB 消息
                mongodb_message = self._build_mongodb_message(
                    element_id=element_id,
                    element_type=element_type,
                    element=element
                )
                mongodb_messages.append(mongodb_message)

                # 构建自包含 element payload（供 ParseEndMessage 携带，split 阶段直接消费，不读库）
                # 内容字段保留 MinerU 原始形态（image_caption/footnote、table_caption/footnote 为 list），
                # 由下游 build_parse_result_from_payload 统一做 list→str 转换。
                elements_payload.append({
                    "element_id": element_id,
                    "element_index": element_index,
                    "page_index": page_idx,
                    "element_type": element_type,
                    "page_position": bbox or None,
                    "text_level": mysql_message.get("text_level"),
                    "bucket_name": bucket_name,
                    "image_file_path": image_file_path,
                    "image_file_name": mysql_message.get("image_file_name"),
                    "image_file_type": mysql_message.get("image_file_type"),
                    "image_file_format": mysql_message.get("image_file_format"),
                    "image_file_suffix": mysql_message.get("image_file_suffix"),
                    "text": element.get("text", ""),
                    "image_caption": element.get("image_caption"),
                    "image_footnote": element.get("image_footnote"),
                    "table_body": element.get("table_body", ""),
                    "table_caption": element.get("table_caption"),
                    "table_footnote": element.get("table_footnote"),
                })
        
        return mysql_messages, mongodb_messages, elements_payload
    
    async def _upload_image(
        self,
        element: Dict,
        user_id: str,
        file_id: str,
        session_id: str,
        element_id: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        上传图片到 MinIO
        
        Args:
            element: 元素数据
            user_id: 用户ID
            file_id: 文件ID
            session_id: 会话ID
            element_id: 元素ID
            
        Returns:
            Tuple[bucket_name, image_path]: 桶名称和图片路径
        """
        try:
            # 从 element 中提取图片数据
            image_base64 = element.get("image_base64")
            if not image_base64:
                return None, None
            
            # 解码 base64
            image_bytes = base64.b64decode(image_base64)
            
            # 生成图片文件名
            img_path = element.get("img_path", "")
            if img_path:
                image_name = Path(img_path).name
            else:
                image_name = f"{element_id}.png"
            
            # 上传到 MinIO
            storage_path = await self.storage_manager.upload_image(
                image_bytes=image_bytes,
                user_id=user_id,
                session_id=session_id,
                file_id=file_id,
                image_name=image_name
            )
            
            # 解析 storage_path: bucket/path/to/image
            parts = storage_path.split("/", 1)
            bucket_name = parts[0] if len(parts) > 1 else "knowledge-files"
            image_path = parts[1] if len(parts) > 1 else storage_path
            
            logger.debug(f"图片上传成功: {storage_path}")
            return bucket_name, image_path
            
        except Exception as e:
            logger.error(f"图片上传失败: {e}")
            return None, None
    
    def _build_mysql_message(
        self,
        element_id: str,
        element_type: str,
        element_index: int,
        page_idx: int,
        bbox: list,
        element: Dict,
        knowledge_base_info: Dict,
        creator: str,
        bucket_name: Optional[str] = None,
        image_file_path: Optional[str] = None,
        document_id: str = ""
    ) -> Dict:
        """构建 MySQL 插入消息"""
        # MinerU bbox 原样入库：[x0,y0,x1,y1]，0~1000 归一化，左上角原点
        page_position = None
        if bbox and len(bbox) == 4:
            page_position = json.dumps(bbox)
        
        # 提取 text_level（仅 text 类型）
        text_level = element.get("text_level") if element_type == "text" else None
        
        # 提取图片相关字段（仅 image 类型）
        image_file_name = None
        image_file_type = None
        image_file_format = None
        image_file_suffix = None
        
        if element_type == "image":
            img_path = element.get("img_path", "")
            if img_path:
                path_obj = Path(img_path)
                image_file_name = path_obj.name
                image_file_suffix = path_obj.suffix
                image_file_type = path_obj.suffix.lstrip('.')
        
        # 构建消息
        message = {
            "element_id": element_id,
            "document_id": document_id,
            "element_index": element_index,
            "page_index": page_idx,
            "element_type": element_type,
            "page_position": page_position,
            "text_level": text_level,
            "bucket_name": bucket_name,
            "image_file_path": image_file_path,
            "image_file_name": image_file_name,
            "image_file_type": image_file_type,
            "image_file_format": image_file_format,
            "image_file_suffix": image_file_suffix,
            # KnowledgeMixin 字段
            "knowledge_base_id": knowledge_base_info.get("knowledge_base_id"),
            "knowledge_base_name": knowledge_base_info.get("knowledge_base_name"),
            "parent_knowledge_base_id": knowledge_base_info.get("parent_knowledge_base_id"),
            "parent_knowledge_base_name": knowledge_base_info.get("parent_knowledge_base_name"),
            "knowledge_type": knowledge_base_info.get("knowledge_type"),
            # BaseModel 字段
            "creator": creator,
            "updater": creator
        }
        
        return message
    
    def _build_mongodb_message(
        self,
        element_id: str,
        element_type: str,
        element: Dict
    ) -> Dict:
        """构建 MongoDB 插入消息"""
        # 根据类型提取内容
        content = {}
        
        if element_type == "text":
            content = {
                "text": element.get("text", "")
            }
        elif element_type == "image":
            content = {
                "image_caption": element.get("image_caption", []),
                "image_footnote": element.get("image_footnote", [])
            }
        elif element_type == "table":
            content = {
                "table_caption": element.get("table_caption", []),
                "table_footnote": element.get("table_footnote", []),
                "table_body": element.get("table_body", "")
            }
        elif element_type == "equation":
            content = {
                "text": element.get("text", ""),
                "text_format": element.get("text_format", "")
            }
        elif element_type == "code_block":
            content = {
                "text": element.get("text", ""),
                "code_language": element.get("code_language", "")
            }
        elif element_type == "discarded":
            content = {
                "text": element.get("text", "")
            }
        else:
            content = {
                "text": element.get("text", "")
            }
        
        # 构建消息
        message = {
            "_id": element_id,
            "type": element_type,
            "content": content
        }
        
        return message
    
    # ========== 私有方法: 数据转换 ==========
    
    def _convert_to_parse_result(
        self,
        parse_result: Dict,
        mysql_count: int,
        mongodb_count: int,
        user_id: str,
        file_id: str,
        document_id: str,
        filename: str,
        storage_path: str,
        knowledge_base_id: str,
        knowledge_base_name: str
    ) -> ParseResult:
        """
        将解析结果转换为标准 ParseResult
        
        Args:
            parse_result: FileParser.parse() 的返回结果
            mysql_count: MySQL 消息数量
            mongodb_count: MongoDB 消息数量
            user_id: 用户ID
            file_id: 文件ID
            filename: 文件名
            storage_path: 存储路径
            knowledge_base_id: 知识库ID
            knowledge_base_name: 知识库名称
            
        Returns:
            ParseResult: 标准解析结果
        """
        # 解析状态
        status = ParseStatus.SUCCESS if parse_result.get("status") == "success" else ParseStatus.FAILED
        
        # 提取统计信息
        total_pages = parse_result.get("total_pages", 0)

        # 检测文档主语言：MinerU 输出不带语言信息，从全文 md_content 按脚本统计推断。
        # 采样前 20000 字符以控制开销；结果沿 ParseEndMessage → split → section_summary 透传。
        from src.utils.language_detector import detect_language
        md_content = parse_result.get("content", "") or ""
        document_language = detect_language(md_content[:20000], fallback="unknown")
        if document_language == "unknown":
            logger.warning("文档语言检测未识别到可识别脚本，回退 unknown")
        else:
            logger.debug(f"文档语言检测: {document_language} (sample_chars={min(len(md_content), 20000)})")

        # 构建 ParseResult
        document_metadata = {}
        converted_pdf_storage_path = parse_result.get("converted_pdf_storage_path")
        if converted_pdf_storage_path:
            document_metadata["converted_pdf_storage_path"] = converted_pdf_storage_path
        if parse_result.get("file_type"):
            document_metadata["file_type"] = parse_result.get("file_type")

        result = ParseResult(
            user_id=user_id,
            file_id=file_id,
            document_id=document_id,
            filename=filename,
            status=status,
            elements=[],  # 数据将通过 Kafka 异步写入数据库
            document_metadata=document_metadata,
            # parser_source 标记解析来源（markdown 内联解析产 "markdown"），
            # 沿 ParseEndMessage → split → SplitEndMessage 透传，section_summary
            # 建树阶段据此分派策略；未标记的（MinerU 链路）默认 "mineru"
            parse_tool=parse_result.get("parser_source") or "mineru",
            total_pages=total_pages,
            total_chars=len(md_content),
            document_language=document_language,
            storage_path=storage_path,
            knowledge_base_id=knowledge_base_id,
            knowledge_base_name=knowledge_base_name
        )
        
        logger.debug(
            f"解析结果转换完成: status={status}, "
            f"mysql_messages={mysql_count}, "
            f"mongodb_messages={mongodb_count}"
        )
        
        return result
