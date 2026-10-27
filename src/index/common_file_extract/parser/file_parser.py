#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : file_parser.py
@Author  : caixiongjiang
@Date    : 2025/12/31 14:27
@Function: 
    文件解析器 - 通用文件解析入口（纯解析逻辑）
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import Dict, Optional, Union
from pathlib import Path

from loguru import logger


class FileParser:
    """
    通用文件解析器（无需实例化）
    
    功能：
    1. 根据文件扩展名自动路由到对应解析器
    2. 懒加载 Parser（按需创建）
    3. 返回标准化的解析结果
    
    使用方式：
        ```python
        # 直接调用，无需实例化
        result = await FileParser.parse("/path/to/file.pdf")
        ```
    
    职责边界：
    - ✅ 文件类型检测
    - ✅ 路由到具体 Parser
    - ✅ 纯解析逻辑
    - ❌ 数据库存储（由 FileParserService 负责）
    - ❌ 文件下载（由 FileParserService 负责）
    """
    
    # 支持的文件扩展名映射
    SUPPORTED_EXTENSIONS = {
        # PDF
        '.pdf': 'pdf',
        '.PDF': 'pdf',
        # Word
        '.docx': 'word',
        '.DOCX': 'word',
        '.doc': 'word',
        '.DOC': 'word',
        # Excel
        '.xlsx': 'excel',
        '.XLSX': 'excel',
        '.xls': 'excel',
        '.XLS': 'excel',
        # PowerPoint
        '.pptx': 'ppt',
        '.PPTX': 'ppt',
        '.ppt': 'ppt',
        '.PPT': 'ppt',
        # 文本
        '.txt': 'txt',
        '.TXT': 'txt',
        # Markdown
        '.md': 'markdown',
        '.MD': 'markdown',
        '.markdown': 'markdown',
        '.MARKDOWN': 'markdown',
        # HTML
        '.html': 'html',
        '.HTML': 'html',
        '.htm': 'html',
        '.HTM': 'html',
        # 数据
        '.json': 'json',
        '.JSON': 'json',
        '.csv': 'csv',
        '.CSV': 'csv',
    }
    
    @staticmethod
    def detect_file_type(file_path: Union[str, Path]) -> str:
        """
        检测文件类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件类型（pdf, word, excel, ppt, txt, markdown, json, csv）
            
        Raises:
            ValueError: 不支持的文件类型
        """
        file_path = Path(file_path)
        suffix = file_path.suffix
        
        if suffix not in FileParser.SUPPORTED_EXTENSIONS:
            supported_types = list(set(FileParser.SUPPORTED_EXTENSIONS.values()))
            raise ValueError(
                f"不支持的文件类型: {suffix}. "
                f"支持的类型: {supported_types}"
            )
        
        return FileParser.SUPPORTED_EXTENSIONS[suffix]
    
    @staticmethod
    async def parse(
        file_path: Union[str, Path],
        file_name: Optional[str] = None
    ) -> Dict:
        """
        通用文件解析入口（静态方法）
        
        Args:
            file_path: 文件路径
            file_name: 文件名（可选，如果不提供则从 file_path 提取）
            
        Returns:
            Dict: 解析结果
            {
                "status": "success",
                "file_name": "example.pdf",
                "file_type": "pdf",
                "total_pages": 10,
                "struct_content": {
                    "root": [...]  # 解析结果
                }
            }
            
        Raises:
            ValueError: 不支持的文件类型
            Exception: 解析失败
        """
        file_path = Path(file_path)
        if file_name is None:
            file_name = file_path.name
        
        logger.info(f"🚀 开始解析文件: {file_name}")
        
        try:
            # 1. 检测文件类型
            file_type = FileParser.detect_file_type(file_path)
            logger.debug(f"📂 文件类型: {file_type}")
            
            # 2. 根据文件类型路由到对应解析器
            if file_type == "pdf":
                parse_result = await FileParser._parse_pdf(file_path, file_name)
            elif file_type == "word":
                parse_result = await FileParser._parse_word(file_path, file_name)
            elif file_type == "excel":
                parse_result = await FileParser._parse_excel(file_path, file_name)
            elif file_type == "ppt":
                parse_result = await FileParser._parse_ppt(file_path, file_name)
            elif file_type == "txt":
                parse_result = await FileParser._parse_txt(file_path, file_name)
            elif file_type == "markdown":
                parse_result = await FileParser._parse_markdown(file_path, file_name)
            elif file_type == "html":
                parse_result = await FileParser._parse_html(file_path, file_name)
            elif file_type == "json":
                parse_result = await FileParser._parse_json(file_path, file_name)
            elif file_type == "csv":
                parse_result = await FileParser._parse_csv(file_path, file_name)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
            
            # 3. 添加通用元信息
            parse_result["status"] = "success"
            parse_result["file_name"] = file_name
            parse_result["file_type"] = file_type
            
            logger.info(
                f"✅ 文件解析完成: {file_name}, "
                f"总页数: {parse_result.get('total_pages', 0)}"
            )
            
            return parse_result
            
        except Exception as e:
            logger.error(f"❌ 文件解析失败: {file_name}, 错误: {e}")
            raise Exception(f"文件解析失败: {e}")
    
    # ========== 私有解析方法（懒加载 Parser）==========
    
    @staticmethod
    async def _parse_pdf(file_path: Path, file_name: str) -> Dict:
        """
        解析 PDF 文件
        
        Args:
            file_path: 文件路径
            file_name: 文件名
            
        Returns:
            Dict: 解析结果
        """
        from src.index.common_file_extract.parser.pdf_parser import PDFParser
        from src.client.mineru import Mineru2Client
        from src.utils.config_manager import get_config_manager
        
        # 懒加载：创建 Mineru 客户端（api_url 由 env 注入）
        config = get_config_manager()
        mineru_cfg = config.get_mineru_config()
        mineru_config = {
            "api_url": mineru_cfg.get("api_url"),
            "timeout": mineru_cfg.get("timeout", 300)
        }
        mineru_client = Mineru2Client(mineru_config=mineru_config)

        # 懒加载：创建 PDF Parser
        pdf_parser = PDFParser(
            mineru_client=mineru_client,
            max_pages_per_request=mineru_cfg.get("max_pages_per_request", 2),
            max_concurrent_requests=mineru_cfg.get("max_concurrent_requests", 5)
        )
        
        # 调用解析
        return await pdf_parser.parse(file_path, file_name)
    
    @staticmethod
    async def _parse_word(file_path: Path, file_name: str) -> Dict:
        """
        解析 Word 文件

        策略：Word → LibreOffice 转 PDF → MinerU 解析（与 PDF/PPT 链路一致，获得真实像素 bbox）

        Args:
            file_path: 文件路径
            file_name: 文件名

        Returns:
            Dict: 解析结果
        """
        from src.index.common_file_extract.parser.word_parser import WordParser
        from src.client.mineru import Mineru2Client
        from src.utils.config_manager import get_config_manager

        # 懒加载：创建 Mineru 客户端（与 PDF/PPT 解析共用）
        config = get_config_manager()
        mineru_cfg = config.get_mineru_config()
        mineru_config = {
            "api_url": mineru_cfg.get("api_url"),
            "timeout": mineru_cfg.get("timeout", 300)
        }
        mineru_client = Mineru2Client(mineru_config=mineru_config)

        # 读取 LibreOffice 配置（soffice 路径跨平台：macOS / Linux 不同）
        libreoffice_cfg = config.get_libreoffice_config()
        soffice_path = libreoffice_cfg.get("soffice_path", "soffice")
        convert_timeout = libreoffice_cfg.get("convert_timeout", 120)

        # 懒加载：创建 Word Parser
        word_parser = WordParser(
            mineru_client=mineru_client,
            max_pages_per_request=mineru_cfg.get("max_pages_per_request", 2),
            max_concurrent_requests=mineru_cfg.get("max_concurrent_requests", 5),
            soffice_path=soffice_path,
            convert_timeout=convert_timeout,
        )

        # 调用解析
        return await word_parser.parse(file_path, file_name)
    
    @staticmethod
    async def _parse_excel(file_path: Path, file_name: str) -> Dict:
        """
        解析 Excel 文件

        策略：xlsx 直读（openpyxl）；xls 先 LibreOffice 转 xlsx 再进同一链路。
        工作表=Section（sheet 名标题元素），工作表内按 TableRegion 产出表格元素。
        """
        from src.index.common_file_extract.parser.excel_parser import ExcelParser
        from src.utils.config_manager import get_config_manager

        config = get_config_manager()
        libreoffice_cfg = config.get_libreoffice_config()
        excel_parser = ExcelParser(
            soffice_path=libreoffice_cfg.get("soffice_path", "soffice"),
            convert_timeout=libreoffice_cfg.get("convert_timeout", 120),
        )
        return await excel_parser.parse(file_path, file_name)
    
    @staticmethod
    async def _parse_ppt(file_path: Path, file_name: str) -> Dict:
        """
        解析 PowerPoint 文件

        策略：PPT → LibreOffice 转 PDF → MinerU 解析（与 PDF 链路一致，获得真实像素 bbox）

        Args:
            file_path: 文件路径
            file_name: 文件名

        Returns:
            Dict: 解析结果
        """
        from src.index.common_file_extract.parser.ppt_parser import PPTParser
        from src.client.mineru import Mineru2Client
        from src.utils.config_manager import get_config_manager

        # 懒加载：创建 Mineru 客户端（与 PDF 解析共用）
        config = get_config_manager()
        mineru_cfg = config.get_mineru_config()
        mineru_config = {
            "api_url": mineru_cfg.get("api_url"),
            "timeout": mineru_cfg.get("timeout", 300)
        }
        mineru_client = Mineru2Client(mineru_config=mineru_config)

        # 读取 LibreOffice 配置（soffice 路径跨平台：macOS / Linux 不同）
        libreoffice_cfg = config.get_libreoffice_config()
        soffice_path = libreoffice_cfg.get("soffice_path", "soffice")
        convert_timeout = libreoffice_cfg.get("convert_timeout", 120)

        # 懒加载：创建 PPT Parser
        ppt_parser = PPTParser(
            mineru_client=mineru_client,
            max_pages_per_request=mineru_cfg.get("max_pages_per_request", 2),
            max_concurrent_requests=mineru_cfg.get("max_concurrent_requests", 5),
            soffice_path=soffice_path,
            convert_timeout=convert_timeout,
        )

        # 调用解析
        return await ppt_parser.parse(file_path, file_name)
    
    @staticmethod
    async def _parse_txt(file_path: Path, file_name: str) -> Dict:
        """
        解析 TXT 文件
        
        Args:
            file_path: 文件路径
            file_name: 文件名
            
        Returns:
            Dict: 解析结果
        """
        from src.index.common_file_extract.parser.txt_parser import TXTParser
        
        # 懒加载：创建 TXT Parser
        txt_parser = TXTParser()
        
        # 调用解析
        return await txt_parser.parse(file_path, file_name)
    
    @staticmethod
    async def _parse_markdown(file_path: Path, file_name: str) -> Dict:
        """
        解析 Markdown 文件
        
        Args:
            file_path: 文件路径
            file_name: 文件名
            
        Returns:
            Dict: 解析结果
        """
        from src.index.common_file_extract.parser.md_parser import MarkdownParser

        # 懒加载：创建 Markdown Parser
        md_parser = MarkdownParser()

        # 调用解析
        return await md_parser.parse(file_path, file_name)

    @staticmethod
    async def _parse_html(file_path: Path, file_name: str) -> Dict:
        """
        解析 HTML 文件

        Args:
            file_path: 文件路径
            file_name: 文件名

        Returns:
            Dict: 解析结果
        """
        from src.index.common_file_extract.parser.html_parser import HtmlParser

        # 懒加载：创建 HTML Parser
        html_parser = HtmlParser()

        # 调用解析
        return await html_parser.parse(file_path, file_name)
    
    @staticmethod
    async def _parse_json(file_path: Path, file_name: str) -> Dict:
        """
        解析 JSON 文件
        
        Args:
            file_path: 文件路径
            file_name: 文件名
            
        Returns:
            Dict: 解析结果
        """
        # JSON 解析比较简单，直接实现
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        return {
            "total_pages": 1,
            "struct_content": {
                "root": [{
                    "page_idx": 0,
                    "page_info": [{
                        "type": "text",
                        "element_index": 0,
                        "text": json.dumps(content, ensure_ascii=False, indent=2),
                        "bbox": []
                    }]
                }]
            }
        }
    
    @staticmethod
    async def _parse_csv(file_path: Path, file_name: str) -> Dict:
        """
        解析 CSV 文件

        表体序列化为 HTML（与 MinerU/excel 契约一致，引用浮层可渲染表格），
        首行即表头（split_large_table 每片重复表头）。
        """
        import csv
        import html as _html

        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        trs = [
            "<tr>" + "".join(f"<td>{_html.escape(cell)}</td>" for cell in row) + "</tr>"
            for row in rows
        ]
        caption = file_path.stem

        return {
            "total_pages": 1,
            "struct_content": {
                "root": [{
                    "page_idx": 0,
                    "page_info": [{
                        "type": "table",
                        "element_index": 0,
                        "table_body": "<table>" + "".join(trs) + "</table>",
                        "table_caption": [caption] if caption else [],
                        "bbox": []
                    }]
                }]
            }
        }
    
    # ========== 批量解析 ==========
    
    @staticmethod
    async def parse_multiple(
        file_paths: list[Union[str, Path]]
    ) -> list[Dict]:
        """
        批量解析多个文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            list[Dict]: 解析结果列表
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = await FileParser.parse(file_path)
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "failed",
                    "file_name": Path(file_path).name,
                    "error": str(e)
                })
        
        return results
