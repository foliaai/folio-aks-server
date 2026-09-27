#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : base.py
@Author  : caixiongjiang
@Date    : 2026/01/06
@Function: 
    MySQL 连接管理器基类，定义统一的数据库连接管理接口
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from loguru import logger


class BaseMySQLManager(ABC):
    """MySQL 连接管理器基类（抽象类）"""
    
    def __init__(self):
        """初始化连接管理器"""
        self.engine: Optional[Engine] = None
        self.SessionLocal = None
        self._initialized: bool = False
    
    @abstractmethod
    def _create_engine(self) -> Engine:
        """创建数据库引擎（子类实现）"""
        pass
    
    @abstractmethod
    def get_db_url(self) -> str:
        """获取数据库连接 URL（子类实现）"""
        pass
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        获取数据库会话的上下文管理器
        
        使用方法:
        ```python
        with manager.get_session() as session:
            result = session.query(Model).all()
        ```
        
        Yields:
            Session: 数据库会话对象
        """
        if not self._initialized:
            raise RuntimeError("连接管理器尚未初始化，请先调用初始化方法")
        
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"数据库会话发生错误: {e}")
            raise
        finally:
            session.close()
    
    def create_database(self) -> None:
        """创建数据库（如果不存在）"""
        # SQLite 不需要显式创建数据库，MySQL Server 需要
        pass
    
    def create_tables(self) -> None:
        """创建所有表结构（如果不存在）"""
        if not self.engine:
            raise RuntimeError("数据库引擎未初始化")
        
        try:
            # 导入 Base，确保所有模型都被注册
            from src.db.mysql.models.base_model import Base
            # 导入所有 Schema 以确保它们被注册
            import src.db.mysql.models.base
            import src.db.mysql.models.extract
            import src.db.mysql.models.business
            import src.db.mysql.models.conversation
            
            Base.metadata.create_all(self.engine)
            logger.info("成功创建数据库表结构")
        except Exception as e:
            logger.error(f"创建数据库表结构失败: {e}")
            raise
    
    def init_db(self) -> None:
        """初始化数据库和表结构"""
        try:
            self.create_database()
            self.create_tables()
            logger.info("数据库初始化完成")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            logger.info("数据库连接已关闭")
    
    def __enter__(self):
        """
        支持 with 上下文管理器
        
        使用方法:
        ```python
        with get_mysql_manager("sqlite") as manager:
            with manager.get_session() as session:
                result = session.query(Model).all()
        ```
        
        Returns:
            self: 管理器实例
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文时关闭连接池
        
        Args:
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常追踪
        """
        self.close()
        return False  # 不抑制异常
    
    def health_check(self) -> bool:
        """
        健康检查：验证数据库连接是否正常
        
        Returns:
            bool: 连接正常返回 True，否则返回 False
        """
        try:
            with self.get_session() as session:
                from sqlalchemy import text
                session.execute(text("SELECT 1"))
                session.commit()
            return True
        except Exception as e:
            logger.error(f"数据库健康检查失败: {e}")
            return False
