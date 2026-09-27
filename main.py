#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : main.py
@Author  : caixiongjiang
@Date    : 2025/12/29 11:40
@Function: 
    FastAPI 应用入口
    启动方式: uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
@Modify History:
    2026/02/18 - 重写为 FastAPI 应用，集成 Knowledge API 路由
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.utils.log_config import setup_logging_from_env

setup_logging_from_env()

import os

from api.routers import chat_router, knowledge_router
from src.db.kafka.connection.factory import close_kafka_manager
from src.db.mongodb.mongodb_manager import get_mongodb_manager
from src.db.mysql.connection.factory import get_mysql_manager
from src.db.milvus import get_milvus_manager
from src.db.redis.connection.factory import RedisManagerFactory


def _get_allowed_origins() -> list[str]:
    """读取允许跨域访问的前端来源地址"""
    from src.utils.env_manager import get_env_manager
    return get_env_manager().get_cors_allowed_origins()


def _init_milvus() -> None:
    """初始化 Milvus 连接并确保所有集合存在"""
    os.environ.setdefault("MILVUS_AUTO_CREATE_COLLECTION", "true")
    from src.db.milvus.repositories import (
        ChunkRepository, SectionRepository,
        EnhancedChunkRepository,
        FileSummaryRepository, SectionSummaryRepository, AtomicQARepository,
        SPORepository, TagRepository,
    )
    for cls in [
        ChunkRepository, SectionRepository,
        EnhancedChunkRepository,
        FileSummaryRepository, SectionSummaryRepository, AtomicQARepository,
        SPORepository, TagRepository,
    ]:
        try:
            cls()
        except Exception as e:
            logger.warning(f"Milvus 集合初始化跳过 {cls.__name__}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理：启动时初始化所有数据库连接，关闭时统一释放"""
    logger.info("应用启动中...")

    # 模型调用观测由自托管 LiteLLM Proxy 端写入 PostgreSQL，本地无需初始化。

    get_mysql_manager().init_db()
    logger.info("MySQL 初始化完成（自动建表）")

    from src.service.skill.registry_singleton import init_registry
    init_registry()
    logger.info("SkillRegistry 初始化完成")

    mongo_manager = await get_mongodb_manager()
    logger.info("MongoDB 初始化完成（Beanie ODM + 索引）")

    _init_milvus()
    logger.info("Milvus 初始化完成（连接 + 集合自动创建）")

    yield

    logger.info("应用关闭中，释放资源...")
    get_mysql_manager().close()
    await mongo_manager.disconnect()
    get_milvus_manager().disconnect()
    await close_kafka_manager()
    await RedisManagerFactory.close_all()
    logger.info("所有资源已释放")


app = FastAPI(
    title="Agentic Knowledge System",
    description="智能知识管理系统 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(knowledge_router)
app.include_router(chat_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
