#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : log_config.py
@Author  : caixiongjiang
@Date    : 2025/12/30
@Function: 
    全局日志配置 - 将标准 logging 拦截到 loguru
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
import sys
import logging
from pathlib import Path
from loguru import logger
from typing import Optional


class InterceptHandler(logging.Handler):
    """
    拦截标准 logging 的 Handler，将日志重定向到 loguru
    """
    def emit(self, record: logging.LogRecord) -> None:
        # 获取对应的 loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到调用者的栈帧
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: str = "app.log",
    rotation: str = "500 MB",
    retention: str = "10 days",
    compression: str = "zip",
    enable_console: bool = True,
    enable_file: bool = True,
    diagnose: bool = False,
    backtrace: bool = True,
    enqueue: bool = True
) -> None:
    """
    配置全局日志系统，将标准 logging 拦截到 loguru
    
    :param log_level: 日志级别，可选 TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
    :param log_dir: 日志文件目录，默认为项目根目录下的 logs 文件夹
    :param log_file: 日志文件名，默认 app.log
    :param rotation: 日志轮转策略，默认 500 MB
                     支持: "500 MB", "12:00", "1 week", "10 days" 等
    :param retention: 日志保留时间，默认 10 天
                     支持: "10 days", "1 month", "1 year" 等
    :param compression: 压缩格式，默认 zip，可选 gz, bz2, xz, lzma, tar, tar.gz 等
    :param enable_console: 是否启用控制台输出，默认 True
    :param enable_file: 是否启用文件输出，默认 True
    :param diagnose: 是否启用诊断模式（显示变量值），调试时可开启，默认 False
    :param backtrace: 是否显示完整的异常堆栈，默认 True
    :param enqueue: 是否使用队列异步写入（线程安全），默认 True
    """
    
    # 移除 loguru 的默认 handler
    logger.remove()
    
    # 日志格式
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # 添加控制台输出
    if enable_console:
        logger.add(
            sys.stderr,
            format=log_format,
            level=log_level,
            colorize=True,
            backtrace=backtrace,
            diagnose=diagnose,
            enqueue=enqueue
        )
    
    # 添加文件输出
    if enable_file:
        # 确定日志目录
        if log_dir is None:
            # 默认使用项目根目录下的 logs 文件夹
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"
        else:
            log_dir = Path(log_dir)
        
        # 创建日志目录
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        log_path = log_dir / log_file
        
        # 添加文件 handler
        logger.add(
            log_path,
            format=log_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=backtrace,
            diagnose=diagnose,
            enqueue=enqueue,
            encoding="utf-8"
        )
        
        logger.info(f"日志文件配置完成: {log_path}")
    
    # 拦截标准 logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # 拦截所有使用标准 logging 的第三方库
    for logger_name in logging.root.manager.loggerDict.keys():
        logging.getLogger(logger_name).handlers = []
        logging.getLogger(logger_name).propagate = True
    
    # 配置 root logger
    logging.getLogger().handlers = [InterceptHandler()]
    logging.getLogger().setLevel(0)
    
    # 抑制第三方库的噪声日志（含 SQLAlchemy 执行 SQL 的 INFO）
    for name in [
        "sqlalchemy",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "sqlalchemy.dialects",
        "pymongo",
        "pymongo.topology",
        "pymongo.connection",
        "pymongo.serverSelection",
        "pymongo.command",
        "pymongo.pool",
        "motor",
        "urllib3",
        "urllib3.connectionpool",
        "hpack",
        "httpcore",
        "httpx",
        "multipart",
        # Kafka 客户端：aiokafka 在 DEBUG 下会逐条打印协议请求/响应
        # （ApiVersion / Metadata / JoinGroup / FetchRequest…），量极大，
        # dev 模式必须压到 WARNING，否则 start_all_workers 会持续刷屏。
        "aiokafka",
        "kafka",
        "kafka-python",
        # LiteLLM：DEBUG 下会打印完整请求 kwargs 与响应原文（embedding 的
        # 输入文本、chat 全量 messages 都在内），既刷屏又泄露数据内容；
        # "LiteLLM" 是其 verbose_logger 的 logger 名（请求/响应值日志的主源），
        # Router / Proxy 为派生 logger，"litellm" 覆盖模块命名空间。
        # WARNING 及以上的真实异常仍会透出。
        "litellm",
        "LiteLLM",
        "LiteLLM Router",
        "LiteLLM Proxy",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)
    
    logger.info(f"日志系统初始化完成 - 级别: {log_level}")


def get_logger():
    """
    获取 loguru logger 实例
    
    使用方式:
        from src.utils.log_config import get_logger
        logger = get_logger()
        logger.info("这是一条日志")
    
    :return: loguru logger 实例
    """
    return logger


def setup_logging_from_env() -> None:
    """
    根据环境变量自动配置日志系统（唯一的加载入口）。

    依据 APP_ENV 选择策略，日志级别可被 LOG_LEVEL 覆盖：
      - development: DEBUG，控制台 + 文件（dev.log），diagnose=True，backtrace=True
      - production:  INFO，控制台 + 文件（prod.log，按天轮转、保留 30 天），diagnose=False，backtrace=False
      - testing:    DEBUG，仅控制台，diagnose=True，backtrace=True

    生产环境保留控制台输出，便于 `docker logs` 查看。
    """
    from src.utils.env_manager import get_env_manager

    env = get_env_manager()
    app_env = env.get_app_env()
    log_level = env.get_log_level()

    if app_env == "production":
        setup_logging(
            log_level=log_level,
            enable_console=True,
            enable_file=True,
            log_file="prod.log",
            rotation="1 day",
            retention="30 days",
            diagnose=False,
            backtrace=False,
        )
    elif app_env == "testing":
        setup_logging(
            log_level=log_level,
            enable_console=True,
            enable_file=False,
            diagnose=True,
            backtrace=True,
        )
    else:  # development（默认）
        setup_logging(
            log_level=log_level,
            enable_console=True,
            enable_file=True,
            log_file="dev.log",
            diagnose=True,
            backtrace=True,
        )


if __name__ == "__main__":
    # 测试日志配置
    print("=" * 60)
    print("开始测试日志配置...")
    print("=" * 60)
    
    setup_logging_from_env()
    
    # 测试 1: loguru 直接使用
    logger.info("✅ 测试 1: loguru 基础日志")
    logger.info("这是 loguru 的 info 日志")
    logger.success("这是 loguru 的 success 日志")
    logger.warning("这是 loguru 的 warning 日志")
    logger.error("这是 loguru 的 error 日志")
    
    # 测试 2: 标准 logging（会被拦截到 loguru）
    logger.info("\n✅ 测试 2: 标准 logging 拦截")
    std_logger = logging.getLogger(__name__)
    std_logger.info("这是标准 logging 的 info 日志（已被拦截到 loguru）")
    std_logger.warning("这是标准 logging 的 warning 日志（已被拦截到 loguru）")
    std_logger.error("这是标准 logging 的 error 日志（已被拦截到 loguru）")
    
    # 测试 3: 异常日志（这是故意触发的异常用于演示）
    logger.info("\n✅ 测试 3: 异常日志（下面的异常是故意触发的用于演示）")
    try:
        result = 1 / 0
    except ZeroDivisionError:
        logger.exception("捕获到除零异常 - 这是正常的测试输出")
    
    logger.success("\n🎉 所有测试完成！日志配置正常工作！")
    logger.info(f"日志文件保存位置: logs/dev.log")
