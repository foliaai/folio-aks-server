#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : env_manager.py
@Author  : caixiongjiang
@Date    : 2025/12/30 17:30
@Function: 
    环境变量管理器，负责加载和管理所有敏感信息配置
@Modify History:
         
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from loguru import logger


class EnvManager:
    """环境变量管理器"""
    
    # 默认环境变量文件路径（相对于项目根目录）
    DEFAULT_ENV_PATH = Path(__file__).parent.parent.parent / ".env"
    
    # 必需的环境变量列表（根据实际使用情况动态判断）
    _critical_vars = {
        "APP_SECRET_KEY",
        "APP_ENV",
    }
    
    def __init__(self, env_file: Optional[str] = None):
        """
        初始化环境变量管理器
        
        Args:
            env_file: .env 文件路径，如果为None则使用默认路径
        """
        self._env_file = env_file
        self._env_vars: Dict[str, str] = {}
        self._load_env()
    
    def _load_env(self) -> None:
        """加载环境变量"""
        # 确定环境变量文件路径
        if self._env_file:
            env_path = Path(self._env_file)
        else:
            env_path = self.DEFAULT_ENV_PATH
        
        # 检查文件是否存在
        if not env_path.exists():
            logger.warning(f"环境变量文件不存在: {env_path}，将仅使用系统环境变量")
            return
        
        # 加载.env文件
        load_dotenv(env_path, override=True)
        logger.info(f"已加载环境变量文件: {env_path}")
        
        # 缓存环境变量
        self._cache_env_vars()
    
    def _cache_env_vars(self) -> None:
        """缓存环境变量到内存"""
        self._env_vars = dict(os.environ)
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        获取环境变量
        
        Args:
            key: 环境变量名
            default: 默认值
            
        Returns:
            环境变量值
        """
        return os.getenv(key, default)
    
    def get_required(self, key: str) -> str:
        """
        获取必需的环境变量，如果不存在则抛出异常
        
        Args:
            key: 环境变量名
            
        Returns:
            环境变量值
            
        Raises:
            ValueError: 如果环境变量不存在
        """
        value = self.get(key)
        if value is None:
            raise ValueError(f"必需的环境变量未设置: {key}")
        return value
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        获取整数类型的环境变量
        
        Args:
            key: 环境变量名
            default: 默认值
            
        Returns:
            整数值
        """
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"环境变量 {key} 无法转换为整数: {value}，使用默认值 {default}")
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        获取布尔类型的环境变量
        
        Args:
            key: 环境变量名
            default: 默认值
            
        Returns:
            布尔值
        """
        value = self.get(key)
        if value is None:
            return default
        
        # 支持多种布尔值表示
        true_values = {"true", "1", "yes", "on", "t", "y"}
        false_values = {"false", "0", "no", "off", "f", "n"}
        
        value_lower = value.lower().strip()
        if value_lower in true_values:
            return True
        elif value_lower in false_values:
            return False
        else:
            logger.warning(f"环境变量 {key} 的值 '{value}' 无法识别为布尔值，使用默认值 {default}")
            return default
    
    def get_list(self, key: str, separator: str = ",", default: Optional[List[str]] = None) -> List[str]:
        """
        获取列表类型的环境变量
        
        Args:
            key: 环境变量名
            separator: 分隔符
            default: 默认值
            
        Returns:
            列表
        """
        value = self.get(key)
        if value is None:
            return default or []
        
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def validate_required_vars(self, required_vars: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        验证必需的环境变量是否都已设置
        
        Args:
            required_vars: 必需的环境变量列表，如果为None则使用默认列表
            
        Returns:
            验证结果字典，key为变量名，value为是否存在
        """
        if required_vars is None:
            required_vars = list(self._critical_vars)
        
        results = {}
        for var in required_vars:
            results[var] = self.get(var) is not None
        
        return results
    
    def check_health(self) -> bool:
        """
        检查环境变量健康状态
        
        Returns:
            是否健康
        """
        validation_results = self.validate_required_vars()
        missing_vars = [var for var, exists in validation_results.items() if not exists]
        
        if missing_vars:
            logger.error(f"缺少必需的环境变量: {', '.join(missing_vars)}")
            return False
        
        logger.info("环境变量健康检查通过")
        return True
    
    # ==================== 数据库认证信息获取 ====================
    
    def get_milvus_auth(self) -> Dict[str, str]:
        """获取Milvus认证信息"""
        return {
            "user": self.get("MILVUS_USER", "root"),
            "password": self.get("MILVUS_PASSWORD", ""),
            "token": self.get("MILVUS_TOKEN", ""),
        }
    
    def get_mongodb_auth(self) -> Dict[str, str]:
        """获取MongoDB认证信息"""
        uri = self.get("MONGODB_URI")
        if uri:
            return {"uri": uri}
        
        return {
            "user": self.get("MONGODB_USER", ""),
            "password": self.get("MONGODB_PASSWORD", ""),
            "auth_source": self.get("MONGODB_AUTH_SOURCE", "admin"),
        }
    
    def get_mysql_auth(self) -> Dict[str, str]:
        """获取MySQL认证信息"""
        return {
            "user": self.get("MYSQL_USER", "root"),
            "password": self.get("MYSQL_PASSWORD", ""),
        }
    
    def get_neo4j_auth(self) -> Dict[str, str]:
        """获取Neo4j认证信息"""
        return {
            "user": self.get("NEO4J_USER", "neo4j"),
            "password": self.get("NEO4J_PASSWORD", ""),
        }
    
    def get_redis_auth(self) -> Dict[str, str]:
        """获取Redis认证信息"""
        return {
            "username": self.get("REDIS_USERNAME", ""),
            "password": self.get("REDIS_PASSWORD", ""),
        }
    
    def get_minio_auth(self) -> Dict[str, str]:
        """获取MinIO认证信息"""
        return {
            "access_key": self.get("MINIO_ACCESS_KEY", "minioadmin"),
            "secret_key": self.get("MINIO_SECRET_KEY", ""),
        }
    
    def get_kafka_auth(self) -> Dict[str, Optional[str]]:
        """
        获取Kafka认证信息

        Returns:
            包含SASL和SSL认证信息的字典
        """
        return {
            # SASL 认证
            "sasl_username": self.get("KAFKA_SASL_USERNAME"),
            "sasl_password": self.get("KAFKA_SASL_PASSWORD"),
            # SSL 证书
            "ssl_cafile": self.get("KAFKA_SSL_CAFILE"),
            "ssl_certfile": self.get("KAFKA_SSL_CERTFILE"),
            "ssl_keyfile": self.get("KAFKA_SSL_KEYFILE"),
            "ssl_password": self.get("KAFKA_SSL_PASSWORD"),
        }

    # ==================== 主机/端点配置（从环境变量加载，区分开发/生产） ====================
    #
    # 所有基础设施服务的主机/端点一律由环境变量提供（.env / .env.production），
    # config.toml 不再承载任何 IP/主机相关字段。缺失时 fail-fast。

    def get_milvus_host(self) -> str:
        """获取 Milvus 主机（dev: localhost / prod: milvus-standalone）"""
        return self.get_required("MILVUS_HOST")

    def get_mongodb_host(self) -> str:
        """获取 MongoDB 主机（dev: localhost / prod: mongodb）"""
        return self.get_required("MONGODB_HOST")

    def get_mysql_host(self) -> str:
        """获取 MySQL 主机（dev: localhost / prod: services_mysql）"""
        return self.get_required("MYSQL_HOST")

    def get_neo4j_uri(self) -> str:
        """获取 Neo4j 完整 URI（如 bolt://localhost:7687 / bolt://neo4j:7687）"""
        return self.get_required("NEO4J_URI")

    def get_redis_host(self) -> str:
        """获取 Redis 主机（dev: localhost / prod: services_redis）"""
        return self.get_required("REDIS_HOST")

    def get_redis_db(self) -> int:
        """获取 Redis db 编号（dev: 3 / prod: 2）。用于 dev/prod 在同一 Redis 实例上的 db 级隔离。"""
        return int(self.get("REDIS_DB", "2"))

    def get_minio_endpoint(self) -> str:
        """获取 MinIO 端点（host:port，如 localhost:9000 / milvus-minio:9000）"""
        return self.get_required("MINIO_ENDPOINT")

    def get_kafka_bootstrap_servers(self) -> List[str]:
        """获取 Kafka bootstrap servers 列表（逗号分隔，如 kafka:9092）"""
        servers = self.get_list("KAFKA_BOOTSTRAP_SERVERS")
        if not servers:
            raise ValueError("必需的环境变量未设置: KAFKA_BOOTSTRAP_SERVERS")
        return servers

    def get_sparse_embedding_api_base(self) -> str:
        """获取 BGE-M3 稀疏向量服务 API base（如 http://localhost:18085/v1）"""
        return self.get_required("SPARSE_EMBEDDING_API_BASE")

    def get_mineru_api_url(self) -> str:
        """获取 MinerU 服务 URL（如 http://192.168.19.232:18000）"""
        return self.get_required("MINERU_API_URL")

    # ---- 数据 namespace（dev/prod 隔离；dev 用 dev_*，prod 用 default） ----
    def get_mysql_database(self) -> str:
        """获取 MySQL 数据库名（dev: dev_default / prod: default）"""
        return self.get_required("MYSQL_DATABASE")

    def get_mongodb_database(self) -> str:
        """获取 MongoDB 数据库名（dev: dev_default / prod: default）"""
        return self.get_required("MONGODB_DATABASE")

    def get_milvus_database(self) -> str:
        """获取 Milvus 数据库名（dev: dev_default / prod: default）"""
        return self.get_required("MILVUS_DATABASE")

    def get_minio_bucket(self) -> str:
        """获取 MinIO 默认桶名（dev: dev-default / prod: default；桶名不允许下划线）"""
        return self.get_required("MINIO_BUCKET")

    def get_kafka_group_id_prefix(self) -> str:
        """获取 Kafka 消费组前缀（dev: aks_dev / prod: aks）"""
        return self.get_required("KAFKA_GROUP_ID_PREFIX")

    def get_kafka_topic_prefix(self) -> str:
        """获取 Kafka Topic 名称前缀（dev: dev_ / prod: 空）。

        用于 dev/prod 在同一 Kafka 集群上的 topic 级隔离。prod 留空以保留
        已有 topic 数据；dev 加前缀后自动创建独立 topic（如 dev_knowledge_base.index.start）。
        """
        return self.get("KAFKA_TOPIC_PREFIX", "")

    # ==================== 模型网关（LiteLLM Proxy / Model Lake 统一接入） ====================
    #
    # 支持自托管 LiteLLM Proxy 以及公司自研 Model Lake（OpenAI 兼容网关）。
    #
    # 环境变量配置：
    # 1. 统一大模型网关控制：
    #    - MODEL_GATEWAY_TYPE: 网关类型（"litellm" 或 "model_lake" / "openai" / "openai_compatible"，默认 "litellm"）
    #    - MODEL_GATEWAY_TIMEOUT: 客户端默认超时秒数（默认 60）
    #    - MODEL_GATEWAY_MAX_RETRIES: 客户端默认重试次数（默认 2）
    #
    # 2. ModelNexus 网关（OpenAI 兼容协议，静态 Key）：
    #    - MODEL_LAKE_BASE   : 网关完整 base，必须配到 /v1 结尾（原样使用，不做拼接）
    #    - MODEL_LAKE_API_KEY: 静态 API Key（mn-sk- 开头，Bearer 直接携带）
    #
    # 3. LiteLLM Proxy 网关（Embedding / Reranker 专用网关，以及未配置 Model Lake 时的 LLM 网关）：
    #    - LITELLM_PROXY_URL : LiteLLM Proxy base URL
    #    - LITELLM_PROXY_KEY : LiteLLM Proxy virtual key

    def get_model_gateway_type(self) -> str:
        """获取大模型网关类型（litellm / model_lake / openai / openai_compatible），默认 litellm"""
        raw = self.get("MODEL_GATEWAY_TYPE") or "litellm"
        return raw.strip().lower()

    def get_config_profile(self) -> str:
        """配置档案：跟随网关类型。"""
        from src.utils.config_profile import resolve_config_profile
        return resolve_config_profile()

    def get_model_gateway_url(self) -> Optional[str]:
        """获取大模型网关 base URL。

        - model_lake (及 openai/openai_compatible): 取 MODEL_LAKE_BASE（必须配到 /v1 结尾，原样使用）
        - litellm: 取 LITELLM_PROXY_URL
        """
        raw_type = self.get_model_gateway_type()
        if raw_type in ("model_lake", "openai", "openai_compatible"):
            ml_base = self.get("MODEL_LAKE_BASE")
            if ml_base and ml_base.strip():
                return ml_base.strip().rstrip("/")
            return None
        return self.get("LITELLM_PROXY_URL")

    def get_model_gateway_key(self) -> Optional[str]:
        """获取大模型网关 API Key。

        - model_lake: 静态 Key ``MODEL_LAKE_API_KEY``（mn-sk- 开头）
        - litellm 返回 LITELLM_PROXY_KEY
        """
        raw_type = self.get_model_gateway_type()
        if raw_type in ("model_lake", "openai", "openai_compatible"):
            raw = self.get("MODEL_LAKE_API_KEY")
            return raw.strip() if raw and raw.strip() else None
        return self.get("LITELLM_PROXY_KEY")

    def get_model_gateway_timeout(self, default: float = 60.0) -> float:
        """客户端默认超时（秒）：``MODEL_GATEWAY_TIMEOUT``。"""
        raw = self.get("MODEL_GATEWAY_TIMEOUT")
        if raw is None or not str(raw).strip():
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning(
                f"环境变量 MODEL_GATEWAY_TIMEOUT 无法转换为数字: {raw}，使用默认值 {default}"
            )
            return default

    def get_model_gateway_max_retries(self, default: int = 2) -> int:
        """客户端默认重试次数：``MODEL_GATEWAY_MAX_RETRIES``。"""
        raw = self.get("MODEL_GATEWAY_MAX_RETRIES")
        if raw is None or not str(raw).strip():
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning(
                f"环境变量 MODEL_GATEWAY_MAX_RETRIES 无法转换为整数: {raw}，使用默认值 {default}"
            )
            return default

    def get_embedding_gateway_url(self) -> Optional[str]:
        """获取 Embedding 模型网关 URL（走 LiteLLM Proxy）"""
        return self.get("LITELLM_PROXY_URL")

    def get_embedding_gateway_key(self) -> Optional[str]:
        """获取 Embedding 模型网关 Key（走 LiteLLM Proxy）"""
        return self.get("LITELLM_PROXY_KEY")

    def get_reranker_gateway_url(self) -> Optional[str]:
        """获取 Reranker 模型网关 URL（走 LiteLLM Proxy）"""
        return self.get("LITELLM_PROXY_URL")

    def get_reranker_gateway_key(self) -> Optional[str]:
        """获取 Reranker 模型网关 Key（走 LiteLLM Proxy）"""
        return self.get("LITELLM_PROXY_KEY")

    def get_litellm_proxy_url(self) -> Optional[str]:
        """获取自托管 LiteLLM Proxy base URL（如 ``http://litellm:4000``）"""
        return self.get("LITELLM_PROXY_URL")

    def get_litellm_proxy_key(self) -> Optional[str]:
        """获取调用 LiteLLM Proxy 用的 virtual key / token"""
        return self.get("LITELLM_PROXY_KEY")

    # ==================== 本地直连 API Keys（未走模型网关的服务） ====================

    def get_sparse_embedding_api_key(self) -> Optional[str]:
        """获取稀疏向量 Embedding 服务 API Key（BGE-M3 本地部署，未走模型网关）"""
        return self.get("SPARSE_EMBEDDING_API_KEY")

    # ==================== 第三方服务 ====================

    def get_mineru_api_key(self) -> Optional[str]:
        """获取 MinerU 服务 API Key（PDF 解析）"""
        return self.get("MINERU_API_KEY")
    
    # ==================== 系统配置 ====================
    
    def get_app_env(self) -> str:
        """获取应用环境"""
        return self.get("APP_ENV", "development")
    
    def is_debug(self) -> bool:
        """是否为调试模式"""
        return self.get_bool("DEBUG", True)

    def get_log_level(self) -> str:
        """获取日志级别（dev 默认 DEBUG / prod 默认 INFO；可被 LOG_LEVEL 覆盖）。"""
        default = "INFO" if self.get_app_env() == "production" else "DEBUG"
        return self.get("LOG_LEVEL", default).upper()

    def get_cors_allowed_origins(self) -> list[str]:
        """获取允许跨域访问的前端来源列表。未设置或为空时回退到默认本地开发来源。"""
        raw = self.get("CORS_ALLOWED_ORIGINS")
        default_origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:4000",
            "http://127.0.0.1:4000",
            "http://localhost:4001",
            "http://127.0.0.1:4001",
            "http://192.168.201.14:4001",
        ]
        if not raw or not raw.strip():
            return default_origins
        origins = [item.strip() for item in raw.split(",") if item.strip()]
        return origins if origins else default_origins
    
    def get_app_secret_key(self) -> str:
        """获取应用密钥"""
        return self.get_required("APP_SECRET_KEY")


# 创建全局单例
_env_manager_instance: Optional[EnvManager] = None


def get_env_manager(env_file: Optional[str] = None) -> EnvManager:
    """
    获取环境变量管理器单例
    
    Args:
        env_file: .env文件路径
        
    Returns:
        EnvManager实例
    """
    global _env_manager_instance
    
    if _env_manager_instance is None:
        _env_manager_instance = EnvManager(env_file)
    
    return _env_manager_instance


# 便捷函数
def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """获取环境变量的便捷函数"""
    return get_env_manager().get(key, default)


def get_required_env(key: str) -> str:
    """获取必需环境变量的便捷函数"""
    return get_env_manager().get_required(key)
