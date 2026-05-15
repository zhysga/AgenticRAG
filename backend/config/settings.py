"""
后端配置设置
"""
import os
from typing import Optional, List, Dict, Any
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用设置"""
    
    # 应用配置
    app_name: str = "LangGraph多智能体RAG系统"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # 数据库配置
    database_url: str = "sqlite:///./storage/langgraph_rag.db"
    
    # 向量数据库配置 - 统一存储到 storage 目录
    chroma_persist_directory: str = "./storage/chroma_db"
    faiss_persist_directory: str = "./storage/faiss_db"
    
    # 嵌入模型配置
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "auto"  # auto, cpu, cuda
    # 本地嵌入模型路径（优先），为空则使用 embedding_model_name
    # 默认本地离线模型路径，可被环境变量 EMBEDDING_MODEL_PATH 覆盖
    embedding_model_path: Optional[str] = "./storage/hf_cache/sentence-transformers/all-MiniLM-L6-v2"
    
    # 重排序配置
    reranker_model_name: str = "BAAI/bge-reranker-base"
    reranker_enabled: bool = True  # 启用重排序
    # 本地重排序模型路径（优先），为空则使用 reranker_model_name
    reranker_model_path: Optional[str] = "./storage/hf_cache/BAAI/bge-reranker-base"
    
    # 文件存储配置 - 统一存储到 storage 目录
    knowledge_base_dir: str = "./storage/knowledge_bases"
    file_upload_dir: str = "./storage/uploads"
    chat_history_dir: str = "./storage/chat_history"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: List[str] = [
        "text/plain",
        "text/markdown",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]
    
    # 分块配置
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50
    
    # 检索配置
    retrieval_enable_hybrid: bool = True  # 启用混合检索以提升检索质量
    retrieval_top_k: int = 5  # 检索返回的最大文档数
    retrieval_similarity_threshold: float = 0.7  # 相似度阈值
    max_top_k: int = 20
    
    # 智能体配置
    default_max_turns: int = 10
    max_max_turns: int = 50
    default_temperature: float = 0.7
    
    # 认证配置
    api_key_header: str = "X-API-Key"
    secret_key: str = "your-secret-key-here"
    
    # 日志配置 - 统一存储到 storage/logs 目录
    log_level: str = "INFO"
    log_file: str = "./storage/logs/backend.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 缓存配置
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1小时
    
    # CORS配置
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # 限流配置
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 秒
    
    # 外部API配置
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    
    # 监控配置
    metrics_enabled: bool = True
    health_check_interval: int = 30

    # 离线/缓存设置（强制离线模式）
    transformers_offline: bool = True  # 强制离线模式，避免网络请求
    hf_cache_dir: Optional[str] = "./storage/hf_cache"  # HuggingFace缓存目录
    
    # Pydantic v2 配置方式
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8", 
        case_sensitive=False
    )


# 全局设置实例
settings = Settings()


def get_settings() -> Settings:
    """获取设置实例"""
    return settings


def update_settings(**kwargs) -> Settings:
    """更新设置"""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings


def validate_settings() -> bool:
    """验证设置"""
    try:
        # 验证端口范围
        if not (1 <= settings.port <= 65535):
            print(f"端口 {settings.port} 超出有效范围 (1-65535)")
            return False
        
        # 验证文件大小
        if settings.max_file_size <= 0:
            print(f"最大文件大小必须大于0: {settings.max_file_size}")
            return False
        
        # 验证分块大小
        if settings.default_chunk_size <= 0:
            print(f"默认分块大小必须大于0: {settings.default_chunk_size}")
            return False
        
        # 验证Top-K范围 (使用retrieval_top_k代替default_top_k)
        if not (1 <= settings.retrieval_top_k <= settings.max_top_k):
            print(f"检索Top-K超出有效范围: {settings.retrieval_top_k}")
            return False
        
        # 验证温度范围
        if not (0.0 <= settings.default_temperature <= 2.0):
            print(f"默认温度超出有效范围: {settings.default_temperature}")
            return False
        
        # 验证最大轮次
        if not (1 <= settings.default_max_turns <= settings.max_max_turns):
            print(f"默认最大轮次超出有效范围: {settings.default_max_turns}")
            return False
        
        return True
        
    except Exception as e:
        print(f"设置验证失败: {e}")
        return False


# 环境变量映射
ENV_MAPPING = {
    "BACKEND_HOST": "host",
    "BACKEND_PORT": "port",
    "DATABASE_URL": "database_url",
    "CHROMA_PERSIST_DIRECTORY": "chroma_persist_directory",
    "EMBEDDING_MODEL_NAME": "embedding_model_name",
    "EMBEDDING_MODEL_PATH": "embedding_model_path",
    "RERANKER_MODEL_NAME": "reranker_model_name",
    "RERANKER_MODEL_PATH": "reranker_model_path",
    "UPLOAD_DIRECTORY": "upload_directory",
    "LOG_LEVEL": "log_level",
    "LOG_FILE": "log_file",
    "OPENAI_API_KEY": "openai_api_key",
    "OPENAI_BASE_URL": "openai_base_url",
    "TRANSFORMERS_OFFLINE": "transformers_offline",
    "HF_CACHE_DIR": "hf_cache_dir"
}