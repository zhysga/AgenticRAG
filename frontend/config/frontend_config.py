"""
前端配置管理
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FrontendConfig:
    """前端配置"""
    
    # 后端API配置
    backend_url: str = "http://localhost:8000"
    api_key: str = ""
    timeout: int = 300
    
    # Gradio配置
    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = 7860
    gradio_share: bool = False
    gradio_debug: bool = False
    
    # 页面配置
    kb_manager_port: int = 7860
    agent_studio_port: int = 7861
    qa_collab_port: int = 7862
    
    # UI配置
    theme: str = "soft"
    title: str = "LangGraph多智能体RAG系统"
    description: str = "基于LangGraph的多智能体RAG工作流系统"
    
    # 文件上传配置 - 统一存储到 storage 目录
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: list = None
    upload_directory: str = "./storage/uploads"
    
    # 缓存配置
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1小时
    
    # 日志配置 - 统一存储到 storage/logs 目录
    log_level: str = "INFO"
    log_file: str = "./storage/logs/frontend.log"
    
    def __post_init__(self):
        """初始化后处理"""
        if self.allowed_file_types is None:
            self.allowed_file_types = [
                "text/plain",
                "text/markdown",
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword"
            ]
        
        # 确保上传目录存在
        os.makedirs(self.upload_directory, exist_ok=True)
        
        # 确保日志目录存在
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'FrontendConfig':
        """从环境变量创建配置"""
        return cls(
            backend_url=os.getenv("BACKEND_URL", "http://localhost:8000"),
            api_key=os.getenv("API_KEY", ""),
            timeout=int(os.getenv("TIMEOUT", "30")),
            gradio_server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
            gradio_server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
            gradio_share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
            gradio_debug=os.getenv("GRADIO_DEBUG", "false").lower() == "true",
            kb_manager_port=int(os.getenv("KB_MANAGER_PORT", "7860")),
            agent_studio_port=int(os.getenv("AGENT_STUDIO_PORT", "7861")),
            qa_collab_port=int(os.getenv("QA_COLLAB_PORT", "7862")),
            theme=os.getenv("THEME", "soft"),
            title=os.getenv("TITLE", "LangGraph多智能体RAG系统"),
            description=os.getenv("DESCRIPTION", "基于LangGraph的多智能体RAG工作流系统"),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", str(100 * 1024 * 1024))),
            upload_directory=os.getenv("UPLOAD_DIRECTORY", "./storage/uploads"),
            cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "./storage/logs/frontend.log")
        )
    
    @classmethod
    def from_file(cls, config_file: str) -> 'FrontendConfig':
        """从配置文件创建配置"""
        import json
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            return cls(**config_data)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "backend_url": self.backend_url,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "gradio_server_name": self.gradio_server_name,
            "gradio_server_port": self.gradio_server_port,
            "gradio_share": self.gradio_share,
            "gradio_debug": self.gradio_debug,
            "kb_manager_port": self.kb_manager_port,
            "agent_studio_port": self.agent_studio_port,
            "qa_collab_port": self.qa_collab_port,
            "theme": self.theme,
            "title": self.title,
            "description": self.description,
            "max_file_size": self.max_file_size,
            "allowed_file_types": self.allowed_file_types,
            "upload_directory": self.upload_directory,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "log_level": self.log_level,
            "log_file": self.log_file
        }
    
    def _json_default(self, obj):
        """处理JSON序列化中的特殊类型"""
        import json
        from datetime import datetime
        
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        import json
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=self._json_default)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def validate(self) -> bool:
        """验证配置"""
        try:
            # 验证端口范围
            ports = [
                self.gradio_server_port,
                self.kb_manager_port,
                self.agent_studio_port,
                self.qa_collab_port
            ]
            
            for port in ports:
                if not (1 <= port <= 65535):
                    print(f"端口 {port} 超出有效范围 (1-65535)")
                    return False
            
            # 验证URL格式
            if not self.backend_url.startswith(("http://", "https://")):
                print(f"后端URL格式错误: {self.backend_url}")
                return False
            
            # 验证文件大小
            if self.max_file_size <= 0:
                print(f"最大文件大小必须大于0: {self.max_file_size}")
                return False
            
            # 验证缓存TTL
            if self.cache_ttl <= 0:
                print(f"缓存TTL必须大于0: {self.cache_ttl}")
                return False
            
            return True
            
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False


# 默认配置实例
default_config = FrontendConfig()
