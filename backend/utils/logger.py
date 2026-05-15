"""
日志工具模块 - 统一使用 storage/logs 目录
"""
import logging
import json
import os
from datetime import datetime
from typing import Optional
from pathlib import Path


class ExtendedFormatter(logging.Formatter):
    """在基础格式后追加 extra 字段(JSON)的日志格式化器"""
    RESERVED = {
        'name','msg','args','levelname','levelno','pathname','filename','module',
        'exc_info','exc_text','stack_info','lineno','funcName','created','msecs',
        'relativeCreated','thread','threadName','process','processName','asctime'
    }

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        try:
            extras = {k: v for k, v in record.__dict__.items() if k not in self.RESERVED}
            if extras:
                try:
                    extras_json = json.dumps(extras, ensure_ascii=False, default=str)
                except Exception:
                    extras_json = str({k: str(v) for k, v in extras.items()})
                return f"{base} | extras={extras_json}"
        except Exception:
            pass
        return base


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """设置日志器"""
    
    # 默认格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    # 不向上级（root）传播，避免终端输出
    logger.propagate = False
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = ExtendedFormatter(format_string)
    
    # 关闭控制台输出：不添加 StreamHandler
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取带命名空间的日志器，确保归属于 'langgraph_rag' 树以继承其处理器"""
    namespace = f"langgraph_rag.{name}" if not name.startswith("langgraph_rag") else name
    return logging.getLogger(namespace)


class LoggerMixin:
    """日志混入类"""
    
    @property
    def logger(self) -> logging.Logger:
        """获取日志器"""
        return get_logger(self.__class__.__name__)


# 默认日志配置 - 统一存储到 storage/logs 目录
DEFAULT_LOG_LEVEL = "INFO"

# 计算项目根目录并设置日志路径
_PROJECT_ROOT = Path(__file__).parent.parent.parent  # backend/utils/logger.py -> 项目根目录
_STORAGE_LOGS_DIR = _PROJECT_ROOT / "storage" / "logs"
_STORAGE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = str(_STORAGE_LOGS_DIR / "backend.log")

# 设置根日志器
root_logger = setup_logger(
    "langgraph_rag",
    level=DEFAULT_LOG_LEVEL,
    log_file=DEFAULT_LOG_FILE
)
