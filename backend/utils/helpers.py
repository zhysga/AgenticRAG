"""辅助函数"""
import json
import hashlib
from typing import Any, Dict, List
from datetime import datetime

def generate_id(prefix: str = "") -> str:
    """生成唯一ID"""
    timestamp = datetime.now().isoformat()
    hash_object = hashlib.md5(timestamp.encode())
    unique_id = hash_object.hexdigest()[:12]
    return f"{prefix}{unique_id}" if prefix else unique_id

def safe_json_loads(data: str, default: Any = None) -> Any:
    """安全地加载JSON数据"""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default
