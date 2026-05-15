"""验证器工具"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

def validate_dict(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """验证字典是否包含必需的键"""
    return all(key in data for key in required_keys)

def validate_list(data: Any, min_length: int = 0, max_length: Optional[int] = None) -> bool:
    """验证列表数据"""
    if not isinstance(data, list):
        return False
    
    if len(data) < min_length:
        return False
    
    if max_length is not None and len(data) > max_length:
        return False
    
    return True
