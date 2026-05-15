"""
认证工具模块
"""
from fastapi import HTTPException, Depends, Header
from typing import Optional
import os


def get_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """获取API密钥"""
    # 简化实现，实际应用中应该验证API密钥
    if not x_api_key:
        # 开发环境默认用户
        return "default_user"
    
    # 这里应该验证API密钥的有效性
    # 简化实现，直接返回用户ID
    return x_api_key


def get_current_user(api_key: str = Depends(get_api_key)) -> str:
    """获取当前用户"""
    # 简化实现，实际应用中应该从API密钥解析用户信息
    return api_key


def verify_api_key(api_key: str) -> bool:
    """验证API密钥"""
    # 简化实现，实际应用中应该查询数据库或缓存
    return bool(api_key)


def get_user_permissions(user_id: str) -> list:
    """获取用户权限"""
    # 简化实现，实际应用中应该查询数据库
    return ["read", "write", "admin"]


def check_permission(user_id: str, permission: str) -> bool:
    """检查用户权限"""
    permissions = get_user_permissions(user_id)
    return permission in permissions
