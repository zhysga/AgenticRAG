"""
系统配置管理路由
"""
from fastapi import APIRouter, Depends, HTTPException
import uuid
from typing import Dict, Any

from backend.utils.auth import get_current_user
from backend.config.settings import get_settings, update_settings, Settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/get")
async def get_config(current_user: str = Depends(get_current_user)):
    try:
        s: Settings = get_settings()
        return {
            "status": "success",
            "message": "获取配置成功",
            "request_id": str(uuid.uuid4()),
            "data": s.model_dump()
        }
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update")
async def update_config(payload: Dict[str, Any], current_user: str = Depends(get_current_user)):
    try:
        s = update_settings(**payload)
        return {
            "status": "success",
            "message": "更新配置成功",
            "request_id": str(uuid.uuid4()),
            "data": s.model_dump()
        }
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))