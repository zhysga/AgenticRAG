"""
智能体管理路由
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
import uuid

from backend.models.agent import (
    AgentCreate, AgentCreateResponse,
    AgentListRequest, AgentListResponse,
    AgentUpdateRequest, AgentUpdateResponse,
    AgentTestRequest, AgentTestResponse,
    AgentDeleteRequest, AgentDeleteResponse
)
from backend.services.agent_service import AgentService
from backend.utils.auth import get_current_user
from backend.utils.logger import get_logger
from backend.dependencies import get_agent_service

logger = get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["智能体管理"])


@router.post("/create", response_model=AgentCreateResponse)
async def create_agent(
    agent_data: AgentCreate,
    current_user: str = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service)
):
    """创建智能体"""
    try:
        logger.info(f"创建智能体请求: {agent_data.name}")
        
        # 创建智能体
        agent_info = agent_service.create_agent(agent_data)
        
        return AgentCreateResponse(
            status="success",
            message="智能体创建成功",
            request_id=str(uuid.uuid4()),
            data=agent_info
        )
        
    except Exception as e:
        logger.error(f"创建智能体失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=AgentListResponse)
async def list_agents(
    page: int = 1,
    size: int = 10,
    agent_type: Optional[str] = None,
    routing_tags: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service)
):
    """列出智能体"""
    try:
        logger.info(f"列出智能体请求: page={page}, size={size}")
        
        # 解析路由标签
        tag_list = None
        if routing_tags:
            tag_list = [tag.strip() for tag in routing_tags.split(",") if tag.strip()]
        
        # 获取智能体列表
        agents = agent_service.list_agents(
            page=page,
            size=size,
            agent_type=agent_type,
            routing_tags=tag_list
        )
        
        # 计算总数（简化实现）
        total = len(agents)
        
        return AgentListResponse(
            status="success",
            message="获取智能体列表成功",
            request_id=str(uuid.uuid4()),
            data=agents,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
        
    except Exception as e:
        logger.error(f"列出智能体失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get/{agent_id}")
async def get_agent(
    agent_id: str,
    current_user: str = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service)
):
    """获取智能体信息"""
    try:
        logger.info(f"获取智能体请求: agent_id={agent_id}")
        
        # 获取智能体信息
        agent_info = agent_service.get_agent(agent_id)
        
        if not agent_info:
            raise HTTPException(status_code=404, detail="智能体不存在")
        
        return {
            "status": "success",
            "message": "获取智能体信息成功",
            "request_id": str(uuid.uuid4()),
            "data": agent_info.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取智能体失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update", response_model=AgentUpdateResponse)
async def update_agent(
    update_data: AgentUpdateRequest,
    current_user: str = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service)
):
    """更新智能体"""
    try:
        logger.info(f"更新智能体请求: agent_id={update_data.agent_id}")
        
        # 更新智能体
        updated_agent = agent_service.update_agent(
            agent_id=update_data.agent_id,
            updates=update_data.updates
        )
        
        if not updated_agent:
            raise HTTPException(status_code=404, detail="智能体不存在")
        
        return AgentUpdateResponse(
            status="success",
            message="智能体更新成功",
            request_id=str(uuid.uuid4()),
            data=updated_agent
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新智能体失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete", response_model=AgentDeleteResponse)
async def delete_agent(
    delete_data: AgentDeleteRequest,
    current_user: str = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service)
):
    """删除智能体"""
    try:
        logger.info(f"删除智能体请求: agent_id={delete_data.agent_id}")
        
        # 删除智能体
        success = agent_service.delete_agent(delete_data.agent_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="智能体不存在")
        
        return AgentDeleteResponse(
            status="success",
            message="智能体删除成功",
            request_id=str(uuid.uuid4()),
            data={"deleted": True}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除智能体失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test", response_model=AgentTestResponse)
async def test_agent(
    test_data: AgentTestRequest,
    current_user: str = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service)
):
    """测试智能体"""
    try:
        logger.info(f"测试智能体请求: agent_id={test_data.agent_id}")
        
        # 测试智能体
        test_result = agent_service.test_agent(
            agent_id=test_data.agent_id,
            test_query=test_data.test_query,
            test_context=test_data.test_context
        )
        
        return AgentTestResponse(
            status="success",
            message="智能体测试成功",
            request_id=str(uuid.uuid4()),
            data=test_result
        )
        
    except Exception as e:
        logger.error(f"测试智能体失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_agent_stats(
    current_user: str = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service)
):
    """获取智能体统计信息"""
    try:
        logger.info("获取智能体统计请求")
        
        # 获取统计信息
        stats = agent_service.get_agent_stats()
        
        return {
            "status": "success",
            "message": "获取统计信息成功",
            "request_id": str(uuid.uuid4()),
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available")
async def get_available_agents(
    current_user: str = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service)
):
    """获取可用智能体列表"""
    try:
        logger.info("获取可用智能体列表请求")
        
        # 获取可用智能体
        agents = agent_service.get_available_agents()
        
        return {
            "status": "success",
            "message": "获取可用智能体列表成功",
            "request_id": str(uuid.uuid4()),
            "data": agents
        }
        
    except Exception as e:
        logger.error(f"获取可用智能体列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
