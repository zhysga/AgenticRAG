"""
系统与资源统计路由
"""
from fastapi import APIRouter, Depends, HTTPException
import uuid
from typing import Dict, Any

from backend.utils.auth import get_current_user
from backend.dependencies import get_agent_service, get_knowledge_base_service
from backend.services.agent_service import AgentService
from backend.services.knowledge_base_service import KnowledgeBaseService
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/agents")
async def get_agent_stats(
    current_user: str = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service)
):
    try:
        stats = agent_service.get_agent_stats()
        return {
            "status": "success",
            "message": "获取智能体统计信息成功",
            "request_id": str(uuid.uuid4()),
            "data": stats
        }
    except Exception as e:
        logger.error(f"获取智能体统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge_bases")
async def get_knowledge_base_stats(
    current_user: str = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    try:
        kbs = kb_service.list_knowledge_bases()
        total_docs = 0
        total_chunks = 0
        kb_stats: Dict[str, Any] = {
            "total_kb": len(kbs),
            "knowledge_bases": []
        }
        for kb in kbs:
            s = kb_service.get_knowledge_base_stats(kb.kb_id)
            kb_stats["knowledge_bases"].append({
                "kb_id": kb.kb_id,
                "name": kb.name,
                **s
            })
            total_docs += s.get("document_count", 0)
            total_chunks += s.get("chunk_count", 0)
        kb_stats["total_documents"] = total_docs
        kb_stats["total_chunks"] = total_chunks

        return {
            "status": "success",
            "message": "获取知识库统计信息成功",
            "request_id": str(uuid.uuid4()),
            "data": kb_stats
        }
    except Exception as e:
        logger.error(f"获取知识库统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system")
async def get_system_stats(
    current_user: str = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    try:
        agent_stats = agent_service.get_agent_stats()
        kbs = kb_service.list_knowledge_bases()
        total_docs = 0
        total_chunks = 0
        for kb in kbs:
            s = kb_service.get_knowledge_base_stats(kb.kb_id)
            total_docs += s.get("document_count", 0)
            total_chunks += s.get("chunk_count", 0)

        system_stats = {
            "knowledge_bases": len(kbs),
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "agents_total": agent_stats.get("total_agents", 0),
            "agents_active": agent_stats.get("active_agents", 0)
        }

        return {
            "status": "success",
            "message": "获取系统统计信息成功",
            "request_id": str(uuid.uuid4()),
            "data": system_stats
        }
    except Exception as e:
        logger.error(f"获取系统统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))