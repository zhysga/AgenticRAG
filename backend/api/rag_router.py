"""
RAG检索路由
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
import uuid
import logging
from pydantic import BaseModel

from backend.services.rag_service import RAGService
from backend.dependencies import get_rag_service
from backend.utils.auth import get_current_user
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG检索"])


class RAGQueryRequest(BaseModel):
    query: str
    kb_ids: List[str]
    top_k: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = {}


class RAGQueryResponse(BaseModel):
    status: str
    message: str
    request_id: str
    data: Optional[Dict[str, Any]] = None


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    current_user: str = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """执行RAG检索查询"""
    try:
        logger.info(f"RAG查询请求: query='{request.query[:50]}...', kb_ids={request.kb_ids}")
        
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="查询内容不能为空")
        
        if not request.kb_ids:
            raise HTTPException(status_code=400, detail="必须指定知识库ID")
        
        # 执行检索
        results = await rag_service.retrieve(
            query=request.query,
            kb_ids=request.kb_ids,
            filters=request.filters or {},
            top_k=request.top_k or 5
        )
        
        # 重排序（如果有重排序器）
        if results:
            results = rag_service.rerank(request.query, results)
        
        logger.info(f"RAG查询完成，返回{len(results)}个结果")
        
        return RAGQueryResponse(
            status="success",
            message="RAG查询成功",
            request_id=str(uuid.uuid4()),
            data={
                "query": request.query,
                "results": results,
                "result_count": len(results)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"RAG查询失败: {str(e)}")


@router.get("/query")
async def rag_query_get(
    query: str,
    kb_ids: str,  # 逗号分隔的KB ID列表
    top_k: Optional[int] = 5,
    current_user: str = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """执行RAG检索查询（GET方法）"""
    try:
        logger.info(f"RAG查询请求(GET): query='{query[:50]}...', kb_ids={kb_ids}")
        
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="查询内容不能为空")
        
        if not kb_ids:
            raise HTTPException(status_code=400, detail="必须指定知识库ID")
        
        # 解析KB ID列表
        kb_id_list = [kb_id.strip() for kb_id in kb_ids.split(",")]
        
        # 执行检索
        results = await rag_service.retrieve(
            query=query,
            kb_ids=kb_id_list,
            filters={},
            top_k=top_k
        )
        
        # 重排序（如果有重排序器）
        if results:
            results = rag_service.rerank(query, results)
        
        logger.info(f"RAG查询完成，返回{len(results)}个结果")
        
        return {
            "status": "success",
            "message": "RAG查询成功",
            "request_id": str(uuid.uuid4()),
            "data": {
                "query": query,
                "results": results,
                "result_count": len(results)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"RAG查询失败: {str(e)}")


@router.get("/health")
async def rag_health_check(
    rag_service: RAGService = Depends(get_rag_service)
):
    """RAG服务健康检查"""
    try:
        # 简单的健康检查 - 尝试访问向量存储
        vector_store_healthy = rag_service.vector_store is not None
        embedding_healthy = rag_service.embedding_client is not None
        
        return {
            "status": "success",
            "message": "RAG服务运行正常" if (vector_store_healthy and embedding_healthy) else "RAG服务部分组件异常",
            "data": {
                "vector_store": "healthy" if vector_store_healthy else "unhealthy",
                "embedding_client": "healthy" if embedding_healthy else "unhealthy",
                "reranker": "available" if rag_service.reranker else "not_configured"
            }
        }
    except Exception as e:
        logger.error(f"RAG健康检查失败: {e}")
        return {
            "status": "error",
            "message": "RAG服务检查失败",
            "data": {"error": str(e)}
        }