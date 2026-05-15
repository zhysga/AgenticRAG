"""
知识库管理路由
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from backend.models.knowledge_base import (
    KnowledgeBaseCreate, KnowledgeBaseCreateResponse,
    KnowledgeBaseListRequest, KnowledgeBaseListResponse,
    FileUploadRequest, FileUploadResponse, FileUploadPayload,
    FileListRequest, FileListResponse,
    ReindexRequest, ReindexResponse,
    KnowledgeBaseDeleteRequest, KnowledgeBaseDeleteResponse
)
from backend.services.knowledge_base_service import KnowledgeBaseService
from backend.services.rag_service import RAGService
from backend.utils.auth import get_current_user
from backend.utils.logger import get_logger
from backend.dependencies import get_knowledge_base_service

logger = get_logger(__name__)

router = APIRouter(prefix="/kb", tags=["知识库管理"])


@router.post("/create", response_model=KnowledgeBaseCreateResponse)
async def create_knowledge_base(
    kb_data: KnowledgeBaseCreate,
    current_user: str = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """创建知识库"""
    try:
        logger.info(f"创建知识库请求: {kb_data.name}")
        
        # 设置用户ID
        kb_data.user_id = current_user
        
        # 创建知识库
        kb_info = kb_service.create_knowledge_base(kb_data)
        
        return KnowledgeBaseCreateResponse(
            status="success",
            message="知识库创建成功",
            request_id=str(uuid.uuid4()),
            data=kb_info
        )
        
    except Exception as e:
        logger.error(f"创建知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    page: int = 1,
    size: int = 10,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    labels: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """列出知识库"""
    try:
        logger.info(f"列出知识库请求: page={page}, size={size}")
        
        # 构建过滤条件
        filters = {"user_id": user_id or current_user}
        if project_id:
            filters["project_id"] = project_id
        if labels:
            filters["labels"] = labels.split(",")
        
        # 获取知识库列表（使用命名参数，避免 filters 误传为 scope）
        kbs = kb_service.list_knowledge_bases(
            page=page,
            size=size,
            scope=None,
            user_id=filters.get("user_id"),
            project_id=filters.get("project_id"),
            labels=filters.get("labels")
        )
        
        return KnowledgeBaseListResponse(
            status="success",
            message="获取知识库列表成功",
            request_id=str(uuid.uuid4()),
            data=kbs,
            total=len(kbs),
            page=page,
            size=size,
            pages=(len(kbs) + size - 1) // size
        )
        
    except Exception as e:
        logger.error(f"列出知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=FileUploadResponse)
async def upload_files(
    payload: FileUploadPayload,
    background: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """上传文件到知识库（JSON）"""
    try:
        logger.info(f"上传文件请求: kb_id={payload.kb_id}, files_count={len(payload.files)}")

        # 将当前用户附加到每个文件的元数据
        file_data = []
        for f in payload.files:
            f = dict(f)
            f.setdefault("labels", payload.labels)
            f["user_id"] = current_user
            file_data.append(f)

        # 上传文件（不在此同步索引）
        uploaded_files = kb_service.upload_files(
            kb_id=payload.kb_id,
            files=file_data,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            # 重要：此处不要同步索引，避免请求超时，改由后台任务处理
            auto_index=False,
        )

        # 如需自动索引，则后台异步执行，避免阻塞请求
        if payload.auto_index:
            background.add_task(kb_service.reindex_knowledge_base, payload.kb_id, None)

        return FileUploadResponse(
            status="success",


            message=("文件上传成功（已启动后台索引）" if payload.auto_index else "文件上传成功（已准备索引）"),
            request_id=str(uuid.uuid4()),
            data=uploaded_files
        )

    except Exception as e:
        logger.error(f"上传文件失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files", response_model=FileListResponse)
async def get_knowledge_base_files(
    kb_id: str,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    labels: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """获取知识库文件列表"""
    try:
        logger.info(f"获取知识库文件请求: kb_id={kb_id}")
        
        # 构建过滤条件
        filters = {"user_id": user_id or current_user}
        if project_id:
            filters["project_id"] = project_id
        if labels:
            filters["labels"] = labels.split(",")
        
        # 获取文件列表（对齐服务方法签名）
        files = kb_service.list_files(
            kb_id=kb_id,
            page=1,
            size=100,
            file_type=None,
            labels=filters.get("labels")
        )
        
        return FileListResponse(
            status="success",
            message="获取文件列表成功",
            request_id=str(uuid.uuid4()),
            data=files
        )
        
    except Exception as e:
        logger.error(f"获取文件列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex", response_model=ReindexResponse)
async def reindex_knowledge_base(
    reindex_data: ReindexRequest,
    background: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """重新索引知识库（纯异步处理，避免超时）"""
    try:
        logger.info(f"重新索引请求: kb_id={reindex_data.kb_id}")
        
        # 仅启动后台索引任务，避免同步阻塞
        background.add_task(
            kb_service.reindex_knowledge_base_async,
            kb_id=reindex_data.kb_id,
            file_ids=reindex_data.file_ids
        )
        
        return ReindexResponse(
            status="accepted",
            message="重新索引任务已启动，请稍后查询状态",
            request_id=str(uuid.uuid4()),
            data={
                "status": "processing", 
                "kb_id": reindex_data.kb_id,
                "message": "索引任务正在后台执行"
            }
        )
        
    except Exception as e:
        logger.error(f"重新索引失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reindex/status/{kb_id}")
async def get_reindex_status(
    kb_id: str,
    current_user: str = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """获取重新索引状态"""
    try:
        logger.info(f"获取索引状态: kb_id={kb_id}")
        
        status = kb_service.get_reindex_status(kb_id)
        
        return {
            "status": "success",
            "message": "获取状态成功",
            "request_id": str(uuid.uuid4()),
            "data": status
        }
        
    except Exception as e:
        logger.error(f"获取索引状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete", response_model=KnowledgeBaseDeleteResponse)
async def delete_knowledge_base(
    delete_data: KnowledgeBaseDeleteRequest,
    current_user: str = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """删除知识库"""
    try:
        logger.info(f"删除知识库请求: kb_id={delete_data.kb_id}")
        
        if not delete_data.confirm:
            raise HTTPException(status_code=400, detail="请确认删除操作")
        
        # 删除知识库
        result = kb_service.delete_knowledge_base(delete_data.kb_id)
        
        return KnowledgeBaseDeleteResponse(
            status="success",
            message="知识库删除成功",
            request_id=str(uuid.uuid4()),
            data={"success": result}
        )
        
    except Exception as e:
        logger.error(f"删除知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{kb_id}")
async def get_knowledge_base_stats(
    kb_id: str,
    current_user: str = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """获取知识库统计信息"""
    try:
        logger.info(f"获取知识库统计请求: kb_id={kb_id}")
        
        # 获取统计信息
        stats = kb_service.get_knowledge_base_stats(kb_id)
        
        return {
            "status": "success",
            "message": "获取统计信息成功",
            "request_id": str(uuid.uuid4()),
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
