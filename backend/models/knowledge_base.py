"""
知识库相关数据模型
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from .base import BaseResponse, PaginatedResponse, FilterParams, PaginationParams


class KnowledgeBaseScope(str, Enum):
    """知识库作用域"""
    USER = "user"
    PROJECT = "project"
    GLOBAL = "global"


class KnowledgeBaseCreate(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: Optional[str] = Field(None, max_length=500, description="知识库描述")
    labels: List[str] = Field(default_factory=list, description="标签列表")
    scope: KnowledgeBaseScope = Field(KnowledgeBaseScope.USER, description="作用域")
    user_id: Optional[str] = Field(None, description="用户ID")
    project_id: Optional[str] = Field(None, description="项目ID")


class KnowledgeBaseInfo(BaseModel):
    """知识库信息"""
    kb_id: str = Field(..., description="知识库ID")
    name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")
    labels: List[str] = Field(default_factory=list, description="标签列表")
    scope: KnowledgeBaseScope = Field(..., description="作用域")
    user_id: Optional[str] = Field(None, description="用户ID")
    project_id: Optional[str] = Field(None, description="项目ID")
    file_count: int = Field(0, description="文件数量")
    chunk_count: int = Field(0, description="分块数量")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    status: str = Field("active", description="状态")


class KnowledgeBaseCreateResponse(BaseResponse):
    """创建知识库响应"""
    data: KnowledgeBaseInfo = Field(..., description="知识库信息")


class KnowledgeBaseListRequest(BaseModel):
    """知识库列表请求"""
    pagination: PaginationParams = Field(default_factory=PaginationParams, description="分页参数")
    filters: FilterParams = Field(default_factory=FilterParams, description="过滤参数")


class KnowledgeBaseListResponse(PaginatedResponse):
    """知识库列表响应"""
    data: List[KnowledgeBaseInfo] = Field(..., description="知识库列表")


class FileUploadRequest(BaseModel):
    """文件上传请求"""
    kb_id: str = Field(..., description="知识库ID")
    labels: List[str] = Field(default_factory=list, description="文件标签")
    chunk_size: int = Field(512, ge=100, le=2000, description="分块大小")
    chunk_overlap: int = Field(50, ge=0, le=200, description="分块重叠")
    auto_index: bool = Field(True, description="自动索引")


class FileUploadPayload(FileUploadRequest):
    """文件上传JSON载荷，包含文件内容列表"""
    files: List[Dict[str, Any]] = Field(default_factory=list, description="文件数据列表")


class FileInfo(BaseModel):
    """文件信息"""
    file_id: str = Field(..., description="文件ID")
    file_name: str = Field(..., description="文件名")
    file_type: str = Field(..., description="文件类型")
    file_size: int = Field(..., description="文件大小")
    labels: List[str] = Field(default_factory=list, description="文件标签")
    file_path: Optional[str] = Field(None, description="文件在后端持久化后的绝对路径")
    chunk_count: int = Field(0, description="分块数量")
    upload_time: datetime = Field(..., description="上传时间")
    index_status: str = Field("pending", description="索引状态")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文件元数据")


class FileUploadResponse(BaseResponse):
    """文件上传响应"""
    data: List[FileInfo] = Field(..., description="上传的文件信息")


class FileListRequest(BaseModel):
    """文件列表请求"""
    kb_id: str = Field(..., description="知识库ID")
    filters: FilterParams = Field(default_factory=FilterParams, description="过滤参数")


class FileListResponse(BaseResponse):
    """文件列表响应"""
    data: List[FileInfo] = Field(..., description="文件列表")


class ReindexRequest(BaseModel):
    """重新索引请求"""
    kb_id: str = Field(..., description="知识库ID")
    file_ids: Optional[List[str]] = Field(None, description="指定文件ID列表，为空则重新索引所有文件")


class ReindexResponse(BaseResponse):
    """重新索引响应"""
    data: Dict[str, Any] = Field(..., description="索引结果")
    
    class Config:
        # 允许任意状态值，支持异步处理的"accepted"状态
        extra = "allow"


class KnowledgeBaseDeleteRequest(BaseModel):
    """删除知识库请求"""
    kb_id: str = Field(..., description="知识库ID")
    confirm: bool = Field(False, description="确认删除")


class KnowledgeBaseDeleteResponse(BaseResponse):
    """删除知识库响应"""
    data: Dict[str, Any] = Field(..., description="删除结果")
