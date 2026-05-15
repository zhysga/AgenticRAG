"""
基础数据模型定义
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ResponseStatus(str, Enum):
    """响应状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    ACCEPTED = "accepted"  # 添加accepted状态用于异步处理


class BaseResponse(BaseModel):
    """基础响应模型"""
    status: Union[ResponseStatus, str] = Field(..., description="响应状态")  # 允许字符串类型
    message: str = Field(..., description="响应消息")
    request_id: str = Field(..., description="请求ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")


class PaginationParams(BaseModel):
    """分页参数"""
    page: int = Field(1, ge=1, description="页码")
    size: int = Field(10, ge=1, le=100, description="每页大小")


class PaginatedResponse(BaseResponse):
    """分页响应模型"""
    total: int = Field(..., description="总记录数")
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页大小")
    pages: int = Field(..., description="总页数")


class FilterParams(BaseModel):
    """过滤参数"""
    user_id: Optional[str] = Field(None, description="用户ID")
    project_id: Optional[str] = Field(None, description="项目ID")
    labels: Optional[List[str]] = Field(None, description="标签列表")
    file_type: Optional[str] = Field(None, description="文件类型")
    custom_filters: Optional[Dict[str, Any]] = Field(None, description="自定义过滤条件")


class Citation(BaseModel):
    """知识溯源引用"""
    kb_id: str = Field(..., description="知识库ID")
    kb_name: str = Field(..., description="知识库名称")
    file_name: str = Field(..., description="文件名")
    chunk_position: int = Field(..., description="分块位置")
    score: float = Field(..., description="相似度分数")
    preview: str = Field(..., description="内容预览")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class IntermediateStep(BaseModel):
    """中间步骤"""
    step_type: str = Field(..., description="步骤类型")
    agent_name: Optional[str] = Field(None, description="智能体名称")
    content: str = Field(..., description="步骤内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="步骤时间戳")
    metadata: Optional[Dict[str, Any]] = Field(None, description="步骤元数据")
