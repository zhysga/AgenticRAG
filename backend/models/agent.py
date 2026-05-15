"""
智能体相关数据模型
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from .base import BaseResponse, PaginatedResponse, PaginationParams


class AgentType(str, Enum):
    """智能体类型"""
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"
    TOOL_CALLER = "tool_caller"
    CUSTOM = "custom"


class AgentCreate(BaseModel):
    """创建智能体请求"""
    name: str = Field(..., min_length=1, max_length=100, description="智能体名称")
    description: Optional[str] = Field(None, max_length=500, description="智能体描述")
    agent_type: AgentType = Field(AgentType.CUSTOM, description="智能体类型")
    system_prompt: str = Field(..., min_length=10, max_length=2000, description="系统提示词")
    tools: List[str] = Field(default_factory=list, description="可用工具列表")
    bind_kb_ids: List[str] = Field(default_factory=list, description="绑定的知识库ID列表")
    routing_tags: List[str] = Field(default_factory=list, description="路由标签")
    max_turns: int = Field(10, ge=1, le=50, description="最大轮次")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    llm_config: Optional[Dict[str, Any]] = Field(None, description="模型配置")


class AgentInfo(BaseModel):
    """智能体信息"""
    agent_id: str = Field(..., description="智能体ID")
    name: str = Field(..., description="智能体名称")
    description: Optional[str] = Field(None, description="智能体描述")
    agent_type: AgentType = Field(..., description="智能体类型")
    system_prompt: str = Field(..., description="系统提示词")
    tools: List[str] = Field(default_factory=list, description="可用工具列表")
    bind_kb_ids: List[str] = Field(default_factory=list, description="绑定的知识库ID列表")
    routing_tags: List[str] = Field(default_factory=list, description="路由标签")
    max_turns: int = Field(..., description="最大轮次")
    temperature: float = Field(..., description="温度参数")
    llm_config: Optional[Dict[str, Any]] = Field(None, description="模型配置")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    status: str = Field("active", description="状态")


class AgentCreateResponse(BaseResponse):
    """创建智能体响应"""
    data: AgentInfo = Field(..., description="智能体信息")


class AgentListRequest(BaseModel):
    """智能体列表请求"""
    pagination: PaginationParams = Field(default_factory=PaginationParams, description="分页参数")
    agent_type: Optional[AgentType] = Field(None, description="智能体类型过滤")
    routing_tags: Optional[List[str]] = Field(None, description="路由标签过滤")


class AgentListResponse(PaginatedResponse):
    """智能体列表响应"""
    data: List[AgentInfo] = Field(..., description="智能体列表")


class AgentUpdateRequest(BaseModel):
    """更新智能体请求"""
    agent_id: str = Field(..., description="智能体ID")
    updates: Dict[str, Any] = Field(..., description="更新字段")


class AgentUpdateResponse(BaseResponse):
    """更新智能体响应"""
    data: AgentInfo = Field(..., description="更新后的智能体信息")


class AgentTestRequest(BaseModel):
    """智能体测试请求"""
    agent_id: str = Field(..., description="智能体ID")
    test_query: str = Field(..., min_length=1, max_length=500, description="测试查询")
    test_context: Optional[Dict[str, Any]] = Field(None, description="测试上下文")


class AgentTestResponse(BaseResponse):
    """智能体测试响应"""
    data: Dict[str, Any] = Field(..., description="测试结果")


class AgentDeleteRequest(BaseModel):
    """删除智能体请求"""
    agent_id: str = Field(..., description="智能体ID")


class AgentDeleteResponse(BaseResponse):
    """删除智能体响应"""
    data: Dict[str, bool] = Field(..., description="删除结果")
