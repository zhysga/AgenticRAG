"""
聊天问答相关数据模型 - 前后端数据交换格式定义

这个文件定义了前后端通信时使用的数据结构和格式。
使用Pydantic模型确保数据验证和类型安全。

前后端数据交换流程：
1. 前端发送请求 → Pydantic模型验证 → 后端处理 → 返回响应
2. 后端返回响应 → Pydantic模型序列化 → 前端接收数据

关键特性：
- 数据验证：确保输入数据的正确性
- 类型安全：Python类型注解
- 自动文档生成：FastAPI自动生成API文档
- 序列化/反序列化：JSON ↔ Python对象转换
"""

from typing import List, Optional, Dict, Any  # 类型注解
from pydantic import BaseModel, Field  # Pydantic数据模型和字段验证
from datetime import datetime  # 时间处理
from .base import BaseResponse, FilterParams, Citation, IntermediateStep, PaginationParams, PaginatedResponse


class ChatAskRequest(BaseModel):
    """
    问答请求模型 - 前端发送给后端的提问请求
    
    这个模型定义了前端调用/chat/ask接口时需要发送的数据结构。
    后端使用这个模型验证和解析前端发送的JSON数据。
    
    前后端数据流：
    前端JSON → ChatAskRequest模型验证 → 后端业务处理
    """
    agent_id: Optional[str] = Field(None, description="智能体ID，指定使用哪个智能体回答问题")
    agent_profile: Optional[Dict[str, Any]] = Field(None, description="智能体配置，动态调整智能体行为")
    query: str = Field(..., min_length=1, max_length=1000, description="用户查询内容，必填字段")
    session_id: Optional[str] = Field(None, description="会话ID，用于关联同一会话的多轮对话")
    top_k: int = Field(5, ge=1, le=20, description="检索Top-K，控制返回的相关文档数量")
    filters: FilterParams = Field(default_factory=FilterParams, description="过滤条件，按知识库、文件等过滤结果")
    rerank: bool = Field(True, description="是否重排序，提升检索结果的相关性")
    stream: bool = Field(False, description="是否流式响应，实时返回答案片段")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="温度参数，控制AI回答的创造性")
    max_turns: Optional[int] = Field(None, ge=1, le=50, description="最大轮次，限制多智能体协作的轮数")


class ChatAnswer(BaseModel):
    """聊天答案"""
    answer: str = Field(..., description="答案内容")
    citations: List[Citation] = Field(default_factory=list, description="知识溯源")
    intermediate_steps: List[IntermediateStep] = Field(default_factory=list, description="中间步骤")
    agent_used: Optional[str] = Field(None, description="使用的智能体")
    session_id: str = Field(..., description="会话ID")
    query_id: str = Field(..., description="查询ID")
    processing_time: float = Field(..., description="处理时间（秒）")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class ChatAskResponse(BaseResponse):
    """问答响应"""
    data: ChatAnswer = Field(..., description="问答结果")


 


class ChatHistoryRequest(BaseModel):
    """聊天历史请求"""
    session_id: str = Field(..., description="会话ID")
    pagination: PaginationParams = Field(default_factory=PaginationParams, description="分页参数")


class ChatHistoryItem(BaseModel):
    """聊天历史项"""
    query_id: str = Field(..., description="查询ID")
    query: str = Field(..., description="查询内容")
    answer: str = Field(..., description="答案内容")
    citations: List[Citation] = Field(default_factory=list, description="知识溯源")
    agent_used: Optional[str] = Field(None, description="使用的智能体")
    timestamp: datetime = Field(..., description="时间戳")
    processing_time: float = Field(..., description="处理时间")


class ChatHistoryResponse(PaginatedResponse):
    """聊天历史响应"""
    data: List[ChatHistoryItem] = Field(..., description="聊天历史列表")


class ChatSessionInfo(BaseModel):
    """聊天会话信息"""
    session_id: str = Field(..., description="会话ID")
    agent_id: Optional[str] = Field(None, description="智能体ID")
    created_at: datetime = Field(..., description="创建时间")
    last_activity: datetime = Field(..., description="最后活动时间")
    message_count: int = Field(0, description="消息数量")
    status: str = Field("active", description="会话状态")


class ChatSessionListRequest(BaseModel):
    """聊天会话列表请求"""
    pagination: PaginationParams = Field(default_factory=PaginationParams, description="分页参数")
    agent_id: Optional[str] = Field(None, description="智能体ID过滤")
    status: Optional[str] = Field(None, description="状态过滤")


class ChatSessionListResponse(PaginatedResponse):
    """聊天会话列表响应"""
    data: List[ChatSessionInfo] = Field(..., description="会话列表")


class ChatSessionDeleteRequest(BaseModel):
    """删除聊天会话请求"""
    session_id: str = Field(..., description="会话ID")
    confirm: bool = Field(False, description="确认删除")


class ChatSessionDeleteResponse(BaseResponse):
    """删除聊天会话响应"""
    data: Dict[str, Any] = Field(..., description="删除结果")


class ChatStats(BaseModel):
    """聊天统计信息数据"""
    total_sessions: int = Field(..., description="总会话数")
    active_sessions: int = Field(..., description="活跃会话数")
    total_messages: int = Field(..., description="总消息数")
    avg_messages_per_session: float = Field(..., description="每会话平均消息数")
    last_activity: Optional[datetime] = Field(None, description="最近活动时间")


class ChatStatsResponse(BaseResponse):
    """聊天统计响应"""
    data: ChatStats = Field(..., description="统计数据")


# 向后兼容的ChatRequest定义
class ChatRequest(ChatAskRequest):
    """聊天请求（向后兼容）"""
    pass
