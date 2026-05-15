"""
LangGraph状态定义
"""
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from backend.models.base import Citation, IntermediateStep


class AgentState(TypedDict):
    """智能体协作状态"""
    # 消息历史
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 智能体信息
    selected_agent: Optional[str]
    agent_config: Optional[Dict[str, Any]]
    
    # 工作记忆
    working_memory: Dict[str, Any]
    
    # 知识库上下文
    kb_context: List[Dict[str, Any]]
    citations: List[Citation]
    
    # 工具使用记录
    used_tools: List[str]
    tool_results: Dict[str, Any]
    
    # 协作控制
    turn_count: int
    max_turns: int
    current_step: str
    
    # 中间步骤
    intermediate_steps: List[IntermediateStep]
    
    # 查询信息
    query: str
    query_id: str
    session_id: str
    
    # 配置参数
    top_k: int
    filters: Dict[str, Any]
    rerank: bool
    temperature: float
    
    # 状态标志
    is_complete: bool
    error_message: Optional[str]
    
    # 元数据
    metadata: Dict[str, Any]


class RouterState(TypedDict):
    """路由状态"""
    query: str
    query_type: str
    intent: str
    selected_agents: List[str]
    routing_reason: str
    confidence: float


class PlannerState(TypedDict):
    """规划状态"""
    query: str
    sub_tasks: List[Dict[str, Any]]
    execution_plan: List[str]
    required_tools: List[str]
    estimated_turns: int


class RetrieverState(TypedDict):
    """检索状态"""
    query: str
    kb_ids: List[str]
    filters: Dict[str, Any]
    top_k: int
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    citations: List[Citation]


class AgentExecutionState(TypedDict):
    """智能体执行状态"""
    agent_name: str
    agent_type: str
    input_content: str
    output_content: str
    reasoning: str
    confidence: float
    tools_used: List[str]
    execution_time: float


class SynthesizerState(TypedDict):
    """整合状态"""
    agent_outputs: List[Dict[str, Any]]
    final_answer: str
    final_citations: List[Citation]
    synthesis_reasoning: str
    quality_score: float


class StopCheckerState(TypedDict):
    """终止检查状态"""
    current_answer: str
    quality_metrics: Dict[str, float]
    should_continue: bool
    improvement_suggestions: List[str]
    termination_reason: str
