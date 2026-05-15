"""
LangGraph编排服务
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from backend.core.state import AgentState
from backend.core.nodes import (
    RouterNode, PlannerNode, RetrieverToolNode,
    AgentExecutionNode, SynthesizerNode, StopCheckerNode, DirectQANode
)
from backend.services.rag_service import RAGService
from backend.services.agent_service import AgentService
from backend.utils.logger import get_logger
from backend.utils.qa_logger import get_qa_logger
import time

logger = get_logger(__name__)
qa_logger = get_qa_logger()


class LangGraphService:
    """LangGraph编排服务"""
    
    def __init__(
        self,
        rag_service: RAGService,
        agent_service: AgentService,
        llm_client
    ):
        self.rag_service = rag_service
        self.agent_service = agent_service
        self.llm_client = llm_client
        
        # 初始化节点
        self.router_node = RouterNode(agent_service)
        self.planner_node = PlannerNode()
        self.retriever_tool = RetrieverToolNode(rag_service)
        self.agent_executor = AgentExecutionNode(agent_service, llm_client)
        self.synthesizer = SynthesizerNode()
        self.stop_checker = StopCheckerNode()
        self.direct_qa_node = DirectQANode(llm_client)
        
        # 构建工作流图（默认多智能体协作流程）
        self.workflow = self._build_workflow()
        # 构建简化工作流（仅RAG+直接问答）
        self.simple_workflow = self._build_simple_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建工作流图 - 多智能体流水线协作模式"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("router", self._router_wrapper)
        workflow.add_node("planner", self._planner_wrapper)
        workflow.add_node("retriever", self._retriever_wrapper)
        workflow.add_node("analyst", self._agent_wrapper("analyst"))
        workflow.add_node("writer", self._agent_wrapper("writer"))
        workflow.add_node("critic", self._agent_wrapper("critic"))
        workflow.add_node("tool_caller", self._agent_wrapper("tool_caller"))
        workflow.add_node("synthesizer", self._synthesizer_wrapper)
        workflow.add_node("stop_checker", self._stop_checker_wrapper)
        
        # 设置入口点
        workflow.set_entry_point("router")
        
        # 添加边：构建流水线式协作流程
        workflow.add_edge("router", "planner")
        workflow.add_edge("planner", "retriever")
        
        # 流水线协作：retriever → analyst → writer → critic → synthesizer
        workflow.add_edge("retriever", "analyst")
        workflow.add_edge("analyst", "writer")
        workflow.add_edge("writer", "critic")
        workflow.add_edge("critic", "synthesizer")
        
        # 整合器到终止检查
        workflow.add_edge("synthesizer", "stop_checker")
        
        # 条件边：终止检查决定是否继续优化
        workflow.add_conditional_edges(
            "stop_checker",
            self._should_continue,
            {
                "continue": "analyst",  # 回到分析智能体继续优化
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _build_simple_workflow(self) -> StateGraph:
        """构建简化工作流 - 仅RAG检索+直接问答（max_turns=1时使用）"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("retriever", self._retriever_wrapper)
        workflow.add_node("direct_qa", self._direct_qa_wrapper)
        
        # 设置入口点
        workflow.set_entry_point("retriever")
        
        # 简单的线性流程：retriever → direct_qa → END
        workflow.add_edge("retriever", "direct_qa")
        workflow.add_edge("direct_qa", END)
        
        return workflow.compile()
    
    async def execute_workflow(
        self,
        query: str,
        agent_id: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        chat_messages: Optional[List[BaseMessage]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行工作流"""
        start_time = time.time()
        workflow_stages = []
        
        try:
            logger.info(
                "[backend] langgraph.execute_workflow: start",
                extra={
                    "request_id": request_id,
                    "agent_id": agent_id,
                    "top_k": kwargs.get("top_k"),
                    "rerank": kwargs.get("rerank"),
                    "temperature": kwargs.get("temperature"),
                    "max_turns": kwargs.get("max_turns"),
                    "filters_keys": list((kwargs.get("filters") or {}).keys()),
                },
            )
            
            # 创建初始状态（注入历史消息）
            initial_state = self._create_initial_state(
                query=query,
                agent_id=agent_id,
                agent_config=agent_config,
                session_id=session_id,
                request_id=request_id,
                chat_messages=chat_messages,
                **kwargs
            )
            
            query_id = initial_state["query_id"]
            session_id = initial_state["session_id"]
            max_turns = initial_state.get("max_turns", 10) or 10
            
            # 根据max_turns选择工作流
            if max_turns == 1:
                # 使用简化工作流：仅RAG检索+直接问答
                logger.info("[backend] langgraph.execute_workflow: using simple workflow (max_turns=1)", extra={"request_id": request_id})
                result = await self.simple_workflow.ainvoke(initial_state, config={"recursion_limit": 10})
            else:
                # 使用完整的多智能体协作流程
                logger.info("[backend] langgraph.execute_workflow: using full workflow", extra={"max_turns": max_turns, "request_id": request_id})
                user_recursion_limit = kwargs.get("recursion_limit")
                safe_recursion_limit = user_recursion_limit or max(30, max_turns * 4)
                result = await self.workflow.ainvoke(initial_state, config={"recursion_limit": safe_recursion_limit})
            
            # 处理结果
            final_result = self._process_result(result)
            
            # 计算总耗时
            total_duration_ms = (time.time() - start_time) * 1000
            final_result["processing_time"] = total_duration_ms / 1000
            
            # 记录完整QA会话日志
            qa_logger.log_qa_session(
                query_id=query_id,
                session_id=session_id,
                query=query,
                workflow_stages=workflow_stages,
                intermediate_steps=final_result.get("intermediate_steps", []),
                final_answer=final_result.get("answer", ""),
                citations=final_result.get("citations", []),
                total_duration_ms=total_duration_ms,
                status="success",
                metadata={
                    "agent_id": agent_id,
                    "max_turns": max_turns,
                    "top_k": kwargs.get("top_k", 5),
                    "rerank": kwargs.get("rerank", True)
                }
            )
            
            logger.info(
                "[backend] langgraph.execute_workflow: done",
                extra={
                    "request_id": request_id,
                    "query_id": final_result.get("query_id"),
                    "session_id": final_result.get("session_id"),
                    "total_duration_ms": total_duration_ms,
                    "citations_count": len(final_result.get("citations", [])),
                    "intermediate_steps_count": len(final_result.get("intermediate_steps", [])),
                    "answer_len": len(final_result.get("answer", "")),
                },
            )
            return final_result
            
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            
            # 记录失败日志
            total_duration_ms = (time.time() - start_time) * 1000
            try:
                qa_logger.log_qa_session(
                    query_id=initial_state.get("query_id", "unknown"),
                    session_id=initial_state.get("session_id", "unknown"),
                    query=query,
                    workflow_stages=workflow_stages,
                    intermediate_steps=[],
                    final_answer="",
                    citations=[],
                    total_duration_ms=total_duration_ms,
                    status="error",
                    error_message=str(e)
                )
            except:
                pass
            
            raise
    
    def _create_initial_state(
        self,
        query: str,
        agent_id: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        chat_messages: Optional[List[BaseMessage]] = None,
        **kwargs
    ) -> AgentState:
        """创建初始状态
        
        Args:
            query: 当前用户查询
            agent_id: 智能体ID
            agent_config: 智能体配置
            session_id: 会话ID
            chat_messages: 历史对话消息列表（用于多轮对话上下文）
            **kwargs: 其他参数
        """
        query_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        
        # 获取智能体配置
        if agent_id:
            agent_info = self.agent_service.get_agent(agent_id)
            if agent_info:
                agent_config = agent_config or {}
                agent_config.update(agent_info)
        
        # 构建消息列表：历史消息 + 当前问题
        base_messages: List[BaseMessage] = []
        if chat_messages:
            base_messages.extend(chat_messages)
            logger.info(f"注入历史消息: {len(chat_messages)} 条")
        base_messages.append(HumanMessage(content=query))
        
        state = AgentState(
            messages=base_messages,
            selected_agent=agent_id,
            agent_config=agent_config,
            working_memory={},
            kb_context=[],
            citations=[],
            used_tools=[],
            tool_results={},
            turn_count=0,
            max_turns=kwargs.get("max_turns", 10),
            current_step="router",
            intermediate_steps=[],
            query=query,
            query_id=query_id,
            session_id=session_id,
            top_k=kwargs.get("top_k", 5),
            filters=(
                (lambda f: (
                    (lambda f2: (
                        f2.update({"custom_filters": {}}) if not isinstance(f2.get("custom_filters"), dict) else None,
                        f2
                    )[-1])(f if isinstance(f, dict) else {})
                ))(kwargs.get("filters", {}))
            ),
            rerank=kwargs.get("rerank", True),
            temperature=kwargs.get("temperature", 0.7),
            is_complete=False,
            error_message=None,
            metadata={"request_id": kwargs.get("request_id")}
        )
        try:
            filters = state.get("filters") or {}
            cf = filters.get("custom_filters") or {}
            kb_ids = None
            if isinstance(cf, dict):
                kb_ids = cf.get("kb_ids")
            logger.info(
                "[backend] langgraph.initial_state",
                extra={
                    "request_id": kwargs.get("request_id"),
                    "query_id": query_id,
                    "session_id": session_id,
                    "filters_keys": list(filters.keys()),
                    "kb_ids_count": (len(kb_ids) if isinstance(kb_ids, list) else (1 if kb_ids else 0)),
                },
            )
        except Exception:
            pass
        return state
    
    def _process_result(self, result: AgentState) -> Dict[str, Any]:
        """处理工作流结果"""
        # 优先使用整合器输出作为最终答案
        final_answer = ""
        working_memory = result.get("working_memory", {}) or {}
        synthesis = working_memory.get("synthesis") or {}
        
        # 兼容 Pydantic 对象或 dict
        if hasattr(synthesis, 'get'):
            synth_answer = synthesis.get("final_answer")
            synth_citations = synthesis.get("final_citations")
        else:
            synth_answer = None
            synth_citations = None
        
        if synth_answer:
            final_answer = synth_answer
        elif result.get("intermediate_steps"):
            # 回退到最后一个智能体的输出
            final_answer = result["intermediate_steps"][-1].content
        
        # 提取引用（优先使用整合器产出）
        citations = synth_citations if synth_citations is not None else result.get("citations", [])
        
        # 提取中间步骤
        intermediate_steps = result.get("intermediate_steps", [])
        
        response = {
            "query_id": result.get("query_id", ""),
            "session_id": result.get("session_id", ""),
            "answer": final_answer,
            "citations": [citation.dict() for citation in citations],
            "intermediate_steps": [step.dict() for step in intermediate_steps],
            "agent_used": result.get("selected_agent"),
            "processing_time": 0.0,  # 实际计算处理时间
            "metadata": result.get("metadata", {})
        }
        
        # 附加整合信息便于溯源（非必需字段）
        if synthesis:
            try:
                response["synthesis"] = synthesis
            except Exception:
                pass
        
        return response
    
    def _router_wrapper(self, state: AgentState) -> AgentState:
        """路由节点包装器"""
        try:
            router_result = self.router_node(state)
            
            # 更新状态
            state["current_step"] = "planner"
            state["selected_agent"] = router_result.get("selected_agents", [""])[0]
            state["working_memory"]["routing_info"] = router_result
            
            return state
            
        except Exception as e:
            logger.error(f"路由节点执行失败: {e}")
            state["error_message"] = str(e)
            return state
    
    def _planner_wrapper(self, state: AgentState) -> AgentState:
        """规划节点包装器"""
        try:
            planner_result = self.planner_node(state)
            
            # 更新状态
            state["current_step"] = "retriever"
            state["working_memory"]["plan"] = planner_result
            # 仅在用户未提供时使用估算轮次；否则尊重用户配置，并进行边界保护
            estimated_turns = planner_result.get("estimated_turns", 10)
            user_max_turns = state.get("max_turns")
            state["max_turns"] = max(1, min(user_max_turns or estimated_turns, 50))
            
            return state
            
        except Exception as e:
            logger.error(f"规划节点执行失败: {e}")
            state["error_message"] = str(e)
            return state
    
    async def _retriever_wrapper(self, state: AgentState) -> AgentState:
        """检索工具包装器"""
        try:
            retriever_result = await self.retriever_tool(state)
            try:
                logger.info(
                    "[backend] langgraph.retriever: completed",
                    extra={
                        "query_id": state.get("query_id"),
                        "kb_ids_count": len(retriever_result.get("kb_ids", []) or []),
                        "retrieved_count": len(retriever_result.get("retrieved_docs", []) or []),
                        "reranked_count": len(retriever_result.get("reranked_docs", []) or []),
                        "citations_count": len(retriever_result.get("citations", []) or []),
                    },
                )
            except Exception:
                pass
            
            # TypedDict 实际上就是普通字典，直接使用即可
            # 更新状态
            state["current_step"] = "agent_execution"
            state["kb_context"] = retriever_result.get("reranked_docs", [])
            state["citations"] = retriever_result.get("citations", [])
            
            return state
            
        except Exception as e:
            logger.error(f"检索工具执行失败: {e}")
            import traceback
            logger.error(f"详细错误堆栈: {traceback.format_exc()}")
            # 确保即使出错也返回有效状态
            state["error_message"] = str(e)
            state["kb_context"] = []
            state["citations"] = []
            return state
    
    def _agent_wrapper(self, agent_name: str):
        """智能体包装器"""
        def wrapper(state: AgentState) -> AgentState:
            try:
                agent_result = self.agent_executor(state, agent_name)
                
                # AgentExecutionNode 返回 AgentExecutionState 对象，需要转换为字典
                if hasattr(agent_result, 'dict'):
                    agent_dict = agent_result.dict()
                elif hasattr(agent_result, '__dict__'):
                    agent_dict = agent_result.__dict__
                else:
                    agent_dict = agent_result
                
                # 创建中间步骤记录
                from backend.models.base import IntermediateStep
                step = IntermediateStep(
                    step_type="agent_output",
                    agent_name=agent_name,
                    content=agent_dict.get("output_content", ""),
                    metadata={
                        "reasoning": agent_dict.get("reasoning", ""),
                        "confidence": agent_dict.get("confidence", 0.8),
                        "execution_time": agent_dict.get("execution_time", 0)
                    }
                )
                
                # 更新状态（创建新列表而非直接修改）
                current_steps = state.get("intermediate_steps", [])
                new_steps = current_steps + [step]
                
                state["current_step"] = "synthesizer"
                state["intermediate_steps"] = new_steps
                state["turn_count"] = state.get("turn_count", 0) + 1
                
                return state
                
            except Exception as e:
                logger.error(f"智能体 {agent_name} 执行失败: {e}")
                import traceback
                traceback.print_exc()
                state["error_message"] = str(e)
                return state
        
        return wrapper
    
    def _synthesizer_wrapper(self, state: AgentState) -> AgentState:
        """整合器包装器"""
        try:
            synthesizer_result = self.synthesizer(state)
            
            # 转换 SynthesizerState 对象为字典
            if hasattr(synthesizer_result, 'dict'):
                synth_dict = synthesizer_result.dict()
            elif hasattr(synthesizer_result, '__dict__'):
                synth_dict = synthesizer_result.__dict__
            else:
                synth_dict = synthesizer_result
            
            # 更新状态
            state["current_step"] = "stop_checker"
            working_memory = state.get("working_memory", {})
            working_memory["synthesis"] = synth_dict
            state["working_memory"] = working_memory
            
            return state
            
        except Exception as e:
            logger.error(f"整合器执行失败: {e}")
            import traceback
            traceback.print_exc()
            state["error_message"] = str(e)
            return state
    
    def _stop_checker_wrapper(self, state: AgentState) -> AgentState:
        """终止检查包装器"""
        try:
            stop_result = self.stop_checker(state)
            
            # 转换 StopCheckerState 对象为字典
            if hasattr(stop_result, 'dict'):
                stop_dict = stop_result.dict()
            elif hasattr(stop_result, '__dict__'):
                stop_dict = stop_result.__dict__
            else:
                stop_dict = stop_result
            
            should_continue = stop_dict.get("should_continue", False)
            
            # 更新状态
            state["current_step"] = "complete" if not should_continue else "continue"
            state["is_complete"] = not should_continue
            working_memory = state.get("working_memory", {})
            working_memory["stop_check"] = stop_dict
            state["working_memory"] = working_memory
            
            logger.info(f"终止检查: should_continue={should_continue}, is_complete={not should_continue}")
            
            return state
            
        except Exception as e:
            logger.error(f"终止检查执行失败: {e}")
            import traceback
            traceback.print_exc()
            state["error_message"] = str(e)
            return state
    
    # 注意：_route_to_agents() 已废弃，现采用固定流水线：analyst → writer → critic
    
    def _should_continue(self, state: AgentState) -> str:
        """判断是否继续"""
        if state.get("is_complete", False):
            return "end"
        else:
            return "continue"
    
    def _direct_qa_wrapper(self, state: AgentState) -> AgentState:
        """直接问答包装器 - 用于简化RAG流程"""
        try:
            # 调用直接问答节点
            qa_result = self.direct_qa_node(state)
            
            # 转换为字典
            if hasattr(qa_result, 'dict'):
                qa_dict = qa_result.dict()
            elif hasattr(qa_result, '__dict__'):
                qa_dict = qa_result.__dict__
            else:
                qa_dict = qa_result
            
            # 创建中间步骤记录
            from backend.models.base import IntermediateStep
            step = IntermediateStep(
                step_type="agent_output",
                agent_name="direct_qa",
                content=qa_dict.get("output_content", ""),
                metadata={
                    "reasoning": qa_dict.get("reasoning", ""),
                    "confidence": qa_dict.get("confidence", 0.85),
                    "execution_time": qa_dict.get("execution_time", 0)
                }
            )
            
            # 更新状态
            state["current_step"] = "complete"
            state["intermediate_steps"] = [step]
            state["is_complete"] = True
            
            logger.info(f"直接问答完成，答案长度: {len(qa_dict.get('output_content', ''))}")
            
            return state
            
        except Exception as e:
            logger.error(f"直接问答执行失败: {e}")
            import traceback
            traceback.print_exc()
            state["error_message"] = str(e)
            state["is_complete"] = True
            return state
