"""
LangGraph节点实现
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from backend.core.state import (
    AgentState, RouterState, PlannerState, RetrieverState,
    AgentExecutionState, SynthesizerState, StopCheckerState
)
from backend.models.base import IntermediateStep, Citation
from backend.services.rag_service import RAGService
from backend.services.agent_service import AgentService
from backend.utils.logger import get_logger
from backend.utils.qa_logger import get_qa_logger

logger = get_logger(__name__)
qa_logger = get_qa_logger()


class RouterNode:
    """智能路由节点"""
    
    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service
    
    def __call__(self, state: AgentState) -> RouterState:
        """执行路由逻辑"""
        try:
            query = state["query"]
            
            # 分析查询意图
            intent = self._analyze_intent(query)
            
            # 选择合适智能体
            selected_agents = self._select_agents(query, intent)
            
            # 计算路由置信度
            confidence = self._calculate_confidence(query, selected_agents)
            
            return RouterState(
                query=query,
                query_type=intent["type"],
                intent=intent["description"],
                selected_agents=selected_agents,
                routing_reason=intent["reasoning"],
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"路由节点执行失败: {e}")
            raise
    
    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """分析查询意图"""
        # 简单的意图分析逻辑，实际应用中可以使用更复杂的NLP模型
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["分析", "分析", "analyze", "analysis"]):
            return {
                "type": "analysis",
                "description": "需要分析类智能体处理",
                "reasoning": "查询包含分析相关关键词"
            }
        elif any(word in query_lower for word in ["写作", "写", "write", "writing"]):
            return {
                "type": "writing",
                "description": "需要写作类智能体处理",
                "reasoning": "查询包含写作相关关键词"
            }
        elif any(word in query_lower for word in ["评价", "批评", "critique", "review"]):
            return {
                "type": "critique",
                "description": "需要批评类智能体处理",
                "reasoning": "查询包含评价相关关键词"
            }
        else:
            return {
                "type": "general",
                "description": "需要通用智能体处理",
                "reasoning": "查询类型不明确，使用通用处理"
            }
    
    def _select_agents(self, query: str, intent: Dict[str, Any]) -> List[str]:
        """选择合适智能体"""
        # 根据意图和查询内容选择智能体
        agents = self.agent_service.get_available_agents()
        
        selected = []
        for agent in agents:
            if intent["type"] in agent.get("routing_tags", []):
                selected.append(agent["agent_id"])
        
        # 如果没有匹配的智能体，选择默认智能体
        if not selected:
            default_agents = [agent["agent_id"] for agent in agents if "default" in agent.get("routing_tags", [])]
            if default_agents:
                selected = default_agents[:1]  # 选择第一个默认智能体
            else:
                # 如果没有默认智能体，选择第一个可用的智能体
                if agents:
                    selected = [agents[0]["agent_id"]]
                else:
                    # 如果没有可用智能体，返回空列表并记录警告
                    logger.warning("没有可用的智能体")
        
        return selected
    
    def _calculate_confidence(self, query: str, selected_agents: List[str]) -> float:
        """计算路由置信度"""
        if not selected_agents:
            return 0.0
        
        # 简单的置信度计算
        base_confidence = 0.7
        if len(selected_agents) == 1:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)


class PlannerNode:
    """任务规划节点"""
    
    def __call__(self, state: AgentState) -> PlannerState:
        """执行规划逻辑"""
        try:
            query = state["query"]
            selected_agents = state.get("selected_agent", [])
            
            # 分解子任务
            sub_tasks = self._decompose_tasks(query)
            
            # 制定执行计划
            execution_plan = self._create_execution_plan(sub_tasks, selected_agents)
            
            # 确定所需工具
            required_tools = self._identify_required_tools(sub_tasks)
            
            # 估算执行轮次
            estimated_turns = self._estimate_turns(sub_tasks)
            
            return PlannerState(
                query=query,
                sub_tasks=sub_tasks,
                execution_plan=execution_plan,
                required_tools=required_tools,
                estimated_turns=estimated_turns
            )
            
        except Exception as e:
            logger.error(f"规划节点执行失败: {e}")
            raise
    
    def _decompose_tasks(self, query: str) -> List[Dict[str, Any]]:
        """分解子任务"""
        # 简单的任务分解逻辑
        tasks = [
            {
                "task_id": str(uuid.uuid4()),
                "description": "检索相关知识",
                "type": "retrieval",
                "priority": 1
            },
            {
                "task_id": str(uuid.uuid4()),
                "description": "分析检索内容",
                "type": "analysis",
                "priority": 2
            },
            {
                "task_id": str(uuid.uuid4()),
                "description": "生成答案",
                "type": "generation",
                "priority": 3
            }
        ]
        
        return tasks
    
    def _create_execution_plan(self, sub_tasks: List[Dict[str, Any]], selected_agents: List[str]) -> List[str]:
        """创建执行计划"""
        plan = []
        
        for task in sub_tasks:
            if task["type"] == "retrieval":
                plan.append("retrieve_knowledge")
            elif task["type"] == "analysis":
                plan.append("analyze_content")
            elif task["type"] == "generation":
                plan.append("generate_answer")
        
        return plan
    
    def _identify_required_tools(self, sub_tasks: List[Dict[str, Any]]) -> List[str]:
        """识别所需工具"""
        tools = []
        
        for task in sub_tasks:
            if task["type"] == "retrieval":
                tools.append("rag_retriever")
            elif task["type"] == "analysis":
                tools.append("text_analyzer")
            elif task["type"] == "generation":
                tools.append("text_generator")
        
        return list(set(tools))  # 去重
    
    def _estimate_turns(self, sub_tasks: List[Dict[str, Any]]) -> int:
        """估算执行轮次"""
        return len(sub_tasks) + 1  # 基础轮次 + 额外轮次


class RetrieverToolNode:
    """RAG检索工具节点"""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
    
    async def __call__(self, state: AgentState) -> RetrieverState:
        """执行检索逻辑"""
        try:
            # 确保 state 不为 None
            if state is None:
                raise ValueError("State 不能为 None")
            
            query = state.get("query") or ""
            if not query:
                raise ValueError("查询文本不能为空")
            
            # 获取并确保数据类型安全
            agent_config = state.get("agent_config")
            if not isinstance(agent_config, dict):
                agent_config = {}
            
            filters = state.get("filters")
            if not isinstance(filters, dict):
                filters = {}

            # 优先使用智能体配置中的绑定知识库
            kb_ids = agent_config.get("bind_kb_ids", [])
            # 若智能体未绑定知识库，则尝试从过滤条件中获取
            if not kb_ids:
                # 兼容多种字段命名
                custom_filters = filters.get("custom_filters")
                if not isinstance(custom_filters, dict):
                    custom_filters = {}
                candidate_ids = (
                    filters.get("kb_ids")
                    or filters.get("knowledge_base_ids")
                    or custom_filters.get("kb_ids")
                )
                if candidate_ids:
                    # 保证返回的是列表类型
                    kb_ids = candidate_ids if isinstance(candidate_ids, list) else [candidate_ids]

            # 从元数据中提取 request_id（若存在）
            _meta = state.get("metadata") or {}
            _request_id = None
            if isinstance(_meta, dict):
                _request_id = _meta.get("request_id")

            try:
                logger.info(
                    "[backend] retriever_node: params",
                    extra={
                        "query_id": state.get("query_id"),
                        "top_k": state.get("top_k"),
                        "rerank": state.get("rerank"),
                        "kb_ids": kb_ids,
                        "filters_keys": list(filters.keys()),
                        "request_id": _request_id,
                    },
                )
            except Exception:
                pass

            top_k = state.get("top_k", 5)
            
            # 执行检索
            import time as _t
            _t0 = _t.time()
            retrieved_docs = await self.rag_service.retrieve(
                query=query,
                kb_ids=kb_ids,
                filters=filters,
                top_k=top_k,
                request_id=_request_id,
            )
            _retrieve_ms = (_t.time() - _t0) * 1000
            retrieved_docs = retrieved_docs if retrieved_docs is not None else []
            # 过滤掉 None 值，防止后续访问属性时报错
            retrieved_docs = [d for d in retrieved_docs if d is not None]
            try:
                logger.info(
                    "[backend] retriever_node: retrieved",
                    extra={
                        "query_id": state.get("query_id"),
                        "elapsed_ms": _retrieve_ms,
                        "retrieved_count": len(retrieved_docs),
                        "request_id": _request_id,
                    },
                )
            except Exception:
                pass
            
            # 重排序（如果启用）
            reranked_docs = retrieved_docs
            if state.get("rerank", True) and retrieved_docs:
                _t1 = _t.time()
                reranked_result = self.rag_service.rerank(query, retrieved_docs, request_id=_request_id)
                reranked_docs = reranked_result if reranked_result is not None else retrieved_docs
                _rerank_ms = (_t.time() - _t1) * 1000
                try:
                    logger.info(
                        "[backend] retriever_node: reranked",
                        extra={
                            "query_id": state.get("query_id"),
                            "elapsed_ms": _rerank_ms,
                            "reranked_count": len(reranked_docs),
                            "request_id": _request_id,
                        },
                    )
                except Exception:
                    pass
            # 重排结果同样需要去除 None
            reranked_docs = [d for d in reranked_docs if d is not None]
            
            # 生成引用
            citations = self._generate_citations(reranked_docs)
            
            return RetrieverState(
                query=query,
                kb_ids=kb_ids,
                filters=filters,
                top_k=top_k,
                retrieved_docs=retrieved_docs,
                reranked_docs=reranked_docs,
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"检索工具节点执行失败: {e}")
            raise
    
    def _generate_citations(self, docs: List[Dict[str, Any]]) -> List[Citation]:
        """生成引用信息"""
        citations = []
        # 过滤 None，确保后续安全访问
        docs = [d for d in docs if d is not None]
        
        for doc in docs:
            citation = Citation(
                kb_id=doc.get("kb_id", ""),
                kb_name=doc.get("kb_name", ""),
                file_name=doc.get("file_name", ""),
                chunk_position=doc.get("chunk_index", 0),
                score=doc.get("score", 0.0),
                preview=doc.get("content", "")[:200] + "...",
                metadata=doc.get("metadata", {})
            )
            citations.append(citation)
        
        return citations


class ContextQualityAssessor:
    """上下文质量评估器，用于评估检索上下文的质量并调整提示词"""
    
    @staticmethod
    def assess_context_quality(context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估上下文质量，返回质量指标和调整建议"""
        if not context:
            return {
                "quality_score": 0.0,
                "has_relevant_content": False,
                "content_density": 0.0,
                "recommendations": ["建议重新表述问题或添加更多上下文"],
                "adjustment_strategy": "no_context"
            }
        
        # 计算基本质量指标
        total_length = sum(len(doc.get("content", "")) for doc in context)
        avg_length = total_length / len(context) if context else 0
        content_density = min(avg_length / 1000, 1.0)  # 归一化到0-1
        
        # 检查内容相关性（简单启发式方法）
        relevance_indicators = 0
        for doc in context:
            content = doc.get("content", "").lower()
            # 检查是否包含常见相关词汇
            if any(indicator in content for indicator in ["问题", "解决", "方法", "步骤", "原因", "影响"]):
                relevance_indicators += 1
        
        relevance_score = relevance_indicators / len(context) if context else 0
        
        # 计算综合质量分数
        quality_score = (content_density * 0.4 + relevance_score * 0.6)
        
        # 确定调整策略
        if quality_score >= 0.7:
            adjustment_strategy = "high_quality"
            recommendations = ["使用标准提示词，强调深度分析"]
        elif quality_score >= 0.4:
            adjustment_strategy = "medium_quality"
            recommendations = ["使用增强提示词，建议补充信息"]
        else:
            adjustment_strategy = "low_quality"
            recommendations = ["使用简化提示词，明确指出信息不足"]
        
        return {
            "quality_score": quality_score,
            "has_relevant_content": relevance_score > 0.3,
            "content_density": content_density,
            "recommendations": recommendations,
            "adjustment_strategy": adjustment_strategy
        }


class PromptTemplate:
    """提示词模板系统，支持结构化提示和上下文自适应"""
    
    @staticmethod
    def build_agent_prompt(agent_name: str, system_prompt: str, query: str, 
                          context: List[Dict[str, Any]], has_context: bool) -> str:
        """构建智能体提示词，根据上下文情况自适应调整"""
        
        # 评估上下文质量
        context_quality = ContextQualityAssessor.assess_context_quality(context)
        
        # 基础提示词结构
        base_structure = (
            "# 角色与使命\n{role_mission}\n\n"
            "# 上下文信息\n{context_section}\n\n"
            "# 任务要求\n{task_requirements}\n\n"
            "# 输出格式\n{output_format}\n\n"
            "# 约束条件\n{constraints}"
        )
        
        # 根据智能体类型定制内容
        if agent_name == "analyst":
            role_mission = system_prompt or "你是资深分析师，负责从多维度提取关键信息并进行结构化分析。"
            task_requirements = f"分析以下问题：{query}"
            output_format = (
                "## 分析报告\n"
                "### 1. 核心摘要（不超过3句话）\n"
                "### 2. 关键信息点（分点列出，每点包含依据和来源）\n"
                "### 3. 多维度分析（按目标/现状/问题/成因/影响/方案展开）\n"
                "### 4. 信息缺口与澄清需求\n"
                "### 5. 风险评估与置信度（0-1分）"
            )
            constraints = (
                "- 严格遵循MECE原则（相互独立，完全穷尽）\n"
                "- 所有结论必须有证据支持\n"
                "- 明确区分事实与推断\n"
                "- 标注信息来源和置信度"
            )
            
        elif agent_name == "writer":
            role_mission = system_prompt or "你是专业写作助手，负责将分析结果转化为清晰、连贯的结构化回答。"
            task_requirements = f"基于分析结果回答以下问题：{query}"
            output_format = (
                "## 结构化答案\n"
                "### 1. 执行摘要（不超过150字）\n"
                "### 2. 背景与问题定义\n"
                "### 3. 主要结论（分点列出，每点包含依据和引用）\n"
                "### 4. 详细论据与分析\n"
                "### 5. 建议与下一步行动\n"
                "### 6. 风险与限制因素"
            )
            constraints = (
                "- 遵循金字塔原理（结论先行，以上统下，归类分组，逻辑递进）\n"
                "- 保持段落间逻辑连贯性\n"
                "- 关键论点必须标注来源\n"
                "- 语言简洁专业，面向技术/业务混合受众"
            )
            
        elif agent_name == "critic":
            role_mission = system_prompt or "你是质量评审专家，负责评估内容质量并提供改进建议。"
            task_requirements = f"评估以下问题相关内容的质量：{query}"
            output_format = (
                "## 质量评估报告\n"
                "### 1. 总体评分（0-5分）及理由\n"
                "### 2. 分维度评估\n"
                "- 完整性（0-5分）：\n"
                "- 准确性（0-5分）：\n"
                "- 相关性（0-5分）：\n"
                "- 结构清晰度（0-5分）：\n"
                "### 3. 问题清单（按严重程度排序）\n"
                "### 4. 具体改进建议\n"
                "### 5. 评审结论（通过/需修改）"
            )
            constraints = (
                "- 评估标准明确一致\n"
                "- 问题定位精准，建议具体可行\n"
                "- 标记潜在幻觉或无依据断言\n"
                "- 提供优先级排序的修复方案"
            )
            
        elif agent_name == "tool_caller":
            role_mission = system_prompt or "你是流程说明专家，负责清晰描述系统工具调用流程与协作关系。"
            task_requirements = f"说明处理以下问题时系统的工具调用流程：{query}"
            output_format = (
                "## 系统流程说明\n"
                "### 1. 流程图（Mermaid格式）\n"
                "### 2. 逐步流程说明\n"
                "### 3. 工具输入输出概览\n"
                "### 4. 智能体协作关系\n"
                "### 5. 最佳实践与常见问题"
            )
            constraints = (
                "- 以ReAct范式解释流程\n"
                "- 明确每步的输入输出和错误处理\n"
                "- 说明智能体间的协作接口\n"
                "- 遵循数据最小化和安全原则"
            )
            
        else:  # custom
            role_mission = system_prompt or "你是通用助手，根据系统提示完成任务。"
            task_requirements = f"请回答以下问题：{query}"
            output_format = "## 答案\n\n请提供清晰、结构化的回答，并尽可能引用依据。"
            constraints = "- 遵循系统提示中的角色定位和要求\n- 确保回答准确、相关、完整"
        
        # 根据上下文情况调整上下文部分
        if has_context and context:
            context_text = "\n\n".join([doc.get("content", "") for doc in context[:5]])
            
            # 根据上下文质量调整提示
            if context_quality["adjustment_strategy"] == "high_quality":
                context_note = "请基于上述高质量检索内容进行深入分析，充分利用丰富的上下文信息。"
            elif context_quality["adjustment_strategy"] == "medium_quality":
                context_note = "请基于上述检索内容进行分析，同时注意识别信息缺口并提出补充建议。"
            else:  # low_quality
                context_note = "上述检索内容质量有限，请谨慎分析并明确指出信息不足之处。"
            
            context_section = (
                f"### 检索上下文（共{len(context)}条相关文档，质量评分：{context_quality['quality_score']:.2f}）\n"
                f"{context_text}\n\n"
                f"{context_note}\n\n"
                "确保关键结论有来源支持，并标注信息质量评估。"
            )
        else:
            context_section = (
                "### 注意事项\n"
                "当前未检索到相关上下文信息，请基于你的专业知识直接回答。"
                "如果问题需要特定领域知识，请明确指出你的知识边界和局限性。"
                "请提供基于通用知识的最佳回答，并建议如何获取更准确的信息。"
            )
        
        # 构建最终提示词
        return base_structure.format(
            role_mission=role_mission,
            context_section=context_section,
            task_requirements=task_requirements,
            output_format=output_format,
            constraints=constraints
        )


class AgentExecutionNode:
    """智能体执行节点"""
    
    def __init__(self, agent_service: AgentService, llm_client):
        self.agent_service = agent_service
        self.llm = llm_client
        self.effectiveness_tracker = PromptEffectivenessTracker()
    
    def __call__(self, state: AgentState, agent_name: str) -> AgentExecutionState:
        """执行智能体逻辑"""
        try:
            agent_config = state.get("agent_config", {})
            input_content = state.get("kb_context", [])
            query = state["query"]
            
            # 获取智能体配置（按名称查找）
            agent_info = self.agent_service.get_agent_by_name(agent_name)
            if not agent_info:
                raise ValueError(f"智能体 {agent_name} 不存在")
            
            # 将 AgentInfo 对象转换为字典
            agent_dict = agent_info.dict()
            
            # 执行智能体逻辑
            start_time = datetime.now()
            
            if agent_name == "analyst":
                output_content, reasoning = self._execute_analyst(query, input_content, agent_dict)
            elif agent_name == "writer":
                output_content, reasoning = self._execute_writer(query, input_content, agent_dict)
            elif agent_name == "critic":
                output_content, reasoning = self._execute_critic(query, input_content, agent_dict)
            elif agent_name == "tool_caller":
                output_content, reasoning = self._execute_tool_caller(query, input_content, agent_dict)
            else:
                output_content, reasoning = self._execute_custom(query, input_content, agent_dict)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 记录智能体执行日志
            query_id = state.get("query_id", "unknown")
            qa_logger.log_agent_execution(
                query_id=query_id,
                agent_name=agent_name,
                agent_type=agent_dict.get("agent_type", "custom"),
                input_content=str(input_content),
                output_content=output_content,
                reasoning=reasoning,
                confidence=0.8,
                execution_time=execution_time,
                tools_used=agent_dict.get("tools", []),
                metadata={
                    "system_prompt": agent_dict.get("system_prompt", "")[:200],
                    "temperature": agent_dict.get("temperature", 0.7),
                    "max_turns": agent_dict.get("max_turns", 10)
                }
            )
            
            return AgentExecutionState(
                agent_name=agent_name,
                agent_type=agent_dict.get("agent_type", "custom"),
                input_content=str(input_content),
                output_content=output_content,
                reasoning=reasoning,
                confidence=0.8,  # 默认置信度
                tools_used=[],
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"智能体执行节点失败: {e}")
            raise
    
    def _execute_analyst(self, query: str, context: List[Dict[str, Any]], agent_info: Dict[str, Any]) -> tuple:
        """执行分析智能体，使用结构化提示词模板"""
        system_prompt = agent_info.get("system_prompt", "")
        has_context = context and len(context) > 0
        
        # 使用新的提示词模板系统
        prompt = PromptTemplate.build_agent_prompt(
            agent_name="analyst",
            system_prompt=system_prompt,
            query=query,
            context=context,
            has_context=has_context
        )
        
        reasoning = "基于知识库检索结果进行分析" if has_context else "知识库无相关内容，使用大模型直接回答"
        
        try:
            resp = self.llm.complete(prompt, temperature=0.3, max_tokens=1024)
            return resp.text, reasoning
        except Exception as e:
            logger.warning(f"分析LLM调用失败，使用智能降级: {e}")
            # 智能降级：基于上下文质量提供简化分析
            if has_context:
                return self._fallback_analysis_with_context(query, context), "LLM调用失败，使用简化分析"
            else:
                return self._fallback_analysis_without_context(query), "LLM调用失败，使用通用知识分析"
    
    def _fallback_analysis_with_context(self, query: str, context: List[Dict[str, Any]]) -> str:
        """基于上下文的智能降级分析"""
        context_snippets = [doc.get("content", "")[:200] + "..." for doc in context[:3]]
        return (
            f"## 分析报告（简化版）\n\n"
            f"### 核心摘要\n"
            f"基于{len(context)}条相关文档，对问题'{query}'的初步分析。\n\n"
            f"### 关键信息点\n"
            f"1. 检索到{len(context)}条相关文档\n"
            f"2. 主要内容片段：\n"
            + "\n".join([f"- {snippet}" for snippet in context_snippets]) +
            f"\n\n### 信息缺口\n"
            f"由于系统限制，无法提供深入分析。建议重新提问或提供更多上下文。"
        )
    
    def _fallback_analysis_without_context(self, query: str) -> str:
        """无上下文的智能降级分析"""
        return (
            f"## 分析报告（简化版）\n\n"
            f"### 核心摘要\n"
            f"针对问题'{query}'，未检索到相关上下文信息。\n\n"
            f"### 关键信息点\n"
            f"1. 知识库中无直接相关内容\n"
            f"2. 建议尝试以下方法：\n"
            f"   - 重新表述问题\n"
            f"   - 添加更多上下文信息\n"
            f"   - 使用更具体的关键词\n\n"
            f"### 信息缺口\n"
            f"当前知识库可能缺少相关领域的文档，建议扩充知识库内容。"
        )
    
    def _execute_writer(self, query: str, context: List[Dict[str, Any]], agent_info: Dict[str, Any]) -> tuple:
        """执行写作智能体，使用结构化提示词模板"""
        system_prompt = agent_info.get("system_prompt", "")
        has_context = context and len(context) > 0
        
        # 使用新的提示词模板系统
        prompt = PromptTemplate.build_agent_prompt(
            agent_name="writer",
            system_prompt=system_prompt,
            query=query,
            context=context,
            has_context=has_context
        )
        
        reasoning = "基于知识库检索结果生成答案" if has_context else "知识库无相关内容，使用大模型直接生成答案"
        
        try:
            resp = self.llm.complete(prompt, temperature=0.7, max_tokens=1536)
            return resp.text, reasoning
        except Exception as e:
            logger.warning(f"写作LLM调用失败，使用智能降级: {e}")
            # 智能降级：基于上下文质量提供简化回答
            if has_context:
                return self._fallback_writing_with_context(query, context), "LLM调用失败，使用简化回答"
            else:
                return self._fallback_writing_without_context(query), "LLM调用失败，使用通用知识回答"
    
    def _fallback_writing_with_context(self, query: str, context: List[Dict[str, Any]]) -> str:
        """基于上下文的智能降级写作"""
        context_snippets = [doc.get("content", "")[:200] + "..." for doc in context[:3]]
        return (
            f"## 结构化答案（简化版）\n\n"
            f"### 执行摘要\n"
            f"基于{len(context)}条相关文档，对问题'{query}'的初步回答。\n\n"
            f"### 主要结论\n"
            f"1. 检索到{len(context)}条相关文档\n"
            f"2. 主要内容摘要：\n"
            + "\n".join([f"- {snippet}" for snippet in context_snippets]) +
            f"\n\n### 建议与下一步\n"
            f"由于系统限制，无法提供详细分析。建议重新提问或提供更多上下文信息。"
        )
    
    def _fallback_writing_without_context(self, query: str) -> str:
        """无上下文的智能降级写作"""
        return (
            f"## 结构化答案（简化版）\n\n"
            f"### 执行摘要\n"
            f"针对问题'{query}'，未检索到相关上下文信息。\n\n"
            f"### 主要结论\n"
            f"1. 知识库中无直接相关内容\n"
            f"2. 建议尝试以下方法：\n"
            f"   - 重新表述问题\n"
            f"   - 添加更多上下文信息\n"
            f"   - 使用更具体的关键词\n\n"
            f"### 建议与下一步\n"
            f"当前知识库可能缺少相关领域的文档，建议扩充知识库内容后再次查询。"
        )
    
    def _execute_critic(self, query: str, context: List[Dict[str, Any]], agent_info: Dict[str, Any]) -> tuple:
        """执行批评智能体，使用结构化提示词模板"""
        system_prompt = agent_info.get("system_prompt", "")
        has_context = context and len(context) > 0
        
        # 使用新的提示词模板系统
        prompt = PromptTemplate.build_agent_prompt(
            agent_name="critic",
            system_prompt=system_prompt,
            query=query,
            context=context,
            has_context=has_context
        )
        
        reasoning = "评估基于知识库的回答质量" if has_context else "评估大模型直接回答的质量"
        
        try:
            resp = self.llm.complete(prompt, temperature=0.4, max_tokens=1024)
            return resp.text, reasoning
        except Exception as e:
            logger.warning(f"批评LLM调用失败，使用智能降级: {e}")
            # 智能降级：基于上下文质量提供简化评估
            if has_context:
                return self._fallback_critic_with_context(query, context), "LLM调用失败，使用简化评估"
            else:
                return self._fallback_critic_without_context(query), "LLM调用失败，使用通用评估"
    
    def _fallback_critic_with_context(self, query: str, context: List[Dict[str, Any]]) -> str:
        """基于上下文的智能降级评估"""
        return (
            f"## 质量评估报告（简化版）\n\n"
            f"### 总体评分：2.5/5\n\n"
            f"### 分维度评估\n"
            f"- 完整性：2/5（基于{len(context)}条文档，内容有限）\n"
            f"- 准确性：3/5（基于检索内容，但未经深入验证）\n"
            f"- 相关性：3/5（内容与问题相关，但可能不够全面）\n"
            f"- 结构清晰度：2/5（由于系统限制，结构不完整）\n\n"
            f"### 主要问题\n"
            f"1. 系统限制导致无法提供完整评估\n"
            f"2. 建议重新提问以获得更详细的分析\n\n"
            f"### 评审结论\n"
            f"需修改：建议重新生成或提供更多上下文信息"
        )
    
    def _fallback_critic_without_context(self, query: str) -> str:
        """无上下文的智能降级评估"""
        return (
            f"## 质量评估报告（简化版）\n\n"
            f"### 总体评分：1.5/5\n\n"
            f"### 分维度评估\n"
            f"- 完整性：1/5（无相关上下文信息）\n"
            f"- 准确性：2/5（基于通用知识，可能不够准确）\n"
            f"- 相关性：2/5（可能无法完全匹配问题需求）\n"
            f"- 结构清晰度：1/5（由于系统限制，结构不完整）\n\n"
            f"### 主要问题\n"
            f"1. 知识库中无相关内容\n"
            f"2. 建议扩充知识库或重新表述问题\n\n"
            f"### 评审结论\n"
            f"需修改：建议先添加相关文档到知识库，然后重新查询"
        )
    
    def _execute_tool_caller(self, query: str, context: List[Dict[str, Any]], agent_info: Dict[str, Any]) -> tuple:
        """执行工具调用智能体，使用结构化提示词模板"""
        system_prompt = agent_info.get("system_prompt", "")
        has_context = context and len(context) > 0
        
        # 使用新的提示词模板系统
        prompt = PromptTemplate.build_agent_prompt(
            agent_name="tool_caller",
            system_prompt=system_prompt,
            query=query,
            context=context,
            has_context=has_context
        )
        
        reasoning = "说明工具调用与流程"
        
        try:
            resp = self.llm.complete(prompt, temperature=0.3, max_tokens=800)
            return resp.text, reasoning
        except Exception as e:
            logger.warning(f"工具说明LLM调用失败，使用智能降级: {e}")
            # 智能降级：提供基本流程说明
            return self._fallback_tool_caller(query, context), "LLM调用失败，使用简化流程说明"
    
    def _fallback_tool_caller(self, query: str, context: List[Dict[str, Any]]) -> str:
        """工具调用智能体的智能降级"""
        return (
            f"## 系统流程说明（简化版）\n\n"
            f"### 流程图（Mermaid）\n"
            f"```mermaid\n"
            f"graph TD\n"
            f"    A[用户问题: {query[:30]}...] --> B[知识检索]\n"
            f"    B --> C[内容分析]\n"
            f"    C --> D[答案生成]\n"
            f"    D --> E[质量评估]\n"
            f"    E --> F[最终答案]\n"
            f"```\n\n"
            f"### 流程说明\n"
            f"1. **知识检索**：从知识库中检索相关文档（共{len(context)}条）\n"
            f"2. **内容分析**：分析检索内容，提取关键信息\n"
            f"3. **答案生成**：基于分析结果生成结构化答案\n"
            f"4. **质量评估**：评估答案质量并提供改进建议\n"
            f"5. **最终答案**：整合各阶段结果，提供最终答案\n\n"
            f"### 工具输入输出\n"
            f"- 检索工具：输入问题，输出相关文档\n"
            f"- 分析工具：输入文档，输出关键信息\n"
            f"- 生成工具：输入关键信息，输出答案\n"
            f"- 评估工具：输入答案，输出质量评估\n\n"
            f"### 注意事项\n"
            f"由于系统限制，当前流程说明为简化版本。完整流程可能包含更多细节和交互。"
        )
    
    def _execute_custom(self, query: str, context: List[Dict[str, Any]], agent_info: Dict[str, Any]) -> tuple:
        """执行自定义智能体，使用结构化提示词模板"""
        system_prompt = agent_info.get("system_prompt", "")
        has_context = context and len(context) > 0
        
        # 使用新的提示词模板系统
        prompt = PromptTemplate.build_agent_prompt(
            agent_name="custom",
            system_prompt=system_prompt,
            query=query,
            context=context,
            has_context=has_context
        )
        
        reasoning = "依据系统提示的通用回答"
        
        try:
            resp = self.llm.complete(prompt, temperature=0.6, max_tokens=1200)
            return resp.text, reasoning
        except Exception as e:
            logger.warning(f"自定义LLM调用失败，使用智能降级: {e}")
            # 智能降级：提供基本回答
            return self._fallback_custom(query, context), "LLM调用失败，使用简化回答"
    
    def _fallback_custom(self, query: str, context: List[Dict[str, Any]]) -> str:
        """自定义智能体的智能降级"""
        if context and len(context) > 0:
            context_snippets = [doc.get("content", "")[:200] + "..." for doc in context[:3]]
            return (
                f"## 答案（简化版）\n\n"
                f"针对问题'{query}'，基于{len(context)}条相关文档的初步回答。\n\n"
                f"### 主要内容\n"
                + "\n".join([f"- {snippet}" for snippet in context_snippets]) +
                f"\n\n### 注意事项\n"
                f"由于系统限制，当前回答为简化版本。建议重新提问以获得更详细的回答。"
            )
        else:
            return (
                f"## 答案（简化版）\n\n"
                f"针对问题'{query}'，未检索到相关上下文信息。\n\n"
                f"### 建议\n"
                f"1. 尝试重新表述问题\n"
                f"2. 添加更多上下文信息\n"
                f"3. 使用更具体的关键词\n\n"
                f"### 注意事项\n"
                f"当前知识库可能缺少相关领域的文档，建议扩充知识库内容后再次查询。"
            )


class SynthesizerNode:
    """结果整合节点"""
    
    def __call__(self, state: AgentState) -> SynthesizerState:
        """执行整合逻辑"""
        try:
            agent_outputs = state.get("intermediate_steps", [])
            
            # 整合智能体输出
            final_answer = self._synthesize_answer(agent_outputs)
            
            # 整合引用
            final_citations = state.get("citations", [])
            
            # 生成整合推理
            synthesis_reasoning = self._generate_synthesis_reasoning(agent_outputs)
            
            # 计算质量分数
            quality_score = self._calculate_quality_score(final_answer, final_citations)
            
            return SynthesizerState(
                agent_outputs=agent_outputs,
                final_answer=final_answer,
                final_citations=final_citations,
                synthesis_reasoning=synthesis_reasoning,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"整合节点执行失败: {e}")
            raise
    
    def _synthesize_answer(self, agent_outputs: List[IntermediateStep]) -> str:
        """整合答案"""
        if not agent_outputs:
            return "抱歉，我无法生成答案。"
        
        # 提取各智能体的输出并以明显分隔符拼接
        blocks: List[str] = []
        step_index = 0
        for step in agent_outputs:
            if step.step_type != "agent_output":
                continue
            step_index += 1
            agent_name = getattr(step, "agent_name", None) or "unknown"
            header = (
                f"\n\n============================================================\n"
                f"<<<< STEP {step_index} | AGENT: {agent_name} >>>>\n"
                f"============================================================\n"
            )
            footer = (
                f"\n============================================================\n"
                f"<<<< END OF AGENT: {agent_name} (STEP {step_index}) >>>>\n"
                f"============================================================\n"
            )
            blocks.append(f"{header}{step.content}{footer}")
        
        if blocks:
            return "".join(blocks)
        # 兜底
        return "基于检索到的信息，我为您提供以下答案..."
    
    def _generate_synthesis_reasoning(self, agent_outputs: List[IntermediateStep]) -> str:
        """生成整合推理"""
        reasoning = f"我整合了{len(agent_outputs)}个智能体的输出，"
        reasoning += "通过分析、写作、批评和工具调用的协作，"
        reasoning += "生成了最终的答案。"
        
        return reasoning
    
    def _calculate_quality_score(self, answer: str, citations: List[Citation]) -> float:
        """计算质量分数"""
        score = 0.0
        
        # 基于答案长度
        if len(answer) > 100:
            score += 0.3
        
        # 基于引用数量
        if len(citations) > 0:
            score += 0.4
        
        # 基于引用质量
        if citations:
            avg_score = sum(c.score for c in citations) / len(citations)
            score += min(avg_score * 0.3, 0.3)
        
        return min(score, 1.0)


class PromptEffectivenessTracker:
    """提示词效果跟踪器，用于评估提示词效果并持续优化"""
    
    def __init__(self):
        self.effectiveness_history = {}
    
    def track_prompt_effectiveness(self, agent_name: str, prompt_type: str, 
                                 context_quality: float, response_length: int, 
                                 response_quality: float) -> None:
        """跟踪提示词效果"""
        key = f"{agent_name}_{prompt_type}"
        
        if key not in self.effectiveness_history:
            self.effectiveness_history[key] = {
                "count": 0,
                "avg_context_quality": 0.0,
                "avg_response_length": 0,
                "avg_response_quality": 0.0,
                "total_score": 0.0
            }
        
        history = self.effectiveness_history[key]
        history["count"] += 1
        
        # 更新平均值
        alpha = 0.1  # 学习率，控制历史数据的权重
        history["avg_context_quality"] = (
            alpha * context_quality + 
            (1 - alpha) * history["avg_context_quality"]
        )
        history["avg_response_length"] = (
            alpha * response_length + 
            (1 - alpha) * history["avg_response_length"]
        )
        history["avg_response_quality"] = (
            alpha * response_quality + 
            (1 - alpha) * history["avg_response_quality"]
        )
        
        # 计算综合分数
        history["total_score"] = (
            history["avg_context_quality"] * 0.3 +
            min(history["avg_response_length"] / 500, 1.0) * 0.3 +  # 归一化响应长度
            history["avg_response_quality"] * 0.4
        )
    
    def get_effectiveness_report(self) -> Dict[str, Any]:
        """获取提示词效果报告"""
        if not self.effectiveness_history:
            return {"message": "暂无提示词效果数据"}
        
        # 按综合分数排序
        sorted_prompts = sorted(
            self.effectiveness_history.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True
        )
        
        best_prompts = sorted_prompts[:3]
        worst_prompts = sorted_prompts[-3:]
        
        return {
            "best_performing": [
                {"prompt": key, "score": data["total_score"], "details": data}
                for key, data in best_prompts
            ],
            "worst_performing": [
                {"prompt": key, "score": data["total_score"], "details": data}
                for key, data in worst_prompts
            ],
            "recommendations": self._generate_recommendations(best_prompts, worst_prompts)
        }
    
    def _generate_recommendations(self, best_prompts: List, worst_prompts: List) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 分析最佳提示词的共同特点
        if best_prompts:
            best_agents = [prompt[0].split("_")[0] for prompt, _ in best_prompts]
            best_types = [prompt[0].split("_")[1] for prompt, _ in best_prompts]
            
            if len(set(best_agents)) == 1:
                recommendations.append(f"智能体 '{best_agents[0]}' 的提示词表现最佳，可将其模式推广到其他智能体")
            
            if len(set(best_types)) == 1:
                recommendations.append(f"'{best_types[0]}' 类型的提示词效果最好，建议优先使用此类提示词")
        
        # 分析最差提示词的问题
        if worst_prompts:
            worst_agents = [prompt[0].split("_")[0] for prompt, _ in worst_prompts]
            worst_types = [prompt[0].split("_")[1] for prompt, _ in worst_prompts]
            
            if len(set(worst_agents)) == 1:
                recommendations.append(f"智能体 '{worst_agents[0]}' 的提示词需要重点优化")
            
            if len(set(worst_types)) == 1:
                recommendations.append(f"'{worst_types[0]}' 类型的提示词效果不佳，建议调整此类提示词结构")
        
        # 通用建议
        recommendations.append("定期评估提示词效果，根据实际表现持续优化")
        recommendations.append("针对不同质量上下文使用差异化提示词策略")
        
        return recommendations


class DirectQANode:
    """直接问答节点 - 用于max_turns=1时的简化RAG问答"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def __call__(self, state: AgentState) -> AgentExecutionState:
        """基于检索结果直接生成答案"""
        try:
            query = state.get("query", "")
            context = state.get("kb_context", [])
            
            # 构建提示词
            prompt = self._build_direct_qa_prompt(query, context)
            
            # 调用LLM生成答案
            start_time = datetime.now()
            try:
                resp = self.llm.complete(prompt, temperature=0.7, max_tokens=2048)
                answer = resp.text
            except Exception as e:
                logger.warning(f"LLM调用失败: {e}")
                answer = self._fallback_answer(query, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 生成推理说明
            reasoning = f"基于{len(context)}条检索文档直接生成答案" if context else "无检索结果，基于通用知识回答"
            
            return AgentExecutionState(
                agent_name="direct_qa",
                agent_type="qa",
                input_content=str(context),
                output_content=answer,
                reasoning=reasoning,
                confidence=0.85,
                tools_used=["rag_retriever"],
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"直接问答节点执行失败: {e}")
            raise
    
    def _build_direct_qa_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """构建直接问答提示词"""
        if context and len(context) > 0:
            # 有检索结果
            context_text = "\n\n".join([
                f"【文档{i+1}】\n{doc.get('content', '')}"
                for i, doc in enumerate(context[:5])
            ])
            
            prompt = f"""你是一个专业的问答助手。请基于以下检索到的文档内容，回答用户的问题。

# 检索文档（共{len(context)}条）

{context_text}

# 用户问题

{query}

# 回答要求

1. **准确性**：答案必须基于上述文档内容，不要编造信息
2. **完整性**：全面回答问题的各个方面
3. **引用**：在关键结论处标注来源（如：[文档1]、[文档2]）
4. **结构化**：使用清晰的段落和标题组织答案
5. **简洁性**：避免冗余，直击要点

请开始回答："""
        else:
            # 无检索结果
            prompt = f"""你是一个专业的问答助手。用户的问题是：

{query}

# 回答要求

1. **知识边界**：由于未检索到相关文档，请明确说明这一点
2. **通用知识**：如果问题属于通用知识范畴，可基于你的知识库回答
3. **建议**：提供用户如何获取更准确答案的建议
4. **诚实**：不要编造或猜测特定领域的专业信息

请开始回答："""
        
        return prompt
    
    def _fallback_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        """LLM失败时的降级答案"""
        if context and len(context) > 0:
            # 提取文档摘要
            snippets = [doc.get("content", "")[:200] + "..." for doc in context[:3]]
            return f"""## 问答结果

针对问题：**{query}**

### 检索到的相关内容

基于{len(context)}条相关文档：

""" + "\n\n".join([f"{i+1}. {snippet}" for i, snippet in enumerate(snippets)]) + f"""

### 说明

由于系统限制，无法生成完整答案。以上是检索到的相关内容摘要，您可以基于这些信息查找更详细的答案。

建议：
- 重新表述问题以获得更精确的检索结果
- 查看原始文档获取完整信息
"""
        else:
            return f"""## 问答结果

针对问题：**{query}**

### 说明

抱歉，未能检索到相关文档内容。

建议：
1. 确认问题中的关键词是否准确
2. 尝试使用不同的表述方式
3. 检查知识库中是否包含相关内容
4. 如果是通用问题，可以尝试搜索引擎或其他资源
"""


class StopCheckerNode:
    """终止检查节点"""
    
    def __call__(self, state: AgentState) -> StopCheckerState:
        """执行终止检查逻辑"""
        try:
            # 优先使用整合器的最终答案（已拼接所有智能体输出）
            synth = (state.get("working_memory", {}) or {}).get("synthesis") or {}
            synth_answer = synth.get("final_answer") if hasattr(synth, 'get') else None
            if synth_answer:
                current_answer = synth_answer
            else:
                current_answer = state.get("intermediate_steps", [])[-1].content if state.get("intermediate_steps") else ""
            turn_count = state.get("turn_count", 0)
            max_turns = state.get("max_turns", 10)
            
            # 计算质量指标
            quality_metrics = self._calculate_quality_metrics(current_answer, state)
            
            # 判断是否应该继续
            should_continue = self._should_continue(quality_metrics, turn_count, max_turns)
            
            # 生成改进建议
            improvement_suggestions = self._generate_improvement_suggestions(quality_metrics)
            
            # 确定终止原因
            termination_reason = self._determine_termination_reason(should_continue, turn_count, max_turns)
            
            return StopCheckerState(
                current_answer=current_answer,
                quality_metrics=quality_metrics,
                should_continue=should_continue,
                improvement_suggestions=improvement_suggestions,
                termination_reason=termination_reason
            )
            
        except Exception as e:
            logger.error(f"终止检查节点执行失败: {e}")
            raise
    
    def _calculate_quality_metrics(self, answer: str, state: AgentState) -> Dict[str, float]:
        """计算质量指标"""
        metrics = {}
        
        # 答案长度指标
        metrics["answer_length"] = len(answer) / 1000.0  # 归一化到0-1
        
        # 引用数量指标（优先使用整合器的最终引用）
        working_memory = state.get("working_memory", {}) or {}
        synth = working_memory.get("synthesis") or {}
        if hasattr(synth, 'get') and synth.get("final_citations") is not None:
            citations_count = len(synth.get("final_citations") or [])
        else:
            citations_count = len(state.get("citations", []))
        metrics["citations_count"] = min(citations_count / 5.0, 1.0)
        
        # 轮次指标
        turn_count = state.get("turn_count", 0)
        metrics["turn_efficiency"] = 1.0 - (turn_count / 10.0)
        
        return metrics
    
    def _should_continue(self, quality_metrics: Dict[str, float], turn_count: int, max_turns: int) -> bool:
        """判断是否应该继续"""
        # 当用户将 max_turns 设置为 1 时，表示仅希望执行单智能体（直接由大模型回答），
        # 因此无需等待完整的多智能体协作流程，首轮执行完即可终止。
        if max_turns <= 1:
            return False

        # 否则仍按原有多智能体协作逻辑处理。
        # 确保至少完成一轮完整协作（analyst → writer → critic = 3个智能体）
        MIN_AGENTS_EXECUTED = 3
        if turn_count < MIN_AGENTS_EXECUTED:
            return False  # 首轮必须完成，不继续迭代
        
        # 达到最大轮次，停止
        if turn_count >= max_turns:
            return False
        
        # 计算平均质量
        avg_quality = sum(quality_metrics.values()) / len(quality_metrics) if quality_metrics else 0
        
        # 质量足够高且已完成至少一轮，停止
        if turn_count >= MIN_AGENTS_EXECUTED and avg_quality >= 0.6:
            return False
        
        # 答案长度足够且已完成至少一轮，停止
        if turn_count >= MIN_AGENTS_EXECUTED and quality_metrics.get("answer_length", 0) >= 0.4:
            return False
        
        # 否则继续优化（但不超过最大轮次）
        return turn_count < max_turns
    
    def _generate_improvement_suggestions(self, quality_metrics: Dict[str, float]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if quality_metrics.get("answer_length", 0) < 0.5:
            suggestions.append("答案内容较短，建议补充更多详细信息")
        
        if quality_metrics.get("citations_count", 0) < 0.3:
            suggestions.append("引用数量不足，建议检索更多相关文档")
        
        if quality_metrics.get("turn_efficiency", 0) < 0.5:
            suggestions.append("执行效率较低，建议优化协作流程")
        
        return suggestions
    
    def _determine_termination_reason(self, should_continue: bool, turn_count: int, max_turns: int) -> str:
        """确定终止原因"""
        if not should_continue:
            if turn_count >= max_turns:
                return "达到最大轮次限制"
            else:
                return "答案质量满足要求"
        else:
            return "需要继续优化"
