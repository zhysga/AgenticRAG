"""
问答协作路由 - 后端API接口定义

这个文件定义了问答协作相关的所有API端点，是前后端通信的核心桥梁。
前端通过调用这些API与后端的LangGraph多智能体系统进行交互。

关键功能：
1. 提问接口（支持流式和非流式响应）
2. 聊天历史管理
3. 会话管理
4. 统计信息获取

前后端结合机制：
- 前端通过HTTP请求调用这些API端点
- 后端使用FastAPI框架处理请求并返回JSON响应
- 支持CORS跨域请求，允许前端应用访问后端API
- 使用依赖注入管理服务实例
"""

# 导入必要的库和模块
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse  # 用于流式响应
from typing import List, Optional, Dict, Any
import uuid  # 生成唯一ID
from datetime import datetime
import json  # JSON处理
import time
from enum import Enum

# 导入数据模型 - 定义API请求和响应的数据结构
from backend.models.chat import (
    ChatAskRequest, ChatAskResponse, ChatAnswer,
    ChatHistoryRequest, ChatHistoryResponse,
    ChatSessionListRequest, ChatSessionListResponse,
    ChatSessionDeleteRequest, ChatSessionDeleteResponse,
    ChatStatsResponse, ChatStats
)

# 导入服务层 - 业务逻辑处理
from backend.services.langgraph_service import LangGraphService  # LangGraph工作流服务
from backend.services.chat_service import ChatService  # 聊天服务

# 导入依赖注入函数 - 管理服务实例生命周期
from backend.dependencies import get_langgraph_service, get_chat_service

# 导入认证和日志工具
from backend.utils.auth import get_current_user  # 用户认证
from backend.utils.logger import get_logger  # 日志记录
from backend.models.base import Citation, IntermediateStep  # 基础数据模型


def _json_default(obj):
    """JSON序列化默认处理函数，用于处理datetime和Enum类型"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# 创建日志记录器实例
logger = get_logger(__name__)

# 创建FastAPI路由器实例
# prefix="/chat" 表示所有路由都以/chat开头
# tags=["问答协作"] 用于API文档分组
router = APIRouter(prefix="/chat", tags=["问答协作"])


@router.post("/ask", response_model=ChatAskResponse)
async def ask_question(
    ask_data: ChatAskRequest,
    langgraph_service: LangGraphService = Depends(get_langgraph_service),
    chat_service: ChatService = Depends(get_chat_service),
    current_user: str = Depends(get_current_user),
    request: Request = None,
):
    """提问（支持流式返回，当ask_data.stream为True时返回NDJSON流）"""
    try:
        req_id = (request.headers.get("X-Request-ID") if request else None) or str(uuid.uuid4())
        logger.info(
            "[backend] /chat/ask: received",
            extra={
                "request_id": req_id,
                "user_id": current_user,
                "stream": ask_data.stream,
                "agent_id": ask_data.agent_id,
                "top_k": ask_data.top_k,
                "rerank": ask_data.rerank,
                "temperature": ask_data.temperature,
                "max_turns": ask_data.max_turns,
                "filters_keys": list((ask_data.filters.dict(exclude_none=True) or {}).keys()),
            },
        )
        
        # 获取历史消息（用于多轮对话上下文）
        chat_messages = None
        if ask_data.session_id:
            chat_messages = chat_service.get_message_history(ask_data.session_id)

        if ask_data.stream:
            request_id = str(uuid.uuid4())

            def sse(event: str, data: Any) -> str:
                return f"event: {event}\n" + f"data: {json.dumps(data, ensure_ascii=False, default=_json_default)}\n\n"

            def build_chunk_raw(
                *,
                content: str,
                is_finish: bool,
                message_type: str = "answer",
                reply_id: Optional[str] = None,
            ) -> Dict[str, Any]:
                return {
                    "index": 0,
                    "seq_id": 0,
                    "is_finish": is_finish,
                    "message": {
                        "role": "assistant",
                        "type": message_type,
                        "section_id": str(uuid.uuid4()),
                        "content_type": "text",
                        "content": content,
                        "reasoning_content": "",
                        "message_status": "available",
                        "message_id": str(uuid.uuid4()),
                        "reply_id": reply_id or request_id,
                        "extra_info": {
                            # 前端可选字段，不强制要求
                            "local_message_id": "",
                        },
                    },
                }

            async def generate_stream():
                try:
                    # 发送 ack 消息，标记开始
                    yield sse(
                        "message",
                        build_chunk_raw(content="", is_finish=False, message_type="ack"),
                    )

                    # 执行工作流（异步，传入历史消息）
                    result = await langgraph_service.execute_workflow(
                        query=ask_data.query,
                        agent_id=ask_data.agent_id,
                        agent_config=ask_data.agent_profile,
                        session_id=ask_data.session_id,
                        top_k=ask_data.top_k,
                        filters=ask_data.filters.dict(exclude_none=True),
                        rerank=ask_data.rerank,
                        temperature=ask_data.temperature,
                        max_turns=ask_data.max_turns,
                        request_id=req_id,
                        chat_messages=chat_messages,
                    )

                    # 返回最终答案为一次性 message 事件
                    final_answer = result.get("answer", "")
                    
                    # 保存聊天历史（用于后续多轮对话）
                    session_id = result.get("session_id", ask_data.session_id or str(uuid.uuid4()))
                    chat_service.save_chat_history(
                        session_id=session_id,
                        query=ask_data.query,
                        answer=final_answer,
                        citations=result.get("citations", []),
                        agent_used=result.get("agent_used"),
                        processing_time=result.get("processing_time", 0.0)
                    )
                    
                    yield sse(
                        "message",
                        build_chunk_raw(content=final_answer, is_finish=True, message_type="answer"),
                    )

                    # 终止事件
                    yield sse("done", "")

                except HTTPException as e:
                    # 业务错误事件
                    yield sse("error", {"code": e.status_code, "msg": e.detail})
                except Exception as e:
                    logger.error(f"流式问答处理异常: {e}")
                    yield sse("error", {"code": 500, "msg": str(e)})

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # 非流式：直接返回完整结果（传入历史消息）
        result = await langgraph_service.execute_workflow(
            query=ask_data.query,
            agent_id=ask_data.agent_id,
            agent_config=ask_data.agent_profile,
            session_id=ask_data.session_id,
            top_k=ask_data.top_k,
            filters=ask_data.filters.dict(exclude_none=True),
            rerank=ask_data.rerank,
            temperature=ask_data.temperature,
            max_turns=ask_data.max_turns,
            request_id=req_id,
            chat_messages=chat_messages,
        )
        try:
            logger.info(
                "[backend] /chat/ask: workflow complete",
                extra={
                    "request_id": req_id,
                    "session_id": result.get("session_id"),
                    "query_id": result.get("query_id"),
                    "citations_count": len(result.get("citations", [])),
                    "intermediate_steps_count": len(result.get("intermediate_steps", [])),
                    "answer_len": len(result.get("answer", "")),
                },
            )
        except Exception:
            pass

        # 保存聊天历史（用于后续多轮对话）
        final_session_id = result.get("session_id", ask_data.session_id or str(uuid.uuid4()))
        chat_service.save_chat_history(
            session_id=final_session_id,
            query=ask_data.query,
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            agent_used=result.get("agent_used"),
            processing_time=result.get("processing_time", 0.0)
        )

        # 构建结构化答案
        answer_obj = ChatAnswer(
            answer=result.get("answer", ""),
            citations=[Citation(**c) for c in result.get("citations", [])],
            intermediate_steps=[IntermediateStep(**s) for s in result.get("intermediate_steps", [])],
            agent_used=result.get("agent_used"),
            session_id=final_session_id,
            query_id=result.get("query_id", ""),
            processing_time=float(result.get("processing_time", 0.0)),
            metadata=result.get("metadata")
        )

        return ChatAskResponse(
            status="success",
            message="问答成功",
            request_id=str(uuid.uuid4()),
            data=answer_obj
        )

    except HTTPException:
        # 传递业务HTTP错误
        raise
    except Exception as e:
        logger.error(f"提问失败: {e}")
        # 返回统一结构的错误
        raise HTTPException(status_code=500, detail="内部服务器错误")


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    page: int = 1,
    size: int = 10,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: str = Depends(get_current_user)
):
    """获取聊天历史"""
    try:
        logger.info(f"获取聊天历史请求: session_id={session_id}")
        
        # 获取聊天历史
        history, total = chat_service.get_chat_history(
            session_id=session_id,
            page=page,
            size=size
        )
        
        return ChatHistoryResponse(
            status="success",
            message="获取聊天历史成功",
            request_id=str(uuid.uuid4()),
            data=history,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
        
    except Exception as e:
        logger.error(f"获取聊天历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=ChatSessionListResponse)
async def list_chat_sessions(
    page: int = 1,
    size: int = 10,
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: str = Depends(get_current_user)
):
    """列出聊天会话"""
    try:
        logger.info(f"列出聊天会话请求: page={page}, size={size}")
        
        # 获取会话列表
        sessions, total = chat_service.list_chat_sessions(
            page=page,
            size=size,
            agent_id=agent_id,
            status=status,
            user_id=current_user
        )
        
        return ChatSessionListResponse(
            status="success",
            message="获取会话列表成功",
            request_id=str(uuid.uuid4()),
            data=sessions,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
        
    except Exception as e:
        logger.error(f"列出聊天会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete_session", response_model=ChatSessionDeleteResponse)
async def delete_chat_session(
    delete_data: ChatSessionDeleteRequest,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: str = Depends(get_current_user)
):
    """删除聊天会话"""
    try:
        logger.info(f"删除聊天会话请求: session_id={delete_data.session_id}")
        
        if not delete_data.confirm:
            raise HTTPException(status_code=400, detail="请确认删除操作")
        
        # 删除会话
        result = chat_service.delete_chat_session(delete_data.session_id)
        
        return ChatSessionDeleteResponse(
            status="success",
            message="会话删除成功",
            request_id=str(uuid.uuid4()),
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除聊天会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=ChatStatsResponse)
async def get_chat_stats(
    chat_service: ChatService = Depends(get_chat_service),
    current_user: str = Depends(get_current_user)
):
    """获取聊天统计信息"""
    try:
        logger.info("获取聊天统计请求")
        
        # 获取统计信息
        stats = chat_service.get_chat_stats(user_id=current_user)
        
        return ChatStatsResponse(
            status="success",
            message="获取统计信息成功",
            request_id=str(uuid.uuid4()),
            data=ChatStats(**stats)
        )
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def ask_question_stream(
    ask_data: ChatAskRequest,
    langgraph_service: LangGraphService = Depends(get_langgraph_service),
    chat_service: ChatService = Depends(get_chat_service),
    current_user: str = Depends(get_current_user)
):
    """流式提问（SSE事件：message/done/error，与前端解析对齐）"""
    try:
        logger.info(f"流式提问请求: query={ask_data.query[:50]}...")

        ask_data.stream = True
        request_id = str(uuid.uuid4())
        
        # 获取历史消息（用于多轮对话上下文）
        chat_messages = None
        if ask_data.session_id:
            chat_messages = chat_service.get_message_history(ask_data.session_id)

        def sse(event: str, data: Any) -> str:
            return f"event: {event}\n" + f"data: {json.dumps(data, ensure_ascii=False, default=_json_default)}\n\n"

        def build_chunk_raw(
            *,
            content: str,
            is_finish: bool,
            message_type: str = "answer",
            reply_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            return {
                "index": 0,
                "seq_id": 0,
                "is_finish": is_finish,
                "message": {
                    "role": "assistant",
                    "type": message_type,
                    "section_id": str(uuid.uuid4()),
                    "content_type": "text",
                    "content": content,
                    "reasoning_content": "",
                    "message_status": "available",
                    "message_id": str(uuid.uuid4()),
                    "reply_id": reply_id or request_id,
                    "extra_info": {
                        "local_message_id": "",
                    },
                },
            }

        async def generate_stream():
            try:
                # ack
                yield sse("message", build_chunk_raw(content="", is_finish=False, message_type="ack"))

                result = await langgraph_service.execute_workflow(
                    query=ask_data.query,
                    agent_id=ask_data.agent_id,
                    agent_config=ask_data.agent_profile,
                    session_id=ask_data.session_id,
                    top_k=ask_data.top_k,
                    filters=ask_data.filters.dict(),
                    rerank=ask_data.rerank,
                    temperature=ask_data.temperature,
                    max_turns=ask_data.max_turns,
                    chat_messages=chat_messages,
                )

                final_answer = result.get("answer", "")
                
                # 保存聊天历史（用于后续多轮对话）
                session_id = result.get("session_id", ask_data.session_id or str(uuid.uuid4()))
                chat_service.save_chat_history(
                    session_id=session_id,
                    query=ask_data.query,
                    answer=final_answer,
                    citations=result.get("citations", []),
                    agent_used=result.get("agent_used"),
                    processing_time=result.get("processing_time", 0.0)
                )
                
                yield sse("message", build_chunk_raw(content=final_answer, is_finish=True, message_type="answer"))

                yield sse("done", "")

            except HTTPException as e:
                yield sse("error", {"code": e.status_code, "msg": e.detail})
            except Exception as e:
                logger.error(f"流式问答处理异常: {e}")
                yield sse("error", {"code": 500, "msg": str(e)})

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"流式提问失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")
