"""
聊天服务 - 支持会话历史持久化存储
"""
import logging
import json
import os
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import uuid

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from backend.models.chat import ChatHistoryItem, ChatSessionInfo
from backend.utils.logger import get_logger
from backend.config.settings import settings

logger = get_logger(__name__)


class ChatService:
    """聊天服务 - 支持持久化存储"""
    
    def __init__(self):
        """初始化聊天服务"""
        # 持久化存储目录
        self._storage_dir = Path(settings.chat_history_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 会话索引文件
        self._sessions_file = self._storage_dir / "sessions_index.json"
        
        # 内存缓存
        self._chat_history: Dict[str, List[Dict[str, Any]]] = {}
        self._chat_sessions: Dict[str, Dict[str, Any]] = {}
        
        # 启动时加载已有数据
        self._load_from_disk()
    
    def _load_from_disk(self):
        """从磁盘加载会话数据"""
        try:
            # 加载会话索引
            if self._sessions_file.exists():
                with open(self._sessions_file, 'r', encoding='utf-8') as f:
                    sessions_data = json.load(f)
                    for session_id, session_info in sessions_data.items():
                        # 转换时间字符串
                        if 'created_at' in session_info and isinstance(session_info['created_at'], str):
                            session_info['created_at'] = datetime.fromisoformat(session_info['created_at'])
                        if 'last_activity' in session_info and isinstance(session_info['last_activity'], str):
                            session_info['last_activity'] = datetime.fromisoformat(session_info['last_activity'])
                        self._chat_sessions[session_id] = session_info
                logger.info(f"从磁盘加载了 {len(self._chat_sessions)} 个会话索引")
            
            # 加载每个会话的聊天历史
            for session_id in self._chat_sessions.keys():
                history_file = self._storage_dir / f"{session_id}.json"
                if history_file.exists():
                    with open(history_file, 'r', encoding='utf-8') as f:
                        history_data = json.load(f)
                        # 转换时间字符串
                        for item in history_data:
                            if 'timestamp' in item and isinstance(item['timestamp'], str):
                                item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        self._chat_history[session_id] = history_data
            
            logger.info(f"从磁盘加载了 {len(self._chat_history)} 个会话的聊天历史")
                        
        except Exception as e:
            logger.error(f"从磁盘加载会话数据失败: {e}")
    
    def _save_sessions_index(self):
        """保存会话索引到磁盘"""
        try:
            sessions_data = {}
            for session_id, session_info in self._chat_sessions.items():
                sessions_data[session_id] = {
                    'session_id': session_info.get('session_id', session_id),
                    'created_at': session_info.get('created_at', datetime.now()).isoformat() if isinstance(session_info.get('created_at'), datetime) else session_info.get('created_at'),
                    'last_activity': session_info.get('last_activity', datetime.now()).isoformat() if isinstance(session_info.get('last_activity'), datetime) else session_info.get('last_activity'),
                    'message_count': session_info.get('message_count', 0),
                    'status': session_info.get('status', 'active'),
                    'agent_id': session_info.get('agent_id'),
                    'user_id': session_info.get('user_id')
                }
            
            with open(self._sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存会话索引失败: {e}")
    
    def _save_session_history(self, session_id: str):
        """保存单个会话的聊天历史到磁盘"""
        try:
            history_data = self._chat_history.get(session_id, [])
            if not history_data:
                return
            
            # 转换为可序列化格式
            serializable_history = []
            for item in history_data:
                serializable_item = {
                    'query_id': item.get('query_id'),
                    'query': item.get('query'),
                    'answer': item.get('answer'),
                    'citations': item.get('citations', []),
                    'agent_used': item.get('agent_used'),
                    'timestamp': item.get('timestamp').isoformat() if isinstance(item.get('timestamp'), datetime) else item.get('timestamp'),
                    'processing_time': item.get('processing_time', 0.0)
                }
                serializable_history.append(serializable_item)
            
            history_file = self._storage_dir / f"{session_id}.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"保存会话历史到磁盘: {session_id}")
                
        except Exception as e:
            logger.error(f"保存会话历史失败: {session_id}, {e}")
    
    def get_chat_history(
        self,
        session_id: str,
        page: int = 1,
        size: int = 10
    ) -> Tuple[List[ChatHistoryItem], int]:
        """获取聊天历史"""
        try:
            logger.info(f"获取聊天历史: session_id={session_id}, page={page}, size={size}")
            
            # 从内存中获取聊天历史（实际项目中应该从数据库获取）
            history_data = self._chat_history.get(session_id, [])
            
            # 分页处理
            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paginated_history = history_data[start_idx:end_idx]
            
            # 转换为模型对象
            history_items = []
            for item in paginated_history:
                history_items.append(ChatHistoryItem(
                    query_id=item.get("query_id", str(uuid.uuid4())),
                    query=item.get("query", ""),
                    answer=item.get("answer", ""),
                    citations=item.get("citations", []),
                    agent_used=item.get("agent_used"),
                    timestamp=item.get("timestamp", datetime.now()),
                    processing_time=item.get("processing_time", 0.0)
                ))
            
            total = len(history_data)
            return history_items, total
            
        except Exception as e:
            logger.error(f"获取聊天历史失败: {e}")
            raise
    
    def list_chat_sessions(
        self,
        page: int = 1,
        size: int = 10,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[List[ChatSessionInfo], int]:
        """列出聊天会话"""
        try:
            logger.info(f"列出聊天会话: page={page}, size={size}, agent_id={agent_id}")
            
            # 从内存中获取会话列表（实际项目中应该从数据库获取）
            sessions_data = list(self._chat_sessions.values())
            
            # 过滤逻辑
            if agent_id:
                sessions_data = [s for s in sessions_data if s.get("agent_id") == agent_id]
            if status:
                sessions_data = [s for s in sessions_data if s.get("status") == status]
            if user_id:
                sessions_data = [s for s in sessions_data if s.get("user_id") == user_id]
            
            # 分页处理
            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paginated_sessions = sessions_data[start_idx:end_idx]
            
            # 转换为模型对象
            session_infos = []
            for session in paginated_sessions:
                session_infos.append(ChatSessionInfo(
                    session_id=session.get("session_id", str(uuid.uuid4())),
                    agent_id=session.get("agent_id"),
                    created_at=session.get("created_at", datetime.now()),
                    last_activity=session.get("last_activity", datetime.now()),
                    message_count=session.get("message_count", 0),
                    status=session.get("status", "active")
                ))
            
            total = len(sessions_data)
            return session_infos, total
            
        except Exception as e:
            logger.error(f"列出聊天会话失败: {e}")
            raise
    
    def delete_chat_session(self, session_id: str) -> Dict[str, Any]:
        """删除聊天会话"""
        try:
            logger.info(f"删除聊天会话: session_id={session_id}")
            
            # 从内存中删除会话
            if session_id in self._chat_sessions:
                del self._chat_sessions[session_id]
            
            # 删除对应的聊天历史
            if session_id in self._chat_history:
                del self._chat_history[session_id]
            
            # 从磁盘删除会话历史文件
            history_file = self._storage_dir / f"{session_id}.json"
            if history_file.exists():
                history_file.unlink()
                logger.info(f"删除会话历史文件: {history_file}")
            
            # 更新会话索引
            self._save_sessions_index()
            
            return {"deleted": True, "session_id": session_id}
            
        except Exception as e:
            logger.error(f"删除聊天会话失败: {e}")
            raise
    
    def get_chat_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取聊天统计信息"""
        try:
            logger.info(f"获取聊天统计: user_id={user_id}")
            
            # 从内存中计算统计信息（实际项目中应该从数据库计算）
            sessions_data = list(self._chat_sessions.values())
            
            # 过滤用户会话
            if user_id:
                user_sessions = [s for s in sessions_data if s.get("user_id") == user_id]
            else:
                user_sessions = sessions_data
            
            # 计算统计信息
            stats = {
                "total_sessions": len(user_sessions),
                "active_sessions": len([s for s in user_sessions if s.get("status") == "active"]),
                "total_messages": sum(s.get("message_count", 0) for s in user_sessions),
                "avg_messages_per_session": 0,
                "last_activity": max([s.get("last_activity", datetime.now()) for s in user_sessions]) if user_sessions else None
            }
            
            if stats["total_sessions"] > 0:
                stats["avg_messages_per_session"] = stats["total_messages"] / stats["total_sessions"]
            
            return stats
            
        except Exception as e:
            logger.error(f"获取聊天统计失败: {e}")
            raise
    
    def save_chat_history(
        self,
        session_id: str,
        query: str,
        answer: str,
        citations: List[Dict[str, Any]] = None,
        agent_used: Optional[str] = None,
        processing_time: float = 0.0
    ) -> str:
        """保存聊天历史"""
        try:
            query_id = str(uuid.uuid4())
            
            # 创建聊天历史项
            history_item = {
                "query_id": query_id,
                "query": query,
                "answer": answer,
                "citations": citations or [],
                "agent_used": agent_used,
                "timestamp": datetime.now(),
                "processing_time": processing_time
            }
            
            # 保存到内存（实际项目中应该保存到数据库）
            if session_id not in self._chat_history:
                self._chat_history[session_id] = []
            self._chat_history[session_id].append(history_item)
            
            # 更新会话信息
            if session_id not in self._chat_sessions:
                self._chat_sessions[session_id] = {
                    "session_id": session_id,
                    "created_at": datetime.now(),
                    "last_activity": datetime.now(),
                    "message_count": 0,
                    "status": "active"
                }
            
            session = self._chat_sessions[session_id]
            session["last_activity"] = datetime.now()
            session["message_count"] += 1
            
            # 持久化到磁盘
            self._save_session_history(session_id)
            self._save_sessions_index()
            
            logger.info(f"保存聊天历史: session_id={session_id}, query_id={query_id}")
            return query_id
            
        except Exception as e:
            logger.error(f"保存聊天历史失败: {e}")
            raise
    
    def get_message_history(
        self,
        session_id: str,
        max_rounds: int = 5
    ) -> List[BaseMessage]:
        """
        获取指定会话的消息历史，转换为 LangChain BaseMessage 列表
        
        用于注入到 LangGraph 的 AgentState.messages 中，实现多轮对话上下文记忆
        
        Args:
            session_id: 会话ID
            max_rounds: 最大历史轮次，防止上下文过长（默认5轮）
            
        Returns:
            List[BaseMessage]: HumanMessage 和 AIMessage 交替的消息列表
        """
        try:
            # 获取该会话的历史记录
            history_items = self._chat_history.get(session_id, [])
            
            if not history_items:
                logger.debug(f"会话 {session_id} 无历史记录")
                return []
            
            # 截取最近 N 轮，避免上下文过长
            recent_items = history_items[-max_rounds:]
            
            # 转换为 LangChain 消息格式
            messages: List[BaseMessage] = []
            for item in recent_items:
                query = item.get("query", "")
                answer = item.get("answer", "")
                
                if query:
                    messages.append(HumanMessage(content=query))
                if answer:
                    messages.append(AIMessage(content=answer))
            
            logger.info(f"获取会话历史消息: session_id={session_id}, rounds={len(recent_items)}, messages={len(messages)}")
            return messages
            
        except Exception as e:
            logger.error(f"获取消息历史失败: {e}")
            return []  # 出错时返回空列表，不影响当前请求