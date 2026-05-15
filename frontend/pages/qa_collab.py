"""
问答协作Gradio应用

支持多轮对话上下文记忆：
- 在问答界面显示历史问题
- 在回答区域显示完整对话历史
- 新建对话按钮生成新会话ID
- 聊天历史界面列出所有会话
"""
import gradio as gr
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import logging

from services.api_client import APIClient
from config.frontend_config import FrontendConfig

# 配置
config = FrontendConfig()
api_client = APIClient(config.backend_url)


class QACollaboration:
    """问答协作管理器 - 支持多轮对话上下文记忆"""
    
    def __init__(self):
        self.current_session_id = str(uuid.uuid4())  # 默认生成一个会话ID
        self.conversation_history = []  # 当前会话的对话历史 [(query, answer), ...]
        self.available_agents = []
        self.logger = logging.getLogger(__name__)
    
    def new_conversation(self) -> Tuple[str, str, str]:
        """
        新建对话 - 生成新的会话ID，清空对话历史
        
        Returns:
            (新会话ID, 清空的输入框, 清空的对话显示)
        """
        self.current_session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.logger.info(f"[frontend] new_conversation: session_id={self.current_session_id}")
        return self.current_session_id, "", self._format_conversation_display()
    
    def _format_conversation_display(self) -> str:
        """格式化对话历史为显示文本"""
        if not self.conversation_history:
            return "💬 开始新的对话...\n\n提示：输入问题后点击「提问」开始对话"
        
        display_text = f"💬 当前会话: {self.current_session_id[:8]}...\n\n"
        display_text += "=" * 50 + "\n"
        
        for i, (query, answer, metadata) in enumerate(self.conversation_history, 1):
            display_text += f"\n🙋 问题 {i}:\n{query}\n\n"
            display_text += f"🤖 回答 {i}:\n{answer}\n"
            if metadata:
                display_text += f"\n📊 [处理时间: {metadata.get('processing_time', 0):.2f}秒"
                if metadata.get('agent_used'):
                    display_text += f" | 智能体: {metadata.get('agent_used')}"
                display_text += "]\n"
            display_text += "\n" + "-" * 50 + "\n"
        
        return display_text
    
    def ask_question(
        self,
        query: str,
        agent_id: str,
        kb_ids_csv: str,
        top_k: int,
        filters: str,
        rerank: bool,
        temperature: float,
        max_turns: int
    ) -> Tuple[str, str]:
        """
        提问方法 - 支持多轮对话上下文记忆
        
        Returns:
            (清空的输入框, 更新后的对话历史显示)
        """
        try:
            request_id = str(uuid.uuid4())
            self.logger.info(
                "[frontend] ask_question: start",
                extra={
                    "request_id": request_id,
                    "agent_id": agent_id,
                    "top_k": top_k,
                    "rerank": rerank,
                    "temperature": temperature,
                    "max_turns": max_turns,
                },
            )
            # 1. 前端输入验证
            if not query.strip():
                self.logger.warning(
                    "[frontend] ask_question: empty query",
                    extra={"request_id": request_id},
                )
                return "", self._format_conversation_display() + "\n\n❌ 请输入问题"
            
            # 2. 解析过滤条件（前端数据处理）并合并界面输入的 kb_ids
            filter_dict = {}
            if filters.strip():
                try:
                    filter_dict = json.loads(filters)  # JSON字符串转字典
                except json.JSONDecodeError:
                    self.logger.error(
                        "[frontend] ask_question: invalid filters json",
                        extra={"request_id": request_id, "filters_raw": filters[:500]},
                    )
                    return "", self._format_conversation_display() + "\n\n❌ 过滤条件格式错误，请使用JSON格式"

            # 合并 kb_ids（来自独立输入框，逗号分隔）到 filters.custom_filters.kb_ids
            merged_kb_count = 0
            if isinstance(kb_ids_csv, str) and kb_ids_csv.strip():
                ids = [s.strip() for s in kb_ids_csv.split(',') if s.strip()]
                if ids:
                    filter_dict.setdefault("custom_filters", {})
                    if isinstance(filter_dict["custom_filters"], dict):
                        existing = filter_dict["custom_filters"].get("kb_ids")
                        if isinstance(existing, list):
                            merged = list(dict.fromkeys(existing + ids))
                        elif isinstance(existing, str):
                            merged = list(dict.fromkeys([existing] + ids))
                        else:
                            merged = ids
                        filter_dict["custom_filters"]["kb_ids"] = merged
                        merged_kb_count = len(merged)

            # 记录 filters 解析与合并后的统计
            try:
                cf = filter_dict.get("custom_filters") or {}
                kb_ids = cf.get("kb_ids") if isinstance(cf, dict) else None
                self.logger.info(
                    "[frontend] ask_question: parsed/merged filters",
                    extra={
                        "request_id": request_id,
                        "filters_keys": list(filter_dict.keys()),
                        "kb_ids_count": (len(kb_ids) if isinstance(kb_ids, list) else (1 if kb_ids else 0)),
                        "kb_ids_merged_from_input_count": merged_kb_count,
                    },
                )
            except Exception:
                pass
            
            # 3. 构建符合后端 API格式的请求数据
            # 这个数据结构必须与后端的ChatAskRequest模型匹配
            data = {
                "query": query,           # 用户问题
                "session_id": self.current_session_id,  # 会话ID（多轮对话上下文记忆）
                "top_k": top_k,           # 检索数量
                "filters": filter_dict,  # 过滤条件
                "rerank": rerank,         # 是否重排序
                "temperature": temperature, # AI温度参数
                "max_turns": max_turns,   # 最大轮次
                "stream": False           # 非流式响应
            }
            
            # 4. 如果有指定智能体，添加到请求中
            if agent_id.strip():
                data["agent_id"] = agent_id
            
            # 5. 通过APIClient调用后端API
            # 这是前后端通信的关键步骤
            self.logger.info(
                "[frontend] ask_question: sending request",
                extra={
                    "request_id": request_id,
                    "payload_summary": {
                        "has_agent_id": bool(agent_id.strip()),
                        "top_k": top_k,
                        "rerank": rerank,
                        "max_turns": max_turns,
                        "filters_keys": list((filter_dict or {}).keys()),
                    },
                },
            )
            response = api_client.ask_question(data, request_id=request_id)
            
            # 6. 处理后端响应
            if response.get("status") == "success":
                # 从响应中提取数据
                result_data = response["data"]
                try:
                    self.logger.info(
                        "[frontend] ask_question: success",
                        extra={
                            "request_id": request_id,
                            "session_id": result_data.get("session_id"),
                            "query_id": result_data.get("query_id"),
                            "citations_count": len(result_data.get("citations", [])),
                            "intermediate_steps_count": len(result_data.get("intermediate_steps", [])),
                            "answer_len": len(result_data.get("answer", "")),
                        },
                    )
                except Exception:
                    pass
                
                # 7. 更新对话历史并格式化显示
                answer = result_data.get('answer', '无回答')
                metadata = {
                    'processing_time': result_data.get('processing_time', 0.0),
                    'agent_used': result_data.get('agent_used'),
                    'query_id': result_data.get('query_id'),
                    'citations': result_data.get('citations', []),
                }
                
                # 添加到对话历史
                self.conversation_history.append((query, answer, metadata))
                
                # 更新会话ID（保持与后端一致）
                if result_data.get('session_id'):
                    self.current_session_id = result_data.get('session_id')
                
                # 返回清空的输入框和更新后的对话显示
                return "", self._format_conversation_display()
            else:
                self.logger.error(
                    "[frontend] ask_question: backend error",
                    extra={
                        "request_id": request_id,
                        "backend_status": response.get("status"),
                        "backend_message": response.get("message"),
                    },
                )
                return "", self._format_conversation_display() + f"\n\n❌ 提问失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            self.logger.exception(
                "[frontend] ask_question: exception",
                extra={"request_id": request_id},
            )
            return "", self._format_conversation_display() + f"\n\n❌ 提问失败: {str(e)}"
    
    def list_all_sessions(self) -> str:
        """列出所有聊天会话及其历史 - 上方显示完整会话ID，下方显示聊天内容"""
        try:
            sessions_found = []
            
            # 方法1: 从后端API获取会话列表
            try:
                response = api_client.list_chat_sessions(page=1, size=100)
                if response.get("status") == "success":
                    backend_sessions = response.get("data", [])
                    for session in backend_sessions:
                        session_id = session.get('session_id')
                        if session_id and session_id not in [s[0] for s in sessions_found]:
                            sessions_found.append((session_id, session))
            except Exception as e:
                self.logger.warning(f"获取后端会话列表失败: {e}")
            
            # 方法2: 添加当前前端实例的会话（如果有对话历史）
            if self.conversation_history and self.current_session_id:
                if self.current_session_id not in [s[0] for s in sessions_found]:
                    sessions_found.append((self.current_session_id, {
                        'session_id': self.current_session_id,
                        'message_count': len(self.conversation_history),
                        'status': 'active (当前会话)'
                    }))
            
            if not sessions_found:
                result_text = "📝 暂无聊天记录\n\n"
                result_text += "💡 提示：在「智能问答」页面开始对话后，历史记录会显示在这里。\n\n"
                result_text += "━" * 50 + "\n"
                result_text += "当前会话ID（完整）:\n"
                result_text += f"{self.current_session_id}\n"
                return result_text
            
            # ============ 构建输出 ============
            result_text = "📋 所有聊天会话记录\n"
            result_text += f"共找到 {len(sessions_found)} 个会话\n\n"
            
            # ===== 第一部分：会话ID列表（完整显示）=====
            result_text += "━" * 50 + "\n"
            result_text += "📌 会话ID列表（可复制）:\n"
            result_text += "━" * 50 + "\n\n"
            
            for idx, (session_id, session_info) in enumerate(sessions_found, 1):
                status = session_info.get('status', 'active')
                msg_count = session_info.get('message_count', 0)
                result_text += f"[{idx}] {session_id}\n"
                result_text += f"    状态: {status} | 消息数: {msg_count}\n\n"
            
            # ===== 第二部分：聊天内容详情 =====
            result_text += "\n" + "━" * 50 + "\n"
            result_text += "💬 聊天内容详情:\n"
            result_text += "━" * 50 + "\n"
            
            for idx, (session_id, session_info) in enumerate(sessions_found, 1):
                result_text += f"\n【会话 {idx}】\n"
                result_text += f"ID: {session_id}\n"
                result_text += "─" * 40 + "\n"
                
                # 获取该会话的聊天内容
                chat_content = []
                
                # 如果是当前会话，使用本地数据
                if session_id == self.current_session_id and self.conversation_history:
                    for query, answer, _ in self.conversation_history:
                        chat_content.append({'query': query, 'answer': answer})
                else:
                    # 从后端获取历史
                    try:
                        history_resp = api_client.get_chat_history(session_id, page=1, size=50)
                        if history_resp.get("status") == "success":
                            chat_content = history_resp.get("data", [])
                    except Exception:
                        pass
                
                if chat_content:
                    for i, item in enumerate(chat_content, 1):
                        query = item.get('query', '')
                        answer = item.get('answer', '')
                        # 显示摘要
                        query_preview = query[:100] + ('...' if len(query) > 100 else '')
                        answer_preview = answer[:150] + ('...' if len(answer) > 150 else '')
                        result_text += f"  对话{i}:\n"
                        result_text += f"  🙋 问: {query_preview}\n"
                        result_text += f"  🤖 答: {answer_preview}\n\n"
                else:
                    result_text += "  (暂无聊天内容)\n\n"
            
            result_text += "━" * 50 + "\n"
            result_text += "💡 复制上方会话ID → 左侧输入框 → 点击「查看会话详情」查看完整内容\n"
            
            return result_text
                
        except Exception as e:
            return f"❌ 获取失败: {str(e)}"
    
    def get_session_history(self, session_id: str) -> str:
        """获取指定会话的完整聊天历史"""
        try:
            if not session_id.strip():
                return "❌ 请输入会话ID"
            
            result_text = f"💬 会话详情\n"
            result_text += f"{'=' * 60}\n"
            result_text += f"会话ID: {session_id}\n"
            result_text += f"{'=' * 60}\n\n"
            
            history = []
            
            # 如果是当前会话，使用本地数据
            if session_id == self.current_session_id and self.conversation_history:
                for query, answer, metadata in self.conversation_history:
                    history.append({
                        'query': query,
                        'answer': answer,
                        'processing_time': metadata.get('processing_time', 0) if metadata else 0,
                        'agent_used': metadata.get('agent_used') if metadata else None
                    })
            else:
                # 从后端获取历史
                response = api_client.get_chat_history(session_id, page=1, size=100)
                if response.get("status") == "success":
                    history = response.get("data", [])
            
            if not history:
                return f"📝 会话 {session_id} 暂无聊天记录\n\n💡 提示：请确认会话ID是否正确"
            
            result_text += f"共 {len(history)} 轮对话\n\n"
            
            for i, item in enumerate(history, 1):
                result_text += f"{'─' * 50}\n"
                result_text += f"【第 {i} 轮对话】\n\n"
                result_text += f"🙋 用户提问:\n{item.get('query', '')}\n\n"
                result_text += f"🤖 智能体回答:\n{item.get('answer', '')}\n\n"
                
                # 显示元数据
                if item.get('agent_used'):
                    result_text += f"📊 使用智能体: {item.get('agent_used')}\n"
                if item.get('processing_time'):
                    result_text += f"⏱️ 处理时间: {item.get('processing_time', 0):.2f}秒\n"
                if item.get('timestamp'):
                    result_text += f"🕐 时间: {item.get('timestamp')}\n"
                result_text += "\n"
            
            result_text += f"{'─' * 50}\n"
            result_text += f"\n💡 提示: 如需继续此会话的对话，请在「智能问答」页面使用相同的会话ID\n"
            
            return result_text
            
        except Exception as e:
            return f"❌ 获取失败: {str(e)}"
    
    def load_session(self, session_id: str) -> Tuple[str, str, str]:
        """加载指定会话到当前对话 - 可继续对话"""
        try:
            if not session_id.strip():
                return self.current_session_id, "", "❌ 请输入会话ID"
            
            # 从后端获取历史
            response = api_client.get_chat_history(session_id, page=1, size=100)
            if response.get("status") == "success":
                history = response.get("data", [])
                
                # 更新当前会话
                self.current_session_id = session_id
                self.conversation_history = []
                
                for item in history:
                    metadata = {
                        'processing_time': item.get('processing_time', 0),
                        'agent_used': item.get('agent_used')
                    }
                    self.conversation_history.append((
                        item.get('query', ''),
                        item.get('answer', ''),
                        metadata
                    ))
                
                return session_id, "", self._format_conversation_display()
            else:
                return self.current_session_id, "", f"❌ 加载会话失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return self.current_session_id, "", f"❌ 加载失败: {str(e)}"
    
    def delete_session(self, session_id: str) -> str:
        """删除指定会话"""
        try:
            if not session_id.strip():
                return "❌ 请输入要删除的会话ID"
            
            data = {"session_id": session_id, "confirm": True}
            response = api_client.delete_chat_session(session_id)
            
            if response.get("status") == "success":
                return f"✅ 会话 {session_id} 已删除"
            else:
                return f"❌ 删除失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 删除失败: {str(e)}"
    
    def list_available_agents(self) -> str:
        """列出可用智能体"""
        try:
            response = api_client.list_agents()
            
            if response.get("status") == "success":
                agents = response["data"]
                if not agents:
                    return "🤖 暂无可用智能体"
                
                agent_list = "🤖 可用智能体:\n\n"
                for agent in agents:
                    if agent.get('status') == 'active':
                        agent_list += f"• {agent['name']} (ID: {agent['agent_id']})\n"
                        agent_list += f"  类型: {agent.get('agent_type', '未知')}\n"
                        agent_list += f"  描述: {agent.get('description', '无')}\n"
                        agent_list += f"  路由标签: {', '.join(agent.get('routing_tags', []))}\n"
                        agent_list += f"  绑定知识库: {', '.join(agent.get('bind_kb_ids', []))}\n\n"
                
                return agent_list
            else:
                return f"❌ 获取失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 获取失败: {str(e)}"
    
    def get_available_knowledge_bases(self) -> str:
        """获取可用知识库"""
        try:
            response = api_client.list_knowledge_bases()
            
            if response.get("status") == "success":
                kbs = response["data"]
                if not kbs:
                    return "📝 暂无知识库"
                
                kb_list = "📚 可用知识库:\n\n"
                for kb in kbs:
                    kb_list += f"• {kb['name']} (ID: {kb['kb_id']})\n"
                    kb_list += f"  描述: {kb.get('description', '无')}\n"
                    kb_list += f"  标签: {', '.join(kb.get('labels', []))}\n"
                    kb_list += f"  文件数: {kb.get('file_count', 0)}\n"
                    kb_list += f"  分块数: {kb.get('chunk_count', 0)}\n\n"
                
                return kb_list
            else:
                return f"❌ 获取失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 获取失败: {str(e)}"


def create_qa_collaboration_interface():
    """创建问答协作界面 - 支持多轮对话上下文记忆"""
    qa_collab = QACollaboration()
    
    with gr.Blocks(title="问答协作", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 💬 多智能体问答协作系统")
        gr.Markdown("*支持多轮对话上下文记忆 - 系统会记住之前的对话内容*")
        
        with gr.Tabs():
            # 问答协作标签页
            with gr.Tab("智能问答"):
                # 会话管理区域
                with gr.Row():
                    with gr.Column(scale=3):
                        session_display = gr.Textbox(
                            label="当前会话ID",
                            value=qa_collab.current_session_id,
                            interactive=False,
                            info="同一会话ID下的对话会保持上下文记忆"
                        )
                    with gr.Column(scale=1):
                        new_conv_btn = gr.Button("🆕 新建对话", variant="secondary", size="lg")
                
                with gr.Row():
                    # 左侧：输入区域
                    with gr.Column(scale=2):
                        # 问题输入区域
                        query_input = gr.Textbox(
                            label="请输入您的问题",
                            placeholder="例如：请分析一下人工智能的发展趋势\n\n💡 提示：在同一会话中，您可以基于之前的问答继续追问",
                            lines=3
                        )
                        
                        # 智能体选择
                        agent_id_input = gr.Textbox(
                            label="智能体ID（可选，留空则自动选择）",
                            placeholder="例如：agent_001",
                            value=""
                        )
                        # 指定知识库（逗号分隔）
                        kb_ids_input = gr.Textbox(
                            label="知识库ID（逗号分隔，可选）",
                            placeholder="例如：kb_id_1,kb_id_2",
                            value=""
                        )
                        
                        # 高级参数
                        with gr.Accordion("高级参数", open=False):
                            top_k = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="检索Top-K"
                            )
                            
                            # 知识库选择提示
                            gr.Markdown("""
                            ### 📚 如何指定知识库
                            在下方的过滤条件中添加 **kb_ids** 数组。支持 **单个** 或 **多个** 知识库 ID：
                            
                            *单个 ID 示例*
                            ```json
                            {"custom_filters": {"kb_ids": ["kb_id_1"]}}
                            ```
                            
                            *多个 ID 示例*
                            ```json
                            {"custom_filters": {"kb_ids": ["kb_id_1", "kb_id_2", "kb_id_3"]}}
                            ```
                            💡 提示：在"知识库参考"标签页查看并复制可用的知识库 ID。
                            """)
                            
                            filters = gr.Textbox(
                                label="过滤条件（JSON格式）",
                                placeholder='{"custom_filters": {"kb_ids": ["a86ea986-70f5-43b6-b3b9-fa58838973fa"]}}',
                                lines=3,
                                info="必须包含custom_filters.kb_ids才能检索知识库内容"
                            )
                            rerank = gr.Checkbox(
                                label="启用重排序",
                                value=True
                            )
                            temperature = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="温度参数"
                            )
                            max_turns = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label="最大轮次"
                            )
                        
                        ask_btn = gr.Button("💬 提问", variant="primary", size="lg")
                    
                    # 右侧：对话历史显示区域
                    with gr.Column(scale=3):
                        conversation_display = gr.Textbox(
                            label="对话历史（显示当前会话的所有问答）",
                            value=qa_collab._format_conversation_display(),
                            lines=25,
                            interactive=False
                        )
                
                # 新建对话按钮事件
                new_conv_btn.click(
                    fn=qa_collab.new_conversation,
                    outputs=[session_display, query_input, conversation_display]
                )
                
                # 提问按钮事件
                ask_btn.click(
                    fn=qa_collab.ask_question,
                    inputs=[
                        query_input, agent_id_input, kb_ids_input, top_k, filters,
                        rerank, temperature, max_turns
                    ],
                    outputs=[query_input, conversation_display]
                )
            
            # 聊天历史标签页 - 列出所有会话
            with gr.Tab("聊天历史"):
                gr.Markdown("### 📋 所有聊天会话记录")
                gr.Markdown("*点击「刷新」查看所有历史会话，输入会话ID可查看详情或加载继续对话*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        refresh_sessions_btn = gr.Button("🔄 刷新所有会话", variant="primary", size="lg")
                        
                        gr.Markdown("---")
                        gr.Markdown("**📖 查看/加载会话**")
                        session_id_input = gr.Textbox(
                            label="会话ID",
                            placeholder="粘贴会话ID",
                            value=""
                        )
                        view_session_btn = gr.Button("📖 查看会话详情", variant="secondary")
                        load_session_btn = gr.Button("▶️ 加载会话继续对话", variant="primary")
                        
                        gr.Markdown("---")
                        gr.Markdown("**🗑️ 删除会话**")
                        delete_session_id = gr.Textbox(
                            label="要删除的会话ID",
                            placeholder="输入要删除的会话ID",
                            value=""
                        )
                        delete_session_btn = gr.Button("🗑️ 删除会话", variant="stop")
                    
                    with gr.Column(scale=3):
                        history_output = gr.Textbox(
                            label="聊天历史详情",
                            lines=30,
                            interactive=False,
                            value="👆 点击「刷新所有会话」查看历史记录\n\n💡 使用说明:\n1. 点击「刷新所有会话」查看所有历史对话\n2. 复制会话ID到左侧输入框\n3. 点击「查看会话详情」查看完整对话内容\n4. 点击「加载会话继续对话」在智能问答页面继续该会话"
                        )
                
                # 刷新所有会话
                refresh_sessions_btn.click(
                    fn=qa_collab.list_all_sessions,
                    outputs=history_output
                )
                
                # 查看特定会话
                view_session_btn.click(
                    fn=qa_collab.get_session_history,
                    inputs=session_id_input,
                    outputs=history_output
                )
                
                # 加载会话继续对话（更新智能问答页面）
                load_session_btn.click(
                    fn=qa_collab.load_session,
                    inputs=session_id_input,
                    outputs=[session_display, query_input, conversation_display]
                )
                
                # 删除会话
                delete_session_btn.click(
                    fn=qa_collab.delete_session,
                    inputs=delete_session_id,
                    outputs=history_output
                )
            
            # 智能体参考标签页
            with gr.Tab("智能体参考"):
                with gr.Row():
                    with gr.Column():
                        agent_refresh_btn = gr.Button("刷新智能体列表", variant="primary")
                        agent_reference = gr.Textbox(
                            label="可用智能体",
                            lines=15,
                            interactive=False
                        )
                
                agent_refresh_btn.click(
                    fn=qa_collab.list_available_agents,
                    outputs=agent_reference
                )
            
            # 知识库参考标签页
            with gr.Tab("知识库参考"):
                with gr.Row():
                    with gr.Column():
                        kb_refresh_btn = gr.Button("刷新知识库列表", variant="primary")
                        kb_reference = gr.Textbox(
                            label="可用知识库",
                            lines=15,
                            interactive=False
                        )
                
                kb_refresh_btn.click(
                    fn=qa_collab.get_available_knowledge_bases,
                    outputs=kb_reference
                )
    
    return interface


if __name__ == "__main__":
    interface = create_qa_collaboration_interface()
    interface.launch(server_name="0.0.0.0", server_port=7862)
