"""
智能体定义Gradio应用
"""
import gradio as gr
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from services.api_client import APIClient
from config.frontend_config import FrontendConfig

# 配置
config = FrontendConfig()
api_client = APIClient(config.backend_url)


class AgentStudio:
    """智能体工作室"""
    
    def __init__(self):
        self.current_agent_id = None
        self.available_kbs = []
    
    def create_agent(
        self,
        name: str,
        description: str,
        agent_type: str,
        system_prompt: str,
        tools: str,
        bind_kb_ids: str,
        routing_tags: str,
        max_turns: int,
        temperature: float
    ) -> str:
        """创建智能体"""
        try:
            # 解析工具列表
            tool_list = [tool.strip() for tool in tools.split(",") if tool.strip()]
            
            # 解析绑定的知识库ID
            kb_id_list = [kb_id.strip() for kb_id in bind_kb_ids.split(",") if kb_id.strip()]
            
            # 解析路由标签
            tag_list = [tag.strip() for tag in routing_tags.split(",") if tag.strip()]
            
            # 构建请求数据
            data = {
                "name": name,
                "description": description,
                "agent_type": agent_type,
                "system_prompt": system_prompt,
                "tools": tool_list,
                "bind_kb_ids": kb_id_list,
                "routing_tags": tag_list,
                "max_turns": max_turns,
                "temperature": temperature
            }
            
            # 调用API
            response = api_client.create_agent(data)
            
            if response.get("status") == "success":
                self.current_agent_id = response["data"]["agent_id"]
                return f"✅ 智能体创建成功！\n智能体ID: {self.current_agent_id}\n名称: {name}"
            else:
                return f"❌ 创建失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 创建失败: {str(e)}"
    
    def list_agents(self) -> str:
        """列出智能体"""
        try:
            response = api_client.list_agents()
            
            if response.get("status") == "success":
                agents = response["data"]
                if not agents:
                    return "🤖 暂无智能体"
                
                agent_list = "🤖 智能体列表:\n\n"
                for agent in agents:
                    agent_list += f"• {agent['name']} (ID: {agent['agent_id']})\n"
                    agent_list += f"  类型: {agent.get('agent_type', '未知')}\n"
                    agent_list += f"  描述: {agent.get('description', '无')}\n"
                    agent_list += f"  工具: {', '.join(agent.get('tools', []))}\n"
                    agent_list += f"  绑定知识库: {', '.join(agent.get('bind_kb_ids', []))}\n"
                    agent_list += f"  路由标签: {', '.join(agent.get('routing_tags', []))}\n"
                    agent_list += f"  最大轮次: {agent.get('max_turns', 10)}\n"
                    agent_list += f"  温度: {agent.get('temperature', 0.7)}\n"
                    agent_list += f"  创建时间: {agent.get('created_at', '未知')}\n\n"
                
                return agent_list
            else:
                return f"❌ 获取失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 获取失败: {str(e)}"
    
    def test_agent(self, agent_id: str, test_query: str) -> str:
        """测试智能体"""
        try:
            if not agent_id or not test_query:
                return "❌ 请输入智能体ID和测试查询"
            
            # 构建请求数据
            data = {
                "agent_id": agent_id,
                "test_query": test_query
            }
            
            # 调用API
            response = api_client.test_agent(data)
            
            if response.get("status") == "success":
                test_result = response["data"]
                result_text = f"✅ 智能体测试成功！\n\n"
                result_text += f"智能体: {test_result.get('agent_name', '未知')}\n"
                result_text += f"测试查询: {test_result.get('test_query', '')}\n"
                result_text += f"系统提示: {test_result.get('system_prompt', '')}\n"
                result_text += f"可用工具: {', '.join(test_result.get('tools', []))}\n"
                result_text += f"绑定知识库: {', '.join(test_result.get('bind_kb_ids', []))}\n"
                result_text += f"路由标签: {', '.join(test_result.get('routing_tags', []))}\n"
                result_text += f"测试时间: {test_result.get('test_time', '未知')}\n"
                
                return result_text
            else:
                return f"❌ 测试失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 测试失败: {str(e)}"
    
    def update_agent(self, agent_id: str, updates: str) -> str:
        """更新智能体"""
        try:
            if not agent_id or not updates:
                return "❌ 请输入智能体ID和更新内容"
            
            # 解析更新内容
            try:
                updates_dict = json.loads(updates)
            except json.JSONDecodeError:
                return "❌ 更新内容格式错误，请使用JSON格式"
            
            # 调用API
            response = api_client.update_agent(agent_id, updates_dict)
            
            if response.get("status") == "success":
                return f"✅ 智能体 {agent_id} 更新成功！"
            else:
                return f"❌ 更新失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 更新失败: {str(e)}"
    
    def delete_agent(self, agent_id: str, confirm: bool) -> str:
        """删除智能体"""
        try:
            if not confirm:
                return "❌ 请确认删除操作"
            
            response = api_client.delete_agent(agent_id)
            
            if response.get("status") == "success":
                if self.current_agent_id == agent_id:
                    self.current_agent_id = None
                return f"✅ 智能体 {agent_id} 删除成功！"
            else:
                return f"❌ 删除失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 删除失败: {str(e)}"
    
    def get_available_knowledge_bases(self) -> str:
        """获取可用知识库列表"""
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
                    kb_list += f"  文件数: {kb.get('file_count', 0)}\n\n"
                
                return kb_list
            else:
                return f"❌ 获取失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 获取失败: {str(e)}"


def create_agent_studio_interface():
    """创建智能体工作室界面"""
    agent_studio = AgentStudio()
    
    with gr.Blocks(title="智能体工作室", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 智能体工作室")
        
        with gr.Tabs():
            # 创建智能体标签页
            with gr.Tab("创建智能体"):
                with gr.Row():
                    with gr.Column(scale=1):
                        agent_name = gr.Textbox(
                            label="智能体名称",
                            placeholder="请输入智能体名称",
                            value=""
                        )
                        agent_description = gr.Textbox(
                            label="智能体描述",
                            placeholder="请输入智能体描述",
                            lines=2
                        )
                        agent_type = gr.Dropdown(
                            choices=["analyst", "writer", "critic", "tool_caller", "custom"],
                            value="custom",
                            label="智能体类型"
                        )
                        system_prompt = gr.Textbox(
                            label="系统提示词",
                            placeholder="请输入系统提示词，定义智能体的行为和能力",
                            lines=5,
                            value="你是一个智能助手，能够帮助用户回答问题并提供有用的信息。"
                        )
                        tools = gr.Textbox(
                            label="可用工具（逗号分隔）",
                            placeholder="例如: rag_retriever,text_analyzer,text_generator",
                            value="rag_retriever,text_analyzer"
                        )
                        bind_kb_ids = gr.Textbox(
                            label="绑定知识库ID（逗号分隔）",
                            placeholder="例如: kb_001,kb_002",
                            value=""
                        )
                        routing_tags = gr.Textbox(
                            label="路由标签（逗号分隔）",
                            placeholder="例如: analysis,writing,general",
                            value="general"
                        )
                        max_turns = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1,
                            label="最大轮次"
                        )
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="温度参数"
                        )
                        create_btn = gr.Button("创建智能体", variant="primary")
                    
                    with gr.Column(scale=1):
                        create_result = gr.Textbox(
                            label="创建结果",
                            lines=15,
                            interactive=False
                        )
                
                create_btn.click(
                    fn=agent_studio.create_agent,
                    inputs=[
                        agent_name, agent_description, agent_type, system_prompt,
                        tools, bind_kb_ids, routing_tags, max_turns, temperature
                    ],
                    outputs=create_result
                )
            
            # 管理智能体标签页
            with gr.Tab("管理智能体"):
                with gr.Row():
                    with gr.Column(scale=1):
                        list_btn = gr.Button("刷新智能体列表", variant="primary")
                        agent_list_result = gr.Textbox(
                            label="智能体列表",
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        agent_id_input = gr.Textbox(
                            label="智能体ID",
                            placeholder="请输入智能体ID"
                        )
                        test_query = gr.Textbox(
                            label="测试查询",
                            placeholder="请输入测试查询",
                            lines=2
                        )
                        test_btn = gr.Button("测试智能体", variant="secondary")
                        update_content = gr.Textbox(
                            label="更新内容（JSON格式）",
                            placeholder='{"temperature": 0.8, "max_turns": 15}',
                            lines=3
                        )
                        update_btn = gr.Button("更新智能体", variant="secondary")
                        delete_btn = gr.Button("删除智能体", variant="stop")
                        delete_confirm = gr.Checkbox(label="确认删除")
                        
                        management_result = gr.Textbox(
                            label="操作结果",
                            lines=10,
                            interactive=False
                        )
                
                list_btn.click(
                    fn=agent_studio.list_agents,
                    outputs=agent_list_result
                )
                
                test_btn.click(
                    fn=agent_studio.test_agent,
                    inputs=[agent_id_input, test_query],
                    outputs=management_result
                )
                
                update_btn.click(
                    fn=agent_studio.update_agent,
                    inputs=[agent_id_input, update_content],
                    outputs=management_result
                )
                
                delete_btn.click(
                    fn=agent_studio.delete_agent,
                    inputs=[agent_id_input, delete_confirm],
                    outputs=management_result
                )
            
            # 知识库参考标签页
            with gr.Tab("知识库参考"):
                with gr.Row():
                    with gr.Column():
                        kb_refresh_btn = gr.Button("刷新知识库列表", variant="primary")
                        kb_reference = gr.Textbox(
                            label="可用知识库",
                            lines=20,
                            interactive=False
                        )
                
                kb_refresh_btn.click(
                    fn=agent_studio.get_available_knowledge_bases,
                    outputs=kb_reference
                )
    
    return interface


if __name__ == "__main__":
    interface = create_agent_studio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7861)
