"""
前端启动脚本

这个脚本用于启动不同的前端页面：
- 知识库管理页面 (kb_manager)
- 智能体工作室页面 (agent_studio) 
- 问答协作页面 (qa_collab)

使用方法：
python start_frontend.py --page kb          # 启动知识库管理页面 (端口7860)
python start_frontend.py --page agent       # 启动智能体工作室页面 (端口7861)
python start_frontend.py --page qa          # 启动问答协作页面 (端口7862)
python start_frontend.py --help            # 显示帮助信息
"""

import argparse
import sys
import os
from typing import Optional

# 添加当前目录到Python路径，确保可以导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 设置环境变量，确保导入路径正确
os.chdir(current_dir)

def start_kb_manager():
    """启动知识库管理页面"""
    try:
        # 直接导入模块，避免相对导入问题
        import sys
        import os
        
        # 确保导入路径正确
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # 直接导入模块
        from pages import kb_manager
        from config import frontend_config
        
        config = frontend_config.FrontendConfig()
        interface = kb_manager.create_kb_manager_interface()
        
        print("🚀 启动知识库管理页面...")
        print(f"📊 服务地址: http://{config.gradio_server_name}:{config.kb_manager_port}")
        print("⏳ 正在启动，请稍候...")
        
        interface.launch(
            server_name=config.gradio_server_name,
            server_port=config.kb_manager_port,
            share=config.gradio_share,
            debug=config.gradio_debug
        )
        
    except Exception as e:
        print(f"❌ 启动知识库管理页面失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def start_agent_studio():
    """启动智能体工作室页面"""
    try:
        # 直接导入模块，避免相对导入问题
        import sys
        import os
        
        # 确保导入路径正确
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # 直接导入模块
        from pages import agent_studio
        from config import frontend_config
        
        config = frontend_config.FrontendConfig()
        interface = agent_studio.create_agent_studio_interface()
        
        print("🚀 启动智能体工作室页面...")
        print(f"🤖 服务地址: http://{config.gradio_server_name}:{config.agent_studio_port}")
        print("⏳ 正在启动，请稍候...")
        
        interface.launch(
            server_name=config.gradio_server_name,
            server_port=config.agent_studio_port,
            share=config.gradio_share,
            debug=config.gradio_debug
        )
        
    except Exception as e:
        print(f"❌ 启动智能体工作室页面失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def start_qa_collab():
    """启动问答协作页面"""
    try:
        # 直接导入模块，避免相对导入问题
        import sys
        import os
        
        # 确保导入路径正确
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # 直接导入模块
        from pages import qa_collab
        from config import frontend_config
        
        config = frontend_config.FrontendConfig()
        interface = qa_collab.create_qa_collaboration_interface()
        
        print("🚀 启动问答协作页面...")
        print(f"💬 服务地址: http://{config.gradio_server_name}:{config.qa_collab_port}")
        print("⏳ 正在启动，请稍候...")
        
        interface.launch(
            server_name=config.gradio_server_name,
            server_port=config.qa_collab_port,
            share=config.gradio_share,
            debug=config.gradio_debug
        )
        
    except Exception as e:
        print(f"❌ 启动问答协作页面失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def show_help():
    """显示帮助信息"""
    help_text = """
LangGraph多智能体RAG系统 - 前端启动脚本

使用方法:
    python start_frontend.py --page <页面类型>

可用页面类型:
    kb      知识库管理页面 (端口: 7860)
    agent   智能体工作室页面 (端口: 7861)
    qa      问答协作页面 (端口: 7862)

示例:
    python start_frontend.py --page kb
    python start_frontend.py --page agent
    python start_frontend.py --page qa

环境变量配置:
    可以通过环境变量修改默认配置，详见 config/frontend_config.py

注意:
    1. 请确保后端服务已启动 (http://localhost:8000)
    2. 每个页面使用不同的端口，避免冲突
    3. 使用 --help 查看此帮助信息
"""
    print(help_text)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="前端启动脚本", add_help=False)
    parser.add_argument('--page', type=str, help='要启动的页面类型 (kb/agent/qa)')
    parser.add_argument('--help', action='store_true', help='显示帮助信息')
    
    args = parser.parse_args()
    
    if args.help or not args.page:
        show_help()
        return
    
    # 检查后端服务是否可用
    try:
        import requests
        response = requests.get("http://localhost:8000/auth/health", timeout=5)
        if response.status_code == 200:
            print("✅ 后端服务连接正常")
        else:
            print("⚠️  后端服务响应异常，但继续启动前端...")
    except Exception as e:
        print(f"⚠️  无法连接到后端服务: {e}")
        print("💡 请确保后端服务已启动 (http://localhost:8000)")
        print("🚀 继续启动前端，但部分功能可能受限...")
    
    # 根据页面类型启动相应的页面
    if args.page.lower() == 'kb':
        start_kb_manager()
    elif args.page.lower() == 'agent':
        start_agent_studio()
    elif args.page.lower() == 'qa':
        start_qa_collab()
    else:
        print(f"❌ 未知的页面类型: {args.page}")
        print("💡 使用 --help 查看可用选项")
        sys.exit(1)

if __name__ == "__main__":
    main()