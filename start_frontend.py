#!/usr/bin/env python3
"""
前端服务启动脚本
"""
import sys
import os
import argparse
import threading
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "frontend"))
from frontend.config.frontend_config import FrontendConfig
cfg = FrontendConfig.from_env()

def check_dependencies():
    """检查依赖"""
    try:
        import gradio
        import requests
        print("✅ 前端依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def check_backend():
    """检查后端服务"""
    try:
        import requests
        response = requests.get("http://localhost:8000/auth/health", timeout=5)
        if response.status_code == 200:
            print("✅ 后端服务运行正常")
            return True
        else:
            print("❌ 后端服务响应异常")
            return False
    except Exception as e:
        print(f"❌ 后端服务连接失败: {e}")
        print("请先启动后端服务: python start_backend.py")
        return False

def start_kb_manager():
    """启动知识库管理页面"""
    try:
        import asyncio
        # 在新线程中创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        from frontend.pages.kb_manager import create_kb_manager_interface
        interface = create_kb_manager_interface()
        print(f"📚 启动知识库管理页面: http://{cfg.gradio_server_name}:{cfg.kb_manager_port}")
        interface.launch(
            server_name=cfg.gradio_server_name,
            server_port=cfg.kb_manager_port,
            share=cfg.gradio_share,
            debug=cfg.gradio_debug,
            show_api=False,
            prevent_thread_lock=True,
        )
    except Exception as e:
        print(f"❌ 知识库管理页面启动失败: {e}")
        import traceback
        traceback.print_exc()

def start_agent_studio():
    """启动智能体工作室页面"""
    try:
        import asyncio
        # 在新线程中创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        from frontend.pages.agent_studio import create_agent_studio_interface
        interface = create_agent_studio_interface()
        print(f"🤖 启动智能体工作室页面: http://{cfg.gradio_server_name}:{cfg.agent_studio_port}")
        interface.launch(
            server_name=cfg.gradio_server_name,
            server_port=cfg.agent_studio_port,
            share=cfg.gradio_share,
            debug=cfg.gradio_debug,
            show_api=False,
            prevent_thread_lock=True,
        )
    except Exception as e:
        print(f"❌ 智能体工作室页面启动失败: {e}")

def start_qa_collab():
    """启动问答协作页面"""
    try:
        import asyncio
        # 在新线程中创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        from frontend.pages.qa_collab import create_qa_collaboration_interface
        interface = create_qa_collaboration_interface()
        print(f"💬 启动问答协作页面: http://{cfg.gradio_server_name}:{cfg.qa_collab_port}")
        interface.launch(
            server_name=cfg.gradio_server_name,
            server_port=cfg.qa_collab_port,
            share=cfg.gradio_share,
            debug=cfg.gradio_debug,
            show_api=False,
            prevent_thread_lock=True,
        )
    except Exception as e:
        print(f"❌ 问答协作页面启动失败: {e}")

def start_all_pages():
    """启动所有页面"""
    print("🚀 启动所有前端页面")
    
    # 创建线程启动各个页面
    threads = []
    
    # 知识库管理页面
    kb_thread = threading.Thread(target=start_kb_manager, daemon=True)
    threads.append(kb_thread)
    
    # 智能体工作室页面
    agent_thread = threading.Thread(target=start_agent_studio, daemon=True)
    threads.append(agent_thread)
    
    # 问答协作页面
    qa_thread = threading.Thread(target=start_qa_collab, daemon=True)
    threads.append(qa_thread)
    
    # 启动所有线程
    for thread in threads:
        thread.start()
        time.sleep(2)  # 避免端口冲突
    
    print("\n📋 前端页面访问地址:")
    print(f"  📚 知识库管理: http://{cfg.gradio_server_name}:{cfg.kb_manager_port}")
    print(f"  🤖 智能体工作室: http://{cfg.gradio_server_name}:{cfg.agent_studio_port}")
    print(f"  💬 问答协作: http://{cfg.gradio_server_name}:{cfg.qa_collab_port}")
    print("\n按 Ctrl+C 停止所有服务")
    
    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 所有前端服务已停止")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动LangGraph多智能体RAG系统前端服务")
    parser.add_argument("--page", choices=["kb", "agent", "qa", "all"], 
                       default="all", help="要启动的页面")
    parser.add_argument("--check-deps", action="store_true", help="仅检查依赖")
    parser.add_argument("--check-backend", action="store_true", help="仅检查后端")
    
    args = parser.parse_args()
    
    print("🎨 LangGraph多智能体RAG系统 - 前端服务")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_deps:
        print("✅ 依赖检查完成")
        return
    
    # 检查后端
    if not check_backend():
        sys.exit(1)
    
    if args.check_backend:
        print("✅ 后端检查完成")
        return
    
    # 启动指定页面
    if args.page == "kb":
        start_kb_manager()
    elif args.page == "agent":
        start_agent_studio()
    elif args.page == "qa":
        start_qa_collab()
    elif args.page == "all":
        start_all_pages()

if __name__ == "__main__":
    main()
