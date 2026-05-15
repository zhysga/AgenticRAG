#!/usr/bin/env python3
"""
后端服务启动脚本
"""
import sys
import os
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录与backend目录到Python路径
project_root = Path(__file__).parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))

def check_dependencies():
    """检查依赖"""
    try:
        import fastapi
        import uvicorn
        import langgraph
        import chromadb
        import sentence_transformers
        import gradio
        print("✅ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def create_directories():
    """创建必要的目录 - 所有存储统一到 storage 目录"""
    directories = [
        "storage",
        "storage/chroma_db",
        "storage/faiss_db",
        "storage/uploads",
        "storage/logs",
        "storage/hf_cache",
        "storage/knowledge_bases",
        "storage/chat_history",
        "storage/data",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 创建目录: {directory}")

def start_backend(host="0.0.0.0", port=8000, reload=False):
    """启动后端服务"""
    print(f"🚀 启动后端服务: http://{host}:{port}")
    
    try:
        import uvicorn
        from backend.config.settings import settings
        # 确保日志目录存在
        Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)

        # 仅文件输出的 uvicorn 日志配置
        uvicorn_log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(message)s",
                    "use_colors": False,
                },
                "access": {
                    "()": "uvicorn.logging.AccessFormatter",
                    "fmt": "%(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s",
                },
            },
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "formatter": "default",
                    "filename": settings.log_file,
                    "encoding": "utf-8",
                },
                "access_file": {
                    "class": "logging.FileHandler",
                    "formatter": "access",
                    "filename": settings.log_file,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["file"], "level": settings.log_level.upper(), "propagate": False},
                "uvicorn.error": {"handlers": ["file"], "level": settings.log_level.upper(), "propagate": False},
                "uvicorn.access": {"handlers": ["access_file"], "level": settings.log_level.upper(), "propagate": False},
            },
        }

        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            log_config=uvicorn_log_config,
        )
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动LangGraph多智能体RAG系统后端服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    parser.add_argument("--check-deps", action="store_true", help="仅检查依赖")
    
    args = parser.parse_args()
    
    print("🔧 LangGraph多智能体RAG系统 - 后端服务")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_deps:
        print("✅ 依赖检查完成")
        return
    
    # 创建目录
    create_directories()
    
    # 启动服务
    start_backend(args.host, args.port, args.reload)

if __name__ == "__main__":
    main()
