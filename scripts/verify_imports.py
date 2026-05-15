#!/usr/bin/env python3
"""验证所有关键模块可以正确导入"""

import sys
import traceback

def test_import(module_path: str) -> bool:
    """测试单个模块导入"""
    try:
        __import__(module_path)
        print(f"✅ {module_path}")
        return True
    except Exception as e:
        print(f"❌ {module_path}: {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("开始验证项目模块导入...")
    print("=" * 60)
    
    # 核心模块
    core_modules = [
        "backend.core.enhanced_rag_engine",
        "backend.core.hybrid_retrieval",
        "backend.core.nodes",
        "backend.core.reranker",
        "backend.core.splitter",
        "backend.core.state",
    ]
    
    # EasyRAG 模块
    easyrag_modules = [
        "backend.core.easyrag.custom.retrievers",
        "backend.core.easyrag.custom.rerankers",
    ]
    
    # 适配器模块
    adapter_modules = [
        "backend.adapters.embedding_adapter",
        "backend.adapters.embedding_client",
        "backend.adapters.reranker",
        "backend.adapters.storage_adapter",
        "backend.adapters.vector_store",
    ]
    
    # 服务模块
    service_modules = [
        "backend.services.agent_service",
        "backend.services.chat_service",
        "backend.services.indexing_service",
        "backend.services.knowledge_base_service",
        "backend.services.langgraph_service",
        "backend.services.rag_service",
    ]
    
    # API 路由
    api_modules = [
        "backend.api.agent_router",
        "backend.api.chat_router",
        "backend.api.config_router",
        "backend.api.kb_router",
        "backend.api.stats_router",
    ]
    
    all_modules = {
        "核心模块": core_modules,
        "EasyRAG 模块": easyrag_modules,
        "适配器模块": adapter_modules,
        "服务模块": service_modules,
        "API 路由": api_modules,
    }
    
    total = 0
    success = 0
    
    for category, modules in all_modules.items():
        print(f"\n📦 {category}")
        print("-" * 60)
        for module in modules:
            total += 1
            if test_import(module):
                success += 1
    
    print("\n" + "=" * 60)
    print(f"验证完成: {success}/{total} 模块导入成功")
    print("=" * 60)
    
    return 0 if success == total else 1

if __name__ == "__main__":
    sys.exit(main())
