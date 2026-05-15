#!/usr/bin/env python3
"""快速测试 EasyRAG 模块导入"""

import sys
import os

# 设置正确的 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_easyrag_retrievers():
    """测试 retrievers 模块"""
    try:
        from backend.core.easyrag.custom.retrievers import (
            BM25Retriever,
            HybridRetriever,
            get_node_content,
            tokenize_and_remove_stopwords
        )
        print("✅ backend.core.easyrag.custom.retrievers - 所有导入成功")
        print(f"   - BM25Retriever: {BM25Retriever}")
        print(f"   - HybridRetriever: {HybridRetriever}")
        print(f"   - get_node_content: {get_node_content}")
        return True
    except Exception as e:
        print(f"❌ backend.core.easyrag.custom.retrievers - 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_easyrag_rerankers():
    """测试 rerankers 模块"""
    try:
        from backend.core.easyrag.custom.rerankers import (
            SentenceTransformerRerank,
            LLMRerank,
            get_node_content
        )
        print("✅ backend.core.easyrag.custom.rerankers - 所有导入成功")
        print(f"   - SentenceTransformerRerank: {SentenceTransformerRerank}")
        print(f"   - LLMRerank: {LLMRerank}")
        print(f"   - get_node_content: {get_node_content}")
        return True
    except Exception as e:
        print(f"❌ backend.core.easyrag.custom.rerankers - 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_modules():
    """测试核心模块"""
    modules_to_test = [
        "backend.core.enhanced_rag_engine",
        "backend.core.hybrid_retrieval",
        "backend.core.splitter",
        "backend.core.reranker",
    ]
    
    success_count = 0
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
            success_count += 1
        except Exception as e:
            print(f"❌ {module}: {e}")
    
    return success_count == len(modules_to_test)

def main():
    print("=" * 70)
    print("测试 EasyRAG 模块导入（修复后）")
    print("=" * 70)
    
    results = []
    
    print("\n📦 测试核心模块")
    print("-" * 70)
    results.append(test_core_modules())
    
    print("\n📦 测试 EasyRAG Retrievers")
    print("-" * 70)
    results.append(test_easyrag_retrievers())
    
    print("\n📦 测试 EasyRAG Rerankers")
    print("-" * 70)
    results.append(test_easyrag_rerankers())
    
    print("\n" + "=" * 70)
    if all(results):
        print("✅ 所有测试通过！")
        print("=" * 70)
        return 0
    else:
        print("❌ 部分测试失败")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
