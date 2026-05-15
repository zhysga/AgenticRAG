#!/usr/bin/env python3
"""EasyRAG 重构后的最终健康检查"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_critical_imports():
    """检查关键模块导入"""
    print("🔍 检查关键模块导入...")
    critical_modules = [
        ("EnhancedRAGEngine", "backend.core.enhanced_rag_engine"),
        ("HybridRetrievalWrapper", "backend.core.hybrid_retrieval"),
        ("SentenceSplitter", "backend.core.splitter"),
        ("RerankerWrapper", "backend.core.reranker"),
        ("BM25Retriever", "backend.core.easyrag.custom.retrievers"),
        ("HybridRetriever", "backend.core.easyrag.custom.retrievers"),
        ("SentenceTransformerRerank", "backend.core.easyrag.custom.rerankers"),
        ("LLMRerank", "backend.core.easyrag.custom.rerankers"),
    ]
    
    failed = []
    for name, module in critical_modules:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed.append((name, str(e)))
    
    return len(failed) == 0, failed

def check_no_pipeline_references():
    """检查是否还有 pipeline 引用"""
    print("\n🔍 检查是否还有 pipeline.ingestion 引用...")
    
    import subprocess
    result = subprocess.run(
        ["grep", "-r", "pipeline.ingestion", "backend/", "--include=*.py"],
        capture_output=True,
        text=True,
        cwd="/home/z/zhy/merge/AIPT3"
    )
    
    if result.returncode == 0:
        print(f"  ❌ 发现 pipeline 引用:\n{result.stdout}")
        return False
    else:
        print("  ✅ 无 pipeline.ingestion 引用")
        return True

def check_no_qdrant_imports():
    """检查是否还有 Qdrant 导入"""
    print("\n🔍 检查是否还有 Qdrant 导入...")
    
    import subprocess
    result = subprocess.run(
        ["grep", "-r", "from.*qdrant", "backend/", "--include=*.py"],
        capture_output=True,
        text=True,
        cwd="/home/z/zhy/merge/AIPT3"
    )
    
    if result.returncode == 0:
        print(f"  ❌ 发现 Qdrant 导入:\n{result.stdout}")
        return False
    else:
        print("  ✅ 无 Qdrant 导入语句")
        return True

def test_splitter_functionality():
    """测试 SentenceSplitter 基本功能"""
    print("\n🔍 测试 SentenceSplitter 功能...")
    try:
        from backend.core.splitter import SentenceSplitter
        from llama_index.core.schema import Document
        
        splitter = SentenceSplitter(cfg={})
        docs = [Document(text="这是一个测试文档。" * 100)]
        nodes = splitter.get_nodes_from_documents(docs)
        
        print(f"  ✅ Splitter 正常工作，生成 {len(nodes)} 个节点")
        return True
    except Exception as e:
        print(f"  ❌ Splitter 测试失败: {e}")
        return False

def test_retrievers():
    """测试检索器初始化"""
    print("\n🔍 测试检索器初始化...")
    try:
        from backend.core.easyrag.custom.retrievers import BM25Retriever
        from llama_index.core.schema import TextNode
        import jieba
        
        nodes = [
            TextNode(text="这是第一个测试节点"),
            TextNode(text="这是第二个测试节点"),
        ]
        
        # jieba.cut 是一个函数，直接传递
        retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            tokenizer=jieba.cut,
            similarity_top_k=5,
        )
        
        print(f"  ✅ BM25Retriever 初始化成功，索引 {len(nodes)} 个节点")
        return True
    except Exception as e:
        print(f"  ❌ BM25Retriever 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("EasyRAG 重构后最终健康检查")
    print("=" * 70)
    
    results = []
    
    # 1. 关键导入
    success, failed = check_critical_imports()
    results.append(("关键模块导入", success))
    
    # 2. Pipeline 引用
    results.append(("Pipeline 清理", check_no_pipeline_references()))
    
    # 3. Qdrant 导入
    results.append(("Qdrant 清理", check_no_qdrant_imports()))
    
    # 4. Splitter 功能
    results.append(("Splitter 功能", test_splitter_functionality()))
    
    # 5. Retriever 功能
    results.append(("Retriever 功能", test_retrievers()))
    
    # 总结
    print("\n" + "=" * 70)
    print("健康检查总结:")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {status} - {name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("🎉 所有检查通过！系统已准备就绪。")
        print("\n下一步:")
        print("  1. 重启后端: python start_backend.py")
        print("  2. 测试知识库问答功能")
        print("  3. 检查日志无 init_engine 错误")
        return 0
    else:
        print("⚠️  部分检查失败，需要进一步修复。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
