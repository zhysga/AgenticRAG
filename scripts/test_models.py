#!/usr/bin/env python3
"""
测试模型加载
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))

def test_embedding_model():
    """测试嵌入模型"""
    print("\n" + "="*60)
    print("测试嵌入模型加载")
    print("="*60)
    
    try:
        from adapters.embedding_client import SentenceTransformerEmbeddingClient
        from config.settings import settings
        
        print(f"\n配置信息:")
        print(f"  embedding_model_name: {settings.embedding_model_name}")
        print(f"  embedding_model_path: {settings.embedding_model_path}")
        print(f"  embedding_device: {settings.embedding_device}")
        print(f"  hf_cache_dir: {settings.hf_cache_dir}")
        print(f"  transformers_offline: {settings.transformers_offline}")
        
        # 使用正确的路径
        model_path = settings.embedding_model_path or settings.embedding_model_name
        print(f"\n实际使用路径: {model_path}")
        
        client = SentenceTransformerEmbeddingClient(model_name=model_path)
        
        # 测试嵌入
        test_text = "这是一个测试文本"
        embedding = client.embed_query(test_text)
        
        print(f"\n✅ 嵌入模型测试成功!")
        print(f"  模型: {client.model_name}")
        print(f"  嵌入维度: {len(embedding)}")
        print(f"  前5个值: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 嵌入模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reranker_model():
    """测试重排序模型"""
    print("\n" + "="*60)
    print("测试重排序模型加载")
    print("="*60)
    
    try:
        from adapters.reranker import create_reranker_adapter
        from config.settings import settings
        
        print(f"\n配置信息:")
        print(f"  reranker_model_name: {settings.reranker_model_name}")
        print(f"  reranker_model_path: {settings.reranker_model_path}")
        print(f"  reranker_enabled: {settings.reranker_enabled}")
        
        if not settings.reranker_enabled:
            print("\n⚠️  重排序已禁用，使用模拟重排序器")
        
        reranker = create_reranker_adapter(
            reranker_type="bge" if settings.reranker_enabled else "mock",
            model_name=settings.reranker_model_name
        )
        
        # 测试重排序
        query = "测试查询"
        documents = ["文档1", "文档2", "文档3"]
        scores = reranker.rerank(query, documents)
        
        print(f"\n✅ 重排序模型测试成功!")
        print(f"  文档数: {len(documents)}")
        print(f"  分数: {scores}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 重排序模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """测试依赖注入"""
    print("\n" + "="*60)
    print("测试依赖注入")
    print("="*60)
    
    try:
        from dependencies import get_embedding_client, get_reranker_adapter
        
        # 测试嵌入客户端
        print("\n获取嵌入客户端...")
        embedding_client = get_embedding_client()
        print(f"✅ 嵌入客户端获取成功: {embedding_client.model_name}")
        
        # 测试重排序器
        print("\n获取重排序器...")
        reranker = get_reranker_adapter()
        print(f"✅ 重排序器获取成功: {reranker.get_model_info()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 依赖注入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("开始模型测试")
    print("="*60)
    
    results = {
        "嵌入模型": test_embedding_model(),
        "重排序模型": test_reranker_model(),
        "依赖注入": test_dependencies(),
    }
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n总体结果: {'✅ 全部通过' if all_passed else '❌ 有失败项'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
