"""
Chroma向量数据库诊断脚本
"""
import chromadb
from chromadb.config import Settings
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend"))

try:
    from backend.adapters.embedding_client import SentenceTransformerEmbeddingClient
    has_embedding = True
except:
    has_embedding = False


def diagnose_chroma():
    """诊断Chroma数据库"""
    
    print("\n" + "=" * 80)
    print("🔍 Chroma向量数据库诊断报告")
    print("=" * 80 + "\n")
    
    try:
        # 连接到Chroma
        chroma_path = "./storage/chroma_db"
        print(f"📁 连接到Chroma数据库: {chroma_path}")
        
        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 列出所有集合
        collections = client.list_collections()
        print(f"\n📚 数据库中共有 {len(collections)} 个集合:")
        
        if not collections:
            print("   ❌ 数据库中没有任何集合！这是问题所在。")
            return
        
        # 检查每个集合
        for i, col in enumerate(collections, 1):
            print(f"\n{'─' * 80}")
            print(f"集合 #{i}: {col.name}")
            print(f"{'─' * 80}")
            
            # 获取集合信息
            count = col.count()
            print(f"  📊 文档数量: {count}")
            
            if count == 0:
                print(f"  ⚠️  集合 '{col.name}' 是空的！")
                continue
            
            # 获取一些示例数据
            try:
                data = col.get(limit=3, include=["documents", "metadatas", "embeddings"])
                
                print(f"\n  📄 示例文档 (前3条):")
                for j, (doc_id, doc, meta) in enumerate(zip(
                    data.get("ids", []),
                    data.get("documents", []),
                    data.get("metadatas", [])
                ), 1):
                    print(f"\n    文档 {j}:")
                    print(f"      ID: {doc_id}")
                    print(f"      内容预览: {doc[:100]}..." if len(doc) > 100 else f"      内容: {doc}")
                    print(f"      元数据: {meta}")
                    
                    # 检查是否有嵌入向量
                    embeddings = data.get("embeddings", [])
                    if embeddings and j-1 < len(embeddings):
                        emb = embeddings[j-1]
                        print(f"      向量维度: {len(emb) if emb else 'None'}")
                
            except Exception as e:
                print(f"  ❌ 获取示例数据失败: {e}")
        
        # 测试查询
        if has_embedding and len(collections) > 0:
            print(f"\n\n{'=' * 80}")
            print("🔬 测试向量查询")
            print("=" * 80)
            
            test_query = "13号，周一"
            print(f"\n测试查询: \"{test_query}\"")
            
            try:
                # 初始化嵌入客户端
                embedding_client = SentenceTransformerEmbeddingClient()
                query_vector = embedding_client.embed_query(test_query)
                print(f"查询向量维度: {len(query_vector)}")
                
                # 在每个集合中搜索
                for col in collections:
                    try:
                        print(f"\n  在集合 '{col.name}' 中搜索...")
                        results = col.query(
                            query_embeddings=[query_vector],
                            n_results=3
                        )
                        
                        if results["documents"] and results["documents"][0]:
                            print(f"    ✅ 找到 {len(results['documents'][0])} 个结果:")
                            for i, (doc, distance) in enumerate(zip(
                                results["documents"][0],
                                results["distances"][0] if results.get("distances") else [0] * len(results["documents"][0])
                            ), 1):
                                print(f"\n      结果 {i}:")
                                print(f"        相似度分数: {distance:.4f}")
                                print(f"        内容预览: {doc[:150]}..." if len(doc) > 150 else f"        内容: {doc}")
                        else:
                            print(f"    ⚠️  没有找到匹配结果")
                            
                    except Exception as e:
                        print(f"    ❌ 查询失败: {e}")
                        
            except Exception as e:
                print(f"\n  ⚠️  无法进行向量查询测试: {e}")
        else:
            print(f"\n  ℹ️  跳过向量查询测试（embedding客户端不可用或无集合）")
        
        print(f"\n\n{'=' * 80}")
        print("✅ 诊断完成")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    diagnose_chroma()
