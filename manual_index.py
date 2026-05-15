#!/usr/bin/env python3
"""
手动索引知识库文件
"""
import sys
import json
sys.path.append('./backend')

from adapters.storage_adapter import StorageAdapter
from adapters.vector_store import ChromaVectorStoreAdapter
from adapters.embedding_client import SentenceTransformerEmbeddingClient
from services.indexing_service import IndexingService
from datetime import datetime

def index_knowledge_base(kb_id: str):
    """为指定知识库建立索引"""
    print(f"开始为知识库 {kb_id} 建立索引...")
    
    # 初始化组件
    storage = StorageAdapter()
    vector_store = ChromaVectorStoreAdapter()
    embedding_client = SentenceTransformerEmbeddingClient()
    indexing_service = IndexingService(vector_store, embedding_client)
    
    # 获取知识库文件
    kb_files_data = storage._load_kb_files()
    
    if kb_id not in kb_files_data:
        print(f"❌ 知识库 {kb_id} 不存在")
        return False
    
    files = kb_files_data[kb_id]
    files_list = list(files.values())
    
    print(f"找到 {len(files_list)} 个文件")
    for f in files_list:
        print(f"  - {f.get('file_name')}: {f.get('file_size')} 字节")
    
    # 执行索引
    print("\n开始索引...")
    result = indexing_service.index_files(kb_id, files_list, force=True)
    
    print(f"\n索引结果:")
    print(f"  状态: {result.get('status')}")
    print(f"  索引文件数: {result.get('indexed_files')}")
    print(f"  总块数: {result.get('total_chunks')}")
    print(f"  消息: {result.get('message', 'N/A')}")
    
    return result.get('status') == 'completed'

def main():
    if len(sys.argv) > 1:
        kb_id = sys.argv[1]
    else:
        # 默认索引当前的知识库
        kb_id = "bc9bb265-6357-464b-8b31-2eb5b476a61e"
    
    print(f"目标知识库: {kb_id}\n")
    
    success = index_knowledge_base(kb_id)
    
    if success:
        print("\n✅ 索引成功！")
    else:
        print("\n❌ 索引失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
