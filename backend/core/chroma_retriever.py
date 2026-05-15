"""
Chroma向量数据库检索器
"""
import logging
import time
from typing import List, Any
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.retrievers import BaseRetriever
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaRetriever(BaseRetriever):
    """Chroma向量数据库检索器"""
    
    def __init__(self, vector_store, embed_model, similarity_top_k: int = 10):
        """初始化Chroma检索器
        
        Args:
            vector_store: 向量存储适配器
            embed_model: 嵌入模型客户端
            similarity_top_k: 相似度top_k
        """
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.similarity_top_k = similarity_top_k
        super().__init__()
        
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """检索实现"""
        try:
            _t0 = time.perf_counter()
            # 获取查询文本和嵌入
            if hasattr(query_bundle, 'embedding') and query_bundle.embedding is not None:
                query_embedding = query_bundle.embedding
            else:
                # 生成查询嵌入
                _te0 = time.perf_counter()
                query_embedding = self.embed_model.embed_query(query_bundle.query_str)
                _embed_ms = int((time.perf_counter() - _te0) * 1000)
                logger.info(f"ChromaRetriever: 查询向量生成完成，用时{_embed_ms}ms")
            
            # 获取知识库ID
            kb_ids = None
            if hasattr(query_bundle, 'metadata') and query_bundle.metadata:
                kb_ids = query_bundle.metadata.get('kb_ids')
            
            if not kb_ids:
                # 使用默认知识库
                kb_ids = ["default"]
            
            logger.info(f"开始检索，知识库: {kb_ids}, 查询: '{query_bundle.query_str[:50]}...'")
            
            # 搜索文档 - 使用新的search方法签名
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                k=self.similarity_top_k,
                kb_ids=kb_ids
            )
            
            _elapsed_ms = int((time.perf_counter() - _t0) * 1000)
            logger.info(
                f"检索完成，返回 {len(search_results)} 个结果",
                extra={"duration_ms": _elapsed_ms}
            )
            
            # 转换为NodeWithScore格式
            nodes = []
            for result in search_results:
                # 创建节点
                node = NodeWithScore(
                    node=TextNode(
                        text=result.get('content', ''),
                        metadata=result.get('metadata', {}),
                        id_=result.get('id', '')
                    ),
                    score=result.get('score', 0.0)
                )
                nodes.append(node)
            
            return nodes
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            
            # 如果指定了知识库ID，尝试逐个知识库检索
            if kb_ids and len(kb_ids) > 0:
                logger.info("尝试逐个知识库检索...")
                all_nodes = []
                
                for kb_id in kb_ids:
                    try:
                        _tkb0 = time.perf_counter()
                        logger.info(f"尝试检索知识库: {kb_id}")
                        results = self.vector_store.search(
                            query_embedding=query_embedding,
                            k=self.similarity_top_k,
                            kb_ids=[kb_id]
                        )
                        
                        for result in results:
                            node = NodeWithScore(
                                node=TextNode(
                                    text=result.get('content', ''),
                                    metadata=result.get('metadata', {}),
                                    id_=result.get('id', '')
                                ),
                                score=result.get('score', 0.0)
                            )
                            all_nodes.append(node)
                        _tkb_ms = int((time.perf_counter() - _tkb0) * 1000)
                        logger.info(
                            f"知识库 {kb_id} 完成检索",
                            extra={"kb_results": len(results), "duration_ms": _tkb_ms}
                        )
                        
                    except Exception as kb_error:
                        logger.warning(f"检索知识库 {kb_id} 失败: {kb_error}")
                        continue
                
                # 按分数排序
                all_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
                logger.info(f"逐个检索完成，总共返回 {len(all_nodes)} 个结果")
                return all_nodes[:self.similarity_top_k]
            
            return []
    
    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """异步检索相关文档"""
        # 对于这个实现，我们直接调用同步版本
        return self._retrieve(query_bundle)