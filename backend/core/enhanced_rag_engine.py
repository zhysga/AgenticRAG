"""Enhanced RAG Engine

High-level orchestrator that glues retrieval, rerank & generation together.
"""
from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional

from llama_index.core.schema import NodeWithScore

from .hybrid_retrieval import HybridRetrievalWrapper
from .reranker import RerankerWrapper
from .splitter import SentenceSplitter
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedRAGEngine:
    """Compose HybridRetriever + Reranker; generation is delegated up-stream."""

    def __init__(self, cfg: Dict[str, Any], embed_model=None, nodes=None, vector_store=None, embedding_client=None, kb_ids=None):
        self.cfg = cfg
        self._hybrid_retriever = None  # 缓存混合检索器实例
        # 优先使用新的参数，回退到旧参数以保持兼容性
        if vector_store and embedding_client:
            # 新参数模式：从向量存储构建节点
            # 获取所有知识库的文档节点用于BM25检索
            all_nodes = []
            if kb_ids:
                try:
                    # 从向量存储获取所有文档
                    all_docs = vector_store.get_all_documents(kb_ids, limit=1000)  # 限制数量避免内存问题
                    
                    # 转换为节点格式
                    from llama_index.core.schema import TextNode, NodeWithScore
                    for doc in all_docs:
                        node = TextNode(
                            text=doc.get("content", ""),
                            metadata=doc.get("metadata", {}),
                            id_=doc.get("chunk_id", "")
                        )
                        all_nodes.append(node)
                    
                    logger.info(f"成功获取 {len(all_nodes)} 个文档节点用于BM25检索")
                except Exception as e:
                    logger.warning(f"获取文档节点失败: {e}，将仅使用向量检索")
                    all_nodes = []  # 使用空列表而不是None，确保HybridRetrievalWrapper能正确处理
            
            self.retrieval = HybridRetrievalWrapper(
                cfg=cfg.get("retrieval", {}), 
                embed_model=embedding_client, 
                vector_store=vector_store, 
                kb_ids=kb_ids,
                nodes=all_nodes
            )
        elif embed_model and nodes:
            # 旧参数模式：直接使用模型和节点
            self.retrieval = HybridRetrievalWrapper(
                cfg=cfg.get("retrieval", {}), 
                embed_model=embed_model, 
                nodes=nodes
            )
        else:
            raise ValueError("必须提供 (vector_store + embedding_client) 或 (embed_model + nodes)")
        
        # 禁用重排序器以避免初始化错误
        reranker_cfg = cfg.get("reranking", {})
        reranker_cfg["enabled"] = False  # 强制禁用重排序器
        self.reranker = RerankerWrapper(reranker_cfg)
        self.splitter = SentenceSplitter(cfg.get("chunking", {}))

    # ------------------------------------------------------------------
    async def retrieve(self, query: str, kb_ids: Optional[List[str]] = None, **kwargs) -> List[NodeWithScore]:
        """检索文档，返回 NodeWithScore 列表"""
        try:
            logger.info(f"开始检索: query='{query}', kb_ids={kb_ids}")
            _t0 = time.perf_counter()
            
            # 执行混合检索
            nodes = await self.retrieval.retrieve(query, kb_ids)
            _t1 = time.perf_counter()
            nodes = self.reranker.postprocess_nodes(nodes, query=query)
            _t2 = time.perf_counter()
            
            logger.info(
                f"检索完成，返回 {len(nodes)} 个节点",
                extra={
                    "durations_ms": {
                        "retrieve_ms": int((_t1 - _t0) * 1000),
                        "rerank_ms": int((_t2 - _t1) * 1000),
                        "total_ms": int((_t2 - _t0) * 1000),
                    }
                },
            )
            return nodes
            
        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    async def query(self, query: str, kb_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        nodes = await self.retrieve(query, kb_ids)
        contexts = [n.node.get_content() for n in nodes]
        return {"nodes": nodes, "contexts": contexts}

    # ------------------------------------------------------------------
    def chunk_documents(self, docs: List[str]):
        """Split text docs to nodes respecting sentence boundaries."""
        return self.splitter.get_nodes_from_documents(docs)

    # Utility: sync wrapper
    def query_sync(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.query(*args, **kwargs))
