"""Hybrid Retrieval Wrapper

Leverages EasyRAG retriever classes while keeping the implementation
Chroma-first.  Qdrant support has been removed, so hybrid mode currently
combines BM25 (sparse) with optional future in-memory dense search.

Usage
-----
>>> wrapper = HybridRetrievalWrapper(cfg, embed_model, nodes)
>>> results  = await wrapper.retrieve("LangChain 原理是什么？")
"""

from __future__ import annotations

import asyncio
from typing import List, Optional
import logging
import time

from llama_index.core.schema import NodeWithScore, QueryBundle
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetrievalWrapper:
    """混合检索包装器"""
    
    def __init__(
        self,
        cfg: dict,
        embed_model=None,
        nodes=None,
        vector_store=None,
        kb_ids=None,
        use_bm25: bool = True  # 新增参数，控制是否使用BM25
    ):
        """
        初始化混合检索器
        
        Args:
            cfg: 检索相关配置
            embed_model: 嵌入模型实例
            nodes: 文档节点列表
            vector_store: 向量存储实例
            kb_ids: 知识库ID列表
            use_bm25: 是否使用BM25稀疏检索器
        """
        self.cfg = cfg
        self.embed_model = embed_model
        self.nodes = nodes
        self.vector_store = vector_store
        self.kb_ids = kb_ids
        self.use_bm25 = use_bm25

        from backend.core.easyrag.custom.retrievers import (
            BM25Retriever,
            HybridRetriever,
        )
        import jieba

        # ----- dense retriever using ChromaDB -----
        if self.vector_store and self.embed_model:
            try:
                from backend.core.chroma_retriever import ChromaRetriever
                self.dense_retriever = ChromaRetriever(
                    vector_store=self.vector_store,
                    embed_model=self.embed_model,
                    similarity_top_k=cfg.get("dense_topk", 10)
                )
                logger.info("ChromaDB密集检索器初始化成功（使用提供的组件）")
            except Exception as e:
                logger.warning(f"ChromaDB密集检索器初始化失败: {e}，将仅使用BM25稀疏检索")
                self.dense_retriever = None
        else:
            try:
                from backend.adapters.vector_store import ChromaVectorStoreAdapter
                from backend.adapters.embedding_client import SentenceTransformerEmbeddingClient
                
                vector_store = ChromaVectorStoreAdapter()
                embed_client = SentenceTransformerEmbeddingClient()
                
                from backend.core.chroma_retriever import ChromaRetriever
                self.dense_retriever = ChromaRetriever(
                    vector_store=vector_store,
                    embed_model=embed_client,
                    similarity_top_k=cfg.get("dense_topk", 10)
                )
                logger.info("ChromaDB密集检索器初始化成功（默认组件）")
            except Exception as e:
                logger.warning(f"ChromaDB密集检索器初始化失败: {e}，将仅使用BM25稀疏检索")
                self.dense_retriever = None

        # ----- sparse retriever (BM25) -----
        if self.use_bm25:
            if self.nodes and len(self.nodes) > 0:
                tokenizer = jieba.Tokenizer()
                self.sparse_retriever = BM25Retriever.from_defaults(
                    nodes=self.nodes,
                    tokenizer=tokenizer.cut,
                    similarity_top_k=cfg.get("sparse_topk", 192),
                    stopwords=[],
                )
                logger.info("BM25稀疏检索器初始化成功（使用提供的节点）")
            else:
                # 没有节点时，禁用BM25检索器
                logger.info("没有提供文档节点，禁用BM25稀疏检索器")
                self.sparse_retriever = None
        else:
            self.sparse_retriever = None
            logger.info("BM25稀疏检索器被禁用")

        # ----- hybrid -----
        if self.dense_retriever or self.sparse_retriever:
            retrieval_type = 3 if self.dense_retriever else 2
            topk = cfg.get("fusion_topk", 256)
            self.hybrid = HybridRetriever(
                dense_retriever=self.dense_retriever,
                sparse_retriever=self.sparse_retriever,
                retrieval_type=retrieval_type,
                topk=topk,
            )
            logger.info(
                f"混合检索器初始化成功，类型: {retrieval_type}",
                extra={
                    "hybrid_config": {
                        "use_dense": bool(self.dense_retriever),
                        "use_sparse": bool(self.sparse_retriever),
                        "fusion_topk": topk,
                        "kb_ids_bound": bool(self.kb_ids),
                        "nodes_count": (len(self.nodes) if self.nodes is not None else 0),
                    }
                },
            )
        else:
            self.hybrid = None
            logger.error("没有可用的检索器，混合检索器初始化失败")

    async def retrieve(self, query: str, kb_ids: Optional[List[str]] = None) -> List[NodeWithScore]:
        """Hybrid retrieval with optional knowledge-base filtering."""
        if not self.hybrid:
            logger.error("混合检索器未初始化，无法执行检索")
            return []
        
        # 创建QueryBundle并添加kb_ids到元数据
        q = QueryBundle(query_str=query)
        if kb_ids:
            # 将kb_ids添加到QueryBundle的元数据中
            q.metadata = {"kb_ids": kb_ids}
        elif self.kb_ids:
            # 使用初始化时的kb_ids
            q.metadata = {"kb_ids": self.kb_ids}
        
        _t0 = time.perf_counter()
        try:
            logger.info(
                "HybridRetrieval.retrieve: start",
                extra={
                    "kb_ids": (kb_ids or self.kb_ids),
                    "query_preview": query[:80],
                },
            )
        except Exception:
            pass

        # NOTE: HybridRetriever.aretrieve is async
        if asyncio.iscoroutinefunction(self.hybrid.aretrieve):
            nodes = await self.hybrid.aretrieve(q)
        # fallback (sync) – unlikely
        else:
            loop = asyncio.get_running_loop()
            nodes = await loop.run_in_executor(None, lambda: self.hybrid.retrieve(q))

        _elapsed_ms = int((time.perf_counter() - _t0) * 1000)
        try:
            logger.info(
                "HybridRetrieval.retrieve: done",
                extra={
                    "returned_nodes": len(nodes) if nodes else 0,
                    "duration_ms": _elapsed_ms,
                },
            )
        except Exception as e:
            logger.exception(e)
        return nodes