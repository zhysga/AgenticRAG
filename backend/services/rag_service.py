"""
RAG检索服务
"""
import logging
import time
from typing import List, Dict, Any, Optional
import uuid
import pickle
import asyncio
from pathlib import Path
from datetime import datetime

from backend.adapters.vector_store import VectorStoreAdapter
from backend.adapters.embedding_client import EmbeddingClient
from backend.adapters.reranker import BGERerankerAdapter, MockRerankerAdapter
from backend.core.enhanced_rag_engine import EnhancedRAGEngine
from backend.core.splitter import SentenceSplitter
from backend.config.settings import settings
from llama_index.core.schema import Document, NodeWithScore
from backend.utils.logger import get_logger
from backend.utils.metrics import CACHE_HIT, time_retrieve

logger = get_logger(__name__)


class RAGService:
    """RAG检索服务"""
    
    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        embedding_client: EmbeddingClient,
        reranker: Optional[BGERerankerAdapter] = None
    ):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.reranker = reranker
        self.engine: Optional[EnhancedRAGEngine] = None
    
    def _init_engine(self, kb_ids: List[str]):
        """初始化引擎"""
        try:
            logger.info("正在初始化检索引擎...")
            # 使用新的配置格式
            cfg = {
                "retrieval": {"dense_topk": 10, "sparse_topk": 192, "fusion_topk": 256},
                "reranking": {"enabled": self.reranker is not None},
                "chunking": {"chunk_size": 512, "chunk_overlap": 50}
            }
            self.engine = EnhancedRAGEngine(
                cfg=cfg,
                vector_store=self.vector_store,
                embedding_client=self.embedding_client,
                kb_ids=kb_ids
            )
            logger.info("检索引擎初始化成功")
        except Exception as e:
            logger.error(f"检索引擎初始化失败: {e}")
            self.engine = None
    
    async def retrieve(
        self,
        query: str,
        kb_ids: List[str],
        filters: Dict[str, Any],
        top_k: int = 5,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """执行检索（增强错误处理）"""
        try:
            if not query or not query.strip():
                logger.error("查询为空")
                return []
            
            logger.info(
                f"开始检索: query='{query[:50]}...', kb_ids={kb_ids}, top_k={top_k}",
                extra={"request_id": request_id},
            )

            # --- Hybrid retrieval path ---
            if settings.retrieval_enable_hybrid:
                logger.info("使用混合检索引擎", extra={"request_id": request_id})
                if self.engine is None:
                    self._init_engine(kb_ids)
                if self.engine is not None:
                    try:
                        with time_retrieve():
                            nodes: List[NodeWithScore] = await self.engine.retrieve(query, kb_ids, request_id=request_id)
                        hybrid_docs = self._nodes_to_dicts(nodes)[:top_k]
                        if hybrid_docs:
                            logger.info(f"混合检索完成，返回{len(hybrid_docs)}个结果", extra={"request_id": request_id})
                            return hybrid_docs
                        else:
                            logger.info("混合检索未返回结果，回退到向量搜索", extra={"request_id": request_id})
                    except Exception as e:
                        logger.error(f"混合检索失败: {e}，回退到向量搜索", extra={"request_id": request_id})
                else:
                    logger.warning("混合检索引擎未初始化，回退到向量搜索", extra={"request_id": request_id})
            
            # 生成查询向量
            logger.info("生成查询向量", extra={"request_id": request_id})
            _t0 = time.perf_counter()
            query_vector = self.embedding_client.embed_query(query)
            _embed_ms = int((time.perf_counter() - _t0) * 1000)
            logger.info(f"查询向量生成完成，用时{_embed_ms}ms", extra={"request_id": request_id})
            
            # 执行向量搜索
            logger.info("执行向量搜索", extra={"request_id": request_id})
            # 过滤掉不应进入 where 的键，避免 Chroma 报“不支持的操作符: kb_ids”
            excluded_keys = {"kb_ids", "knowledge_base_ids", "top_k", "rerank", "custom_filters"}
            raw_filters = filters or {}
            safe_filters = {k: v for k, v in raw_filters.items() if k not in excluded_keys}
            try:
                _fk = list(safe_filters.keys()) if isinstance(safe_filters, dict) else []
                logger.info(
                    f"向量搜索参数: kb_ids_count={len(kb_ids) if kb_ids else 0}, k={top_k}, where_keys={_fk}",
                    extra={"request_id": request_id},
                )
            except Exception:
                pass
            _t1 = time.perf_counter()
            results = self.vector_store.search(
                query_embedding=query_vector,
                k=top_k,
                where=safe_filters,
                kb_ids=kb_ids,
                request_id=request_id,
            )
            _search_ms = int((time.perf_counter() - _t1) * 1000)
            
            logger.info(
                f"检索完成，返回{len(results)}个结果，用时{_search_ms}ms",
                extra={"request_id": request_id},
            )
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True, extra={"request_id": request_id})
            return []
    
    def rerank(self, query: str, docs: List[Dict[str, Any]], request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """执行重排序"""
        try:
            if not self.reranker or not docs:
                return docs
            
            logger.info(f"开始重排序: query={query[:50]}..., docs_count={len(docs)}", extra={"request_id": request_id})
            
            # 提取文档内容
            doc_contents = [doc.get("content", "") for doc in docs]
            
            # 执行重排序
            reranked_scores = self.reranker.rerank(query, doc_contents)
            
            # 更新文档分数
            for i, doc in enumerate(docs):
                doc["score"] = reranked_scores[i]
            
            # 按分数重新排序
            docs.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"重排序完成，返回{len(docs)}个结果", extra={"request_id": request_id})
            return docs
            
        except Exception as e:
            logger.error(f"重排序失败: {e}", extra={"request_id": request_id})
            return docs
    
    def _nodes_to_dicts(self, nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        """将NodeWithScore列表转换为字典列表"""
        try:
            docs = []
            for node in nodes:
                # 兼容两类输入：
                # 1) NodeWithScore(TextNode)（标准混合检索路径）
                # 2) dict（某些检索实现/回退路径可能直接返回字典）
                if isinstance(node, dict):
                    chunk_id = node.get("chunk_id") or node.get("id") or str(uuid.uuid4())
                    docs.append({
                        "chunk_id": chunk_id,
                        "content": node.get("content", ""),
                        "score": node.get("score", 0.0),
                        "metadata": node.get("metadata", {}),
                        "kb_id": node.get("kb_id")
                    })
                    continue
                
                # 标准 NodeWithScore → TextNode
                inner = getattr(node, "node", None)
                if inner is None:
                    # 防御式处理：当结构异常时也要返回可用结果
                    docs.append({
                        "chunk_id": str(uuid.uuid4()),
                        "content": "",
                        "score": getattr(node, "score", 0.0),
                        "metadata": {}
                    })
                    continue
                
                docs.append({
                    "chunk_id": getattr(inner, "id_", str(uuid.uuid4())),
                    "content": getattr(inner, "text", ""),
                    "score": getattr(node, "score", 0.0),
                    "metadata": getattr(inner, "metadata", {})
                })
            return docs
        except Exception as e:
            logger.error(f"节点转换失败: {e}")
            return []

    # ------------------------------------------------------------------
    def _invalidate_nodes_cache(self):
        """Remove nodes cache file to force rebuild next time."""
        try:
            cache_path = Path("./storage/nodes_cache.pkl")
            if cache_path.exists():
                cache_path.unlink()
                logger.info("nodes cache invalidated: %s", cache_path)
        except Exception as e:
            logger.warning(f"invalidate cache failed: {e}")
    
    def add_documents(
        self,
        kb_id: str,
        documents: List[Dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[str]:
        """添加文档到知识库"""
        try:
            logger.info(f"开始添加文档到知识库 {kb_id}: {len(documents)}个文档")
            
            chunk_ids = []
            
            for doc in documents:
                # 分块处理
                chunks = self._chunk_document(doc, chunk_size, chunk_overlap)
                
                # 生成嵌入向量
                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    chunk["chunk_id"] = chunk_id
                    chunk["kb_id"] = kb_id
                    chunk["created_at"] = datetime.now()
                    
                    # 生成向量
                    vector = self.embedding_client.embed_text(chunk["content"])
                    chunk["vector"] = vector
                    
                    # 存储到向量库
                    self.vector_store.add_document(chunk)
                    chunk_ids.append(chunk_id)
            
            logger.info(f"文档添加完成，生成{len(chunk_ids)}个分块")
            # 使节点缓存失效，保证后续混合检索使用最新节点
            self._invalidate_nodes_cache()
            return chunk_ids
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def delete_documents(self, kb_id: str, chunk_ids: Optional[List[str]] = None) -> bool:
        """删除文档"""
        try:
            logger.info(f"开始删除文档: kb_id={kb_id}, chunk_ids={chunk_ids}")
            
            if chunk_ids:
                # 删除指定分块
                for chunk_id in chunk_ids:
                    self.vector_store.delete_document(chunk_id)
            else:
                # 删除整个知识库
                self.vector_store.delete_knowledge_base(kb_id)
            
            logger.info("文档删除完成")
            # 删除后使节点缓存失效
            self._invalidate_nodes_cache()
            return True
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            raise
    
    def get_knowledge_base_stats(self, kb_id: str) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            stats = self.vector_store.get_knowledge_base_stats(kb_id)
            return stats
            
        except Exception as e:
            logger.error(f"获取知识库统计失败: {e}")
            raise
    
    def _chunk_document(self, doc: Dict[str, Any], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """分块处理文档"""
        content = doc.get("content", "")
        if not content:
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        content_length = len(content)

        # Handle short documents that fit within a single chunk
        if content_length <= chunk_size:
            chunk = {
                "content": content,
                "chunk_index": chunk_index,
                "file_name": doc.get("file_name", ""),
                "file_type": doc.get("file_type", ""),
                "metadata": doc.get("metadata", {}),
                "start_pos": 0,
                "end_pos": content_length,
            }
            return [chunk]

        # Sliding window chunking for longer documents
        while start < content_length:
            end = min(start + chunk_size, content_length)
            chunk_content = content[start:end]

            chunk = {
                "content": chunk_content,
                "chunk_index": chunk_index,
                "file_name": doc.get("file_name", ""),
                "file_type": doc.get("file_type", ""),
                "metadata": doc.get("metadata", {}),
                "start_pos": start,
                "end_pos": end,
            }

            chunks.append(chunk)
            chunk_index += 1

            # If we've reached the end of the content, break to avoid
            # creating empty or overlapping chunks endlessly
            if end == content_length:
                break

            # Move the window forward with the specified overlap
            start = end - chunk_overlap if chunk_overlap < chunk_size else end

        return chunks