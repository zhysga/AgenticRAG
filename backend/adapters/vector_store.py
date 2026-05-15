"""
向量存储适配器
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
import os
import shutil
from datetime import datetime

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStoreAdapter:
    """Chroma向量存储适配器"""
    
    def __init__(self, persist_directory: str = "./storage/chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collections = {}
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化Chroma客户端（包含损坏库自动修复）"""
        last_error: Optional[BaseException] = None
        for attempt in range(2):
            try:
                # 初始化前先检查持久化目录的sqlite文件是否完好
                self._ensure_persist_dir_integrity()

                # 确保目录存在
                os.makedirs(self.persist_directory, exist_ok=True)

                # 创建Chroma客户端
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )

                logger.info(f"Chroma客户端初始化成功: {self.persist_directory}")

                # 启动时检查集合健康状态
                self._health_check_collections()
                return

            except BaseException as e:
                last_error = e
                logger.error(f"Chroma客户端初始化失败: {e}")
                # 首次失败尝试自动修复持久化目录
                if attempt == 0 and self._attempt_recover_persist_dir():
                    continue
                break

        # 如果尝试两次依旧失败，抛出最后的异常
        if last_error:
            raise last_error

    def _ensure_persist_dir_integrity(self) -> bool:
        """快速检测 sqlite 持久化文件是否损坏，必要时提前备份重建"""
        db_path = os.path.join(self.persist_directory, "chroma.sqlite3")
        if not os.path.exists(db_path):
            return True

        try:
            import sqlite3

            # 只读方式打开并尝试简单查询，若失败视为损坏
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                conn.execute("PRAGMA schema_version;")
            return True
        except Exception as integrity_err:
            logger.warning(
                "Chroma sqlite 持久化文件可能损坏(%s)，尝试备份并重建: %s",
                db_path,
                integrity_err,
            )
            return self._attempt_recover_persist_dir()

    def _attempt_recover_persist_dir(self) -> bool:
        """在Chroma目录损坏时备份旧目录并重新创建，防止sqlite崩溃"""
        try:
            if not os.path.exists(self.persist_directory):
                return False

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{self.persist_directory}_corrupt_{timestamp}"

            shutil.move(self.persist_directory, backup_dir)
            logger.warning(
                "检测到Chroma持久化目录可能损坏，已备份到 %s，准备重建空目录", backup_dir
            )

            os.makedirs(self.persist_directory, exist_ok=True)
            return True
        except Exception as recover_err:
            logger.error(f"Chroma持久化目录备份/重建失败: {recover_err}")
            return False
    
    def _health_check_collections(self):
        """检查并清理损坏的集合"""
        try:
            collections = self.client.list_collections()
            damaged_collections = []
            
            for collection in collections:
                try:
                    # 尝试简单查询来验证集合健康
                    collection.peek(limit=1)
                except Exception as e:
                    logger.warning(f"集合 {collection.name} 可能已损坏: {e}")
                    damaged_collections.append(collection.name)
            
            # 删除损坏的集合
            for name in damaged_collections:
                try:
                    self.client.delete_collection(name)
                    logger.info(f"已删除损坏的集合: {name}")
                except Exception as e:
                    logger.error(f"删除损坏集合失败 {name}: {e}")
            
            if damaged_collections:
                logger.info(f"健康检查完成，清理了 {len(damaged_collections)} 个损坏集合")
            else:
                logger.debug("集合健康检查通过")
                
        except Exception as e:
            logger.error(f"集合健康检查失败: {e}")
    
    def create_collection(self, collection_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """创建集合（幂等）。若已存在则返回 True。缺失时自动创建。"""
        try:
            if collection_name in self.collections:
                logger.debug(f"集合已存在于缓存中: {collection_name}")
                return True

            # 优先使用 get_or_create_collection（如可用）
            get_or_create = getattr(self.client, "get_or_create_collection", None)
            if callable(get_or_create):
                collection = get_or_create(name=collection_name, metadata=metadata or {})
            else:
                # 兼容旧版本：先尝试 get，再创建
                try:
                    collection = self.client.get_collection(name=collection_name)
                except Exception:
                    collection = self.client.create_collection(name=collection_name, metadata=metadata or {})

            self.collections[collection_name] = collection
            logger.info(f"集合创建成功: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def get_collection(self, collection_name: str):
        """获取集合。如果不存在则自动创建并返回，避免检索时报错。
        
        注意：不使用内存缓存，每次都从ChromaDB获取最新的集合引用，
        避免缓存的集合对象引用过期的UUID导致"Collection does not exist"错误。
        """
        try:
            # 始终从ChromaDB获取最新的集合引用，不使用缓存
            # 这样可以避免缓存的集合对象引用了已删除/重建的集合UUID
            collection = self.client.get_or_create_collection(name=collection_name)
            # 更新缓存（可选，主要用于其他方法的兼容性）
            self.collections[collection_name] = collection
            return collection
        except Exception as e:
            logger.error(f"获取或创建集合失败: {collection_name}, 错误: {e}")
            # 清除可能过期的缓存
            if collection_name in self.collections:
                del self.collections[collection_name]
            return None
    
    def add_document(self, document: Dict[str, Any]) -> bool:
        """添加文档"""
        try:
            kb_id = document.get("kb_id")
            if not kb_id:
                logger.error("文档缺少kb_id")
                return False
            
            collection_name = f"kb_{kb_id}"
            
            # 确保集合存在
            if collection_name not in self.collections:
                self.create_collection(collection_name)
            
            collection = self.get_collection(collection_name)
            if not collection:
                return False
            
            # 准备数据
            chunk_id = document.get("chunk_id")
            content = document.get("content", "")
            vector = document.get("vector")
            metadata = document.get("metadata", {})
            
            if not chunk_id or not content:
                logger.error("文档缺少必要字段")
                return False
            
            # 确保metadata不为空，至少包含基本信息
            if not metadata:
                metadata = {"source": "unknown", "type": "text"}
            
            # 添加文档到集合
            collection.add(
                ids=[chunk_id],
                documents=[content],
                embeddings=[vector] if vector is not None else None,
                metadatas=[metadata]
            )
            
            logger.info(f"文档添加成功: {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 10, 
        where: Optional[Dict[str, Any]] = None,
        kb_ids: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """搜索向量"""
        try:
            import time
            if not query_embedding:
                logger.error("查询向量为空")
                return []
            
            # 构建where条件，支持kb_ids过滤
            search_where = where.copy() if where else {}
            try:
                _wk = list(search_where.keys()) if isinstance(search_where, dict) else []
                logger.info(
                    "VectorStore.search: start",
                    extra={
                        "k": k,
                        "kb_ids_count": (len(kb_ids) if kb_ids else 0),
                        "where_keys": _wk,
                        "request_id": request_id,
                    },
                )
            except Exception:
                pass
            _t0 = time.perf_counter()
            
            # 如果指定了知识库ID，搜索对应集合
            if kb_ids:
                results = []
                for kb_id in kb_ids:
                    collection_name = f"kb_{kb_id}"
                    try:
                        collection = self.get_collection(collection_name)
                        if collection:
                            _tkb0 = time.perf_counter()
                            # 在按集合检索时，不强制要求元数据包含 kb_id，避免因缺失而无结果
                            kb_where = search_where.copy()
                            
                            # 验证where条件语法
                            validated_where = self._validate_where_clause(kb_where)
                            
                            # 搜索集合
                            search_results = collection.query(
                                query_embeddings=[query_embedding],
                                n_results=k,
                                where=validated_where
                            )
                            
                            # 处理结果
                            if search_results['ids'][0]:
                                for i, doc_id in enumerate(search_results['ids'][0]):
                                    result = {
                                        'id': doc_id,
                                        'kb_id': kb_id,
                                        'score': 1.0 - float(search_results['distances'][0][i]),  # 转换为相似度分数
                                        'content': search_results['documents'][0][i],
                                        'metadata': search_results['metadatas'][0][i] if search_results['metadatas'][0] else {}
                                    }
                                    results.append(result)
                            _tkb_ms = int((time.perf_counter() - _tkb0) * 1000)
                            try:
                                logger.info(
                                    "VectorStore.search: per-kb",
                                    extra={
                                        "kb_id": kb_id,
                                        "hits": (len(search_results['ids'][0]) if search_results and search_results.get('ids') else 0),
                                        "duration_ms": _tkb_ms,
                                        "request_id": request_id,
                                    },
                                )
                            except Exception:
                                pass
                    except Exception as e:
                        logger.warning(f"搜索知识库 {kb_id} 失败: {e}")
                        continue
                
                # 按分数排序
                results.sort(key=lambda x: x['score'], reverse=True)
                _elapsed_ms = int((time.perf_counter() - _t0) * 1000)
                try:
                    logger.info(
                        "VectorStore.search: done (kb_ids)",
                        extra={
                            "total_hits": len(results),
                            "duration_ms": _elapsed_ms,
                            "request_id": request_id,
                        },
                    )
                except Exception:
                    pass
                return results[:k]
            
            # 如果没有指定知识库ID，搜索所有集合
            else:
                results = []
                collections = self.client.list_collections()
                try:
                    logger.info(
                        "VectorStore.search: listing collections",
                        extra={"total_collections": len(collections), "request_id": request_id},
                    )
                except Exception:
                    pass
                
                for collection in collections:
                    try:
                        # 只搜索知识库集合
                        if not collection.name.startswith('kb_'):
                            continue
                            
                        # 提取kb_id
                        kb_id = collection.name[3:]  # 移除kb_前缀
                        
                        # 在按集合检索时，不强制要求元数据包含 kb_id，避免因缺失而无结果
                        kb_where = search_where.copy()
                        # 验证where条件语法
                        validated_where = self._validate_where_clause(kb_where)
                        
                        # 搜索集合
                        _tc0 = time.perf_counter()
                        search_results = collection.query(
                            query_embeddings=[query_embedding],
                            n_results=k,
                            where=validated_where
                        )
                        _tc_ms = int((time.perf_counter() - _tc0) * 1000)
                        
                        # 处理结果
                        if search_results['ids'][0]:
                            for i, doc_id in enumerate(search_results['ids'][0]):
                                result = {
                                    'id': doc_id,
                                    'kb_id': kb_id,
                                    'score': 1.0 - float(search_results['distances'][0][i]),
                                    'content': search_results['documents'][0][i],
                                    'metadata': search_results['metadatas'][0][i] if search_results['metadatas'][0] else {}
                                }
                                results.append(result)
                        try:
                            logger.info(
                                "VectorStore.search: per-collection",
                                extra={
                                    "collection": collection.name,
                                    "kb_id": kb_id,
                                    "hits": (len(search_results['ids'][0]) if search_results and search_results.get('ids') else 0),
                                    "duration_ms": _tc_ms,
                                    "request_id": request_id,
                                },
                            )
                        except Exception:
                            pass
                    except Exception as e:
                        logger.warning(f"搜索集合 {collection.name} 失败: {e}")
                        continue
                
                # 按分数排序
                results.sort(key=lambda x: x['score'], reverse=True)
                _elapsed_ms = int((time.perf_counter() - _t0) * 1000)
                try:
                    logger.info(
                        "VectorStore.search: done (all)",
                        extra={
                            "total_hits": len(results),
                            "duration_ms": _elapsed_ms,
                            "request_id": request_id,
                        },
                    )
                except Exception:
                    pass
                return results[:k]
                
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return []
    
    def delete_document(self, chunk_id: str) -> bool:
        """删除文档"""
        try:
            # 需要遍历所有集合找到文档
            for collection_name, collection in self.collections.items():
                try:
                    collection.delete(ids=[chunk_id])
                    logger.info(f"文档删除成功: {chunk_id}")
                    return True
                except:
                    continue
            
            logger.warning(f"文档不存在: {chunk_id}")
            return False
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def delete_knowledge_base(self, kb_id: str) -> bool:
        """删除知识库"""
        try:
            collection_name = f"kb_{kb_id}"
            
            if collection_name in self.collections:
                # 删除集合
                self.client.delete_collection(name=collection_name)
                del self.collections[collection_name]
                logger.info(f"知识库删除成功: {kb_id}")
                return True
            else:
                logger.warning(f"知识库不存在: {kb_id}")
                return False
                
        except Exception as e:
            logger.error(f"删除知识库失败: {e}")
            return False
    
    def get_knowledge_base_stats(self, kb_id: str) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            collection_name = f"kb_{kb_id}"
            collection = self.get_collection(collection_name)
            
            if not collection:
                return {
                    "kb_id": kb_id,
                    "document_count": 0,
                    "status": "not_found"
                }
            
            # 获取集合信息
            count = collection.count()
            
            return {
                "kb_id": kb_id,
                "document_count": count,
                "status": "active"
            }
            
        except Exception as e:
            logger.error(f"获取知识库统计失败: {e}")
            return {
                "kb_id": kb_id,
                "document_count": 0,
                "status": "error"
            }
        
    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            if collection_name in self.collections:
                del self.collections[collection_name]
            self.client.delete_collection(name=collection_name)
            logger.info(f"集合删除成功: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """构建where查询条件"""
        if not filters:
            return None
        
        # 过滤掉不应该在 where 子句中出现的键
        # kb_ids 是用于选择集合的，不是元数据过滤条件
        excluded_keys = {"kb_ids", "knowledge_base_ids", "top_k", "rerank"}
        where_clause = {k: v for k, v in filters.items() if k not in excluded_keys}
        
        # 如果过滤后为空，返回 None
        return where_clause if where_clause else None
    
    def _validate_where_clause(self, where_clause: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """验证where查询条件是否符合ChromaDB 1.0+语法"""
        if not where_clause:
            return None
        
        try:
            # ChromaDB 1.0+ 支持的简单语法：{'field': 'value'}
            # 复杂语法：{'field': {'$eq': 'value'}}, {'field': {'$in': ['val1', 'val2']}}
            # 逻辑操作符：{'$and': [{...}, {...}]}, {'$or': [{...}, {...}]}
            
            validated_clause = {}
            for key, value in where_clause.items():
                if key in ['$and', '$or']:
                    # 验证逻辑操作符
                    if isinstance(value, list):
                        validated_subclauses = []
                        for subclause in value:
                            if isinstance(subclause, dict):
                                validated_sub = self._validate_where_clause(subclause)
                                if validated_sub:
                                    validated_subclauses.append(validated_sub)
                        if validated_subclauses:
                            validated_clause[key] = validated_subclauses
                    else:
                        logger.warning(f"逻辑操作符 {key} 需要列表值，当前值: {value}")
                elif isinstance(value, (str, int, float, bool)):
                    # 简单值直接支持
                    validated_clause[key] = value
                elif isinstance(value, dict):
                    # 验证操作符语法
                    valid_operators = {'$eq', '$ne', '$gt', '$gte', '$lt', '$lte', '$in', '$nin'}
                    validated_ops = {}
                    for op, op_value in value.items():
                        if op in valid_operators:
                            if op in ['$in', '$nin'] and not isinstance(op_value, list):
                                logger.warning(f"操作符 {op} 需要列表值，当前值: {op_value}")
                                continue
                            validated_ops[op] = op_value
                        else:
                            logger.warning(f"不支持的操作符: {op}")
                    if validated_ops:
                        validated_clause[key] = validated_ops
                else:
                    logger.warning(f"不支持的where条件格式: {key} = {value}")
            
            return validated_clause if validated_clause else None
            
        except Exception as e:
            logger.error(f"验证where条件失败: {e}")
            return None


    def get_all_documents(self, kb_ids: List[str], limit: int | None = None, offset: int = 0) -> List[Dict[str, Any]]:
        """批量拉取指定知识库全部文档，用于构造稀疏检索节点。
        仅在启用混合检索（PoC）时调用。大数据量环境请使用分页或缓存。"""
        all_docs: List[Dict[str, Any]] = []
        try:
            for kb_id in kb_ids:
                collection_name = f"kb_{kb_id}"
                
                # 始终从ChromaDB获取最新的集合引用
                try:
                    col = self.client.get_or_create_collection(name=collection_name)
                except Exception as col_error:
                    logger.warning(f"无法获取集合 {collection_name}: {col_error}")
                    continue
                
                if not col:
                    continue
                
                try:
                    # 尝试使用标准的get方法获取数据
                    data = col.get(include=["documents", "metadatas"])
                    all_ids = data.get("ids", [])
                    all_docs_text = data.get("documents", [])
                    all_metas = data.get("metadatas", [])
                    
                    # 检查是否获取到数据
                    if not all_ids:
                        logger.info(f"集合 {collection_name} 为空或无数据")
                        continue
                        
                except Exception as get_error:
                    # 如果get方法失败，记录错误并继续
                    logger.warning(f"col.get() 失败 for {collection_name}: {get_error}")
                    # 尝试检查集合是否真的有数据
                    try:
                        count = col.count()
                        logger.info(f"集合 {collection_name} 文档数量: {count}")
                        if count == 0:
                            continue
                    except Exception:
                        pass
                    continue
                
                start = max(0, offset)
                end = (start + limit) if isinstance(limit, int) and limit > 0 else None
                ids_slice = all_ids[start:end]
                docs_slice = all_docs_text[start:end]
                metas_slice = all_metas[start:end]
                
                for doc_id, doc_text, meta in zip(ids_slice, docs_slice, metas_slice):
                    all_docs.append({
                        "chunk_id": doc_id,
                        "content": doc_text,
                        "metadata": meta or {},
                    })
                
                logger.info(f"从集合 {collection_name} 获取了 {len(ids_slice)} 个文档")
                    
            return all_docs
        except Exception as e:
            logger.error(f"get_all_documents failed: {e}")
            return all_docs


class FAISSVectorStoreAdapter:
    """FAISS向量存储适配器（备用实现）"""
    
    def __init__(self, persist_directory: str = "./storage/faiss_db"):
        self.persist_directory = persist_directory
        self.indices = {}
        self.metadata = {}
        self._initialize_storage()
    
    def _initialize_storage(self):
        """初始化FAISS存储"""
        try:
            import faiss
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info(f"FAISS存储初始化成功: {self.persist_directory}")
        except Exception as e:
            logger.error(f"FAISS存储初始化失败: {e}")
            raise
    
    def create_collection(self, collection_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """创建集合"""
        try:
            if collection_name in self.indices:
                return True
            
            # 创建FAISS索引
            dimension = 384  # sentence-transformers/all-MiniLM-L6-v2的维度
            index = faiss.IndexFlatIP(dimension)  # 内积索引
            
            self.indices[collection_name] = index
            self.metadata[collection_name] = metadata or {}
            
            logger.info(f"FAISS集合创建成功: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建FAISS集合失败: {e}")
            return False
    
    def add_document(self, document: Dict[str, Any]) -> bool:
        """添加文档"""
        try:
            kb_id = document.get("kb_id")
            if not kb_id:
                return False
            
            collection_name = f"kb_{kb_id}"
            
            if collection_name not in self.indices:
                self.create_collection(collection_name)
            
            # 添加向量到索引
            vector = document.get("vector")
            if vector is not None:
                vector_array = np.array([vector], dtype=np.float32)
                self.indices[collection_name].add(vector_array)
            
            logger.info(f"FAISS文档添加成功: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"添加FAISS文档失败: {e}")
            return False
    
    def search(
        self,
        query_vector: List[float],
        kb_ids: List[str],
        filters: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索文档"""
        try:
            results = []
            
            for kb_id in kb_ids:
                collection_name = f"kb_{kb_id}"
                
                if collection_name not in self.indices:
                    continue
                
                # 执行搜索
                query_array = np.array([query_vector], dtype=np.float32)
                scores, indices = self.indices[collection_name].search(query_array, top_k)
                
                # 处理结果
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx != -1:  # 有效结果
                        result = {
                            "chunk_id": f"{kb_id}_{idx}",
                            "content": f"FAISS文档内容 {idx}",
                            "score": float(score),
                            "kb_id": kb_id,
                            "metadata": {}
                        }
                        results.append(result)
            
            # 按分数排序
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"FAISS搜索失败: {e}")
            return []
    
    def delete_document(self, chunk_id: str) -> bool:
        """删除文档（FAISS不支持删除）"""
        logger.warning("FAISS不支持删除单个文档")
        return False
    
    def delete_knowledge_base(self, kb_id: str) -> bool:
        """删除知识库"""
        try:
            collection_name = f"kb_{kb_id}"
            
            if collection_name in self.indices:
                del self.indices[collection_name]
                del self.metadata[collection_name]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"删除FAISS知识库失败: {e}")
            return False
    
    def get_knowledge_base_stats(self, kb_id: str) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            collection_name = f"kb_{kb_id}"
            
            if collection_name in self.indices:
                count = self.indices[collection_name].ntotal
                return {
                    "kb_id": kb_id,
                    "document_count": count,
                    "status": "active"
                }
            else:
                return {
                    "kb_id": kb_id,
                    "document_count": 0,
                    "status": "not_found"
                }
                
        except Exception as e:
            logger.error(f"获取FAISS知识库统计失败: {e}")
            return {
                "kb_id": kb_id,
                "document_count": 0,
                "status": "error"
            }


# 默认使用Chroma
VectorStoreAdapter = ChromaVectorStoreAdapter