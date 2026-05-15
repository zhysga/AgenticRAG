"""
知识库管理服务
"""
import logging
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import asyncio
import threading
from pathlib import Path

from backend.models.knowledge_base import (
    KnowledgeBaseInfo, KnowledgeBaseCreate, KnowledgeBaseScope,
    FileInfo, FileUploadRequest, ReindexRequest
)
from backend.adapters.storage_adapter import StorageAdapter
from backend.adapters.vector_store import VectorStoreAdapter
from backend.adapters.embedding_client import EmbeddingClient
from backend.services.indexing_service import IndexingService
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBaseService:
    """知识库管理服务"""
    
    def __init__(
        self, 
        storage: StorageAdapter, 
        vector_store: VectorStoreAdapter,
        embedding_client: EmbeddingClient = None,
        preload_embeddings: bool = False
    ):
        self.storage = storage
        self.vector_store = vector_store
        self._kb_cache = {}  # 知识库缓存
        self._reindex_status = {}  # 索引状态缓存 {kb_id: status_dict}
        self._status_lock = threading.Lock()  # 状态锁
        
        # 预加载嵌入模型，避免索引时阻塞
        self._embedding_client = None
        self.indexing_service = None
        
        # 默认不预加载，避免首次请求阻塞；在需要索引时延迟初始化
        if preload_embeddings:
            try:
                logger.info("开始预加载嵌入模型...")
                if embedding_client:
                    self._embedding_client = embedding_client
                else:
                    from backend.adapters.embedding_client import SentenceTransformerEmbeddingClient
                    self._embedding_client = SentenceTransformerEmbeddingClient()
                
                # 初始化索引服务
                self.indexing_service = IndexingService(vector_store, self._embedding_client)
                logger.info("嵌入模型和索引服务预加载成功")
                
            except Exception as e:
                logger.error(f"预加载嵌入模型失败: {e}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                # 预加载失败时，保持延迟初始化能力
                self._embedding_client = None
                self.indexing_service = None
    
    def _get_indexing_service(self) -> Optional[IndexingService]:
        """获取索引服务，支持延迟初始化"""
        if self.indexing_service is None:
            try:
                from backend.adapters.embedding_client import SentenceTransformerEmbeddingClient
                logger.info("开始初始化嵌入客户端...")
                self._embedding_client = SentenceTransformerEmbeddingClient()
                logger.info("嵌入客户端初始化成功，开始初始化索引服务...")
                self.indexing_service = IndexingService(self.vector_store, self._embedding_client)
                logger.info("索引服务延迟初始化成功")
            except Exception as e:
                logger.error(f"索引服务初始化失败: {e}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                return None
        return self.indexing_service
    
    def create_knowledge_base(self, kb_data: KnowledgeBaseCreate) -> KnowledgeBaseInfo:
        """创建知识库"""
        try:
            logger.info(f"开始创建知识库: {kb_data.name}")
            
            # 生成知识库ID
            kb_id = str(uuid.uuid4())
            
            # 创建知识库信息
            kb_info = KnowledgeBaseInfo(
                kb_id=kb_id,
                name=kb_data.name,
                description=kb_data.description,
                labels=kb_data.labels,
                scope=kb_data.scope,
                user_id=kb_data.user_id,
                project_id=kb_data.project_id,
                file_count=0,
                chunk_count=0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status="active"
            )
            
            # 存储到数据库
            self.storage.save_knowledge_base(kb_info.dict())
            
            # 更新缓存
            self._kb_cache[kb_id] = kb_info

            # 预创建对应的向量集合，避免后续统计/查询时报 "Collection does not exist"
            try:
                collection_name = f"kb_{kb_id}"
                meta = {"kb_id": kb_id, "created_at": datetime.now().isoformat()}
                created = self.vector_store.create_collection(collection_name, meta)
                if created:
                    logger.info(f"已为知识库创建空集合: {collection_name}")
                else:
                    logger.warning(f"创建知识库集合失败（可忽略，稍后索引会创建）: {collection_name}")
            except Exception as _e:
                # 非阻断：集合会在首次索引时创建
                logger.warning(f"预创建集合异常（非致命）: kb_id={kb_id}, err={_e}")
            
            logger.info(f"知识库创建成功: {kb_id}")
            return kb_info
            
        except Exception as e:
            logger.error(f"创建知识库失败: {e}")
            raise
    
    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBaseInfo]:
        """获取知识库信息"""
        try:
            # 先检查缓存
            if kb_id in self._kb_cache:
                return self._kb_cache[kb_id]
            
            # 从数据库获取
            kb_data = self.storage.get_knowledge_base(kb_id)
            if kb_data:
                kb_info = KnowledgeBaseInfo(**kb_data)
                self._kb_cache[kb_id] = kb_info
                return kb_info
            
            return None
            
        except Exception as e:
            logger.error(f"获取知识库失败: {e}")
            return None
    
    def list_knowledge_bases(
        self,
        page: int = 1,
        size: int = 10,
        scope: Optional[KnowledgeBaseScope] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        labels: Optional[List[str]] = None
    ) -> List[KnowledgeBaseInfo]:
        """列出知识库"""
        try:
            logger.info(f"列出知识库: page={page}, size={size}")
            
            # 构建过滤条件
            filters = {}
            # 兼容不同类型的 scope，避免 'dict' 没有属性 'value' 的异常
            if scope is not None:
                try:
                    if isinstance(scope, KnowledgeBaseScope):
                        filters["scope"] = scope.value
                    elif isinstance(scope, str):
                        filters["scope"] = scope
                    else:
                        # 其他类型统一转字符串以保障兼容
                        filters["scope"] = str(scope)
                except Exception:
                    # 出错时忽略 scope 过滤，确保接口可用
                    logger.warning("scope 处理异常，已忽略该过滤条件")
            if user_id:
                filters["user_id"] = user_id
            if project_id:
                filters["project_id"] = project_id
            if labels:
                # 允许 labels 传入为字符串或列表
                if isinstance(labels, str):
                    filters["labels"] = [l.strip() for l in labels.split(",") if l.strip()]
                else:
                    filters["labels"] = labels
            
            # 从数据库获取
            kb_data_list = self.storage.list_knowledge_bases(
                page=page,
                size=size,
                filters=filters
            )
            
            # 转换为KnowledgeBaseInfo对象
            knowledge_bases = []
            for kb_data in kb_data_list:
                kb_info = KnowledgeBaseInfo(**kb_data)
                knowledge_bases.append(kb_info)
            
            logger.info(f"获取到{len(knowledge_bases)}个知识库")
            return knowledge_bases
            
        except Exception as e:
            logger.error(f"列出知识库失败: {e}")
            return []
    
    def update_knowledge_base(self, kb_id: str, updates: Dict[str, Any]) -> Optional[KnowledgeBaseInfo]:
        """更新知识库"""
        try:
            logger.info(f"开始更新知识库: {kb_id}")
            
            # 获取现有知识库
            kb_info = self.get_knowledge_base(kb_id)
            if not kb_info:
                logger.warning(f"知识库不存在: {kb_id}")
                return None
            
            # 更新字段
            update_data = kb_info.dict()
            for key, value in updates.items():
                if key in update_data:
                    update_data[key] = value
            
            # 更新时间戳
            update_data["updated_at"] = datetime.now()
            
            # 保存到数据库
            self.storage.update_knowledge_base(kb_id, update_data)
            
            # 更新缓存
            updated_kb = KnowledgeBaseInfo(**update_data)
            self._kb_cache[kb_id] = updated_kb
            
            logger.info(f"知识库更新成功: {kb_id}")
            return updated_kb
            
        except Exception as e:
            logger.error(f"更新知识库失败: {e}")
            return None
    
    def delete_knowledge_base(self, kb_id: str) -> bool:
        """删除知识库"""
        try:
            logger.info(f"开始删除知识库: {kb_id}")
            
            # 1) 从向量存储删除集合（kb_<kb_id>）
            self.vector_store.delete_knowledge_base(kb_id)
            
            # 2) 清理该知识库下的文件元数据（kb_files.json）
            if hasattr(self.storage, "delete_kb_files"):
                try:
                    self.storage.delete_kb_files(kb_id)
                except Exception as _e:
                    logger.warning(f"清理知识库文件元数据失败但不阻断主流程: {kb_id}, err={_e}")
            
            # 3) 从知识库索引（knowledge_bases.json）删除 KB
            success = self.storage.delete_knowledge_base(kb_id)
            
            if success:
                # 从缓存删除
                self._kb_cache.pop(kb_id, None)
                logger.info(f"知识库删除成功: {kb_id}")
            else:
                logger.warning(f"知识库删除失败: {kb_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"删除知识库失败: {e}")
            return False
    
    def upload_files(
        self,
        kb_id: str,
        files: List[Dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        auto_index: bool = True
    ) -> List[FileInfo]:
        """上传文件到知识库并保存元数据"""
        try:
            logger.info(f"开始上传文件到知识库: {kb_id}, files={len(files)}")

            # 使用存储适配器持久化文件
            saved_files_data: List[Dict[str, Any]] = []
            if hasattr(self.storage, "add_files"):
                saved_files_data = self.storage.add_files(
                    kb_id=kb_id,
                    files=files,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    auto_index=auto_index,
                )
            else:
                logger.warning("StorageAdapter.add_files 未实现，返回空列表")

            # 转换为 FileInfo 对象
            uploaded_files: List[FileInfo] = []
            for f in saved_files_data:
                try:
                    uploaded_files.append(FileInfo(**f))
                except Exception:
                    # 兼容 upload_time 为字符串
                    if isinstance(f.get("upload_time"), str):
                        f["upload_time"] = datetime.fromisoformat(f["upload_time"])  # type: ignore
                    uploaded_files.append(FileInfo(**f))

            logger.info(f"文件上传完成: {kb_id}, 成功{len(uploaded_files)}")
            
            # 自动触发索引
            if auto_index and uploaded_files:
                logger.info(f"准备自动索引: kb_id={kb_id}, auto_index={auto_index}, files_count={len(uploaded_files)}")
                indexing_service = self._get_indexing_service()
                if indexing_service:
                    logger.info(f"开始自动索引知识库: {kb_id}")
                    logger.info(f"文件数据示例: {saved_files_data[0] if saved_files_data else 'None'}")
                    index_result = indexing_service.index_files(kb_id, saved_files_data)
                    logger.info(f"自动索引结果: {index_result}")
                    
                    # 更新文件的索引状态
                    if index_result.get("status") == "completed":
                        file_chunk_map = index_result.get("file_chunk_map", {})
                        if file_chunk_map:
                            # 构建文件状态映射
                            file_status_map = {}
                            for file_id, chunk_count in file_chunk_map.items():
                                file_status_map[file_id] = {
                                    "index_status": "completed",
                                    "chunk_count": chunk_count
                                }
                            
                            # 批量更新文件状态
                            if file_status_map:
                                self.storage.update_files_status(kb_id, file_status_map)
                                logger.info(f"已更新 {len(file_status_map)} 个文件的状态")
                        logger.info(f"知识库 {kb_id} 自动索引成功")
                    else:
                        logger.warning(f"自动索引失败: {index_result}")
                else:
                    logger.warning("索引服务不可用，跳过自动索引")
            else:
                logger.info(f"跳过自动索引: auto_index={auto_index}, files_count={len(uploaded_files) if uploaded_files else 0}")
            
            return uploaded_files

        except Exception as e:
            logger.error(f"上传文件失败: {e}")
            raise
    
    def list_files(
        self,
        kb_id: str,
        page: int = 1,
        size: int = 10,
        file_type: Optional[str] = None,
        labels: Optional[List[str]] = None
    ) -> List[FileInfo]:
        """列出知识库中的文件"""
        try:
            logger.info(f"列出知识库文件: kb_id={kb_id}, page={page}")
            
            # 构建过滤条件
            filters = {"kb_id": kb_id}
            if file_type:
                filters["file_type"] = file_type
            if labels:
                filters["labels"] = labels
            
            # 从数据库获取，如果适配器未实现则返回空列表
            if hasattr(self.storage, "list_files"):
                files_data = self.storage.list_files(
                    page=page,
                    size=size,
                    filters=filters
                )
            else:
                logger.warning("StorageAdapter.list_files 未实现，返回空列表")
                files_data = []
            
            # 转换为FileInfo对象
            files = []
            for file_data in files_data:
                file_info = FileInfo(**file_data)
                files.append(file_info)
            
            logger.info(f"获取到{len(files)}个文件")
            return files
            
        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            return []
    
    def reindex_knowledge_base(self, kb_id: str, file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """重新索引知识库（同步版本，保持兼容性）"""
        try:
            logger.info(f"开始重新索引知识库: {kb_id}")
            
            # 获取索引服务
            indexing_service = self._get_indexing_service()
            if not indexing_service:
                return {
                    "kb_id": kb_id,
                    "reindexed_files": 0,
                    "total_chunks": 0,
                    "status": "failed",
                    "message": "索引服务初始化失败"
                }
            
            # 获取知识库文件
            kb_files_data = self.storage._load_kb_files()
            if kb_id not in kb_files_data:
                return {
                    "kb_id": kb_id,
                    "reindexed_files": 0,
                    "total_chunks": 0,
                    "status": "failed",
                    "message": "知识库不存在"
                }
            
            files = kb_files_data[kb_id]
            
            # 过滤指定文件（如果提供了file_ids）
            if file_ids:
                files = {fid: fdata for fid, fdata in files.items() if fid in file_ids}
            
            files_list = list(files.values())
            
            # 执行重新索引（强制重建）
            result = indexing_service.reindex_knowledge_base(kb_id, files_list)
            
            # 如果索引成功，更新文件状态和分块数
            if result.get("status") == "completed":
                file_chunk_map = result.get("file_chunk_map", {})
                if file_chunk_map:
                    # 构建文件状态映射
                    file_status_map = {}
                    for file_id, chunk_count in file_chunk_map.items():
                        file_status_map[file_id] = {
                            "index_status": "completed",
                            "chunk_count": chunk_count
                        }
                    
                    # 批量更新文件状态
                    if file_status_map:
                        self.storage.update_files_status(kb_id, file_status_map)
                        logger.info(f"已更新 {len(file_status_map)} 个文件的状态")
                else:
                    logger.warning("索引结果中没有文件分块数信息")
            
            logger.info(f"知识库重新索引完成: {kb_id}")
            return result
            
        except Exception as e:
            logger.error(f"重新索引知识库失败: {e}")
            raise
    
    async def reindex_knowledge_base_async(self, kb_id: str, file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """异步重新索引知识库
        
        Args:
            kb_id: 知识库ID
            file_ids: 要重新索引的文件ID列表，如果为None则重新索引所有文件
            
        Returns:
            Dict: 包含索引结果的字典
        """
        logger.info(f"🔍 开始异步重新索引知识库: {kb_id}, 文件ID: {file_ids or '全部文件'}")
        start_time = datetime.now()
        
        try:
            # 初始化状态
            with self._status_lock:
                self._reindex_status[kb_id] = {
                    "status": "processing",
                    "started_at": start_time.isoformat(),
                    "message": "索引处理中...",
                    "progress": 0,
                    "processed_files": 0,
                    "total_files": 0,
                    "total_chunks": 0
                }
                logger.info(f"✅ 已初始化知识库 {kb_id} 的索引状态")
            
            # 获取索引服务（在后台线程中初始化，避免阻塞事件循环）
            logger.info(f"🔄 正在初始化索引服务...")
            indexing_service = await asyncio.to_thread(self._get_indexing_service)
            if not indexing_service:
                error_msg = "❌ 索引服务初始化失败"
                logger.error(error_msg)
                with self._status_lock:
                    self._reindex_status[kb_id].update({
                        "status": "failed",
                        "message": error_msg,
                        "completed_at": datetime.now().isoformat()
                    })
                return {
                    "kb_id": kb_id,
                    "reindexed_files": 0,
                    "total_chunks": 0,
                    "status": "failed",
                    "message": error_msg
                }
            
            # 获取知识库文件（磁盘IO放到线程池执行，避免阻塞事件循环）
            logger.info(f"📂 正在加载知识库 {kb_id} 的文件数据...")
            kb_files_data = await asyncio.to_thread(self.storage._load_kb_files)
            logger.info(f"✅ 已加载知识库数据，共有 {len(kb_files_data)} 个知识库")
            
            if kb_id not in kb_files_data:
                error_msg = f"❌ 知识库 {kb_id} 不存在"
                logger.error(error_msg)
                with self._status_lock:
                    self._reindex_status[kb_id].update({
                        "status": "failed",
                        "message": error_msg,
                        "completed_at": datetime.now().isoformat()
                    })
                return {
                    "kb_id": kb_id,
                    "reindexed_files": 0,
                    "total_chunks": 0,
                    "status": "failed",
                    "message": error_msg
                }
            
            files = kb_files_data[kb_id]
            total_files = len(files)
            logger.info(f"📊 知识库 {kb_id} 共有 {total_files} 个文件")
            
            # 过滤指定文件（如果提供了file_ids）
            if file_ids:
                files = {fid: fdata for fid, fdata in files.items() if fid in file_ids}
                logger.info(f"🔍 已过滤文件，剩余 {len(files)} 个待处理文件")
            
            files_list = list(files.values())
            total_files = len(files_list)
            
            # 更新状态：开始处理
            with self._status_lock:
                self._reindex_status[kb_id].update({
                    "status": "processing",
                    "message": f"开始处理 {total_files} 个文件...",
                    "total_files": total_files,
                    "processed_files": 0,
                    "total_chunks": 0
                })
            logger.info(f"🔄 已更新索引状态，准备开始处理 {total_files} 个文件")
            
            # 准备文件列表，确保包含必要的字段（包含内容回退）
            files_to_index = []
            for file_id, file_data in files.items():
                if not isinstance(file_data, dict):
                    logger.warning(f"文件 {file_id} 数据格式无效，跳过")
                    continue
                
                file_info = {
                    "file_id": file_id,
                    "file_path": file_data.get("file_path"),
                    "file_name": file_data.get("file_name", f"file_{file_id}"),
                    # 允许索引器在缺少物理路径时使用内联内容
                    "content": (file_data.get("metadata") or {}).get("content"),
                    "metadata": file_data.get("metadata") or {}
                }
                files_to_index.append(file_info)
            
            logger.info(f"📝 已准备 {len(files_to_index)} 个文件进行索引")
            
            # 定义进度回调：在索引批次之间实时更新状态
            def _progress_update(progress_data: Dict[str, Any]):
                try:
                    progress = int(progress_data.get("progress", 0))
                    processed_files = int(progress_data.get("indexed_files", 0))
                    total_chunks = int(progress_data.get("total_chunks", 0))
                    current_batch = progress_data.get('current_batch', 0)
                    total_batches = progress_data.get('total_batches', 1)
                    total_files = progress_data.get('total_files', 1)
                    
                    status_msg = (
                        f"处理中... {progress}% 完成 | "
                        f"文件: {processed_files}/{total_files} | "
                        f"批次: {current_batch}/{total_batches} | "
                        f"文本块: {total_chunks}"
                    )
                    
                    with self._status_lock:
                        self._reindex_status.setdefault(kb_id, {})
                        self._reindex_status[kb_id].update({
                            "progress": progress,
                            "processed_files": processed_files,
                            "total_files": int(progress_data.get("total_files", total_files)),
                            "total_chunks": total_chunks,
                            "message": status_msg,
                            "last_updated": datetime.now().isoformat()
                        })
                    
                    # 每10%或每批次都记录一次进度
                    if progress % 10 == 0 or current_batch == 1 or current_batch == total_batches:
                        logger.info(f"📊 索引进度: {status_msg}")
                        
                except Exception as _e:
                    logger.warning(f"进度回调更新状态失败（不阻断）: {_e}", exc_info=True)

            # 执行重新索引（强制重建）（CPU/IO密集操作放到线程池执行），同时传入进度回调
            logger.info(f"🚀 开始执行重新索引，共 {len(files_to_index)} 个文件，使用批量处理...")
            try:
                # 直接调用 index_files 以支持 progress_callback
                result = await asyncio.to_thread(
                    indexing_service.index_files,
                    kb_id,
                    files_to_index,
                    True,   # force=True 等价于重新索引
                    10,     # batch_size 默认值
                    _progress_update
                )
                logger.info(f"✅ 重新索引完成，结果: {result}")
            except Exception as e:
                error_msg = f"重新索引过程中发生错误: {str(e)}"
                logger.error(error_msg, exc_info=True)
                with self._status_lock:
                    self._reindex_status[kb_id].update({
                        "status": "failed",
                        "message": error_msg,
                        "completed_at": datetime.now().isoformat()
                    })
                return {
                    "kb_id": kb_id,
                    "reindexed_files": 0,
                    "total_chunks": 0,
                    "status": "failed",
                    "message": error_msg
                }
            
            # 如果索引成功，更新文件状态和分块数
            if result.get("status") == "completed":
                file_chunk_map = result.get("file_chunk_map", {})
                if file_chunk_map:
                    # 构建文件状态映射
                    file_status_map = {}
                    for file_id, chunk_count in file_chunk_map.items():
                        file_status_map[file_id] = {
                            "index_status": "completed",
                            "chunk_count": chunk_count
                        }
                    
                    # 批量更新文件状态
                    if file_status_map:
                        self.storage.update_files_status(kb_id, file_status_map)
                        logger.info(f"已更新 {len(file_status_map)} 个文件的状态")
                else:
                    logger.warning("索引结果中没有文件分块数信息")
            
            # 计算总耗时
            duration = (datetime.now() - start_time).total_seconds()
            
            # 更新最终状态
            final_status = result.get("status", "unknown")
            final_message = result.get("message", "索引完成")
            reindexed_files = result.get("indexed_files", 0)
            total_chunks = result.get("total_chunks", 0)
            
            with self._status_lock:
                self._reindex_status[kb_id].update({
                    "status": final_status,
                    "message": f"{final_message} | 总耗时: {duration:.2f}秒",
                    "completed_at": datetime.now().isoformat(),
                    "reindexed_files": reindexed_files,
                    "total_chunks": total_chunks,
                    "progress": 100,
                    "duration_seconds": duration
                })
            
            # 记录完成日志
            logger.info(
                f"🏁 知识库 {kb_id} 重新索引完成！"
                f"状态: {final_status}, "
                f"已处理文件: {reindexed_files}/{total_files}, "
                f"总块数: {total_chunks}, "
                f"耗时: {duration:.2f}秒"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"异步重新索引知识库 {kb_id} 失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # 更新错误状态
            with self._status_lock:
                self._reindex_status[kb_id].update({
                    "status": "failed",
                    "message": error_msg,
                    "completed_at": datetime.now().isoformat(),
                    "error": str(e)
                })
            
            # 返回错误信息
            return {
                "kb_id": kb_id,
                "status": "failed",
                "message": error_msg,
                "error": str(e)
            }
    
    def get_reindex_status(self, kb_id: str) -> Dict[str, Any]:
        """获取重新索引状态"""
        with self._status_lock:
            if kb_id not in self._reindex_status:
                return {
                    "kb_id": kb_id,
                    "status": "not_started",
                    "message": "未开始索引"
                }
            
            status = self._reindex_status[kb_id].copy()
            status["kb_id"] = kb_id
            return status
    
    def get_knowledge_base_stats(self, kb_id: str) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            logger.info(f"获取知识库统计信息: {kb_id}")
            
            # 获取知识库信息
            kb_info = self.get_knowledge_base(kb_id)
            if not kb_info:
                return {
                    "kb_id": kb_id,
                    "file_count": 0,
                    "chunk_count": 0,
                    "index_status": "not_found"
                }
            
            # 获取文件列表（使用list_files方法获取FileInfo对象）
            files = self.list_files(kb_id)
            file_count = len(files)
            
            # 计算统计信息
            total_chunks = 0
            completed_files = 0
            
            for file in files:
                if file.chunk_count:
                    total_chunks += file.chunk_count
                if file.index_status == "completed":
                    completed_files += 1
            
            # 确定索引状态
            if file_count == 0:
                index_status = "empty"
            elif completed_files == file_count:
                index_status = "completed"
            elif completed_files > 0:
                index_status = "partial"
            else:
                index_status = "pending"
            
            logger.info(f"统计信息: kb_id={kb_id}, file_count={file_count}, chunk_count={total_chunks}, index_status={index_status}")
            result = {
                "kb_id": kb_id,
                "file_count": file_count,
                "chunk_count": total_chunks,
                "index_status": index_status
            }
            logger.info(f"返回统计结果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"获取知识库统计信息失败: {e}")
            return {
                "kb_id": kb_id,
                "file_count": 0,
                "chunk_count": 0,
                "index_status": "error"
            }
    
    def search_knowledge_base(
        self,
        kb_id: str,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """在知识库中搜索"""
        try:
            logger.info(f"在知识库中搜索: kb_id={kb_id}, query={query}")
            
            # 生成查询向量
            query_vector = self._embedding_client.embed_query(query)
            
            # 使用向量存储搜索
            search_results = self.vector_store.search(
                query_vector=query_vector,
                kb_ids=[kb_id],
                top_k=top_k,
                filters=filters or {}
            )
            
            logger.info(f"搜索完成，找到{len(search_results)}个结果")
            return search_results
            
        except Exception as e:
            logger.error(f"搜索知识库失败: {e}")
            return []