"""
索引服务 - 负责知识库文件的向量化索引
"""
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from backend.adapters.vector_store import VectorStoreAdapter
from backend.adapters.embedding_client import EmbeddingClient
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class IndexingService:
    """索引服务类"""
    
    def __init__(self, vector_store: VectorStoreAdapter, embedding_client: EmbeddingClient):
        self.vector_store = vector_store
        self.embedding_client = embedding_client

    # ------------------------------------------------------------------
    # 文件内容读取（支持纯文本和二进制文档格式）
    # ------------------------------------------------------------------
    _BINARY_EXTS = frozenset(['.pdf', '.doc', '.docx', '.pptx', '.xlsx'])

    def _read_file_content(self, file_path: str, file_name: str = "") -> Optional[str]:
        """根据文件扩展名选择合适的读取方式，返回纯文本内容。"""
        ext = Path(file_path).suffix.lower()

        # --- PDF ---
        if ext == '.pdf':
            return self._extract_pdf(file_path)

        # --- DOCX ---
        if ext == '.docx':
            return self._extract_docx(file_path)

        # --- DOC (旧格式，需要额外库，暂做提示) ---
        if ext == '.doc':
            logger.warning(f".doc 格式建议转换为 .docx 后上传: {file_name}")
            return self._extract_docx(file_path)  # 尝试当 docx 读取

        # --- 其他二进制格式 ---
        if ext in ('.pptx', '.xlsx'):
            logger.warning(f"暂不支持 {ext} 格式的文本提取: {file_name}")
            return None

        # --- 默认：纯文本 ---
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"以文本模式读取文件失败: {file_path}, 错误: {e}")
            return None

    @staticmethod
    def _extract_pdf(file_path: str) -> Optional[str]:
        """使用 PyPDF2 提取 PDF 文本"""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            texts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
            content = "\n".join(texts)
            if content.strip():
                logger.info(f"PDF 提取成功: {os.path.basename(file_path)} ({len(content)} 字符, {len(reader.pages)} 页)")
                return content
            logger.warning(f"PDF 提取结果为空: {file_path}")
            return None
        except ImportError:
            logger.warning("PyPDF2 未安装，无法解析 PDF。请执行: pip install PyPDF2")
            return None
        except Exception as e:
            logger.error(f"PDF 提取失败: {file_path}, 错误: {e}")
            return None

    @staticmethod
    def _extract_docx(file_path: str) -> Optional[str]:
        """使用 python-docx 提取 DOCX 文本"""
        try:
            from docx import Document
            doc = Document(file_path)
            texts = [para.text for para in doc.paragraphs if para.text.strip()]
            content = "\n".join(texts)
            if content.strip():
                logger.info(f"DOCX 提取成功: {os.path.basename(file_path)} ({len(content)} 字符)")
                return content
            logger.warning(f"DOCX 提取结果为空: {file_path}")
            return None
        except ImportError:
            logger.warning("python-docx 未安装，无法解析 DOCX。请执行: pip install python-docx")
            return None
        except Exception as e:
            logger.error(f"DOCX 提取失败: {file_path}, 错误: {e}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
        """文本分块"""
        if not text:
            return []
        
        # 如果文本很短，直接返回
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        if chunk_size <= 0:
            chunk_size = 512
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 2)

        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # 如果不是最后一块，尝试在空格处分割
            if end < len(text):
                # 向前查找最近的空格或标点符号
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in ' \n\t.,!?;，。！？；':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            next_start = end - chunk_overlap
            if next_start <= start:
                next_start = end
            start = next_start
        
        return chunks
    
    def index_files(
        self, 
        kb_id: str, 
        files: List[Dict[str, Any]], 
        force: bool = False,
        batch_size: int = 10,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """为知识库文件建立索引（优化版本，支持批量处理和进度回调）"""
        try:
            logger.info(f"🚩 开始为知识库 {kb_id} 建立索引 | 文件数: {len(files)} | 批量大小: {batch_size} | force={force}")
            
            # 创建向量集合
            collection_name = f"kb_{kb_id}"
            
            # 检查集合是否存在
            try:
                existing_collection = self.vector_store.get_collection(collection_name)
                if existing_collection and force:
                    logger.info(f"🧹 强制重新索引，删除现有集合: {collection_name}")
                    # Chroma 0.5.x 需显式使用关键字参数 name
                    self.vector_store.client.delete_collection(name=collection_name)
                    existing_collection = None
            except Exception as e:
                logger.warning(f"检查/删除现有集合时出错（将忽略并继续）: {e}")

            # 如需要则创建集合；否则使用已有集合以支持增量写入
            if existing_collection is None:
                metadata = {"kb_id": kb_id, "created_at": datetime.now().isoformat()}
                if not self.vector_store.create_collection(collection_name, metadata):
                    logger.error(f"创建集合失败: {collection_name}")
                    return {
                        "kb_id": kb_id,
                        "indexed_files": 0,
                        "total_chunks": 0,
                        "status": "failed",
                        "message": "创建向量集合失败"
                    }
                collection = self.vector_store.get_collection(collection_name)
            else:
                collection = existing_collection

            # 刷新集合句柄，确保使用最新可用引用
            try:
                collection = self.vector_store.get_collection(collection_name)
            except Exception as _e:
                logger.warning(f"刷新集合句柄失败，将继续使用现有引用: {_e}")

            if not collection:
                logger.error(f"获取集合失败: {collection_name}")
                return {
                    "kb_id": kb_id,
                    "indexed_files": 0,
                    "total_chunks": 0,
                    "status": "failed",
                    "message": "获取向量集合失败"
                }
            
            # 初始化统计量与映射，避免未定义变量
            indexed_files = 0
            total_chunks = 0
            file_chunk_map: Dict[str, int] = {}

            # 批量处理文件
            for batch_start in range(0, len(files), batch_size):
                batch_files = files[batch_start:batch_start + batch_size]
                batch_documents = []
                
                current_batch = batch_start // batch_size + 1
                total_batches = (len(files) - 1) // batch_size + 1
                logger.info(
                    f"📦 处理批次 {current_batch}/{total_batches} | 文件范围: {batch_start+1}-{min(batch_start + batch_size, len(files))}"
                )
                
                for file in batch_files:
                    try:
                        file_id = file.get("file_id")
                        file_path = file.get("file_path")
                        file_name = file.get("file_name", "")
                        
                        if not file_path or not Path(file_path).exists():
                            logger.warning(f"⛔ 文件路径不可用，将尝试使用内联内容 | file_path: {file_path} | 文件ID: {file_id} | 名称: {file_name}")
                        
                        logger.info(f"📝 开始处理文件 | 名称: {file_name} | ID: {file_id} | 路径: {file_path}")
                        
                        # 读取文件内容或使用内联内容
                        content = None
                        if file_path and Path(file_path).exists():
                            logger.debug("读取文件内容...")
                            content = self._read_file_content(file_path, file_name)
                        else:
                            # 回退：使用传入的内联内容（用于存储层没有物理文件路径时）
                            content = file.get("content") or (file.get("metadata") or {}).get("content")
                            if content is None:
                                logger.warning(
                                    f"⛔ 无法获取文件内容 | 无有效 file_path 且缺少内联 content | 文件ID: {file_id} | 名称: {file_name}"
                                )
                                continue
                            logger.info("使用内联内容进行索引（无文件路径）")
                        
                        # 分块
                        logger.debug(f"对文件进行分块处理（chunk_size=默认, chunk_overlap=默认）...")
                        chunks = self.chunk_text(content)
                        logger.info(f"🔪 分块完成 | 文件: {file_name} | 块数: {len(chunks)}")
                        
                        # 构建文档
                        for i, chunk in enumerate(chunks):
                            doc_id = f"{file_id}_{i}"
                            doc = {
                                "id": doc_id,
                                "kb_id": kb_id,
                                "file_id": file_id,
                                "file_name": file_name,
                                "content": chunk,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "metadata": {
                                    "chunk_size": len(chunk),
                                    "file_path": str(file_path),
                                    "indexed_at": datetime.now().isoformat()
                                }
                            }
                            batch_documents.append(doc)
                        
                        # 更新文件状态
                        file_chunk_map[file_id] = len(chunks)
                        indexed_files += 1
                        logger.info(f"✅ 文件处理完成 | 名称: {file_name} | 生成文本块: {len(chunks)} | 已处理文件: {indexed_files}/{len(files)}")
                        
                    except Exception as e:
                        logger.error(f"处理文件时出错 | 路径: {file.get('file_path')} | 错误: {e}", exc_info=True)
                        continue
                
                # 批量生成嵌入向量
                if batch_documents:
                    logger.info(f"🧠 开始批量生成嵌入向量 | 文档块: {len(batch_documents)}")
                    try:
                        # 使用批量嵌入方法提高效率
                        batch_embeddings = self.embedding_client.embed_documents([doc["content"] for doc in batch_documents])
                        logger.info(f"🧠✅ 批量嵌入完成 | 文档块: {len(batch_documents)}")
                    except Exception as e:
                        logger.error(f"批量生成嵌入向量失败，回退到逐个处理: {e}")
                        # 回退到逐个处理
                        batch_embeddings = []
                        for doc in batch_documents:
                            try:
                                embedding = self.embedding_client.embed_query(doc["content"])
                                batch_embeddings.append(embedding)
                            except Exception as e2:
                                logger.error(f"单个嵌入生成失败: {e2}")
                                batch_embeddings.append([0.0] * 384)  # 使用默认维度
                    
                    # 批量添加到集合
                    try:
                        # Chroma 1.0.x: 使用关键字参数确保兼容性
                        collection = self.vector_store.client.get_or_create_collection(
                            name=collection_name,
                            metadata={"kb_id": kb_id, "created_at": datetime.now().isoformat()}
                        )
                        logger.info(f"📥 批量写入向量集合 | 集合: {collection_name} | 文档块: {len(batch_documents)}")
                        collection.add(
                            ids=[doc["id"] for doc in batch_documents],
                            embeddings=batch_embeddings,
                            documents=[doc["content"] for doc in batch_documents],
                            metadatas=[doc["metadata"] for doc in batch_documents]
                        )
                        logger.info(f"📥✅ 批次添加完成 | 写入块: {len(batch_documents)}")
                        total_chunks += len(batch_documents)
                    except Exception as e:
                        logger.error(f"批量添加到集合失败: {e}")
                        # 回退到逐个添加
                        for i in range(len(batch_documents)):
                            try:
                                # 每次添加前刷新collection引用 - 使用关键字参数
                                collection = self.vector_store.client.get_or_create_collection(
                                    name=collection_name,
                                    metadata={"kb_id": kb_id}
                                )
                                collection.add(
                                    ids=[batch_documents[i]["id"]],
                                    embeddings=[batch_embeddings[i]],
                                    documents=[batch_documents[i]["content"]],
                                    metadatas=[batch_documents[i]["metadata"]]
                                )
                                total_chunks += 1
                            except Exception as e2:
                                logger.error(f"单个块添加失败: {e2}")
                
                # 调用进度回调
                if progress_callback:
                    try:
                        progress = min(100, (indexed_files / len(files)) * 100) if len(files) > 0 else 100
                        progress_callback({
                            "kb_id": kb_id,
                            "progress": progress,
                            "indexed_files": indexed_files,
                            "total_files": len(files),
                            "total_chunks": total_chunks,
                            "current_batch": current_batch,
                            "total_batches": total_batches
                        })
                    except Exception as e:
                        logger.warning(f"进度回调失败: {e}")
            
            logger.info(f"🏁 知识库 {kb_id} 索引完成 | 处理文件: {indexed_files}/{len(files)} | 文本块: {total_chunks}")

            # 清理混合检索节点缓存，确保下次检索使用最新文档
            try:
                cache_path = Path("./storage/nodes_cache.pkl")
                if cache_path.exists():
                    cache_path.unlink()
                    logger.info(f"已清理混合检索节点缓存: {cache_path}")
            except Exception as _e:
                logger.warning(f"清理混合检索节点缓存失败（不阻断）: {_e}")

            result = {
                "kb_id": kb_id,
                "indexed_files": indexed_files,
                "total_chunks": total_chunks,
                "status": "completed",
                "message": f"成功索引 {indexed_files} 个文件，{total_chunks} 个文本块",
                "file_chunk_map": file_chunk_map  # 返回每个文件的分块数映射
            }
            logger.info(f"📊 索引任务总结: {json.dumps({k: v for k, v in result.items() if k != 'file_chunk_map'}, ensure_ascii=False)}")
            return result
            
        except Exception as e:
            logger.error(f"知识库 {kb_id} 索引失败: {e}", exc_info=True)
            return {
                "kb_id": kb_id,
                "indexed_files": 0,
                "total_chunks": 0,
                "status": "failed",
                "message": str(e)
            }
    
    def reindex_knowledge_base(self, kb_id: str, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """重新索引知识库（强制重建）"""
        return self.index_files(kb_id, files, force=True)
