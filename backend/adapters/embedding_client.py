"""
嵌入模型客户端适配器
"""
import logging
import os
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from backend.utils.logger import get_logger
from backend.config.settings import settings

logger = get_logger(__name__)


class SentenceTransformerEmbeddingClient:
    """SentenceTransformer嵌入客户端"""
    
    def __init__(self, model_name: str = None):
        # 默认取设置中的模型名称，入参覆盖
        self.model_name = model_name or settings.embedding_model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            # 解析本地路径与离线参数
            model_path = settings.embedding_model_path or self.model_name
            cache_dir = settings.hf_cache_dir
            local_only = bool(settings.transformers_offline)

            if local_only:
                # 兼容不同版本的 Transformers/SentenceTransformers 离线模式
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")

            logger.info(
                "正在加载嵌入模型: %s (offline=%s, cache_dir=%s)",
                model_path,
                local_only,
                cache_dir,
            )

            # 加载模型（支持本地/缓存/离线）
            st_kwargs = {}
            if cache_dir:
                st_kwargs["cache_folder"] = cache_dir

            try:
                self.model = SentenceTransformer(model_path, **st_kwargs)
            except TypeError:
                # 兼容旧版本 sentence-transformers 不支持 cache_folder 参数的情况
                logger.debug("SentenceTransformer 不支持 cache_folder 参数，使用默认加载方式")
                self.model = SentenceTransformer(model_path)

            # 设置设备，支持 settings.embedding_device
            device_pref = (settings.embedding_device or "auto").lower()
            if device_pref == "cpu":
                device = "cpu"
            elif device_pref == "cuda":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = self.model.to(device)
            
            logger.info(f"嵌入模型加载成功: {self.model_name}, 设备: {device}")
            
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """嵌入查询文本"""
        try:
            if not self.model:
                raise ValueError("模型未初始化")
            
            # 生成嵌入向量
            embedding = self.model.encode(query, convert_to_tensor=False)
            
            # 转换为列表
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"查询嵌入失败: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """嵌入文本"""
        try:
            if not self.model:
                raise ValueError("模型未初始化")
            
            # 生成嵌入向量
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # 转换为列表
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"文本嵌入失败: {e}")
            raise
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        try:
            if not self.model:
                raise ValueError("模型未初始化")
            
            # 批量生成嵌入向量
            embeddings = self.model.encode(documents, convert_to_tensor=False)
            
            # 转换为列表
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"文档批量嵌入失败: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "max_seq_length": self.model.max_seq_length if self.model else 0,
            "embedding_dimension": self.model.get_sentence_embedding_dimension() if self.model else 0,
            "device": str(self.model.device) if self.model else "unknown"
        }


class OpenAIEmbeddingClient:
    """OpenAI嵌入客户端（备用实现）"""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化OpenAI客户端"""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI嵌入客户端初始化成功: {self.model_name}")
            
        except Exception as e:
            logger.error(f"OpenAI嵌入客户端初始化失败: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """嵌入查询文本"""
        try:
            response = self.client.embeddings.create(
                input=query,
                model=self.model_name
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI查询嵌入失败: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """嵌入文本"""
        return self.embed_query(text)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        try:
            response = self.client.embeddings.create(
                input=documents,
                model=self.model_name
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI文档批量嵌入失败: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "provider": "openai",
            "api_key_set": bool(self.api_key)
        }


class HuggingFaceEmbeddingClient:
    """HuggingFace嵌入客户端（备用实现）"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name or settings.embedding_model_name
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化HuggingFace模型"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # 解析路径与离线参数
            model_path = settings.embedding_model_path or self.model_name
            cache_dir = settings.hf_cache_dir
            local_only = bool(settings.transformers_offline)

            logger.info(f"正在加载HuggingFace模型: {model_path} (local_files_only={local_only}, cache_dir={cache_dir})")
            
            # 加载模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                local_files_only=local_only,
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                local_files_only=local_only,
            )
            
            # 设置设备
            device_pref = (settings.embedding_device or "auto").lower()
            if device_pref == "cpu":
                device = "cpu"
            elif device_pref == "cuda":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            
            logger.info(f"HuggingFace模型加载成功: {self.model_name}, 设备: {device}")
            
        except Exception as e:
            logger.error(f"HuggingFace模型加载失败: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """嵌入查询文本"""
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("模型未初始化")
            
            # 分词和编码
            inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
            
            # 生成嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS]标记的嵌入
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"HuggingFace查询嵌入失败: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """嵌入文本"""
        return self.embed_query(text)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        try:
            embeddings = []
            for doc in documents:
                embedding = self.embed_text(doc)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"HuggingFace文档批量嵌入失败: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "provider": "huggingface",
            "max_length": self.tokenizer.model_max_length if self.tokenizer else 0
        }


# 默认使用SentenceTransformer
EmbeddingClient = SentenceTransformerEmbeddingClient
