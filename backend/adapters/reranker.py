"""
重排序器适配器
"""
import logging
from typing import List, Optional
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from backend.utils.logger import get_logger
from backend.config.settings import settings

logger = get_logger(__name__)


class BGERerankerAdapter:
    """BGE重排序器适配器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = None):
        self.model_name = model_name or settings.reranker_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化重排序模型"""
        try:
            # 路径与离线参数
            model_path = settings.reranker_model_path or self.model_name
            cache_dir = settings.hf_cache_dir
            local_only = bool(settings.transformers_offline)

            logger.info(f"正在加载重排序模型: {model_path}, 设备: {self.device} (local_files_only={local_only}, cache_dir={cache_dir})")
            
            # 加载模型和分词器（本地优先/离线支持）
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                local_files_only=local_only,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                local_files_only=local_only,
            )
            self.model = self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            logger.info(f"重排序模型加载成功: {self.model_name}")
            
        except Exception as e:
            logger.error(f"重排序模型加载失败: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            重排序分数列表
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("重排序模型未初始化")
            
            if not documents:
                return []
            
            logger.info(f"开始重排序: query='{query[:50]}...', docs_count={len(documents)}")
            
            scores = []
            
            # 对每个文档计算重排序分数
            for doc in documents:
                # 编码查询和文档对
                inputs = self.tokenizer(
                    query,
                    doc,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # 获取模型预测
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    score = outputs.logits.item()
                    scores.append(score)
            
            logger.info(f"重排序完成，返回{len(scores)}个分数")
            return scores
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            raise
    
    def rerank_batch(self, query: str, documents: List[str], batch_size: int = 8) -> List[float]:
        """
        批量重排序（优化性能）
        
        Args:
            query: 查询文本
            documents: 文档列表
            batch_size: 批次大小
            
        Returns:
            重排序分数列表
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("重排序模型未初始化")
            
            if not documents:
                return []
            
            logger.info(f"开始批量重排序: query='{query[:50]}...', docs_count={len(documents)}, batch_size={batch_size}")
            
            scores = []
            
            # 批量处理文档
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                
                # 准备批次输入
                batch_inputs = []
                for doc in batch_docs:
                    inputs = self.tokenizer(
                        query,
                        doc,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    batch_inputs.append(inputs)
                
                # 合并批次输入
                batch_input = self.tokenizer.pad(
                    batch_inputs,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # 批量预测
                with torch.no_grad():
                    outputs = self.model(**batch_input)
                    batch_scores = outputs.logits.squeeze().cpu().numpy()
                    
                    # 处理单个文档的情况
                    if len(batch_docs) == 1:
                        batch_scores = [batch_scores.item()]
                    else:
                        batch_scores = batch_scores.tolist()
                    
                    scores.extend(batch_scores)
            
            logger.info(f"批量重排序完成，返回{len(scores)}个分数")
            return scores
            
        except Exception as e:
            logger.error(f"批量重排序失败: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.model is not None
        }


class MockRerankerAdapter:
    """模拟重排序器适配器（用于测试或备用）"""
    
    def __init__(self):
        logger.info("初始化模拟重排序器")
    
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """模拟重排序（返回随机分数）"""
        try:
            if not documents:
                return []
            
            # 简单的模拟：基于查询和文档的长度计算分数
            scores = []
            query_len = len(query)
            
            for doc in documents:
                # 基于文档长度和查询相关性的简单分数
                doc_len = len(doc)
                score = min(1.0, query_len / (doc_len + 1)) + np.random.uniform(0, 0.3)
                scores.append(min(1.0, score))  # 确保分数在0-1范围内
            
            return scores
            
        except Exception as e:
            logger.error(f"模拟重排序失败: {e}")
            return [0.0] * len(documents)  # 返回默认分数
    
    def rerank_batch(self, query: str, documents: List[str], batch_size: int = 8) -> List[float]:
        """批量重排序"""
        return self.rerank(query, documents)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": "MockReranker",
            "device": "cpu",
            "model_loaded": True,
            "type": "mock"
        }


# 重排序器工厂函数
def create_reranker_adapter(
    reranker_type: str = "bge",
    model_name: str = "BAAI/bge-reranker-base",
    device: Optional[str] = None,
    **kwargs
) -> BGERerankerAdapter:
    """
    创建重排序器适配器
    
    Args:
        reranker_type: 重排序器类型 ("bge", "mock")
        model_name: 模型名称
        device: 设备类型
        **kwargs: 其他参数
        
    Returns:
        重排序器适配器实例
    """
    try:
        if reranker_type == "bge":
            return BGERerankerAdapter(model_name=model_name, device=device)
        elif reranker_type == "mock":
            return MockRerankerAdapter()
        else:
            logger.warning(f"未知的重排序器类型: {reranker_type}，使用默认BGE重排序器")
            return BGERerankerAdapter(model_name=model_name, device=device)
            
    except Exception as e:
        logger.error(f"创建重排序器适配器失败: {e}，回退到模拟重排序器")
        return MockRerankerAdapter()