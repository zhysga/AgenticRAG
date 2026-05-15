"""Reranker Wrapper

Leverages EasyRAG‘s SentenceTransformerRerank 或 LLMRerank, 通过配置切换启用。
优先使用 settings 中的本地模型路径，避免联网下载。
"""
from __future__ import annotations

from typing import List
import os

from llama_index.core.schema import NodeWithScore, QueryBundle


class RerankerWrapper:
    """统一重排序接口。

    cfg 示例::
        {
          "enabled": True,
          "model": "BAAI/bge-reranker-v2-minicpm-layerwise",
          "top_k": 6,
          "device": "auto",
          "type": "llm"   # llm | sbert | none
        }
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}

        # 从全局 settings 读取默认配置与本地模型路径
        try:
            from backend.config.settings import get_settings
            _settings = get_settings()
        except Exception:
            _settings = None

        # 启用/禁用优先级: cfg > settings > True
        enabled = self.cfg.get("enabled")
        if enabled is None and _settings is not None:
            enabled = getattr(_settings, "reranker_enabled", True)
        if enabled is None:
            enabled = True

        if not enabled:
            self._impl = None
            return


        from backend.core.easyrag.custom.rerankers import LLMRerank, SentenceTransformerRerank

        # 模型名称优先级: cfg.model > settings.reranker_model_path(存在则优先) > settings.reranker_model_name > 默认 cross-encoder
        model_name = self.cfg.get("model")
        if model_name is None and _settings is not None:
            local_path = getattr(_settings, "reranker_model_path", None)
            if local_path and os.path.isdir(local_path):
                model_name = local_path
            else:
                model_name = getattr(_settings, "reranker_model_name", None)
        if model_name is None:
            model_name = "cross-encoder/stsb-distilroberta-base"

        top_k = self.cfg.get("top_k", 6)
        device = self.cfg.get("device", None)
        # 类型优先级: cfg.type > 默认 sbert
        r_type = self.cfg.get("type", "sbert")

        if r_type == "llm":
            self._impl = LLMRerank(top_n=top_k, model=model_name, device=device)
        else:
            self._impl = SentenceTransformerRerank(top_n=top_k, model=model_name, device=device)

    # ---------------------------------------------------------------------
    def postprocess_nodes(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        if not self._impl or not nodes:
            return nodes  # rerank disabled or empty
        # EasyRAG expects QueryBundle
        qb = QueryBundle(query_str=query)
        return self._impl.postprocess_nodes(nodes, query_bundle=qb)

