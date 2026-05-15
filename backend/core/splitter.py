"""Sentence Splitter Wrapper.

Uses the vendored EasyRAG implementation when available and falls back to
llama-index' splitter if initialisation fails.
"""
from __future__ import annotations

from typing import List
import logging

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class SentenceSplitter:
    """Wrapper around llama-index SentenceSplitter.
    
    Switched from EasyRAG implementation to llama-index native implementation
    after Qdrant/pipeline cleanup to avoid dependency on archived modules.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}

        from llama_index.core.node_parser import SentenceSplitter as _LISplitter
        
        self._impl = _LISplitter(
            chunk_size=cfg.get("chunk_size", 1024),
            chunk_overlap=cfg.get("chunk_overlap", 200),
            paragraph_separator=cfg.get("paragraph_separator", "\n\n\n"),
        )

    # ------------------------------------------------------------------
    def get_nodes_from_documents(self, docs: List[Document]):
        """Return list of nodes after splitting documents."""
        return self._impl.get_nodes_from_documents(docs)

