"""EmbeddingAdapter bridges existing EmbeddingClient with LlamaIndex BaseEmbedding API."""
from __future__ import annotations

from typing import List, Any

from llama_index.core.base.embeddings.base import BaseEmbedding


class EmbeddingAdapter(BaseEmbedding):
    """Wraps an existing embedding_client (with embed_query / embed_text) to provide
    LlamaIndex-compatible methods get_query_embedding / get_text_embedding.
    """

    client: Any

    def __init__(self, client):
        super().__init__(client=client)

    # ------------------------------------------------------------------
    def get_query_embedding(self, query: str) -> List[float]:  # type: ignore[override]
        return self.client.embed_query(query)  # noqa: B023

    def get_text_embedding(self, text: str) -> List[float]:  # type: ignore[override]
        return self.client.embed_text(text)  # noqa: B023

    # ------------------------------------------------------------------
    # optional batch methods for better perf ------------------------------------------------
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        return [self.get_text_embedding(t) for t in texts]

    def get_query_embeddings(self, queries: List[str]) -> List[List[float]]:  # type: ignore[override]
        return [self.get_query_embedding(q) for q in queries]

    def _get_query_embedding(self, query: str) -> List[float]:  # type: ignore[override]
        """Private helper to comply with BaseEmbedding's expected interface."""
        return self.get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:  # type: ignore[override]
        """Private helper to comply with BaseEmbedding's expected interface."""
        return self.get_text_embedding(text)

    # Async versions ----------------------------------------------------------
    async def _aget_query_embedding(self, query: str) -> List[float]:  # type: ignore[override]
        """Async helper matching BaseEmbedding contract."""
        # Delegates to sync implementation for simplicity
        return self.get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:  # type: ignore[override]
        """Async helper matching BaseEmbedding contract."""
        return self.get_text_embedding(text)
