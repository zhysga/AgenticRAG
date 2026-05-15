import logging
from typing import List, Optional, Callable, cast

import bm25s
from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.schema import NodeWithScore, BaseNode, IndexNode
from llama_index.core.storage.docstore import BaseDocumentStore
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def get_node_content(node: BaseNode, embed_type: int = 0) -> str:
    """Extract content from node based on embed type.
    
    Simplified version after pipeline cleanup. Always returns node.get_content()
    regardless of embed_type since we removed hierarchical/complex embedding logic.
    """
    return node.get_content()


def tokenize_and_remove_stopwords(tokenizer, text, stopwords):
    """Tokenize text and remove stopwords.
    
    Args:
        tokenizer: A callable that tokenizes text (e.g., jieba.cut or tokenizer.cut method)
        text: Input text to tokenize
        stopwords: List of stopwords to remove
    """
    words = tokenizer(text)  # tokenizer is already a callable (e.g., jieba.Tokenizer().cut)
    filtered_words = [word for word in words
                      if word not in stopwords and word != ' ']
    return filtered_words


# using jieba to split sentence and remove meaningless words
class BM25Retriever(BaseRetriever):
    def __init__(
            self,
            nodes: List[BaseNode],
            tokenizer: Optional[Callable[[str], List[str]]],
            similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
            callback_manager: Optional[CallbackManager] = None,
            objects: Optional[List[IndexNode]] = None,
            object_map: Optional[dict] = None,
            verbose: bool = False,
            stopwords: List[str] = [""],
            embed_type: int = 0,
            bm25_type: int = 0,
    ) -> None:
        self._nodes = nodes
        self._tokenizer = tokenizer
        self._similarity_top_k = similarity_top_k
        self.embed_type = embed_type
        self._corpus = [tokenize_and_remove_stopwords(
            self._tokenizer, get_node_content(node, self.embed_type), stopwords=stopwords)
            for node in self._nodes]
        # self._corpus = [self._tokenizer(node.get_content()) for node in self._nodes]
        self.bm25_type = bm25_type
        self.k1 = 1.5
        self.b = 0.75
        self.epsilon = 0.25
        if self.bm25_type == 1:
            self.bm25 = bm25s.BM25(
                k1=self.k1,
                b=self.b,
            )
            self.bm25.index(self._corpus)
        else:
            self.bm25 = BM25Okapi(
                self._corpus,
                k1=self.k1,
                b=self.b,
                epsilon=self.epsilon,
            )
        self.filter_dict = None
        self.stopwords = stopwords
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    def get_scores(self, query, docs=None):
        if docs is None:
            bm25 = self.bm25
        else:
            corpus = [tokenize_and_remove_stopwords(
                self._tokenizer, doc, stopwords=self.stopwords)
                for doc in docs]
            if self.bm25_type == 1:
                bm25 = bm25s.BM25(
                    k1=self.k1,
                    b=self.b,
                )
                bm25.index(corpus)
            else:
                bm25 = BM25Okapi(
                    corpus,
                    k1=self.k1,
                    b=self.b,
                    epsilon=self.epsilon,
                )
        tokenized_query = tokenize_and_remove_stopwords(self._tokenizer, query,
                                                        stopwords=self.stopwords)
        scores = bm25.get_scores(tokenized_query)
        return scores

    @classmethod
    def from_defaults(
            cls,
            index: Optional[VectorStoreIndex] = None,
            nodes: Optional[List[BaseNode]] = None,
            docstore: Optional[BaseDocumentStore] = None,
            tokenizer: Optional[Callable[[str], List[str]]] = None,
            similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
            verbose: bool = False,
            stopwords: List[str] = [""],
            embed_type: int = 0,
            bm25_type: int = 0,  # 0-->official bm25-Okapi 1-->bm25s
    ) -> "BM25Retriever":
        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert (
                nodes is not None
        ), "Please pass exactly one of index, nodes, or docstore."

        tokenizer = tokenizer
        return cls(
            nodes=nodes,
            tokenizer=tokenizer,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
            stopwords=stopwords,
            embed_type=embed_type,
            bm25_type=bm25_type,
        )

    def filter(self, scores):
        top_n = scores.argsort()[::-1]
        nodes: List[NodeWithScore] = []
        for ix in top_n:
            if scores[ix] <= 0:
                break
            flag = True
            if self.filter_dict is not None:
                for key, value in self.filter_dict.items():
                    if self._nodes[ix].metadata[key] != value:
                        flag = False
                        break
            if flag:
                nodes.append(NodeWithScore(node=self._nodes[ix], score=float(scores[ix])))
            if len(nodes) == self._similarity_top_k:
                break

        # add nodes sort in BM25Retriever
        nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if query_bundle.custom_embedding_strs or query_bundle.embedding:
            logger.warning("BM25Retriever does not support embeddings, skipping...")

        query = query_bundle.query_str
        scores = self.get_scores(query)
        nodes = self.filter(scores)

        return nodes


class HybridRetriever(BaseRetriever):
    def __init__(
            self,
            dense_retriever: Optional[BaseRetriever],
            sparse_retriever: Optional[BM25Retriever],
            retrieval_type=1,
            topk=256,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.retrieval_type = retrieval_type  # 1:dense only 2:sparse only 3:hybrid
        self.filters = None
        self.filter_dict = None
        self.topk = topk
        super().__init__()

    @classmethod
    def fusion(self, list_of_list_ranks_system, topk=256):
        all_nodes = []

        node_ids = set()
        for nodes in list_of_list_ranks_system:
            for node in nodes:
                content = node.get_content()
                if content not in node_ids:
                    all_nodes.append(node)
                    node_ids.add(content)
        all_nodes = sorted(all_nodes, key=lambda node: node.score, reverse=True)
        topk = min(len(all_nodes), topk)
        # print("simple fusion后数量:", topk)
        return all_nodes[:topk]

    # 倒数排序融合
    @classmethod
    def reciprocal_rank_fusion(self, list_of_list_ranks_system, K=60, topk=256):
        from collections import defaultdict
        rrf_map = defaultdict(float)
        text_to_node = {}
        for rank_list in list_of_list_ranks_system:
            for rank, item in enumerate(rank_list, 1):
                content = item.get_content()
                text_to_node[content] = item
                rrf_map[content] += 1 / (rank + K)
        sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)

        reranked_nodes: List[NodeWithScore] = []
        for text, score in sorted_items:
            reranked_nodes.append(text_to_node[text])
            reranked_nodes[-1].score = score
        topk = min(topk, len(reranked_nodes))
        # print("rrf fusion后数量:", topk)
        return reranked_nodes[:topk]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        sparse_nodes: List[NodeWithScore] = []
        dense_nodes: List[NodeWithScore] = []

        if self.sparse_retriever and self.retrieval_type != 1:
            self.sparse_retriever.filter_dict = self.filter_dict
            sparse_nodes = await self.sparse_retriever.aretrieve(query_bundle)
            if self.retrieval_type == 2 or not self.dense_retriever:
                return sparse_nodes

        if self.dense_retriever and self.retrieval_type != 2:
            if hasattr(self.dense_retriever, "filters"):
                setattr(self.dense_retriever, "filters", self.filters)
            dense_nodes = await self.dense_retriever.aretrieve(query_bundle)
            if self.retrieval_type == 1:
                return dense_nodes

        if not sparse_nodes:
            return dense_nodes
        if not dense_nodes:
            return sparse_nodes

        all_nodes = self.reciprocal_rank_fusion([sparse_nodes, dense_nodes], topk=self.topk)
        return all_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 不维护 占位
        sparse_nodes: List[NodeWithScore] = []
        dense_nodes: List[NodeWithScore] = []

        if self.sparse_retriever and self.retrieval_type != 1:
            sparse_nodes = self.sparse_retriever.retrieve(query_bundle)
            if self.retrieval_type == 2 or not self.dense_retriever:
                return sparse_nodes

        if self.dense_retriever and self.retrieval_type != 2:
            dense_nodes = self.dense_retriever.retrieve(query_bundle)
            if self.retrieval_type == 1:
                return dense_nodes

        if not sparse_nodes:
            return dense_nodes
        if not dense_nodes:
            return sparse_nodes

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in sparse_nodes + dense_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes
