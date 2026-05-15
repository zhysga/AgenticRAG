"""
依赖注入模块
解决循环导入问题
"""
from typing import Optional
from backend.adapters.storage_adapter import StorageAdapter
from backend.adapters.vector_store import VectorStoreAdapter, ChromaVectorStoreAdapter, FAISSVectorStoreAdapter
from backend.services.knowledge_base_service import KnowledgeBaseService
from backend.services.agent_service import AgentService
from backend.services.rag_service import RAGService
from backend.services.chat_service import ChatService
from backend.services.langgraph_service import LangGraphService
from backend.models.multi_llm_client import MultiLLMClient
from pathlib import Path
from backend.adapters.reranker import create_reranker_adapter
from backend.adapters.embedding_client import SentenceTransformerEmbeddingClient
from backend.config.settings import settings

# 全局服务实例
_storage_adapter: Optional[StorageAdapter] = None
_vector_store_adapter: Optional[VectorStoreAdapter] = None
_knowledge_base_service: Optional[KnowledgeBaseService] = None
_agent_service: Optional[AgentService] = None
_rag_service: Optional[RAGService] = None
_chat_service: Optional[ChatService] = None
_langgraph_service: Optional[LangGraphService] = None
_llm_client: Optional[MultiLLMClient] = None
_reranker_adapter = None
_embedding_client = None


def get_storage_adapter() -> StorageAdapter:
    """获取存储适配器实例"""
    global _storage_adapter
    if _storage_adapter is None:
        _storage_adapter = StorageAdapter()
    return _storage_adapter


def get_vector_store_adapter() -> VectorStoreAdapter:
    """获取向量存储适配器实例"""
    global _vector_store_adapter
    if _vector_store_adapter is None:
        _vector_store_adapter = ChromaVectorStoreAdapter()
    return _vector_store_adapter


def get_knowledge_base_service() -> KnowledgeBaseService:
    """获取知识库服务实例"""
    global _knowledge_base_service
    if _knowledge_base_service is None:
        storage = get_storage_adapter()
        vector_store = get_vector_store_adapter()
        _knowledge_base_service = KnowledgeBaseService(storage, vector_store)
    return _knowledge_base_service


def get_agent_service() -> AgentService:
    """获取智能体服务实例"""
    global _agent_service
    if _agent_service is None:
        storage = get_storage_adapter()
        _agent_service = AgentService(storage)
    return _agent_service


def get_embedding_client():
    """获取嵌入客户端实例"""
    global _embedding_client
    if _embedding_client is None:
        # 使用本地路径优先，如果没有则使用模型名称
        model_path = settings.embedding_model_path or settings.embedding_model_name
        _embedding_client = SentenceTransformerEmbeddingClient(
            model_name=model_path
        )
    return _embedding_client


def get_reranker_adapter():
    """获取重排序器适配器实例"""
    global _reranker_adapter
    if _reranker_adapter is None:
        _reranker_adapter = create_reranker_adapter(
            reranker_type="bge" if settings.reranker_enabled else "mock",
            model_name=settings.reranker_model_name
        )
    return _reranker_adapter


def get_rag_service() -> RAGService:
    """获取RAG服务实例"""
    global _rag_service
    if _rag_service is None:
        vector_store = get_vector_store_adapter()
        embedding_client = get_embedding_client()
        reranker = get_reranker_adapter()
        
        _rag_service = RAGService(
            vector_store=vector_store,
            embedding_client=embedding_client,
            reranker=reranker
        )
    return _rag_service


def get_chat_service() -> ChatService:
    """获取聊天服务实例"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


def get_llm_client() -> MultiLLMClient:
    """获取多模型LLM客户端实例"""
    global _llm_client
    if _llm_client is None:
        project_dir = str(Path(__file__).resolve().parent)
        _llm_client = MultiLLMClient(project_dir=project_dir)
    return _llm_client


def get_langgraph_service() -> LangGraphService:
    """获取LangGraph编排服务实例"""
    global _langgraph_service
    if _langgraph_service is None:
        rag_service = get_rag_service()
        agent_service = get_agent_service()
        llm_client = get_llm_client()
        _langgraph_service = LangGraphService(rag_service, agent_service, llm_client)
    return _langgraph_service