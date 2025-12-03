from functools import lru_cache

from loguru import logger
from openai import AsyncOpenAI, OpenAI
from qdrant_client import AsyncQdrantClient

from ..generation.async_generator import AsyncGenerator
from ..generation.generator import Generator
from ..generation.prompts import SYSTEM_PROMPT
from ..processing.chunking.docling_chunker import DoclingChunker
from ..processing.chunking.recursive_chunker import RecursiveChunker
from ..processing.document_parser import DocumentParser
from ..retrieval.database import UserKnowledgeBase
from ..retrieval.database_manager import QdrantCollectionManager
from ..retrieval.embedding.dense_embedding_model import DenseEmbeddingModel
from ..retrieval.embedding.sparse_embedding_model import SparseEmbeddingModel
from ..retrieval.reranker import FinancialReranker
from .api_key_manager import get_openai_api_key
from .config import (
    DEFAULT_QDRANT_COLLECTION_NAME,
    DEFAULT_QDRANT_STORAGE_PATH,
    DENSE_EMBEDDING_MODEL_NAME,
    RERANKING_MODEL,
    SPARSE_EMBEDDING_MODEL_NAME,
)


# EXPENSIVE: model loading
@lru_cache
def get_dense_embedding_model() -> DenseEmbeddingModel:
    logger.info("Initializing dense embedding model (cached)...")
    return DenseEmbeddingModel(model_name=DENSE_EMBEDDING_MODEL_NAME)


# EXPENSIVE: model loading
@lru_cache
def get_sparse_embedding_model() -> SparseEmbeddingModel:
    logger.info("Initializing sparse embedding model (cached)...")
    return SparseEmbeddingModel(model_name=SPARSE_EMBEDDING_MODEL_NAME)


# Database connection, should be a singleton
@lru_cache
def get_async_qdrant_client() -> AsyncQdrantClient:
    logger.info("Initializing Qdrant client (cached)...")
    return AsyncQdrantClient(path=DEFAULT_QDRANT_STORAGE_PATH)


# EXPENSIVE: DocumentConverter loads Docling models
@lru_cache
def get_document_parser() -> DocumentParser:
    logger.info("Initializing document parser (cached)...")
    return DocumentParser()


# EXPENSIVE: model loading
@lru_cache
def get_reranking_model() -> FinancialReranker:
    logger.info("Initializing reranking model (cached)...")
    return FinancialReranker(model_name=RERANKING_MODEL)


def get_user_knowledge_base(user_id: str) -> UserKnowledgeBase:
    return UserKnowledgeBase(
        user_id=user_id,
        dense_embedding_model=get_dense_embedding_model(),
        sparse_embedding_model=get_sparse_embedding_model(),
        client=get_async_qdrant_client(),
        collection_name=DEFAULT_QDRANT_COLLECTION_NAME,
    )


def get_qdrant_collection_manager() -> QdrantCollectionManager:
    return QdrantCollectionManager(
        client=get_async_qdrant_client(),
        collection_name=DEFAULT_QDRANT_COLLECTION_NAME,
    )


def get_generator(
    model: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> Generator:
    return Generator(
        model=model,
        system_prompt=system_prompt,
        client=get_openai_client(),
    )


def get_async_generator(
    model: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> AsyncGenerator:
    return AsyncGenerator(
        model=model,
        system_prompt=system_prompt,
        client=get_async_openai_client(),
    )


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


def get_async_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_openai_api_key())


def get_recursive_chunker() -> RecursiveChunker:
    return RecursiveChunker()


def get_docling_chunker() -> DoclingChunker:
    return DoclingChunker()
