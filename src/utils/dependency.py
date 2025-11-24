from functools import cache, lru_cache

from loguru import logger
from openai import AsyncOpenAI, OpenAI
from qdrant_client import QdrantClient

from processing.chunking.docling_chunker import DoclingChunker
from processing.chunking.recursive_chunker import RecursiveChunker
from src.generation.async_generator import AsyncGenerator
from src.generation.generator import Generator
from src.generation.prompts import SYSTEM_PROMPT
from src.processing.document_parser import DocumentParser
from src.retrieval.database import UserKnowledgeBase
from src.retrieval.database_manager import QdrantCollectionManager
from src.retrieval.embedding.dense_embedding_model import DenseEmbeddingModel
from src.utils.api_key_manager import get_openai_api_key
from src.utils.config import (
    DEFAULT_QDRANT_COLLECTION_NAME,
    DEFAULT_QDRANT_STORAGE_PATH,
    DENSE_EMBEDDING_MODEL_NAME,
)


@lru_cache
def get_dense_embedding_model() -> DenseEmbeddingModel:
    logger.info("Caching singleton of embedding model...")
    return DenseEmbeddingModel(model_name=DENSE_EMBEDDING_MODEL_NAME)


@lru_cache
def get_qdrant_client() -> QdrantClient:
    logger.info("Caching singleton of Qdrant Client...")
    return QdrantClient(path=DEFAULT_QDRANT_STORAGE_PATH)


@cache
def get_user_knowledge_base(user_id: str) -> UserKnowledgeBase:
    logger.info(f"Caching singleton of User Knowledge Base for user {user_id}...")
    return UserKnowledgeBase(
        user_id=user_id,
        dense_embedding_model=get_dense_embedding_model(),
        client=get_qdrant_client(),
        collection_name=DEFAULT_QDRANT_COLLECTION_NAME,
    )


@lru_cache
def get_qdrant_collection_manager() -> QdrantCollectionManager:
    logger.info("Caching singleton of Qdrant Collection Manager...")
    return QdrantCollectionManager(
        client=get_qdrant_client(),
        collection_name=DEFAULT_QDRANT_COLLECTION_NAME,
    )


@lru_cache
def get_generator(model: str) -> Generator:
    logger.info("Caching singleton of Generator...")
    return Generator(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        client=get_openai_client(),
    )


@lru_cache
def get_async_generator(model: str) -> AsyncGenerator:
    logger.info("Caching singleton of Generator...")
    return AsyncGenerator(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        client=get_async_openai_client(),
    )


@lru_cache
def get_openai_client() -> OpenAI:
    logger.info("Caching singleton of OpenAI Client...")
    return OpenAI(api_key=get_openai_api_key())


@lru_cache
def get_async_openai_client() -> OpenAI:
    logger.info("Caching singleton of OpenAI Client...")
    return AsyncOpenAI(api_key=get_openai_api_key())


@lru_cache
def get_document_parser() -> DocumentParser:
    logger.info("Caching singleton of Document Parser...")
    return DocumentParser()


@lru_cache
def get_recursive_chunker() -> RecursiveChunker:
    logger.info("Caching singleton of Recursive Chunker...")
    return RecursiveChunker()


@lru_cache
def get_docling_chunker() -> DoclingChunker:
    logger.info("Caching singleton of Docling Chunker...")
    return DoclingChunker()
