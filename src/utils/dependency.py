from functools import lru_cache

from loguru import logger
from openai import OpenAI
from qdrant_client import QdrantClient

from src.generation.generator import Generator
from src.generation.prompts import SYSTEM_PROMPT
from src.processing.chuncker import DocumentChunker
from src.processing.document_parser import DocumentParser
from src.retrieval.database import UserKnowledgeBase
from src.retrieval.embedder import EmbeddingModel, FastEmbedModel
from src.utils.api_key_manager import get_openai_api_key
from src.utils.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_QDRANT_STORAGE_PATH,
    FAST_EMBED_DEFAULT_EMBEDDING_MODEL,
    TESTING_OPENAI_MODEL,
)


@lru_cache()
def get_embedder() -> EmbeddingModel:
    logger.info("Caching singleton of embedding model...")
    return FastEmbedModel(model_name=FAST_EMBED_DEFAULT_EMBEDDING_MODEL)


@lru_cache()
def get_qdrant_client() -> QdrantClient:
    logger.info("Caching singleton of Qdrant Client...")
    return QdrantClient(path=DEFAULT_QDRANT_STORAGE_PATH)


@lru_cache()
def get_user_knowledge_base(user_id: str) -> UserKnowledgeBase:
    logger.info(f"Caching singleton of User Knowledge Base for user {user_id}...")
    return UserKnowledgeBase(
        user_id=user_id, embedding_model=get_embedder(), client=get_qdrant_client()
    )


@lru_cache()
def get_generator() -> Generator:
    logger.info("Caching singleton of Generator...")
    return Generator(
        model=TESTING_OPENAI_MODEL,
        system_prompt=SYSTEM_PROMPT,
        client=get_openai_client(),
    )


@lru_cache()
def get_openai_client() -> OpenAI:
    logger.info("Caching singleton of OpenAI Client...")
    return OpenAI(api_key=get_openai_api_key())


@lru_cache()
def get_document_parser() -> DocumentParser:
    logger.info("Caching singleton of Document Parser...")
    return DocumentParser()


@lru_cache()
def get_document_chunker() -> DocumentChunker:
    logger.info("Caching singleton of Document Chunker...")
    return DocumentChunker(
        strategy=DEFAULT_CHUNKING_STRATEGY,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_CHUNK_OVERLAP,
    )
