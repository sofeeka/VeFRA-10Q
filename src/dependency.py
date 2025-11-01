import os
import logging
from functools import lru_cache
from qdrant_client import QdrantClient

from src.generation.generator import Generator
from src.generation.prompts import SYSTEM_PROMPT
from src.processing.document_parser import DocumentParser
from src.processing.chuncker import DocumentChunker
from src.retrieval.embedder import EmbeddingModel, FastEmbedModel
from src.retrieval.database import QdrantDatabase
from src.utils.config import *

logger = logging.getLoger(__name__)


@lru_cache()
def get_embedder() -> EmbeddingModel:
    logger.info("Initialising embedding model...")
    return FastEmbedModel(model_name=FAST_EMBED_DEFAULT_EMBEDDING_MODEL)


@lru_cache()
def get_qdrant_client() -> QdrantClient:
    logger.info("Initializing Qdrant Client...")
    return QdrantClient(path=DEFAULT_QDRANT_STORAGE_PATH)


@lru_cache()
def get_qdrant_database() -> QdrantDatabase:
    logger.info("Initialising Qdrant Database...")
    return QdrantDatabase(embedding_model=get_embedder(), client=get_qdrant_client())


@lru_cache()
def get_generator() -> Generator:
    try:
        logger.info("Initialising Generator...")
        api_key = os.environ["OPENAI_API_KEY"]
        return Generator(api_key=api_key, model=TESTING_OPENAI_MODEL, system_prompt=SYSTEM_PROMPT)
    except KeyError:
        raise ValueError("OPENAI_API_KEY environment variable not set.")


@lru_cache()
def get_document_parser() -> DocumentParser:
    logger.info("Initialising Document Parser...")
    return DocumentParser()


@lru_cache()
def get_document_chunker() -> DocumentChunker:
    logger.info("Initialising Document Chunker...")
    return DocumentChunker(strategy=DEFAULT_CHUNKING_STRATEGY, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP, )
