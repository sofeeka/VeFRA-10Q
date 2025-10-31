import logging
from pydantic import BaseModel, Field
from typing import List, Any, Dict

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.conversions.common_types import ScoredPoint

from src.utils.config import DEFAULT_QDRANT_COLLECTION_NAME, DEFAULT_QDRANT_STORAGE_PATH, DEFAULT_QDRANT_DISTANCE_METRIC, DEFAULT_SEARCH_K
from src.retrieval.embedder import EmbeddingModel


class ChunkPayload(BaseModel):
    """
    A Pydantic model for Qdrant payload.
    """
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QdrantDatabase:  # TODO think of a better way of setting default qdrant collection name

    def __init__(self, embedding_model: EmbeddingModel, path: str = DEFAULT_QDRANT_STORAGE_PATH):
        logging.info("Initializing Qdrant Database...")
        self.embedding_model = embedding_model

        logging.info(f"Creating Qdrant client at {path}...")
        self.client = QdrantClient(path=path)

    def recreate_collection(self, collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME, vector_params: Any = None):
        """
        Recreate a Qdrant collection with specified vector parameters.
        """
        logging.info(f"Recreating collection '{collection_name}'...")

        if vector_params is None:
            vector_params = {
                "size": self.embedding_model.dim,
                "distance": DEFAULT_QDRANT_DISTANCE_METRIC
            }

        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vector_params
        )

    def add_chunks(self, chunks: List[str], collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME):
        embeddings = self.embedding_model.embed(chunks)
        # TODO add better payload and unique ID
        points = [PointStruct(id=i, vector=embeddings[i], payload={
            "text": chunk}) for i, chunk in enumerate(chunks)]

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    def get_search_results(self, query: str, collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME, limit: int = DEFAULT_SEARCH_K) -> list[ScoredPoint]:
        query_vector = self.embedding_model.embed(query)[0]

        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )

        return search_results

    def get_related_chunks(self, query: str, collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME, limit: int = DEFAULT_SEARCH_K) -> List[str]:
        results = self.get_search_results(
            query=query, collection_name=collection_name, limit=limit)
        chunks = [result.payload['text']
                  for result in results
                  if result.payload and 'text' in result.payload]
        return chunks
