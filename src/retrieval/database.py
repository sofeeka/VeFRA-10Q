import uuid
import logging
from pydantic import BaseModel, Field
from typing import List, Any, Dict

from qdrant_client import QdrantClient
import qdrant_client.http.models as types
from qdrant_client.http.models import PointStruct
from qdrant_client.conversions.common_types import ScoredPoint

from src.utils.config import DEFAULT_QDRANT_COLLECTION_NAME, DEFAULT_QDRANT_DISTANCE_METRIC, DEFAULT_SEARCH_K
from src.retrieval.embedder import EmbeddingModel

logger = logging.getLogger(__name__)


class ChunkPayload(BaseModel):
    """
    A Pydantic model for Qdrant payload.
    """
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class UserKnowledgeBase:

    def __init__(
        self,
        user_id: str,
        client: QdrantClient,
        embedding_model: EmbeddingModel,
        collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME
    ):
        logging.info(f"Initializing Knowledge Base for user {user_id}...")

        if not user_id:
            logging.error("No user ID provided to User Knowledge Base.")

        if not embedding_model:
            logging.error(
                "No embedding model provided to User Knowledge Base.")

        if not client:
            logging.error("No Qdrant client provided to User Knowledge Base.")

        self.user_id = user_id
        self.client = client
        self.embedding_model = embedding_model
        self.collection_name = collection_name

    def recreate_collection(self, vector_params: Any = None) -> bool:
        """
        Recreate a Qdrant collection with specified vector parameters.
        """
        logging.info(f"Recreating collection '{self.collection_name}'...")

        if vector_params is None:
            vector_params = {
                "size": self.embedding_model.dim,
                "distance": DEFAULT_QDRANT_DISTANCE_METRIC
            }

        result: bool = self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=vector_params
        )

        return result

    def add_chunks(self, chunks: List[ChunkPayload]) -> bool:
        """
        Embed and add text chunks to the Qdrant collection.
        """

        texts_to_embed = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed(texts_to_embed)

        if not embeddings or len(embeddings) != len(chunks):
            logger.error(
                "Embedding failed or returned mismatched number of embeddings.")
            return False

        points = [
            PointStruct(
                id=chunk.id,
                vector=embeddings[i],
                payload=chunk.model_dump()
            ) for i, chunk in enumerate(chunks)
        ]

        if not points:
            logger.error(
                "Embedded data successfully, but found no points to upsert.")
            return False

        result: types.UpdateResult = self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        status: types.UpdateStatus = result.status

        if status == types.UpdateStatus.COMPLETED:
            logger.info(
                f"Upsert successful (Operation ID: {result.operation_id})")
            return True

        elif status == types.UpdateStatus.ACKNOWLEDGED:
            logger.warning(
                f"Upsert acknowledged, but processing in background (Operation ID: {result.operation_id})")
            return True
        else:
            logger.error(f"Upsert failed with status: {result.status}")
            return False

    def get_search_results(self, query: str, limit: int = DEFAULT_SEARCH_K) -> list[ScoredPoint]:
        """
        Queries the Qdrant collection for similar chunks based on the input query.
        """

        query_vector: List[float] = self.embedding_model.embed(query)[0]

        search_results: List[types.ScoredPoint] = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )

        return search_results

    def get_related_chunks(self, query: str, limit: int = DEFAULT_SEARCH_K) -> List[str]:
        """
        Retrieves text chunks related to the input query.
        """

        results: list[ScoredPoint] = self.get_search_results(
            query=query, limit=limit)

        if not results:
            return []

        chunks: List[str] = [result.payload['text']
                             for result in results
                             if result.payload and 'text' in result.payload]

        return chunks
