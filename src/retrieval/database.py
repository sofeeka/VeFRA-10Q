import uuid
from loguru import logger
from pydantic import BaseModel, Field
from typing import List, Any, Dict

from qdrant_client import QdrantClient
import qdrant_client.http.models as types
from qdrant_client.http.models import PointStruct
from qdrant_client.conversions.common_types import ScoredPoint

from src.utils.config import DEFAULT_QDRANT_COLLECTION_NAME, DEFAULT_QDRANT_DISTANCE_METRIC, DEFAULT_SEARCH_K
from src.retrieval.embedder import EmbeddingModel


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
        logger.info(f"Initializing Knowledge Base for user {user_id}...")

        if not user_id:
            logger.error("No user ID provided to User Knowledge Base.")

        if not embedding_model:
            logger.error(
                "No embedding model provided to User Knowledge Base.")

        if not client:
            logger.error("No Qdrant client provided to User Knowledge Base.")

        self.user_id = user_id
        self.client = client
        self.embedding_model = embedding_model
        self.collection_name = collection_name

    def recreate_collection(self, vector_params: Any = None) -> bool:
        """
        Recreate a Qdrant collection with specified vector parameters.
        """
        logger.info(f"Recreating collection '{self.collection_name}'...")
        # TODO: move to admin or setup script, make params obligatory
        if vector_params is None:
            vector_params = {
                "size": self.embedding_model.dim,
                "distance": DEFAULT_QDRANT_DISTANCE_METRIC
            }

        result: bool = self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=vector_params
        )

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=types.PayloadSchemaType.KEYWORD,
                wait=True
            )
            logger.info(
                f"Created payload index on 'user_id' for collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to create payload index: {e}")
            return False

        return result

    def add_chunks(self, chunks: List[ChunkPayload]) -> bool:
        """
        Embed and add text chunks to the Qdrant collection.
        """

        texts_to_embed: List[str] = [chunk.text for chunk in chunks]
        embeddings: List[List[float]] = self.embedding_model.embed(
            texts_to_embed)

        if not embeddings or len(embeddings) != len(chunks):
            logger.error(
                "Embedding failed or returned mismatched number of embeddings.")
            return False

        points: List[PointStruct] = []
        for i, chunk in enumerate(chunks):
            payload = chunk.model_dump()

            payload['user_id'] = self.user_id
            point = PointStruct(
                id=chunk.id,
                vector=embeddings[i],
                payload=payload
            )
            points.append(point)

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

        user_filter = types.Filter(
            must=[
                types.FieldCondition(
                    key="user_id",
                    match=types.MatchValue(value=self.user_id)
                )
            ]
        )

        search_results: List[types.ScoredPoint] = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=user_filter,
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
