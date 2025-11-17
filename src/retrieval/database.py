import uuid
from typing import Any

import qdrant_client.http.models as types
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.http.models import PointStruct

from src.retrieval.embedding.dense_embedding_model import DenseEmbeddingModel
from src.utils.config import DEFAULT_SEARCH_K, DENSE_DEFAULT
from src.utils.exceptions import DatabaseError, DataInsertionError, SearchError


class ChunkPayload(BaseModel):
    """
    A Pydantic model for Qdrant payload.
    """

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class UserKnowledgeBase:
    def __init__(
        self,
        user_id: str,
        client: QdrantClient,
        dense_embedding_model: DenseEmbeddingModel,
        collection_name: str,
    ):
        logger.info(f"Initializing Knowledge Base for user {user_id}...")

        if not user_id:
            logger.error("No user ID provided to User Knowledge Base.")

        if not dense_embedding_model:
            logger.error("No embedding model provided to User Knowledge Base.")

        if not client:
            logger.error("No Qdrant client provided to User Knowledge Base.")

        self.user_id = user_id
        self.client = client
        self.dense_embedding_model = dense_embedding_model
        self.collection_name = collection_name

    def add_chunks(self, chunks: list[ChunkPayload]) -> bool:
        """
        Embed and add text chunks to the Qdrant collection.
        """

        texts_to_embed: list[str] = [chunk.text for chunk in chunks]
        dense_embeddings: list[list[float]] = self.dense_embedding_model.embed(
            texts_to_embed
        )

        if not dense_embeddings or len(dense_embeddings) != len(chunks):
            logger.error(
                "Embedding failed or returned mismatched number of embeddings."
            )
            raise DatabaseError(
                "Embedding failed or returned mismatched number of embeddings."
            )

        points: list[PointStruct] = []
        for i, chunk in enumerate(chunks):
            payload = chunk.model_dump()

            payload["user_id"] = self.user_id
            point = PointStruct(
                id=chunk.id,
                vector={
                    DENSE_DEFAULT: dense_embeddings[i],
                },
                payload=payload,
            )
            points.append(point)

        if not points:
            logger.error("Embedded data successfully, but found no points to upsert.")
            return DatabaseError(
                "Embedded data successfully, but found no points to upsert."
            )

        result: types.UpdateResult = self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        status: types.UpdateStatus = result.status

        if status == types.UpdateStatus.COMPLETED:
            logger.info(f"Upsert successful (Operation ID: {result.operation_id})")

        elif status == types.UpdateStatus.ACKNOWLEDGED:
            logger.warning(
                f"Upsert acknowledged, but processing in background (Operation ID: {result.operation_id})"
            )

        else:
            logger.error(f"Upsert failed with status: {result.status}")
            raise DataInsertionError(f"Upsert failed with status: {result.status}")

    def get_search_results(
        self, query: str, limit: int = DEFAULT_SEARCH_K
    ) -> list[ScoredPoint]:
        """
        Queries the Qdrant collection for similar chunks based on the input query.
        """

        query_vector: list[float] = self.dense_embedding_model.embed(query)[0]

        user_filter = types.Filter(
            must=[
                types.FieldCondition(
                    key="user_id", match=types.MatchValue(value=self.user_id)
                )
            ]
        )

        search_results: list[types.ScoredPoint] = self.client.search(
            collection_name=self.collection_name,
            query_vector=(DENSE_DEFAULT, query_vector),
            query_filter=user_filter,
            limit=limit,
            with_payload=True,
        )

        if not search_results:
            raise SearchError(
                f"Could not find relevant information for query {query}, for user {self.user_id}"
            )

        return search_results

    def get_related_chunks(
        self, query: str, limit: int = DEFAULT_SEARCH_K
    ) -> list[str]:
        """
        Retrieves text chunks related to the input query.
        """

        results: list[ScoredPoint] = self.get_search_results(query=query, limit=limit)

        if not results:
            return []

        chunks: list[str] = [
            result.payload["text"]
            for result in results
            if result.payload and "text" in result.payload
        ]

        return chunks
