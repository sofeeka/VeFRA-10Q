import asyncio

import qdrant_client.http.models as types
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.http.models import PointStruct

from src.data_models.retrieval import ChunkPayload, DocumentMetadata
from src.retrieval.embedding.dense_embedding_model import DenseEmbeddingModel
from src.retrieval.embedding.sparse_embedding_model import SparseEmbeddingModel
from src.utils.config import DEFAULT_SEARCH_K, DENSE_DEFAULT, SPARSE_DEFAULT
from src.utils.exceptions import VeFRA_DatabaseError, VeFRA_DataInsertionError


class UserKnowledgeBase:
    def __init__(
        self,
        user_id: str,
        client: AsyncQdrantClient,
        dense_embedding_model: DenseEmbeddingModel,
        sparse_embedding_model: SparseEmbeddingModel,
        collection_name: str,
    ):
        if not all([user_id, dense_embedding_model, client]):
            logger.error(
                "Missing one or more required arguments for UserKnowledgeBase.",
                user_id_is_present=bool(user_id),
                embedding_model_is_present=bool(dense_embedding_model),
                sparse_embedding_model_is_present=bool(sparse_embedding_model),
                client_is_present=bool(client),
            )
            raise ValueError("user_id, client, and dense_embedding_model are required.")

        self.user_id = user_id
        self.client = client
        self.dense_embedding_model = dense_embedding_model
        self.sparse_embedding_model = sparse_embedding_model
        self.collection_name = collection_name
        logger.info(
            "Initialized Knowledge Base for user.",
            user_id=self.user_id,
            collection_name=self.collection_name,
        )

    async def add_chunks(self, chunks: list[ChunkPayload]):
        """
        Embed and add text chunks to the Qdrant collection.
        """
        if not chunks:
            logger.warning(
                "add_chunks called with an empty list of chunks.", user_id=self.user_id
            )
            return

        texts_to_embed = [chunk.text for chunk in chunks]
        dense_embeddings = await asyncio.to_thread(
            self.dense_embedding_model.embed, texts_to_embed
        )
        sparse_embeddings = await asyncio.to_thread(
            self.sparse_embedding_model.embed, texts_to_embed
        )

        if not dense_embeddings or len(dense_embeddings) != len(chunks):
            logger.error(
                "Dense embedding failed or returned mismatched number of embeddings.",
                expected_count=len(chunks),
                actual_count=len(dense_embeddings) if dense_embeddings else 0,
            )
            raise VeFRA_DatabaseError(
                "Dense embedding failed or returned mismatched number of embeddings."
            )

        if not sparse_embeddings or len(sparse_embeddings) != len(chunks):
            logger.error(
                "Sparse embedding failed or returned mismatched number of embeddings.",
                expected_count=len(chunks),
                actual_count=len(sparse_embeddings) if sparse_embeddings else 0,
            )
            raise VeFRA_DatabaseError(
                "Sparse embedding failed or returned mismatched number of embeddings."
            )

        points = [
            PointStruct(
                id=chunk.id,
                vector={
                    DENSE_DEFAULT: dense_embeddings[i],
                    SPARSE_DEFAULT: sparse_embeddings[i],
                },
                payload={**chunk.model_dump(), "user_id": self.user_id},
            )
            for i, chunk in enumerate(chunks)
        ]

        result = await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        if result.status == types.UpdateStatus.COMPLETED:
            logger.success(
                "Upsert successful.",
                operation_id=result.operation_id,
                point_count=len(points),
            )
        else:
            logger.error(
                "Upsert failed.",
                status=result.status,
                operation_id=result.operation_id,
            )
            raise VeFRA_DataInsertionError(
                f"Upsert failed with status: {result.status}"
            )

    async def get_search_results(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_K,
        doc_metadata_filter: list[DocumentMetadata] | None = None,
    ) -> list[ScoredPoint]:
        """
        Queries the Qdrant collection for similar chunks based on the input query.
        """
        logger.info(
            "Performing vector search.",
            query=query,
            user_id=self.user_id,
            limit=limit,
            metadata_filter=[m.model_dump() for m in doc_metadata_filter]
            if doc_metadata_filter
            else "None",
        )

        import asyncio

        # Generate both dense and sparse query vectors
        dense_query_vector = (
            await asyncio.to_thread(self.dense_embedding_model.embed, [query])
        )[0]

        sparse_query_vector = (
            await asyncio.to_thread(self.sparse_embedding_model.embed, [query])
        )[0]

        filter_params = {
            "must": [
                types.FieldCondition(
                    key="user_id", match=types.MatchValue(value=self.user_id)
                )
            ]
        }

        if doc_metadata_filter:
            should_clauses = [
                types.Filter(
                    must=[
                        types.FieldCondition(
                            key="metadata.year",
                            match=types.MatchValue(value=doc.year),
                        ),
                        types.FieldCondition(
                            key="metadata.quarter",
                            match=types.MatchValue(value=doc.quarter),
                        ),
                    ]
                )
                for doc in doc_metadata_filter
            ]

            if should_clauses:
                filter_params["should"] = should_clauses

        user_filter = types.Filter(**filter_params)

        # Perform hybrid search using query fusion
        query_response = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                types.Prefetch(
                    query=dense_query_vector,
                    using=DENSE_DEFAULT,
                    limit=limit,
                    filter=user_filter,
                ),
                types.Prefetch(
                    query=sparse_query_vector,
                    using=SPARSE_DEFAULT,
                    limit=limit,
                    filter=user_filter,
                ),
            ],
            query=types.FusionQuery(fusion=types.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        # Extract points from the query response
        search_results = query_response.points

        logger.info(
            "Vector search completed.",
            found_results=len(search_results),
        )

        if not search_results:
            logger.warning(
                "No relevant information found for query.",
                query=query,
                user_id=self.user_id,
            )
            return []

        return search_results

    async def get_related_chunks(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_K,
        doc_metadata_filter: list[DocumentMetadata] | None = None,
    ) -> list[ScoredPoint]:
        """
        Retrieves text chunks related to the input query.
        """

        results = await self.get_search_results(
            query=query,
            limit=limit,
            doc_metadata_filter=doc_metadata_filter,
        )

        if not results:
            return []

        return results
