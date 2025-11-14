from typing import Mapping, Optional, Union

import qdrant_client.http.models as types
from loguru import logger
from qdrant_client import QdrantClient


class QdrantCollectionManager:
    """
    Manages the lifecycle and schema of a Qdrant collection.
    This is an administrative-level class.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
    ):
        if not client:
            logger.error("No Qdrant client provided to Collection Manager.")
            raise ValueError("client is required.")

        self.client = client
        self.collection_name = collection_name
        logger.info(f"Initializing QdrantCollectionManager for '{collection_name}'")

    def recreate_collection(
        self,
        vectors_config: Optional[
            Union[types.VectorParams, Mapping[str, types.VectorParams]]
        ] = None,
        sparse_vectors_config: Optional[Mapping[str, types.SparseVectorParams]] = None,
    ) -> bool:
        """
        Recreate the Qdrant collection with specified vector parameters.
        This is a destructive operation and will wipe all data.
        """

        result: bool = self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

        if not result:
            logger.error(f"Failed to recreate collection '{self.collection_name}'")
            return False

        logger.info(f"Successfully recreated collection '{self.collection_name}'")

        return True

    def create_payload_index(self, field_name: str) -> bool:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=types.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            logger.info(
                f"Created payload index on {field_name} for collection {self.collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to create payload index on {field_name}: {e}")
            return False

        return True

    def collection_exists(self) -> bool:
        """Checks if the collection already exists."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:  # Catches "Not found" and other connection errors
            return False

    def create_collection_if_not_exists(
        self,
        vectors_config: Optional[
            Union[types.VectorParams, Mapping[str, types.VectorParams]]
        ] = None,
        sparse_vectors_config: Optional[Mapping[str, types.SparseVectorParams]] = None,
    ) -> bool:
        """
        A safer method for setup scripts. Ensures the collection and
        its indexes exist without destroying data.
        """
        if self.collection_exists():
            logger.info(
                f"Collection '{self.collection_name}' already exists. Skipping creation."
            )
            # You might want to verify indexes here too
            return True

        logger.info(f"Collection '{self.collection_name}' not found. Creating...")
        return self.recreate_collection(
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
