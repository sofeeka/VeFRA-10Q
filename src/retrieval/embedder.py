from typing import List, Protocol

from fastembed.embedding import DefaultEmbedding
from loguru import logger


class EmbeddingModel(Protocol):
    """A protocol defining the interface for an embedding model."""

    def embed(self, chunks: List[str]) -> List[List[float]]:
        """Takes a list of text chunks and returns a list of embeddings."""
        ...

    @property
    def dim(self) -> int:
        """Returns the dimension (size) of the embeddings."""
        ...


class FastEmbedModel:
    """A concrete implementation of an embedding model using FastEmbed."""

    def __init__(self, model_name: str):
        logger.info("Initializing FastEmbed model...")
        self.model = DefaultEmbedding(model_name=model_name)

        # calculate dimension (needed for Qdrant)
        dummy_embedding = list(self.model.embed("test"))[0]
        self._dim = len(dummy_embedding)
        logger.info(f"Initialized model [{model_name}] with {self._dim} dimensions.")

    def embed(self, chunks: List[str]) -> List[List[float]]:
        """Takes a list of text chunks and returns a list of embeddings."""
        return list(self.model.embed(chunks))

    @property
    def dim(self) -> int:
        """Returns the dimension (size) of the embeddings."""
        return self._dim
