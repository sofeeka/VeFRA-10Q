from typing import List

from fastembed.embedding import DefaultEmbedding
from loguru import logger


class DenseEmbeddingModel:
    """A concrete implementation of an embedding model using FastEmbed."""

    def __init__(self, model_name: str):
        logger.info("Initializing FastEmbed model for dense embeddigns...")
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
