from typing import Protocol


class EmbeddingModel(Protocol):
    """A protocol defining the interface for an embedding model."""

    def embed(self, chunks: list[str]) -> list[list[float]]:
        """Takes a list of text chunks and returns a list of embeddings."""
        ...

    @property
    def dim(self) -> int:
        """Returns the dimension (size) of the embeddings."""
        ...
