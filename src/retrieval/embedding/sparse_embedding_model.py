import torch
from loguru import logger
from qdrant_client import models
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.utils.config import SPARSE_EMBEDDING_MODEL_NAME


class SparseEmbeddingModel:
    """A model for generating SPLADE sparse vectors."""

    def __init__(self, model_name: str = SPARSE_EMBEDDING_MODEL_NAME):
        logger.info(f"Initializing SPLADE model [{model_name}]...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        logger.info("Initialized SPLADE model.")

    def embed(self, chunks: list[str]) -> list[models.SparseVector]:
        """Takes text and returns a list of Qdrant SparseVector objects."""

        output_vectors = []
        for text in chunks:
            with torch.no_grad():
                tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
                output = self.model(**tokens)

                vec = torch.max(
                    torch.log(1 + torch.relu(output.logits))
                    * tokens.attention_mask.unsqueeze(-1),
                    dim=1,
                )[0].squeeze()

            indices = vec.nonzero().squeeze().cpu().tolist()
            values = vec[indices].cpu().tolist()

            if not isinstance(indices, list):
                indices = [indices]
                values = [values]

            output_vectors.append(models.SparseVector(indices=indices, values=values))

        return output_vectors

    @property
    def dim(self) -> int:
        """Not applicable for sparse models in the same way,
        but you could return vocab size."""
        return self.model.config.vocab_size
