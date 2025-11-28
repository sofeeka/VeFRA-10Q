import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FinancialReranker:
    def __init__(self, model_name: str):
        """
        Initializes a Cross-Encoder reranker using HuggingFace Transformers.
        This removes the need for 'ragatouille' or 'langchain'.
        """
        logger.info(f"Loading Reranker model: {model_name}...")

        # Detect device (GPU > MPS (Mac) > CPU)
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            # attn_implementation="flash_attention_2" if self.device == "cuda" else None,
            # attn_implementation="sdpa",
        )
        self.model.to(self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        chunks: list[str],
        top_n: int = 5,
        batch_size=8,
    ) -> list[str]:
        """
        Reranks a list of retrieved chunks using a Cross-Encoder.
        """
        if not chunks:
            logger.warning("No chunks provided to rerank.")
            return []

        pairs = [[query, chunk] for chunk in chunks]

        logger.info(
            f"Starting re-ranking of {len(pairs)} retrieved chunks in batches of {batch_size}."
        )

        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]

            logger.info(f"Processing batch {i // batch_size + 1}...")

            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=8192,
                ).to(self.device)

                outputs = self.model(**inputs, return_dict=True)

                logits = outputs.logits
                if logits.shape[1] > 1:
                    # If model has 2 classes (neg, pos), take the positive score
                    batch_scores = logits[:, 1]
                else:
                    batch_scores = logits.view(-1)

                all_scores.extend(batch_scores.cpu().float().numpy())

        scored_chunks = list(zip(chunks, all_scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        ordered_chunks = [chunk for chunk, score in scored_chunks[:top_n]]

        logger.debug(
            f"Top Score: {scored_chunks[0][1]} | Low Score: {scored_chunks[-1][1]}"
        )
        return ordered_chunks
