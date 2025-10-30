from typing import List, Optional
from pathlib import Path

from utils.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class DocumentChunker:
    def __init__(self, strategy: str = 'recursive', chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP, separators: Optional[List[str]] = None):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap

        if separators is None:
            self.separators = ["\n\n\n\n", "\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str) -> List[str]:
        """Chunk document based on the selected strategy"""

        if self.strategy == 'fixed':
            return self._fixed_chunking(text)

        elif self.strategy == 'overlapping':
            return self._overlapping_chunking(text)

        elif self.strategy == 'recursive':
            return self._recursive_character_chunking(text)

        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _fixed_chunking(self, text: str) -> List[str]:
        """Split text into fixed-size chunks"""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def _overlapping_chunking(self, text: str) -> List[str]:
        """Split text with overlapping windows"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            if end >= len(text):
                break

            start = end - self.overlap

        return chunks

    def _recursive_character_chunking(self, text: str) -> List[str]:
        """Recursively split text using different separators"""

        def _split_text(text, separators, chunk_size):
            if len(text) <= chunk_size:
                return [text]

            for separator in separators:
                if separator in text:
                    parts = text.split(separator)
                    chunks = []
                    current_chunk = ""

                    for part in parts:
                        test_chunk = current_chunk + separator + part if current_chunk else part

                        if len(test_chunk) <= chunk_size:
                            current_chunk = test_chunk
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = part

                    if current_chunk:
                        chunks.append(current_chunk)

                    final_chunks = []
                    for chunk in chunks:
                        if len(chunk) > chunk_size:
                            final_chunks.extend(_split_text(
                                chunk, separators[1:], chunk_size))
                        else:
                            final_chunks.append(chunk)

                    return final_chunks

            # If no separator works, split by characters
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        # Call the helper function with the instance's state
        return _split_text(text, self.separators, self.chunk_size)
