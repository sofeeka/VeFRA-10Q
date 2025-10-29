import re
from typing import List, Optional, Union
from pathlib import Path

from utils.config import DATA_DIR_PATH, TEST_DATA_DIR_PATH, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from document_parser import prepare_document_for_chunking, print_items_from_document


def fixed_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Split text into fixed-size chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def overlapping_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Split text with overlapping windows"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def recursive_character_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, separators: Optional[List[str]] = None) -> List[str]:
    """Recursively split text using different separators"""
    if separators is None:
        separators = ["\n\n\n", "\n\n", "\n", ". ", "  ", " ", ""]

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

                # Recursively split large chunks
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

    return _split_text(text, separators, chunk_size)
