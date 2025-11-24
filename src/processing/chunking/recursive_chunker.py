from docling_core.types.doc import DoclingDocument
from loguru import logger

from src.processing.document_processor import process_document_for_chunking
from src.utils.exceptions import VeFRA_ChunkingError

from .base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    def chunk(self, document: DoclingDocument, user_id: str) -> list[str]:
        """Chunk document based on the selected strategy"""

        logger.info(
            "Processing document for chunking (extracting tables, converting to text)..."
        )
        processed_text = process_document_for_chunking(
            document=document, user_id=user_id
        )
        logger.info("Document processed successfully.")

        logger.info("Recursively splitting the text into chunks...")
        chunks = self._recursive_character_chunking(text=processed_text)

        if not chunks:
            raise VeFRA_ChunkingError(f"Failed to chunk document {document.name}.")
        return chunks

    def _recursive_character_chunking(self, text: str) -> list[str]:
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
                        test_chunk = (
                            current_chunk + separator + part if current_chunk else part
                        )

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
                            final_chunks.extend(
                                _split_text(chunk, separators[1:], chunk_size)
                            )
                        else:
                            final_chunks.append(chunk)

                    return final_chunks

            # If no separator works, split by characters
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        separators = ["\n\n\n\n", "\n\n", "\n", ". ", " ", ""]
        from src.utils.config import DEFAULT_CHUNK_SIZE

        # Call the helper function with the instance's state
        return _split_text(
            text=text, separators=separators, chunk_size=DEFAULT_CHUNK_SIZE
        )
