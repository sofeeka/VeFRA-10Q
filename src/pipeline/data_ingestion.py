import logging
from typing import List

from docling_core.types.doc import DoclingDocument

from src.processing.document_parser import DocumentParser
from src.processing.document_processor import process_documents_for_chunking, process_document_for_chunking
from src.processing.chuncker import DocumentChunker
from src.retrieval.database import UserKnowledgeBase, ChunkPayload

from src.dependency import get_document_parser, get_document_chunker
from src.utils.config import SOURCE_DATA_DIR_PATH

logger = logging.getLogger(__name__)


# TODO: remove duplication, refactor to reuse ingestion of single file
def populate_database_with_docs_in_folder(data_dir_path: str, db: UserKnowledgeBase):
    # PDF -> Docling
    parser: DocumentParser = get_document_parser()
    parsed_docs: List[DoclingDocument] = parser.parse_documents_in_directory(
        directory_path=data_dir_path)

    # Docling -> Processed Text
    processed_texts: List[str] = process_documents_for_chunking(
        documents=parsed_docs)

    # Processed Text -> Text Chunks
    chunker: DocumentChunker = get_document_chunker()
    chunks: List[str] = []
    for text in processed_texts:
        c = chunker.chunk_text(text)
        chunks.extend(c)

    # Text Chunks -> ChunkPayloads
    chunk_payloads: List[ChunkPayload] = [
        ChunkPayload(
            text=chunk,
            metadata={}  # TODO: add metadata
        ) for chunk in chunks
    ]

    # ChunkPayloads -> Database
    db.add_chunks(chunks=chunk_payloads)


def ingest_single_document(file_path: str, db: UserKnowledgeBase) -> bool:
    try:
        # PDF -> Docling
        parser: DocumentParser = get_document_parser()
        parsed_doc: DoclingDocument = parser.parse_document(
            file_path=file_path)

        if not parsed_doc:
            logger.error(f"Failed to parse document: {file_path}")
            return False

        # Docling -> Processed Text
        processed_text: str = process_document_for_chunking(
            document=parsed_doc)

        if not processed_text:
            logger.error(
                f"Parsed, but failed to process document: {file_path}")
            return False

        # Processed Text -> Text Chunks
        chunker: DocumentChunker = get_document_chunker()
        chunks: List[str] = chunker.chunk_text(processed_text)

        if not chunks:
            logger.error(
                f"Parsed, processed, but failed to chunk document: {file_path}")
            return False

        # Text Chunks -> ChunkPayloads
        chunk_payloads: List[ChunkPayload] = [
            ChunkPayload(
                text=chunk,
                metadata={}  # TODO: add metadata
            ) for chunk in chunks
        ]

        # ChunkPayloads -> Database
        result: bool = db.add_chunks(chunks=chunk_payloads)

        if not result:
            logger.error(
                f"Failed to add chunks to database for document: {file_path}")
            return False

        return True

    except Exception as e:
        print(
            f"Unexpected error happened when ingesting document {file_path}: {e}")
        return False


if __name__ == "__main__":
    populate_database_with_docs_in_folder(
        data_dir_path=SOURCE_DATA_DIR_PATH)
