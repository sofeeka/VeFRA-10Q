import logging
from typing import List

from docling_core.types.doc import DoclingDocument

from src.processing.document_parser import DocumentParser
from src.processing.document_processor import process_documents_for_chunking, process_document_for_chunking
from src.processing.chuncker import DocumentChunker
from src.retrieval.database import QdrantDatabase

from src.utils.config import SOURCE_DATA_DIR_PATH

logger = logging.getLogger(__name__)


def populate_database_with_docs_in_folder(data_dir_path: str, db: QdrantDatabase):
    # PDF -> Docling
    parser = DocumentParser()
    parsed_docs: List[DoclingDocument] = parser.parse_documents_in_directory(
        directory_path=data_dir_path)

    # Docling -> Processed Text
    processed_texts: List[str] = process_documents_for_chunking(
        documents=parsed_docs)

    # Processed Text -> Text Chunks
    chunker = DocumentChunker()
    chunks: List[str] = []
    for text in processed_texts:
        c = chunker.chunk_text(text)
        chunks.extend(c)

    # Text Chunks -> Database
    db.add_chunks(chunks=chunks)


def ingest_single_document(file_path: str, db: QdrantDatabase):
    try:
        # PDF -> Docling
        parser = DocumentParser()
        parsed_doc: DoclingDocument = parser.parse_document(
            file_path=file_path)

        if not parsed_doc:
            logger.error(f"Failed to parse document: {file_path}")
            return False

        # Docling -> Processed Text
        processed_text: str = process_document_for_chunking(
            documents=[parsed_doc])

        if not processed_text:
            logger.error(
                f"Parsed, but failed to process document: {file_path}")
            return False

        # Processed Text -> Text Chunks
        chunker = DocumentChunker()
        chunks: List[str] = chunker.chunk_text(processed_text)

        if not chunks:
            logger.error(
                f"Parsed, processed, but failed to chunk document: {file_path}")
            return False

        # Text Chunks -> Database
        # TODO add error handling and feedback about success/failure
        db.add_chunks(chunks=chunks)
        return True

    except Exception as e:
        print(
            f"Unexpected error happened when ingesting document {file_path}: {e}")
        return False


if __name__ == "__main__":
    populate_database_with_docs_in_folder(
        data_dir_path=SOURCE_DATA_DIR_PATH)
