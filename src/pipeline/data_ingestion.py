from typing import List

from docling_core.types.doc import DoclingDocument

from src.processing.document_parser import DocumentParser
from src.processing.document_processor import process_documents_for_chunking, process_document_for_chunking
from src.processing.chuncker import DocumentChunker
from src.retrieval.database import QdrantDatabase

from src.utils.config import SOURCE_DATA_DIR_PATH


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
    # PDF -> Docling
    parser = DocumentParser()
    parsed_doc: DoclingDocument = parser.parse_document(
        file_path=file_path)

    # Docling -> Processed Text
    processed_text: str = process_document_for_chunking(
        documents=[parsed_doc])

    # Processed Text -> Text Chunks
    chunker = DocumentChunker()
    chunks: List[str] = chunker.chunk_text(processed_text)

    # Text Chunks -> Database
    db.add_chunks(chunks=chunks)


if __name__ == "__main__":
    populate_database_with_docs_in_folder(
        data_dir_path=SOURCE_DATA_DIR_PATH)
