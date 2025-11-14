from typing import List

from docling_core.types.doc import DoclingDocument

from src.api.models import FileUploadModel
from src.processing.chuncker import DocumentChunker
from src.processing.document_parser import DocumentParser
from src.processing.document_processor import (
    process_document_for_chunking,
    process_documents_for_chunking,
)
from src.retrieval.database import ChunkPayload, UserKnowledgeBase
from src.utils.dependency import get_document_chunker, get_document_parser


# TODO: remove duplication, refactor to reuse ingestion of single file
def populate_database_with_docs_in_folder(data_dir_path: str, db: UserKnowledgeBase):
    # PDF -> Docling
    parser: DocumentParser = get_document_parser()
    parsed_docs: List[DoclingDocument] = parser.parse_documents_in_directory(
        directory_path=data_dir_path
    )

    # Docling -> Processed Text
    processed_texts: List[str] = process_documents_for_chunking(documents=parsed_docs)

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
            metadata={},  # TODO: add metadata
        )
        for chunk in chunks
    ]

    # ChunkPayloads -> Database
    db.add_chunks(chunks=chunk_payloads)


def ingest_single_document(model: FileUploadModel, db: UserKnowledgeBase):
    """
    Runs the full ingestion pipeline. Reads a PDF file with Docling, processes it
    and saves to user's knowledge base
    """

    filepath = model.filepath
    # PDF -> Docling
    parser: DocumentParser = get_document_parser()
    parsed_doc: DoclingDocument = parser.parse_document(filepath=filepath)

    # Docling -> Processed Text
    processed_text: str = process_document_for_chunking(document=parsed_doc)

    # Processed Text -> Text Chunks
    chunker: DocumentChunker = get_document_chunker()
    chunks: List[str] = chunker.chunk_text(processed_text)

    # Text Chunks -> ChunkPayloads
    chunk_payloads: List[ChunkPayload] = [
        ChunkPayload(
            text=chunk,
            metadata={
                "year": model.parsed_year,
                "quarter": model.parsed_quarter,
                "company": model.parsed_company,  # TODO redundant when user_id is present
            },
        )
        for chunk in chunks
    ]

    # ChunkPayloads -> Database
    db.add_chunks(chunks=chunk_payloads)
