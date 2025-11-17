from loguru import logger

from src.data_models.api import FileUploadModel
from src.processing.document_processor import (
    process_document_for_chunking,
)
from src.retrieval.database import ChunkPayload, UserKnowledgeBase
from src.utils.dependency import get_document_chunker, get_document_parser


def ingest_single_document(model: FileUploadModel, db: UserKnowledgeBase):
    """
    Runs the full ingestion pipeline. Reads a PDF file with Docling, processes it
    and saves to user's knowledge base
    """
    logger.info(
        "Starting document ingestion pipeline.",
        user_id=db.user_id,
        filename=model.file.filename,
    )
    filepath = model.filepath

    # PDF -> Docling
    logger.info("Parsing document...")
    parser = get_document_parser()
    parsed_doc = parser.parse_document(filepath=filepath)
    logger.info("Document parsed successfully.")

    # Docling -> Processed Text
    logger.info("Processing document for chunking (extracting tables)...")
    processed_text = process_document_for_chunking(
        document=parsed_doc, user_id=db.user_id
    )
    logger.info("Document processed successfully.")

    # Processed Text -> Text Chunks
    logger.info("Chunking processed text...")
    chunker = get_document_chunker()
    chunks = chunker.chunk_text(processed_text)
    logger.info("Created {chunk_count} chunks.", chunk_count=len(chunks))

    # Text Chunks -> ChunkPayloads
    chunk_payloads = [
        ChunkPayload(
            text=chunk,
            metadata={
                "year": model.parsed_year,
                "quarter": model.parsed_quarter,
            },
        )
        for chunk in chunks
    ]

    # ChunkPayloads -> Database
    logger.info("Adding chunks to the database...")
    db.add_chunks(chunks=chunk_payloads)
    logger.success(
        "Document ingestion pipeline completed successfully.",
        filename=model.file.filename,
        user_id=db.user_id,
    )
