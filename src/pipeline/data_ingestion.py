from loguru import logger

from src.data_models.api import FileUploadModel
from src.retrieval.database import ChunkPayload, UserKnowledgeBase
from src.utils.dependency import get_document_parser, get_recursive_chunker


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

    # Docling -> Chunks
    logger.info("Chunking parsed document...")
    chunker = get_recursive_chunker()
    chunks = chunker.chunk(document=parsed_doc, user_id=db.user_id)
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
