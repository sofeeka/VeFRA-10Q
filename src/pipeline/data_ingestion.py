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

    filepath = model.filepath
    # PDF -> Docling
    parser = get_document_parser()
    parsed_doc = parser.parse_document(filepath=filepath)

    # Docling -> Processed Text
    processed_text = process_document_for_chunking(
        document=parsed_doc, user_id=db.user_id
    )

    # Processed Text -> Text Chunks
    chunker = get_document_chunker()
    chunks = chunker.chunk_text(processed_text)

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
    db.add_chunks(chunks=chunk_payloads)
