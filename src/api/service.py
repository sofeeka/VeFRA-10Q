import os
from pathlib import Path

from loguru import logger

from src.api.models import FileUploadModel
from src.pipeline.data_ingestion import ingest_single_document
from src.retrieval.database import UserKnowledgeBase
from src.utils.dependency import get_user_knowledge_base
from src.utils.exceptions import FileIOError


async def save_uploaded_document(model: FileUploadModel):
    """
    Save the uploaded file to user directory.
    """

    try:
        uploaded_file_content = await model.file.read()

        with open(model.filepath, "wb") as f:
            f.write(uploaded_file_content)

    except IOError as e:
        logger.error(
            f"Failed to write file {model.file.filename} to {model.filepath}: {e}",
            exc_info=True,
        )
        raise FileIOError(f"Failed to save uploaded document: {e}") from e

    logger.info(f"Saved file to: {model.filepath}")


async def process_document_ingestion(model: FileUploadModel) -> bool:
    """
    Processes the API request after the input as been validated.
    Saves the uploaded document, and triggers the ingestion pipeline.
    """

    await save_uploaded_document(model=model)

    db: UserKnowledgeBase = get_user_knowledge_base(user_id=model.user_id)
    success = ingest_single_document(model=model, db=db)

    if not success:
        _cleanup_document(filepath=model.filepath)

    return success


def _cleanup_document(filepath: str):
    """
    Delete the document. Used when saved the document, but failed the ingestion.
    """

    if filepath and Path(filepath).exists():
        try:
            os.remove(filepath)
            logger.info(f"Cleaned up failed upload at: {filepath}")
        except OSError as oe:
            logger.error(f"Failed to clean up file {filepath}: {oe}")
            raise FileIOError(f"Failed to clean up document at {filepath}") from oe
    else:
        logger.warning(
            f"Attempted to clean up document at {filepath} that does not exist."
        )
