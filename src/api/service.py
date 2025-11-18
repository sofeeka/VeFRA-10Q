import os
from pathlib import Path

from loguru import logger

from src.data_models.api import FileUploadModel
from src.pipeline.data_ingestion import ingest_single_document
from src.utils.dependency import get_user_knowledge_base
from src.utils.exceptions import VeFRA_FileIOError


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
            "Failed to write file to disk.",
            filename=model.file.filename,
            filepath=str(model.filepath),
            exc_info=True,
        )
        raise VeFRA_FileIOError(f"Failed to save uploaded document: {e}") from e

    logger.info(
        "Saved file to disk.",
        filepath=str(model.filepath),
        user_id=model.user_id,
        filename=model.file.filename,
    )


async def process_document_ingestion(model: FileUploadModel):
    """
    Processes the API request after the input as been validated.
    Saves the uploaded document, and triggers the ingestion pipeline.
    """
    logger.info(
        "Starting document ingestion process.",
        filename=model.file.filename,
        user_id=model.user_id,
    )
    await save_uploaded_document(model=model)

    db = get_user_knowledge_base(user_id=model.user_id)
    try:
        ingest_single_document(model=model, db=db)
        logger.success(
            "Document ingestion process completed successfully.",
            filename=model.file.filename,
            user_id=model.user_id,
        )
    except Exception as e:
        logger.error(
            "Ingestion pipeline failed. Cleaning up saved document.",
            filename=model.file.filename,
            user_id=model.user_id,
            exc_info=True,
        )
        _cleanup_document(filepath=model.filepath)
        raise e


def _cleanup_document(filepath: str | Path):
    """
    Delete the document. Used when saved the document, but failed the ingestion.
    """

    if filepath and Path(filepath).exists():
        try:
            os.remove(filepath)
            logger.info(
                "Cleaned up failed upload artifact.",
                filepath=str(filepath),
            )
        except OSError as oe:
            logger.error(
                "Failed to clean up file artifact.",
                filepath=str(filepath),
                exc_info=True,
            )
            raise VeFRA_FileIOError(
                f"Failed to clean up document at {filepath}"
            ) from oe
    else:
        logger.warning(
            "Attempted to clean up a document that does not exist.",
            filepath=str(filepath),
        )
