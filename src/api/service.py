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

    except OSError as e:
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
    import time

    start_time = time.monotonic()

    await save_uploaded_document(model=model)

    db = get_user_knowledge_base(user_id=model.user_id)
    try:
        await ingest_single_document(model=model, db=db)
        duration = time.monotonic() - start_time
        logger.success(
            f"Document ingestion process completed successfully in {duration:.2f} seconds.",
            filename=model.file.filename,
            user_id=model.user_id,
            duration=duration,
        )
    except Exception as e:
        duration = time.monotonic() - start_time
        logger.error(
            f"Ingestion pipeline failed after {duration:.2f} seconds. Cleaning up saved document.",
            filename=model.file.filename,
            user_id=model.user_id,
            exc_info=True,
            duration=duration,
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
