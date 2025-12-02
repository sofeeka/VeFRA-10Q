import os
from pathlib import Path

from loguru import logger

from src.data_models.api import CSVUploadModel, FileUploadModel
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


async def save_uploaded_csv(model: CSVUploadModel, is_temporary: bool = False):
    """
    Save the uploaded CSV file.
    If processed_content is available, saves that. Otherwise saves the original file.
    """
    import tempfile

    if is_temporary:
        # Create a temporary file
        fd, path = tempfile.mkstemp(suffix=".csv", prefix=f"eval_{model.user_id}_")
        os.close(fd)
        model.filepath = Path(path)
    elif model.filepath is None:
        # Fallback if not temporary but no path set (shouldn't happen in current flow but good for safety)
        from src.utils.config import get_user_evaluations_folder

        model.filepath = (
            get_user_evaluations_folder(user_id=model.user_id) / model.file.filename
        )

    try:
        if model.processed_content:
            with open(model.filepath, "wb") as f:
                f.write(model.processed_content)
        else:
            uploaded_file_content = await model.file.read()
            with open(model.filepath, "wb") as f:
                f.write(uploaded_file_content)

    except OSError as e:
        logger.error(
            "Failed to write CSV file to disk.",
            filename=model.file.filename,
            filepath=str(model.filepath),
            exc_info=True,
        )
        raise VeFRA_FileIOError(f"Failed to save uploaded CSV: {e}") from e

    logger.info(
        "Saved CSV file to disk.",
        filepath=str(model.filepath),
        user_id=model.user_id,
        filename=model.file.filename,
        is_temporary=is_temporary,
    )


async def process_csv_upload(model: CSVUploadModel, is_temporary: bool = False):
    """
    Processes the CSV upload request after validation.
    Saves the uploaded CSV file (or processed content) to disk.
    """
    logger.info(
        "Starting CSV upload process.",
        filename=model.file.filename,
        user_id=model.user_id,
        is_temporary=is_temporary,
    )

    await save_uploaded_csv(model=model, is_temporary=is_temporary)

    logger.success(
        "CSV upload process completed successfully.",
        filename=model.file.filename,
        user_id=model.user_id,
        filepath=str(model.filepath),
    )

    return model.filepath
