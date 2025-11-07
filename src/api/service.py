import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.models import FileUploadModel
from src.pipeline.data_ingestion import ingest_single_document
from src.retrieval.database import UserKnowledgeBase
from src.utils.dependency import get_user_knowledge_base


async def save_uploaded_document(model: FileUploadModel):
    """
    Save the uploaded file to user directory.
    """

    try:
        uploaded_file_content = await model.file.read()

        with open(model.file_path, "wb") as f:
            f.write(uploaded_file_content)

    except IOError as e:
        logger.error(
            f"Failed to write file {model.file.filename} to {model.file_path}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to save file on server: {e}"
        )

    logger.info(f"Saved file to: {model.file_path}")


async def process_document_ingestion(model: FileUploadModel) -> JSONResponse:
    """
    Processes the API request after the input as been validated.
    Saves the uploaded document, and triggers the ingestion pipeline.
    """

    await save_uploaded_document(model=model)

    try:
        db: UserKnowledgeBase = get_user_knowledge_base(user_id=model.user_id)
        success = ingest_single_document(model=model, db=db)

        if success:
            return JSONResponse(
                content={
                    "filename": model.file.filename,
                },
                status_code=200,
            )
        else:
            _cleanup_document(file_path=model.filepath)
            raise HTTPException(status_code=500, detail="File processing failed.")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error occured: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occured: {e}")


def _cleanup_document(file_path: str):
    """
    Delete the document. Used when saved the document, but failed the ingestion.
    """

    if file_path and Path(file_path).exists():
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up failed upload: {file_path}")
        except OSError as oe:
            logger.error(f"Failed to clean up file {file_path}: {oe}")
