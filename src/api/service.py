import os
from loguru import logger
from pathlib import Path

from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.pipeline.data_ingestion import ingest_single_document
from src.retrieval.database import UserKnowledgeBase
from src.dependency import get_user_knowledge_base


async def save_uploaded_document(file: UploadFile, permanent_file_path: Path):
    """
    Save the uploaded file to user directory.
    """

    try:
        uploaded_file_content = await file.read()

        with open(permanent_file_path, "wb") as f:
            f.write(uploaded_file_content)

    except IOError as e:
        logger.error(
            f"Failed to write file to {permanent_file_path}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to save file on server: {e}"
        )

    logger.info(f"Saved file to: {permanent_file_path}")


async def process_document_ingestion(file: UploadFile, user_id: str, permanent_file_path: Path) -> JSONResponse:
    """
    Processes the API request after the input as been validated. 
    Saves the uploaded document, and triggers the ingestion pipeline.
    """

    await save_uploaded_document(file=file, permanent_file_path=permanent_file_path)

    db: UserKnowledgeBase = get_user_knowledge_base(user_id=user_id)
    success = ingest_single_document(file_path=permanent_file_path, db=db)

    if success:
        return JSONResponse(
            content={
                "filename": file.filename,
            },
            status_code=200
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="File processing failed."
        )


def cleanup_document(file_path: str):
    """
    Delete the document. Used when saved the document, but failed the ingestion.
    """

    if file_path and Path(file_path).exists():
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up failed upload: {file_path}")
        except OSError as oe:
            logger.error(
                f"Failed to clean up file {file_path}: {oe}")
