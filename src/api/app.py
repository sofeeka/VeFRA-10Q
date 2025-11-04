import os
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.pipeline.query_answering import answer_query
from src.pipeline.data_ingestion import ingest_single_document
from src.retrieval.database import UserKnowledgeBase
from src.dependency import get_user_knowledge_base, get_generator
from src.utils.config import USER_SOURCE_DATA_DIR_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VeFRA PDF Document Ingestion API",
    description="API to accept PDF documents."
)


@app.post("/{user_id}/uploadfile/")
async def create_upload_file(user_id: str, file: UploadFile = File(...)):
    """
    Accepts a single PDF file, saves it temporarily,
    and triggers the ingestion pipeline.
    """

    if ".." in user_id or "/" in user_id or "\\" in user_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid user_id format."
        )
    user_id = user_id.lower()
    user_data_dir = USER_SOURCE_DATA_DIR_PATH / user_id

    filename = Path(file.filename).name
    if not filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .pdf files are accepted."
        )

    permanent_file_path = user_data_dir / filename

    try:
        user_data_dir.mkdir(parents=True, exist_ok=True)

        if permanent_file_path.exists():
            logger.warning(
                f"File conflict: {permanent_file_path} already exists."
            )
            raise HTTPException(
                status_code=409,  # 409 Conflict
                detail=f"File with name '{filename}' already exists for this user. "
                "Please rename the file or delete the existing one first."
            )

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

        logger.info(
            f"Received file: {file.filename}. Saved to: {permanent_file_path}")

        db: UserKnowledgeBase = get_user_knowledge_base(user_id=user_id)
        success = ingest_single_document(file_path=permanent_file_path, db=db)

        if success:
            return JSONResponse(
                content={
                    "filename": file.filename,
                    # does this expose the structure inside of the server?
                    "saved_path": str(permanent_file_path)
                },
                status_code=200
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="File processing failed."
            )

    except HTTPException as e:
        # Re-raise HTTPExceptions directly
        raise e
    except Exception as e:
        logger.error(
            f"Error handling upload for {file.filename}: {e}", exc_info=True)

        if permanent_file_path and Path(permanent_file_path).exists():
            try:
                os.remove(permanent_file_path)
                logger.info(f"Cleaned up failed upload: {permanent_file_path}")
            except OSError as oe:
                logger.error(
                    f"Failed to clean up file {permanent_file_path}: {oe}")

        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}")


@app.post("/{user_id}/generate/")
async def generate(user_id: str, query: str):
    if ".." in user_id or "/" in user_id or "\\" in user_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid user_id format."
        )

    db = get_user_knowledge_base(user_id=user_id)
    rag_generator = get_generator()
    answer = ''
    try:
        answer = answer_query(query=query, db=db, generator=rag_generator)
        return JSONResponse(
            content={
                "answer": answer,
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error. {e}")


if __name__ == "__main__":
    # uvicorn src.api.app:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
