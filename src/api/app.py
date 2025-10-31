import os
import logging
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

from src.pipeline.data_ingestion import ingest_single_document
from src.retrieval.database import QdrantDatabase
from src.retrieval.embedder import FastEmbedModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VeFRA PDF Document Ingestion API",
    description="API to accept PDF documents."
)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Accepts a single PDF file, saves it temporarily,
    and triggers the ingestion pipeline.
    """

    # try to temporarily save the uploaded file, so it can be processed
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            uploaded_file = await file.read()
            temp_file.write(uploaded_file)
            temp_file_path = temp_file.name

        logger.info(
            f"Received file: {file.filename}. Saved to: {temp_file_path}")
        db = QdrantDatabase(embedding_model=FastEmbedModel())
        success = ingest_single_document(file_path=temp_file_path, db=db)

        if success:
            return {
                "filename": file.filename,
                "status": "Processing successful"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="File processing failed."
            )

    except Exception as e:
        logger.error(
            f"Error handling upload for {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}")

    finally:
        # clean up the temporary file
        if 'tmp_file_path' in locals() and Path(temp_file_path).exists():
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temp file: {temp_file_path}")

if __name__ == "__main__":
    # python src/api/app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
