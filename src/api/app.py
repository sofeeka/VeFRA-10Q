import uvicorn
from loguru import logger
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.api.models import FileUploadModel
from src.pipeline.query_answering import answer_query
from api.service import process_document_ingestion, cleanup_document
from src.dependency import get_user_knowledge_base, get_generator
from src.utils.config import USER_SOURCE_DATA_DIR_PATH


app = FastAPI(
    title="VeFRA PDF Document Ingestion API",
    description="API to accept PDF documents."
)


@app.post("/{user_id}/uploadfile/")
async def create_upload_file(input_user_id: str, input_file: UploadFile = File(...)):
    """
    Accepts a single PDF file, saves it,
    and triggers the ingestion pipeline.
    """

    model = None
    try:
        model = FileUploadModel(file=input_file, user_id=input_user_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred.")

    # TODO move to path manager or something similar
    user_data_dir = USER_SOURCE_DATA_DIR_PATH / model.user_id
    permanent_file_path = user_data_dir / model.file.filename

    try:
        response: JSONResponse = await process_document_ingestion(
            file=model.file, user_id=model.user_id, permanent_file_path=permanent_file_path)
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error handling upload for {model.file.filename}: {e}")
        cleanup_document(permanent_file_path)
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
