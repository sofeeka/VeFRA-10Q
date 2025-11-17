import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.api.service import process_document_ingestion
from src.data_models.api import FileUploadModel
from src.evaluation.rag_evaluator import run_evaluation
from src.pipeline.query_answering import answer_query
from src.scripts.logging_config import setup_logging
from src.scripts.setup_database import setup_database
from src.utils.config import MAIN_RESPONSE_GENERATION_MODEL
from src.utils.dependency import get_generator, get_user_knowledge_base
from src.utils.exceptions import VeFRAException


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_database()
    setup_logging()
    yield


app = FastAPI(
    title="VeFRA PDF Document Ingestion API",
    description="API to accept PDF documents.",
    lifespan=lifespan,
)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())

        with logger.contextualize(request_id=request_id):
            logger.info(f"Request started: {request.method} {request.url.path}")

            response = await call_next(request)

            logger.info(f"Request finished: status_code={response.status_code}")
            return response


app.add_middleware(LoggingMiddleware)


@app.post("/{user_id}/uploadfile/")
async def create_upload_file(input_user_id: str, input_file: UploadFile = File(...)):
    """
    Accepts a single PDF file, saves it,
    and triggers the ingestion pipeline.
    """

    model = None
    try:
        model = FileUploadModel(file=input_file, user_id=input_user_id)

        await process_document_ingestion(model=model)
        return JSONResponse(
            content={
                "filename": model.file.filename,
            },
            status_code=200,
        )

    except VeFRAException as e:
        logger.warning(
            "Caught specific VeFRAException during file upload.",
            error_message=e.message,
            error_details=e.details,
            status_code=e.status_code,
        )
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        filename = model.file.filename if model else "unknown"
        logger.error(
            "Unhandled exception during file upload.",
            user_id=input_user_id,
            filename=filename,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error.") from e


@app.post("/{user_id}/generate/")
async def generate(user_id: str, query: str):
    """
    Generates the response to user question using user's knowledge base.
    """
    # TODO create better validation
    if ".." in user_id or "/" in user_id or "\\" in user_id:
        raise HTTPException(status_code=400, detail="Invalid user_id format.")

    try:
        db = get_user_knowledge_base(user_id=user_id)
        rag_generator = get_generator(model=MAIN_RESPONSE_GENERATION_MODEL)
        answer, _ = await asyncio.to_thread(
            answer_query, query=query, db=db, generator=rag_generator
        )
        return JSONResponse(
            content={
                "answer": answer,
            },
            status_code=200,
        )
    except Exception as e:
        logger.error(
            "Unhandled exception during response generation.",
            user_id=user_id,
            query=query,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error.") from e


@app.post("/{user_id}/evaluate/")
async def evaluate(user_id: str):
    """
    Runs the evaluation of the RAG system.
    """
    try:
        df = run_evaluation(
            user_id=user_id
        )  # TODO mention user_id in the benchmark dataset or create a testing user with all docs for this
        ranking = round(df["Ranking for Question with Contexts"].mean(), 2)

        n_correct = df["Correctness"].value_counts()["CORRECT"]
        n_total = df.shape[0]
        correctness = round(n_correct / n_total, 2)

        return JSONResponse(
            content={
                "mean_ranking": ranking,
                "correctness": correctness,
                "n_correct": int(n_correct),
                "n": int(n_total),
            },
            status_code=200,
        )
    except Exception as e:
        logger.error(
            "Unhandled exception during evaluation.",
            user_id=user_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error.") from e
