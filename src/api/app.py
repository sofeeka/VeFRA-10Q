import asyncio
import json
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
    except VeFRAException as e:
        logger.error(
            "VeFRA EXception caught.",
            user_id=user_id,
            query=query,
            exc_info=True,
        )
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message,
        ) from e
    except Exception as e:
        logger.error(
            "Unhandled exception during response generation.",
            user_id=user_id,
            query=query,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        ) from e


@app.post("/{user_id}/evaluate/")
async def evaluate(user_id: str):
    """
    Runs the evaluation of the RAG system.
    """
    try:
        df = await run_evaluation(user_id=user_id)

        n_total = len(df)
        n_successful = df[df["evaluation_status"] == "SUCCESS"].shape[0]

        correctness_scores = df["answer_correctness_score"].dropna()
        mean_correctness = (
            correctness_scores.mean() if not correctness_scores.empty else -1.0
        )

        groundedness_scores = df["groundedness_score"].dropna()
        mean_groundedness = (
            groundedness_scores.mean() if not groundedness_scores.empty else -1.0
        )

        context_coverage_scores = df["context_coverage_score"].dropna()
        mean_context_coverage = (
            context_coverage_scores.mean()
            if not context_coverage_scores.empty
            else -1.0
        )

        numerical_accuracy_scores = df["numerical_accuracy_score"].dropna()
        mean_numerical_accuracy = (
            numerical_accuracy_scores.mean()
            if not numerical_accuracy_scores.empty
            else -1.0
        )

        all_chunk_relevance_scores = []
        for _, row in df.iterrows():
            if row["chunk_relevance_scores"] and row["evaluation_status"] == "SUCCESS":
                # chunk_relevance_scores is a JSON string of list of LLMJudgeScore dicts
                chunk_scores_list = json.loads(row["chunk_relevance_scores"])
                question_chunk_scores = [
                    s["score"]
                    for s in chunk_scores_list
                    if s and "score" in s and s["score"] is not None
                ]
                if question_chunk_scores:
                    all_chunk_relevance_scores.extend(question_chunk_scores)

        mean_chunk_relevance = (
            sum(all_chunk_relevance_scores) / len(all_chunk_relevance_scores)
            if all_chunk_relevance_scores
            else 0.0
        )

        return JSONResponse(
            content={
                "total_questions": n_total,
                "successful_evaluations": n_successful,
                "mean_answer_correctness": round(mean_correctness, 2),
                "mean_groundedness": round(mean_groundedness, 2),
                "mean_context_coverage": round(mean_context_coverage, 2),
                "mean_chunk_relevance": round(mean_chunk_relevance, 2),
                "mean_numerical_accuracy": round(mean_numerical_accuracy, 2),
            },
            status_code=200,
        )
    except Exception as e:
        logger.error(
            "Unhandled exception during evaluation.",
            e=e,
            user_id=user_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error.") from e
