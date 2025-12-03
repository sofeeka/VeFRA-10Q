import os
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from ..data_models.api import CSVUploadModel, FileUploadModel
from ..debug.debug import debug_manager
from ..evaluation.rag_evaluator import run_evaluation
from ..pipeline.query_answering import QueryAnsweringPipeline
from ..scripts.logging_config import setup_logging
from ..scripts.setup_database import setup_database
from ..utils.config import MAIN_RESPONSE_GENERATION_MODEL, PROJECT_ROOT_PATH
from ..utils.dependency import (
    get_async_generator,
    get_reranking_model,
    get_user_knowledge_base,
)
from ..utils.exceptions import VeFRAException
from .input_validation import get_existing_user, validate_user_id
from .service import (
    get_evaluation_results,
    process_csv_upload,
    process_document_ingestion,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_database()
    setup_logging()
    logger.info("Pre-loading expensive models at startup...")
    get_reranking_model()
    logger.info("Models pre-loaded successfully.")
    yield


app = FastAPI(
    title="VeFRA PDF Document Ingestion API",
    description="API to accept PDF documents.",
    lifespan=lifespan,
)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory=PROJECT_ROOT_PATH / "src" / "api" / "templates")


def client_wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


@app.exception_handler(VeFRAException)
async def global_exception_handler(request: Request, e: VeFRAException):
    logger.exception(e.message)

    # If the client wants HTML → render error.html
    if client_wants_html(request):
        response = templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error": e},
            status_code=e.status_code,
        )
        response.body  # force rendering NOW so errors get caught
        return response

    # Otherwise return JSON error
    return JSONResponse(
        {"detail": e.message, "fromExceptionHandler": True}, status_code=e.status_code
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
@limiter.limit("2/minute")
async def create_upload_file(
    user_id: str = Depends(validate_user_id),
    file: UploadFile = File(...),
):
    """
    Accepts a single PDF file, saves it,
    and triggers the ingestion pipeline.
    """

    model = None
    try:
        model = FileUploadModel(file=file, user_id=user_id)

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
        filename = model.file.filename if model and model.file else "unknown"
        logger.error(
            "Unhandled exception during file upload.",
            user_id=user_id,
            filename=filename,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error.") from e


@app.get("/{user_id}/generate/")
@limiter.limit("5/minute")
async def generate(
    query: str,
    user_id: str = Depends(get_existing_user),
):
    """
    Generates the response to user question using user's knowledge base.
    """
    query_answering_pipeline = QueryAnsweringPipeline(
        db=get_user_knowledge_base(user_id=user_id),
        generator=get_async_generator(model=MAIN_RESPONSE_GENERATION_MODEL),
    )
    try:
        answer = await query_answering_pipeline.run(query=query)
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


@app.get("/{user_id}/debug_generate/")
@limiter.limit("2/minute")
async def debug_generate(
    request: Request,
    query: str,
    user_id: str = Depends(get_existing_user),
):
    """
    Generates the response to user question using user's knowledge base.
    """

    try:
        debug_manager.enable()
        debug_manager.clear_data()

        debug_data = debug_manager.get_data()
        debug_data.user = user_id
        debug_data.question = query

        query_answering_pipeline = QueryAnsweringPipeline(
            db=get_user_knowledge_base(user_id=user_id),
            generator=get_async_generator(model=MAIN_RESPONSE_GENERATION_MODEL),
        )

        answer = await query_answering_pipeline.run(query=query)

        debug_data.answer = answer

        response = templates.TemplateResponse(
            "debug.html",
            {
                "request": request,
                "debug_data": debug_data,
            },
        )

        response.body  # force early rendering
        return response

    finally:
        debug_manager.disable()


@app.get("/{user_id}/evaluate/")
@limiter.limit("10/hour")
async def evaluate(user_id: str = Depends(get_existing_user)):
    """
    Runs the evaluation of the RAG system.
    """

    try:
        df = await run_evaluation(user_id=user_id)

        metrics = get_evaluation_results(df=df)
        metrics["evaluation_results"] = df.to_dict(orient="records")

        return JSONResponse(
            content=metrics,
            status_code=200,
        )

    except VeFRAException as e:
        logger.error(
            "VeFRA Exception caught during evaluation.",
            e=e,
            user_id=user_id,
            exc_info=True,
        )
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error(
            "Unhandled exception during evaluation.",
            e=e,
            user_id=user_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error.") from e


@app.post("/{user_id}/evaluate_file/")
@limiter.limit("10/hour")
async def evaluate_file(
    user_id: str = Depends(get_existing_user),
    file: UploadFile = File(...),
):
    """
    Runs the evaluation of the RAG system.
    """

    csv_filepath = None
    temp_filepath = None

    try:
        if file:
            model = CSVUploadModel(file=file, user_id=user_id)

            temp_filepath = await process_csv_upload(model, is_temporary=True)
            csv_filepath = temp_filepath

        df = await run_evaluation(user_id=user_id, csv_filepath=csv_filepath)

        metrics = get_evaluation_results(df=df)
        metrics["evaluation_results"] = df.to_dict(orient="records")

        return JSONResponse(
            content=metrics,
            status_code=200,
        )
    except VeFRAException as e:
        logger.error(
            "VeFRA Exception caught during evaluation.",
            e=e,
            user_id=user_id,
            exc_info=True,
        )
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error(
            "Unhandled exception during evaluation.",
            e=e,
            user_id=user_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error.") from e
    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                logger.info(f"Cleaned up temporary evaluation file: {temp_filepath}")
            except OSError as e:
                logger.warning(f"Failed to cleanup temporary file {temp_filepath}: {e}")
