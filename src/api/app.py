import asyncio

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.models import FileUploadModel
from src.api.service import process_document_ingestion
from src.evaluation.rag_evaluator import run_evaluation
from src.pipeline.query_answering import answer_query
from src.utils.config import MAIN_RESPONSE_GENERATION_MODEL
from src.utils.dependency import get_generator, get_user_knowledge_base
from src.utils.exceptions import VeFRAException

app = FastAPI(
    title="VeFRA PDF Document Ingestion API", description="API to accept PDF documents."
)


@app.post("/{user_id}/uploadfile/")
async def create_upload_file(input_user_id: str, input_file: UploadFile = File(...)):
    """
    Accepts a single PDF file, saves it,
    and triggers the ingestion pipeline.
    """

    try:
        model = FileUploadModel(file=input_file, user_id=input_user_id)

        result: bool = await process_document_ingestion(model=model)
        if result:
            return JSONResponse(
                content={
                    "filename": model.file.filename,
                },
                status_code=200,
            )
        else:
            raise VeFRAException(
                "No errors were raised, but document ingestion pipeline did not succeed."
            )

    except VeFRAException as e:  # TODO move to decorator
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error(f"Error handling upload for {model.file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/{user_id}/generate/")
async def generate(user_id: str, query: str):
    """
    Generates the response to user question using user's knowledge base.
    """
    # TODO create better validation
    if ".." in user_id or "/" in user_id or "\\" in user_id:
        raise HTTPException(status_code=400, detail="Invalid user_id format.")

    db = get_user_knowledge_base(user_id=user_id)
    rag_generator = get_generator(model=MAIN_RESPONSE_GENERATION_MODEL)
    answer = ""
    try:
        answer = await asyncio.to_thread(
            answer_query, query=query, db=db, generator=rag_generator
        )
        return JSONResponse(
            content={
                "answer": answer,
            },
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error. {e}")


@app.post("/{user_id}/evaluate/")
async def evaluate(user_id: str):
    """
    Runs the evaluation of the RAG system.
    """
    try:
        df: pd.DataFrame = run_evaluation(
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
        raise HTTPException(status_code=500, detail=f"Internal server error. {e}")


# TODO remove before demo
if __name__ == "__main__":
    # uvicorn src.api.app:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
