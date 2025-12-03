import json
import os
from pathlib import Path

import pandas as pd
from loguru import logger

from ..data_models.api import CSVUploadModel, FileUploadModel
from ..pipeline.data_ingestion import ingest_single_document
from ..utils.dependency import get_user_knowledge_base
from ..utils.exceptions import VeFRA_FileIOError


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
        from ..utils.config import get_user_evaluations_folder

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


def get_evaluation_results(df: pd.DataFrame) -> dict:
    """
    Extract evaluation results from the DataFrame.
    """
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
        context_coverage_scores.mean() if not context_coverage_scores.empty else -1.0
    )

    context_recall_hit_scores = df["context_recall_hit_score"].dropna()
    mean_context_recall = (
        context_recall_hit_scores.mean()
        if not context_recall_hit_scores.empty
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
        else -1.0
    )

    content = {
        "total_questions": n_total,
        "successful_evaluations": n_successful,
        "mean_answer_correctness": round(mean_correctness, 2),
        "mean_groundedness": round(mean_groundedness, 2),
        "mean_context_coverage": round(mean_context_coverage, 2),
        "context_recall_hit_score": round(mean_context_recall, 2),
        "mean_chunk_relevance": round(mean_chunk_relevance, 2),
        "mean_numerical_accuracy": round(mean_numerical_accuracy, 2),
        "evaluation_results": df.to_dict(orient="records"),
    }
    return content
