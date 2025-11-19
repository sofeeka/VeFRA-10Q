import asyncio
import datetime
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data_models.evaluation import (
    EvaluationQuestion,
    EvaluationResult,
)
from src.evaluation.evaluation_metrics import EvaluationMetrics
from src.generation.generator import Generator
from src.pipeline.query_answering import answer_query
from src.retrieval.database import UserKnowledgeBase
from src.utils.config import (
    EVALUATION_MODEL,
    EVALUATION_RESULTS_ROOT_PATH,
    MAIN_RESPONSE_GENERATION_MODEL,
    MSFT_BENCHMARK,
)
from src.utils.dependency import get_generator, get_user_knowledge_base
from src.utils.exceptions import VeFRA_EvaluationError, VeFRAException

question_id = "Question Id"
ground_truth_answer = "Ground Truth Answer"
query = "Question"


def _get_evaluation_session_filepath(user_id: str, timestamp_str: str) -> Path:
    """Generates a unique filepath for an evaluation session."""
    return (
        EVALUATION_RESULTS_ROOT_PATH / f"eval_session_{user_id}_{timestamp_str}.jsonl"
    )


def _load_or_init_results(
    session_filepath: Path, questions: list[EvaluationQuestion]
) -> tuple[list[EvaluationResult], list[EvaluationQuestion]]:
    """Loads existing results and identifies remaining questions."""
    completed_ids = set()
    results: list[EvaluationResult] = []

    if session_filepath.exists():
        logger.info(f"Resuming evaluation from {session_filepath}")
        with open(session_filepath, encoding="utf-8") as f:
            for line in f:
                try:
                    result_dict = json.loads(line)
                    result = EvaluationResult(**result_dict)
                    results.append(result)
                    completed_ids.add(result.question_id)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Corrupted line in session file: {e}. Skipping line: {line.strip()}",
                        exc_info=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Error loading evaluation result from line: {e}. Skipping line: {line.strip()}",
                        exc_info=True,
                    )

    remaining_questions = [q for q in questions if q.question_id not in completed_ids]
    logger.info(
        f"Loaded {len(results)} completed results. {len(remaining_questions)} questions remaining."
    )
    return results, remaining_questions


def _save_single_result(session_filepath: Path, result: EvaluationResult):
    """Appends a single evaluation result to the session file."""
    with open(session_filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.model_dump(), indent=None) + "\n")


async def _evaluate_single_question(
    question_data: EvaluationQuestion,
    db: UserKnowledgeBase,
    rag_generator: Generator,
    eval_metrics: EvaluationMetrics,
) -> EvaluationResult:
    logger.info(f"Starting evaluation for question_id: {question_data.question_id}")

    # Initial default result with question data
    current_result = EvaluationResult(
        question_id=question_data.question_id,
        query=question_data.query,
        ground_truth_answer=question_data.ground_truth_answer,
        rag_response="",  # Default empty
        retrieved_chunks=[],  # Default empty
        full_context="",  # Default empty
        evaluation_status="FAILED",  # Assume failure until proven success
        error_message=None,
    )

    try:
        # RAG execution: Get response and chunks
        rag_response, retrieved_chunks_list = await asyncio.to_thread(
            answer_query, query=question_data.query, db=db, generator=rag_generator
        )
        full_context = "\n---\n".join(retrieved_chunks_list)

        # Update result with RAG output
        current_result.rag_response = rag_response
        current_result.retrieved_chunks = retrieved_chunks_list
        current_result.full_context = full_context
        current_result.evaluation_status = (
            "SUCCESS"  # Mark as success for RAG generation step
        )

        # Answer Correctness (against ground truth)
        current_result.answer_correctness = (
            await eval_metrics.evaluate_answer_correctness(
                query=question_data.query,
                ground_truth_answer=question_data.ground_truth_answer,
                rag_response=rag_response,
            )
        )

        # Groundedness (response based on context)
        current_result.groundedness = await eval_metrics.evaluate_groundedness(
            rag_response=rag_response,
            full_context=full_context,
        )

        # Context Coverage (overall context sufficiency)
        current_result.context_coverage = await eval_metrics.evaluate_context_coverage(
            query=question_data.query,
            ground_truth_answer=question_data.ground_truth_answer,
            full_context=full_context,
        )

        # Chunk Relevance (per retrieved chunk)
        current_result.chunk_relevance_scores = (
            await eval_metrics.evaluate_chunk_relevance(
                question_data.query,
                retrieved_chunks_list,
            )
        )

        # Financial Numerical Accuracy
        current_result.numerical_accuracy = (
            await eval_metrics.evaluate_financial_fact_accuracy(
                query=question_data.query,
                ground_truth_answer=question_data.ground_truth_answer,
                rag_response=rag_response,
            )
        )

        logger.info(
            f"Finished RAG and all metrics for question_id: {question_data.question_id}",
            status=current_result.evaluation_status,
        )
        return current_result

    except asyncio.CancelledError:
        current_result.evaluation_status = "FAILED"
        current_result.error_message = "Evaluation cancelled."
        logger.warning(
            f"Evaluation for question {question_data.question_id} was cancelled.",
            exc_info=True,
        )
    except VeFRAException as e:
        current_result.evaluation_status = "FAILED"
        current_result.error_message = e.message
        logger.warning(
            f"VeFRAException for question {question_data.question_id}: {e.message}",
            exc_info=True,
        )
    except Exception as e:
        current_result.evaluation_status = "FAILED"
        current_result.error_message = str(e)
        logger.error(
            f"Unhandled exception for question {question_data.question_id}: {e}",
            exc_info=True,
        )

    return current_result


async def run_evaluation(user_id: str) -> pd.DataFrame:
    db = get_user_knowledge_base(user_id=user_id)

    rag_generator = get_generator(model=MAIN_RESPONSE_GENERATION_MODEL)
    eval_generator = get_generator(model=EVALUATION_MODEL)

    eval_metrics = EvaluationMetrics(generator=eval_generator)

    full_df = pd.read_csv(MSFT_BENCHMARK)

    required_cols = [question_id, query, ground_truth_answer]
    if not all(col in full_df.columns for col in required_cols):
        logger.error(
            f"Evaluation dataset must contain columns: {required_cols}. Found: {full_df.columns.tolist()}"
        )
        raise VeFRA_EvaluationError(
            f"Evaluation dataset must contain columns: {required_cols}. Found: {full_df.columns.tolist()}"
        )

    full_df = full_df.dropna(subset=required_cols)

    questions: list[EvaluationQuestion] = [
        EvaluationQuestion(
            question_id=str(row[question_id]),
            query=row[query],
            ground_truth_answer=row[ground_truth_answer],
        )
        for _, row in full_df.iterrows()
    ]
    logger.info(f"Loaded {len(questions)} evaluation questions from {MSFT_BENCHMARK}.")

    # Setup session file for persistence
    current_timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_filepath = _get_evaluation_session_filepath(user_id, current_timestamp_str)
    all_results, remaining_questions = _load_or_init_results(
        session_filepath, questions
    )

    for q_data in remaining_questions:
        result = await _evaluate_single_question(
            question_data=q_data,
            db=db,
            rag_generator=rag_generator,
            eval_metrics=eval_metrics,
        )
        all_results.append(result)
        _save_single_result(session_filepath, result)  # Save after each question

    # Aggregate results into a final DataFrame for analysis
    results_df = pd.DataFrame(
        [r.model_dump_json_optimized() for r in all_results]
    )  # Use custom method for better JSON parsing

    n_total = len(results_df)
    n_success = results_df[results_df["evaluation_status"] == "SUCCESS"].shape[0]

    mean_correctness = results_df["answer_correctness_score"].dropna().mean()
    mean_groundedness = results_df["groundedness_score"].dropna().mean()
    mean_context_coverage = results_df["context_coverage_score"].dropna().mean()

    logger.info(
        f"Evaluation completed. {n_success}/{n_total} questions processed successfully."
    )

    if pd.notna(mean_correctness):
        logger.info(f"Mean Answer Correctness: {mean_correctness:.2f}")
    if pd.notna(mean_groundedness):
        logger.info(f"Mean Groundedness: {mean_groundedness:.2f}")
    if pd.notna(mean_context_coverage):
        logger.info(f"Mean Context Coverage: {mean_context_coverage:.2f}")

    final_results_csv_path = (
        EVALUATION_RESULTS_ROOT_PATH
        / f"final_eval_results_{user_id}_{current_timestamp_str}.csv"
    )
    results_df.to_csv(final_results_csv_path, index=False)
    logger.info(f"Final evaluation results (CSV) saved to {final_results_csv_path}")

    final_results_jsonl_path = (
        EVALUATION_RESULTS_ROOT_PATH
        / f"final_eval_results_{user_id}_{current_timestamp_str}.jsonl"
    )
    with open(final_results_jsonl_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(r.model_dump_json() + "\n")
    logger.info(f"Final evaluation results (JSONL) saved to {final_results_jsonl_path}")

    return results_df
