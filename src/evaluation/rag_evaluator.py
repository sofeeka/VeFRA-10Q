import asyncio
import datetime
import json
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm.asyncio import tqdm

from ..data_models.evaluation import (
    EvaluationQuestion,
    EvaluationResult,
)
from ..debug.debug import debug_manager
from ..generation.prompts import EVALUATION_SYSTEM_PROMPT
from ..pipeline.query_answering import QueryAnsweringPipeline
from ..utils.config import (
    EVALUATION_CONCURRENCY_LIMIT,
    EVALUATION_MODEL,
    MAIN_RESPONSE_GENERATION_MODEL,
    MSFT_BENCHMARK,
    NVDA_BENCHMARK,
    get_user_evaluations_folder,
)
from ..utils.dependency import (
    get_async_generator,
    get_user_knowledge_base,
)
from ..utils.exceptions import VeFRA_EvaluationError, VeFRAException
from .evaluation_service import MetricsEvaluator

question_id = "Question Id"
ground_truth_answer = "Answer"
question = "Question"
ground_truth_context = "Context"


def _get_evaluation_session_filepath(user_id: str, timestamp_str: str) -> Path:
    """Generates a unique filepath for an evaluation session."""
    return (
        get_user_evaluations_folder(user_id=user_id)
        / f"eval_session_{user_id}_{timestamp_str}.jsonl"
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
    metrics_evaluator: MetricsEvaluator,
    pipeline: QueryAnsweringPipeline,
) -> EvaluationResult:
    logger.info(f"Starting evaluation for question_id: {question_data.question_id}")

    # Initial default result with question data
    current_result = EvaluationResult(
        question_id=question_data.question_id,
        query=question_data.question,
        ground_truth_answer=question_data.ground_truth_answer,
        rag_response="",
        retrieved_chunks=[],
        full_context="",
        evaluation_status="FAILED",
        error_message=None,
    )

    try:
        debug_manager.enable()
        debug_manager.clear_data()

        debug_data = debug_manager.get_data()
        debug_data.user = pipeline.db.user_id
        debug_data.question = question_data.question

        rag_response = await pipeline.run(question_data.question)
        retrieved_chunks_list = debug_data.final_retrieved_chunks
        full_context = "\n---\n".join(retrieved_chunks_list)

        current_result.rag_response = rag_response
        current_result.retrieved_chunks = retrieved_chunks_list
        current_result.full_context = full_context
        current_result.evaluation_status = "SUCCESS"

        metric_tasks = {
            "answer_correctness": metrics_evaluator.evaluate_answer_correctness(
                query=question_data.question,
                ground_truth_answer=question_data.ground_truth_answer,
                rag_response=rag_response,
            ),
            "groundedness": metrics_evaluator.evaluate_groundedness(
                rag_response=rag_response, full_context=full_context
            ),
            "context_coverage": metrics_evaluator.evaluate_context_coverage(
                query=question_data.question,
                ground_truth_answer=question_data.ground_truth_answer,
                full_context=full_context,
            ),
            "context_recall_hit": metrics_evaluator.evaluate_context_recall_hit_rate(
                query=question_data.question,
                ground_truth_context=question_data.ground_truth_context,
                full_context=full_context,
            ),
            "chunk_relevance_scores": metrics_evaluator.evaluate_chunk_relevance(
                question_data.query, retrieved_chunks_list
            ),
            "numerical_accuracy": metrics_evaluator.evaluate_financial_fact_accuracy(
                query=question_data.query,
                ground_truth_answer=question_data.ground_truth_answer,
                rag_response=rag_response,
            ),
        }

        results = await asyncio.gather(*metric_tasks.values(), return_exceptions=True)

        # Map results back
        results_map = dict(zip(metric_tasks.keys(), results))

        for key, value in results_map.items():
            if isinstance(value, Exception):
                error_msg = f"Metric calculation '{key}' failed: {value}"
                logger.error(error_msg, question_id=question_data.question_id)
                # We can still proceed, the metric will have a `None` score
            else:
                setattr(current_result, key, value)

        current_result.evaluation_status = "SUCCESS"
        logger.success(
            f"Finished evaluation for question_id: {question_data.question_id}"
        )

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
    finally:
        debug_manager.disable()

    return current_result


async def run_evaluation(
    user_id: str, csv_filepath: str | Path | None = None
) -> pd.DataFrame:
    """
    Runs the full evaluation pipeline concurrently.

    Args:
        user_id: The user ID to run evaluation for
        csv_filepath: Optional path to uploaded CSV file. If not provided, uses default benchmarks.
    """
    logger.info(f"Starting evaluation run for user '{user_id}'.")
    start_time = datetime.datetime.now()

    db = get_user_knowledge_base(user_id=user_id)
    rag_async_generator = get_async_generator(model=MAIN_RESPONSE_GENERATION_MODEL)
    eval_async_generator = get_async_generator(
        model=EVALUATION_MODEL,
        system_prompt=EVALUATION_SYSTEM_PROMPT,
    )

    metrics_evaluator = MetricsEvaluator(generator=eval_async_generator)

    # Determine which CSV file to use
    if csv_filepath:
        benchmark = Path(csv_filepath)
        logger.info(f"Using uploaded CSV file: {benchmark}")
    else:
        # Use default benchmarks based on user_id
        if db.user_id == "msft":
            benchmark = MSFT_BENCHMARK
        else:
            benchmark = NVDA_BENCHMARK
        logger.info(f"Using default benchmark: {benchmark}")

    full_df = pd.read_csv(benchmark)

    required_cols = [question_id, question, ground_truth_answer, ground_truth_context]
    if not all(col in full_df.columns for col in required_cols):
        raise VeFRA_EvaluationError(
            f"Evaluation dataset must contain columns: {required_cols}. Found: {full_df.columns.tolist()}"
        )

    full_df = full_df.dropna(subset=required_cols)

    questions: list[EvaluationQuestion] = [
        EvaluationQuestion(
            question_id=str(row[question_id]),
            question=row[question],
            ground_truth_answer=row[ground_truth_answer],
            ground_truth_context=row.get(ground_truth_context, None),
        )
        for _, row in full_df.iterrows()
    ]
    logger.info(f"Loaded {len(questions)} evaluation questions from {benchmark}.")

    # Setup session file for persistence
    current_timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_filepath = _get_evaluation_session_filepath(user_id, current_timestamp_str)
    all_results, remaining_questions = _load_or_init_results(
        session_filepath, questions
    )

    if not remaining_questions:
        logger.info("No remaining questions to evaluate. Session is already complete.")
    else:
        semaphore = asyncio.Semaphore(EVALUATION_CONCURRENCY_LIMIT)
        pipeline = QueryAnsweringPipeline(db=db, generator=rag_async_generator)

        async def evaluate_and_save(q_data: EvaluationQuestion) -> EvaluationResult:
            async with semaphore:
                result = await _evaluate_single_question(
                    question_data=q_data,
                    metrics_evaluator=metrics_evaluator,
                    pipeline=pipeline,
                )
                _save_single_result(session_filepath, result)
                return result

        tasks = [evaluate_and_save(q) for q in remaining_questions]

        logger.info(
            f"Processing {len(tasks)} questions with concurrency limit {EVALUATION_CONCURRENCY_LIMIT}..."
        )

        processed_results = await tqdm.gather(*tasks, desc="Evaluating Questions")
        all_results.extend(processed_results)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logger.info(f"Evaluation run finished in {duration.total_seconds():.2f} seconds.")

    if not all_results:
        logger.warning("No results to process. Returning empty DataFrame.")
        return pd.DataFrame()

    results_df = pd.DataFrame([r.model_dump_json_optimized() for r in all_results])

    n_total = len(results_df)
    n_success = results_df[results_df["evaluation_status"] == "SUCCESS"].shape[0]

    mean_correctness = results_df["answer_correctness_score"].dropna().mean()
    mean_groundedness = results_df["groundedness_score"].dropna().mean()
    mean_context_coverage = results_df["context_coverage_score"].dropna().mean()
    mean_context_recall_hit = results_df["context_recall_hit_score"].dropna().mean()

    logger.info(
        f"Evaluation completed. {n_success}/{n_total} questions processed successfully."
    )

    if pd.notna(mean_correctness):
        logger.info(f"Mean Answer Correctness: {mean_correctness:.2f}")
    if pd.notna(mean_groundedness):
        logger.info(f"Mean Groundedness: {mean_groundedness:.2f}")
    if pd.notna(mean_context_coverage):
        logger.info(f"Mean Context Coverage: {mean_context_coverage:.2f}")
    if pd.notna(mean_context_recall_hit):
        logger.info(f"Mean Context Recall Hit: {mean_context_recall_hit:.2f}")

    final_results_csv_path = (
        get_user_evaluations_folder(user_id=user_id)
        / f"final_eval_results_{user_id}_{current_timestamp_str}.csv"
    )
    results_df.to_csv(final_results_csv_path, index=False)
    logger.success(f"Final evaluation results (CSV) saved to {final_results_csv_path}")

    logger.info(f"Final evaluation results (JSONL) saved to {session_filepath}")
    return results_df
