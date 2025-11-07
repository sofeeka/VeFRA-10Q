import datetime
import os
from typing import List, Tuple

import pandas as pd
from evidently import DataDefinition, Dataset
from evidently.descriptors import ContextRelevance, CorrectnessLLMEval

from src.pipeline.query_answering import answer_query
from src.utils.api_key_manager import get_openai_api_key
from src.utils.config import TESTING_OPENAI_MODEL
from src.utils.dependency import get_generator, get_user_knowledge_base

os.api_key = openai_api_key = get_openai_api_key()
small_table_link = "https://docs.google.com/spreadsheets/d/1p2yTtVr-xZSpJy9Ypxgx3RL3rq_Ggb7jsZBy81QOk0E/export?format=csv&gid=0"
full_table_link = "https://docs.google.com/spreadsheets/d/1CdunoCRKYYMcVc78v8DTfhPZYdkNqRPQRlCJfPN12Fg/export?format=csv&gid=0"


def run_evaluation() -> pd.DataFrame:
    db = get_user_knowledge_base(user_id="msft")
    generator = get_generator()

    full_df = pd.read_csv(full_table_link)
    full_df.drop(
        columns=["Context"], inplace=True
    )  # TODO change when context is added to all chunks
    full_df = full_df.dropna()
    eval_df = full_df.reset_index(drop=True)

    questions = full_df["Question"]

    # (response, list of chunks)
    generation_result: List[Tuple[str, List[str]]] = [
        answer_query(query=question, db=db, generator=generator)
        for question in questions
    ]

    responses, chunks = zip(*generation_result)

    full_contexts = ["\n---\n".join(chunks_from_1_doc) for chunks_from_1_doc in chunks]

    eval_df = pd.DataFrame(
        {
            "Question": questions,
            "Contexts": full_contexts,
            "Response": responses,
        }
    )

    context_based_evals = Dataset.from_pandas(
        eval_df,
        data_definition=DataDefinition(
            text_columns=["Question", "Contexts", "Response"]
        ),
        descriptors=[
            CorrectnessLLMEval(
                column_name="Response",
                target_output="Question",
                provider="openai",
                model=TESTING_OPENAI_MODEL,
            ),
            ContextRelevance(
                "Question",
                "Context",
                output_scores=True,
                method="llm",
                method_params={"model": TESTING_OPENAI_MODEL, "provider": "openai"},
                aggregation_method="hit",
            ),
        ],
    )

    result_df = context_based_evals.as_dataframe()

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    result_df.to_csv(f"evals_{timestamp}.csv")

    return result_df
