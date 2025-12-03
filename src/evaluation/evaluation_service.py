import asyncio

from loguru import logger

from ..data_models.evaluation import LLMJudgeScore
from ..generation.async_generator import AsyncGenerator
from ..generation.prompts import (
    ANSWER_CORRECTNESS_JUDGE_PROMPT,
    CHUNK_RELEVANCE_JUDGE_PROMPT,
    CONTEXT_COVERAGE_JUDGE_PROMPT,
    FINANCIAL_FACT_ACCURACY_JUDGE_PROMPT,
    GROUNDEDNESS_JUDGE_PROMPT,
)


class MetricsEvaluator:
    def __init__(self, generator: AsyncGenerator):
        self.generator = generator

    async def _call_judge_llm(
        self, prompt_template: str, format_kwargs: dict
    ) -> LLMJudgeScore:
        """Helper to call the LLM judge with a specific prompt template."""
        try:
            prompt = prompt_template.format(**format_kwargs)
            # Use asyncio.to_thread to run sync generator.generate_response in an async context
            parsed_response = await self.generator.generate_response(
                prompt=prompt,
                text_format=LLMJudgeScore,
            )
            judge_output: LLMJudgeScore = parsed_response.output_parsed
            return judge_output
        except Exception as e:
            logger.error(f"LLM judge call failed: {e}", exc_info=True)
            return LLMJudgeScore(score=None, reasoning=f"LLM judge failed: {str(e)}")

    async def evaluate_answer_correctness(
        self, query: str, ground_truth_answer: str, rag_response: str
    ) -> LLMJudgeScore:
        return await self._call_judge_llm(
            ANSWER_CORRECTNESS_JUDGE_PROMPT,
            {
                "query": query,
                "ground_truth_answer": ground_truth_answer,
                "rag_response": rag_response,
            },
        )

    async def evaluate_groundedness(
        self, rag_response: str, full_context: str
    ) -> LLMJudgeScore:
        return await self._call_judge_llm(
            GROUNDEDNESS_JUDGE_PROMPT,
            {"rag_response": rag_response, "full_context": full_context},
        )

    async def evaluate_context_recall_hit_rate(
        self, query: str, ground_truth_context: str, full_context: str
    ) -> LLMJudgeScore:
        hit = ground_truth_context in full_context
        return LLMJudgeScore(score=int(hit), reasoning=f"Hit: {hit}")

    async def evaluate_context_coverage(
        self, query: str, ground_truth_answer: str, full_context: str
    ) -> LLMJudgeScore:
        return await self._call_judge_llm(
            CONTEXT_COVERAGE_JUDGE_PROMPT,
            {
                "query": query,
                "ground_truth_answer": ground_truth_answer,
                "full_context": full_context,
            },
        )

    async def evaluate_chunk_relevance(
        self, query: str, retrieved_chunks: list[str]
    ) -> list[LLMJudgeScore]:
        tasks = [
            self._call_judge_llm(
                CHUNK_RELEVANCE_JUDGE_PROMPT,
                {"query": query, "chunk_text": chunk_text},
            )
            for chunk_text in retrieved_chunks
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunk_scores = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(
                    f"Chunk relevance evaluation failed for a chunk: {res}",
                    exc_info=True,
                )
                chunk_scores.append(
                    LLMJudgeScore(
                        score=None, reasoning=f"Evaluation failed: {str(res)}"
                    )
                )
            else:
                chunk_scores.append(res)  # res is already an LLMJudgeScore object

        return chunk_scores

    async def evaluate_financial_fact_accuracy(
        self, query: str, ground_truth_answer: str, rag_response: str
    ) -> LLMJudgeScore:
        return await self._call_judge_llm(
            FINANCIAL_FACT_ACCURACY_JUDGE_PROMPT,
            {
                "query": query,
                "ground_truth_answer": ground_truth_answer,
                "rag_response": rag_response,
            },
        )
