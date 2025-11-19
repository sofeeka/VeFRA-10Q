import datetime
import json

from pydantic import BaseModel, Field


class EvaluationQuestion(BaseModel):
    """
    Represents a single question in the evaluation dataset.
    """

    question_id: str = Field(..., description="Unique identifier for the question")
    query: str = Field(..., description="The user's original query to the RAG system")
    ground_truth_answer: str = Field(
        ..., description="The expert-provided correct answer"
    )
    ground_truth_context_chunks: list[str] | None = Field(
        None,
        description="Optional: list of ground truth relevant context chunks (if available for precise retrieval evaluation)",
    )
    expected_metadata: dict[str, str] | None = Field(
        None,
        description="Optional: Expected document metadata (e.g., {'year': '2023', 'quarter': 'Q1'}) to evaluate retrieval intent parsing",
    )


class LLMJudgeScore(BaseModel):
    """
    Standard format for an LLM judge's score and reasoning.
    """

    score: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Numerical score from 0.0 to 1.0, or None if evaluation failed",
    )
    reasoning: str | None = Field(
        None, description="Detailed reasoning for the given score"
    )


class EvaluationResult(BaseModel):
    """
    Represents the complete result for a single evaluated question.
    """

    question_id: str
    query: str
    ground_truth_answer: str

    # RAG output
    rag_response: str
    retrieved_chunks: list[str]  # The raw chunks before table insertion
    full_context: str  # fully processed context ready for LLM call

    # Core RAG Metrics (using LLM as Judge)
    answer_correctness: LLMJudgeScore = Field(default_factory=LLMJudgeScore)
    groundedness: LLMJudgeScore = Field(default_factory=LLMJudgeScore)

    context_coverage: LLMJudgeScore = Field(default_factory=LLMJudgeScore)
    chunk_relevance_scores: list[LLMJudgeScore] | None = None

    numerical_accuracy: LLMJudgeScore = Field(
        default_factory=LLMJudgeScore
    )  # TODO think about it

    evaluation_status: str = "SUCCESS"  # SUCCESS, FAILED, SKIPPED
    error_message: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

    def model_dump_json_optimized(self):
        """Custom dump to flatten nested LLMJudgeScore for Pandas DataFrames."""
        data = self.model_dump()
        for key, value in data.copy().items():
            if isinstance(value, dict) and "score" in value and "reasoning" in value:
                data[f"{key}_score"] = value["score"]
                data[f"{key}_reasoning"] = value["reasoning"]
                del data[key]  # Remove original dict field
            elif (
                isinstance(value, list)
                and value
                and all(isinstance(item, dict) and "score" in item for item in value)
            ):
                # flattenning chunk_relevance_scores for simplicity
                data[key] = json.dumps(value)  # Store as JSON string in DF cell
        return data
