import uuid
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator


class Intent(str, Enum):
    SPECIFIC_TIME = "SPECIFIC_TIME"
    GENERAL_QUESTION = "GENERAL_QUESTION"
    LATEST_DOCUMENT = "LATEST_DOCUMENT"


class QuestionValidity(str, Enum):
    RELEVANT = "RELEVANT"
    IRRELEVANT = "IRRELEVANT"


class QuestionValidityModel(BaseModel):
    """
    Model for checking if a question is relevant to 10-Q documents.
    """

    validity: QuestionValidity


class RelevantDocumentsModel(BaseModel):
    """
    Model for metadata extraction from input query.
    """

    intent: Intent
    needed_periods: Annotated[list[str], Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def check_combinations_of_fields(self) -> "RelevantDocumentsModel":
        """
        Does a final check that the combination of fields and values is valid.
        """

        if self.intent == Intent.SPECIFIC_TIME:
            if not self.needed_periods:
                raise ValueError(
                    "For intent SPECIFIC_TIME, 'needed_periods' must be provided."
                )
        elif self.intent in [Intent.GENERAL_QUESTION, Intent.LATEST_DOCUMENT]:
            if self.needed_periods:
                raise ValueError(
                    f"For intent {self.intent}, 'needed_periods' must be null or empty."
                )
        else:
            raise ValueError(f"Invalid intent '{self.intent}'.")

        return self


class ChunkPayload(BaseModel):
    """
    A Pydantic model for Qdrant payload.
    """

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class DocumentMetadata(BaseModel):
    """
    A DTO for document metadata, specifically year and quarter.
    """

    year: str  # YYYY
    quarter: str  # Q1, Q2, Q3


class QueryExpansionModel(BaseModel):
    """
    A DTO for query expansion.
    """

    reworded_query: str
    queries: list[str]


class ExpandedQuery(BaseModel):
    """
    A DTO for expanded query.
    """

    query: str
