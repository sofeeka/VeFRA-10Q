import uuid
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


class Intent(str, Enum):
    WRONG_Q_FALLBACK = "WRONG_Q_FALLBACK"
    IRRELEVANT_QUESTION = "IRRELEVANT_QUESTION"
    SPECIFIC_TIME = "SPECIFIC_TIME"
    GENERAL_QUESTION = "GENERAL_QUESTION"
    LATEST_DOCUMENT = "LATEST_DOCUMENT"


class RelevantDocumentsModel(BaseModel):
    status: Literal["success", "failure"]
    intent: Intent | None = None
    needed_periods: Annotated[list[str], Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def check_combinations_of_fields(self) -> "RelevantDocumentsModel":
        """
        Does a final check that the combination of fields and values is valid.
        """
        if self.status == "failure":
            if self.intent not in [Intent.IRRELEVANT_QUESTION, Intent.WRONG_Q_FALLBACK]:
                raise ValueError(
                    f"For a 'failure' status, intent must be IRRELEVANT_QUESTION or WRONG_Q_FALLBACK, not {self.intent}"
                )
            if self.needed_periods:
                raise ValueError(
                    f"For a 'failure' status, 'needed_periods' must be null or empty, not {self.needed_periods}"
                )

        elif self.status == "success":
            if self.intent == Intent.SPECIFIC_TIME:
                if not self.needed_periods:
                    raise ValueError(
                        "For a 'success' status with intent SPECIFIC_TIME, 'needed_periods' must be provided."
                    )
            elif self.intent in [Intent.GENERAL_QUESTION, Intent.LATEST_DOCUMENT]:
                if self.needed_periods:
                    raise ValueError(
                        f"For a 'success' status with intent {self.intent}, 'needed_periods' must be null or empty."
                    )
            else:
                # This case catches if intent is None or an invalid enum for a success status
                raise ValueError(
                    f"Invalid intent '{self.intent}' for a 'success' status."
                )

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

    queries: list[str]


class ExpandedQuery(BaseModel):
    """
    A DTO for expanded query.
    """

    query: str
