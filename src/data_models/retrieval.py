import uuid
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


class Intent(str, Enum):
    K_10_FALLBACK = "K_10_FALLBACK"
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
            assert self.intent in [Intent.IRRELEVANT_QUESTION, Intent.K_10_FALLBACK]
            assert not self.needed_periods

        elif self.status == "success" and self.intent == Intent.SPECIFIC_TIME:
            assert self.needed_periods

        else:
            assert self.intent in [Intent.GENERAL_QUESTION, Intent.LATEST_DOCUMENT]
            assert not self.needed_periods


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
