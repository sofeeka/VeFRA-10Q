import uuid
from enum import Enum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    K_10_FALLBACK = "K_10_FALLBACK"
    IRRELEVANT_QUESTION = "IRRELEVANT_QUESTION"
    SPECIFIC_TIME = "SPECIFIC_TIME"
    GENERAL_QUESTION = "GENERAL_QUESTION"
    LATEST_DOCUMENT = "LATEST_DOCUMENT"


class RelevantDocumentsModel(BaseModel):
    status: Literal["success", "failure"]
    intent: Optional[Intent] = None
    needed_periods: Optional[Annotated[list[str], Field(min_length=1)]] = None


class ChunkPayload(BaseModel):
    """
    A Pydantic model for Qdrant payload.
    """

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
