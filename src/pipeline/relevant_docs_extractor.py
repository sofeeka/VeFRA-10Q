from enum import Enum
from typing import Annotated, List, Literal, Optional, Tuple

from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel, Field

from src.generation.generator import Generator
from src.generation.prompts import CHOOSING_RELEVANT_DOCUMENTS_PROMPT_BASE_PROMPT
from src.utils.config import CHOOSING_RELEVANT_DOCUMENTS_MODEL, get_user_sources_folder
from src.utils.dependency import get_generator


# TODO test if specifying the doc even helps, maybe relevance check with fallbacks would be enough for this
class Intent(str, Enum):
    K_10_FALLBACK = "K_10_FALLBACK"
    IRRELEVANT_QUESTION = "IRRELEVANT_QUESTION"
    SPECIFIC_TIME = "SPECIFIC_TIME"
    GENERAL_QUESTION = "GENERAL_QUESTION"
    LATEST_DOCUMENT = "LATEST_DOCUMENT"


class RelevantDocumentsModel(BaseModel):
    status: Literal["success", "failure"]
    intent: Optional[Intent] = None
    needed_periods: Optional[Annotated[List[str], Field(min_length=1)]] = None


def get_relevant_docs(question: str, user_id: str) -> RelevantDocumentsModel:
    """
    Vaildates the user query.
    Returns names of relevant documents if successful else or fallback reason.
    """
    year, quarter = get_metadate_from_most_recent_document(user_id=user_id)

    prompt = CHOOSING_RELEVANT_DOCUMENTS_PROMPT_BASE_PROMPT.format(
        year=year,
        quarter=quarter,
        question=question,
    )

    generator: Generator = get_generator(model=CHOOSING_RELEVANT_DOCUMENTS_MODEL)

    response: ParsedResponse = generator.generate_response(
        prompt=prompt, text_format=RelevantDocumentsModel
    )

    output = response.output_parsed

    return output


def get_metadate_from_most_recent_document(user_id: str) -> Tuple[str, str]:
    user_sources_folder = get_user_sources_folder(user_id=user_id)
    paths = list(user_sources_folder.glob("*.pdf"))
    documents = [doc.name for doc in paths]

    latest_document = documents[-1]

    parts = latest_document.split(" ")
    year = parts[0]
    quarter = parts[1]

    return (year, quarter)
