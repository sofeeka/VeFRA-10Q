import datetime
from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel, Field

from src.generation.generator import Generator
from src.generation.prompts import CHOOSING_RELEVANT_DOCUMENTS_PROMPT_BASE_PROMPT
from src.utils.config import CHOOSING_RELEVANT_DOCUMENTS_MODEL, get_user_sources_folder
from src.utils.dependency import get_generator


# TODO test if specifying the doc even helps, maybe relevance check with fallbacks would be enough for this
class FallbackReason(str, Enum):
    TIME_FRAME_UNIDENTIFIED = "TIME_FRAME_UNIDENTIFIED"
    PERIOD_NOT_COVERED = "PERIOD_NOT_COVERED"
    K_10_FALLBACK = "K_10_FALLBACK"
    FUTURE_QUESTION = "FUTURE_QUESTION"
    IRRELEVANT_QUESTION = "IRRELEVANT_QUESTION"


class RelevantDocumentsModel(BaseModel):
    status: Literal["success", "failure"]
    documents: Optional[Annotated[List[str], Field(min_length=1)]] = None
    fallback_reason: Optional[FallbackReason] = None


def get_relevant_docs(question: str, user_id: str) -> Union[List[str], FallbackReason]:
    """
    Vaildates the user query.
    Returns names of relevant documents if successful else or fallback reason.
    """

    user_sources_folder = get_user_sources_folder(user_id=user_id)
    paths = list(user_sources_folder.glob("*.pdf"))
    documents = [doc.name for doc in paths]
    current_date = datetime.datetime.now()

    prompt = CHOOSING_RELEVANT_DOCUMENTS_PROMPT_BASE_PROMPT.format(
        current_date=current_date,
        question=question,
        documents=documents,
    )

    generator: Generator = get_generator(model=CHOOSING_RELEVANT_DOCUMENTS_MODEL)

    response: ParsedResponse = generator.generate_response(
        prompt=prompt, text_format=RelevantDocumentsModel
    )

    output = response.output_parsed

    if output.status == "success":
        return output.documents
    else:
        raise Exception(f"Fallback behaviour triggered: {output.fallback}.")
