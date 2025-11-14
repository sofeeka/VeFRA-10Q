from enum import Enum
from typing import Annotated, List, Literal, Optional, Tuple

from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel, Field

from src.generation.generator import Generator
from src.generation.prompts import CHOOSING_RELEVANT_DOCUMENTS_PROMPT_BASE_PROMPT
from src.utils.config import CHOOSING_RELEVANT_DOCUMENTS_MODEL, get_user_sources_folder
from src.utils.dependency import get_generator


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


def get_output_parsed_for_relevant_document_extraction(
    question: str, user_id: str
) -> RelevantDocumentsModel:
    """ """

    year, quarter = get_metadata_from_most_recent_user_document(user_id=user_id)

    prompt = CHOOSING_RELEVANT_DOCUMENTS_PROMPT_BASE_PROMPT.format(
        year=year,
        quarter=quarter,
        question=question,
    )

    generator: Generator = get_generator(model=CHOOSING_RELEVANT_DOCUMENTS_MODEL)

    response: ParsedResponse = generator.generate_response(
        prompt=prompt, text_format=RelevantDocumentsModel
    )

    return response.output_parsed


def get_relevant_docs(question: str, user_id: str) -> List[Tuple[str, str]]:
    """
    Vaildates the user query to catch fallbacks like irrelevant question.
    If successful returns a list of pairs of years and quarters needed for answering the question.
    If unsuccessful raises an Exception.
    """

    output: ParsedResponse = get_output_parsed_for_relevant_document_extraction(
        question=question, user_id=user_id
    )

    if output.status != "success":  # == "failure"
        if output.intent == Intent.IRRELEVANT_QUESTION:
            raise Exception(
                f"This system is designed to assist people with financial analysis. Question {question} is irrelevant."
            )
        elif output.intent == Intent.K_10_FALLBACK:
            raise Exception(
                f"10 K FALLBACK triggered for question {question}"
            )  # TODO maybe process this for 9 months instead of 12 months, or remove this fallback

    # at this point status == "success"

    match output.intent:
        # all documents should already be available
        case Intent.SPECIFIC_TIME:
            return output.needed_periods

        case Intent.LATEST_DOCUMENT:
            return get_metadata_from_most_recent_user_document(user_id=user_id)

        case Intent.GENERAL_QUESTION:
            # for now I return all available documents,
            # but for the future maybe specify that there simply is no filter here
            return get_filenames_of_all_user_documents(user_id=user_id)

    pass


def extract_year_quarter_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Robustly parses a filename like "2022 Q3 MSFT.pdf" into (year, quarter).
    Returns None if the format is incorrect.
    """
    parts = filename.split(" ")

    if len(parts) < 2:
        raise Exception(
            f"Invalid file name {filename}. Could not parse to get metadata."
        )

    return (parts[0], parts[1])


def get_filenames_of_all_user_documents(user_id: str) -> List[str]:
    """Gets all sorted user document filenames."""
    user_sources_folder = get_user_sources_folder(user_id=user_id)
    paths = list(user_sources_folder.glob("*.pdf"))
    paths.sort()
    return [doc.name for doc in paths]


def get_metadata_from_all_user_documents(user_id: str) -> List[Tuple[str, str]]:
    """Gets metadata from ALL documents, skipping any bad filenames."""
    filenames = get_filenames_of_all_user_documents(user_id=user_id)

    metadata: List[Tuple[str, str]] = []
    for filename in filenames:
        data = extract_year_quarter_from_filename(filename=filename)
        metadata.append(data)

    return metadata


def get_metadata_from_most_recent_user_document(
    user_id: str,
) -> Optional[Tuple[str, str]]:
    """
    Efficiently gets metadata from ONLY the most recent document.
    """
    user_sources_folder = get_user_sources_folder(user_id=user_id)
    most_recent_path = max(user_sources_folder.glob("*.pdf"))

    return extract_year_quarter_from_filename(filename=most_recent_path.name)
