import re

from ..data_models.retrieval import DocumentMetadata, Intent, RelevantDocumentsModel
from ..generation.prompts import (
    CHOOSING_RELEVANT_DOCUMENTS_PROMPT,
    CHOOSING_RELEVANT_DOCUMENTS_SYSTEM_PROMPT,
)
from ..generation.token_counter import TokenCounter
from ..utils.config import CHOOSING_RELEVANT_DOCUMENTS_MODEL, get_user_sources_folder
from ..utils.dependency import get_async_generator
from ..utils.exceptions import VeFRA_MetadataExtractionError

FILENAME_PATTERN = re.compile(r"(\d{4})[\s_-]+(Q[1-3])", re.IGNORECASE)


async def get_output_parsed_for_relevant_document_extraction(
    question: str,
    user_id: str,
    token_counter: TokenCounter | None = None,
) -> RelevantDocumentsModel:
    """ """

    try:
        doc_metadata = get_metadata_from_most_recent_user_document(user_id=user_id)
        year, quarter = doc_metadata.year, doc_metadata.quarter
    except VeFRA_MetadataExtractionError:
        year, quarter = "N/A", "N/A"

    prompt = CHOOSING_RELEVANT_DOCUMENTS_PROMPT.format(
        year=year,
        quarter=quarter,
        question=question,
    )

    generator = get_async_generator(
        model=CHOOSING_RELEVANT_DOCUMENTS_MODEL,
        system_prompt=CHOOSING_RELEVANT_DOCUMENTS_SYSTEM_PROMPT,
        token_counter=token_counter,
    )

    response = await generator.generate_response(
        prompt=prompt, text_format=RelevantDocumentsModel
    )

    return response.output_parsed


async def get_relevant_docs(
    question: str,
    user_id: str,
    token_counter: TokenCounter | None = None,
) -> list[DocumentMetadata] | None:
    """
    Vaildates the user query to catch fallbacks like irrelevant question.
    If successful returns a list of pairs of years and quarters needed for answering the question.
    If unsuccessful raises an Exception.
    """

    output = await get_output_parsed_for_relevant_document_extraction(
        question=question,
        user_id=user_id,
        token_counter=token_counter,
    )

    match output.intent:
        case Intent.SPECIFIC_TIME:
            # all documents should already be available
            return [
                extract_metadata_from_string(filename=period)
                for period in output.needed_periods
            ]

        case Intent.LATEST_DOCUMENT:
            return [get_metadata_from_most_recent_user_document(user_id=user_id)]

        case Intent.GENERAL_QUESTION:
            return None  # fallback to searching all documents


def extract_metadata_from_string(filename: str) -> DocumentMetadata:
    """
    Parses a filename like "2022 Q3 MSFT.pdf" or plain "2022 Q3" into DocumentMetadata.
    Raise exception if it is impossible to parse
    """
    match = FILENAME_PATTERN.search(filename)

    if not match:
        raise VeFRA_MetadataExtractionError(
            f"Could not extract Year/Quarter from {filename}"
        )

    return DocumentMetadata(year=match.group(1), quarter=match.group(2).upper())


def get_metadata_from_most_recent_user_document(
    user_id: str,
) -> DocumentMetadata:
    """
    Efficiently gets metadata from ONLY the most recent document.
    """
    user_sources_folder = get_user_sources_folder(user_id=user_id)
    most_recent_path = max(user_sources_folder.glob("*.pdf"))

    return extract_metadata_from_string(filename=most_recent_path.name)
