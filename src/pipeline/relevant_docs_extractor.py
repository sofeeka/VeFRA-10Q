from typing import Optional

from openai.types.responses.parsed_response import ParsedResponse

from src.data_models.retrieval import Intent, RelevantDocumentsModel
from src.generation.generator import Generator
from src.generation.prompts import CHOOSING_RELEVANT_DOCUMENTS_PROMPT_BASE_PROMPT
from src.utils.config import CHOOSING_RELEVANT_DOCUMENTS_MODEL, get_user_sources_folder
from src.utils.dependency import get_generator
from src.utils.exceptions import GenerationError


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


def get_relevant_docs(question: str, user_id: str) -> list[tuple[str, str]]:
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
            raise GenerationError(
                f"This system is designed to assist people with financial analysis. Question {question} is irrelevant."
            )
        elif output.intent == Intent.K_10_FALLBACK:
            raise GenerationError(f"10 K FALLBACK triggered for question {question}")
        else:
            raise

    # TODO add additional validation that if it is success then Intent has to be SPECIFIC_TIME, LATEST_DOCUMENT, or GENERAL_QUESTION.

    # at this point status == "success"
    match output.intent:
        case Intent.SPECIFIC_TIME:
            # all documents should already be available
            return output.needed_periods

        case Intent.LATEST_DOCUMENT:
            return get_metadata_from_most_recent_user_document(user_id=user_id)

        case Intent.GENERAL_QUESTION:
            # for now I return all available documents,
            # TODO but for the future maybe specify that there simply is no filter here
            return get_filenames_of_all_user_documents(user_id=user_id)

    pass


def extract_year_quarter_from_filename(filename: str) -> Optional[tuple[str, str]]:
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


def get_filenames_of_all_user_documents(user_id: str) -> list[str]:
    """Gets all sorted user document filenames."""
    user_sources_folder = get_user_sources_folder(user_id=user_id)
    paths = list(user_sources_folder.glob("*.pdf"))
    paths.sort()
    return [doc.name for doc in paths]


def get_metadata_from_all_user_documents(user_id: str) -> list[tuple[str, str]]:
    """Gets metadata from ALL documents, skipping any bad filenames."""
    filenames = get_filenames_of_all_user_documents(user_id=user_id)

    metadata: list[tuple[str, str]] = []
    for filename in filenames:
        data = extract_year_quarter_from_filename(filename=filename)
        metadata.append(data)

    return metadata


def get_metadata_from_most_recent_user_document(
    user_id: str,
) -> Optional[tuple[str, str]]:
    """
    Efficiently gets metadata from ONLY the most recent document.
    """
    user_sources_folder = get_user_sources_folder(user_id=user_id)
    most_recent_path = max(user_sources_folder.glob("*.pdf"))

    return extract_year_quarter_from_filename(filename=most_recent_path.name)
