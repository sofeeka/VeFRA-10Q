import re

from src.data_models.retrieval import DocumentMetadata, Intent, RelevantDocumentsModel
from src.generation.prompts import CHOOSING_RELEVANT_DOCUMENTS_PROMPT_BASE_PROMPT
from src.utils.config import CHOOSING_RELEVANT_DOCUMENTS_MODEL, get_user_sources_folder
from src.utils.dependency import get_generator
from src.utils.exceptions import VeFRA_GenerationError, VeFRA_MetadataExtractionError

FILENAME_PATTERN = re.compile(r"(\d{4})[\s_-]+(Q[1-3])", re.IGNORECASE)


def get_output_parsed_for_relevant_document_extraction(
    question: str, user_id: str
) -> RelevantDocumentsModel:
    """ """

    try:
        doc_metadata = get_metadata_from_most_recent_user_document(user_id=user_id)
        year, quarter = doc_metadata.year, doc_metadata.quarter
    except VeFRA_MetadataExtractionError:
        year, quarter = "N/A", "N/A"

    prompt = CHOOSING_RELEVANT_DOCUMENTS_PROMPT_BASE_PROMPT.format(
        year=year,
        quarter=quarter,
        question=question,
    )

    generator = get_generator(model=CHOOSING_RELEVANT_DOCUMENTS_MODEL)

    response = generator.generate_response(
        prompt=prompt, text_format=RelevantDocumentsModel
    )

    return response.output_parsed


def get_relevant_docs(question: str, user_id: str) -> list[DocumentMetadata]:
    """
    Vaildates the user query to catch fallbacks like irrelevant question.
    If successful returns a list of pairs of years and quarters needed for answering the question.
    If unsuccessful raises an Exception.
    """

    output = get_output_parsed_for_relevant_document_extraction(
        question=question, user_id=user_id
    )

    if output.status != "success":  # == "failure"
        if output.intent == Intent.IRRELEVANT_QUESTION:
            raise VeFRA_GenerationError(
                f"This system is designed to assist people with financial analysis. Question {question} is irrelevant."
            )
        elif output.intent == Intent.K_10_FALLBACK:
            raise VeFRA_GenerationError(
                f"10 K FALLBACK triggered for question {question}"
            )
        else:
            raise

    # TODO add additional validation that if it is success then Intent has to be SPECIFIC_TIME, LATEST_DOCUMENT, or GENERAL_QUESTION.

    # at this point status == "success"
    match output.intent:
        case Intent.SPECIFIC_TIME:
            # all documents should already be available
            return [
                extract_metadata_from_string(filename=period)
                for period in output.needed_periods
            ]

        case Intent.LATEST_DOCUMENT:
            return get_metadata_from_most_recent_user_document(user_id=user_id)

        case Intent.GENERAL_QUESTION:
            # for now I return all available documents,
            # TODO but for the future maybe specify that there simply is no filter here
            return get_filenames_of_all_user_documents(user_id=user_id)

    pass


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


def get_filenames_of_all_user_documents(user_id: str) -> list[str]:
    """Gets all sorted user document filenames."""
    user_sources_folder = get_user_sources_folder(user_id=user_id)
    paths = list(user_sources_folder.glob("*.pdf"))
    paths.sort()
    return [doc.name for doc in paths]


def get_metadata_from_all_user_documents(user_id: str) -> list[DocumentMetadata]:
    """Gets metadata from ALL documents, skipping any bad filenames."""
    filenames = get_filenames_of_all_user_documents(user_id=user_id)

    metadata = []
    for filename in filenames:
        doc_metadata = extract_metadata_from_string(filename=filename)
        metadata.append(doc_metadata)

    return metadata


def get_metadata_from_most_recent_user_document(
    user_id: str,
) -> DocumentMetadata:
    """
    Efficiently gets metadata from ONLY the most recent document.
    """
    user_sources_folder = get_user_sources_folder(user_id=user_id)
    most_recent_path = max(user_sources_folder.glob("*.pdf"))

    return extract_metadata_from_string(filename=most_recent_path.name)
