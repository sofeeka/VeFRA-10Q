from dataclasses import dataclass

import pytest

from src.data_models.retrieval import DocumentMetadata
from src.pipeline.relevant_docs_extractor import get_relevant_docs
from src.utils.api_key_manager import get_openai_api_key

try:
    get_openai_api_key()
    API_KEY_AVAILABLE = True
except Exception:
    API_KEY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not API_KEY_AVAILABLE, reason="Requires OPENAI_API_KEY environment variable"
)


@dataclass
class DocumentExtractionTestCase:
    """
    Defines a test case for the document metadata extraction pipeline.
    - query: The input question to the system.
    - expected_documents: The list of DocumentMetadata objects the LLM should identify.
    - expected_exception: The type of exception to expect for fallback cases.
    """

    id: str
    query: str
    expected_documents: list[DocumentMetadata] | None
    expected_exception: type[Exception] | None = None


TEST_CASES = [
    DocumentExtractionTestCase(
        id="specific_time_single_doc",
        query="In Q1 2023, how did Microsoft's operating expenses measure up against its revenue?",
        expected_documents=[DocumentMetadata(year="2023", quarter="Q1")],
    ),
    DocumentExtractionTestCase(
        id="specific_time_comparison",
        query="How did revenue in Q1 2023 compare to Q1 2022?",
        expected_documents=[
            DocumentMetadata(year="2023", quarter="Q1"),
            DocumentMetadata(year="2022", quarter="Q1"),
        ],
    ),
    DocumentExtractionTestCase(
        id="latest_document_qualitative",
        query="What are the current risk factors for the company?",
        # The LLM should identify the "LATEST_DOCUMENT" intent. Our code then
        # uses the mocked filesystem call to determine the latest document.
        expected_documents=[DocumentMetadata(year="2024", quarter="Q2")],
    ),
    DocumentExtractionTestCase(
        id="general_question_broad_search",
        query="Has the company ever mentioned 'AI research' in its reports?",
        # For a general question, the pipeline should return None to search all documents.
        expected_documents=None,
    ),
]


@pytest.mark.asyncio
@pytest.mark.real_api
@pytest.mark.parametrize("case", TEST_CASES, ids=[c.id for c in TEST_CASES])
async def test_get_relevant_docs_with_real_api(
    case: DocumentExtractionTestCase, mocker
):
    """
    Tests the get_relevant_docs pipeline by making a REAL API call to the LLM.

    - It DOES NOT mock the LLM call, testing the actual prompt performance.
    - It DOES mock filesystem interactions to ensure the test is isolated and
      predictable, focusing purely on the LLM's intent parsing capability.
    """
    # We mock the filesystem access because this test's purpose is to
    # validate the LLM interaction, not the file I/O logic. This makes
    # the test deterministic regardless of what files are on disk.
    mocker.patch(
        "src.pipeline.relevant_docs_extractor.get_metadata_from_most_recent_user_document",
        return_value=DocumentMetadata(year="2024", quarter="Q2"),
    )

    user_id = "test_user_real_api"

    if case.expected_exception:
        # If we expect an exception (e.g., for an irrelevant question),
        # we use pytest.raises to assert that it occurs.
        with pytest.raises(case.expected_exception):
            await get_relevant_docs(question=case.query, user_id=user_id)
    else:
        # For successful cases, we execute the pipeline and check the result.
        result = await get_relevant_docs(question=case.query, user_id=user_id)

        if case.expected_documents is None:
            assert result is None
        else:
            assert result is not None
            # Convert to a set of tuples for order-independent comparison,
            # which is more robust for non-deterministic LLM outputs.
            expected_set = {(doc.year, doc.quarter) for doc in case.expected_documents}
            result_set = {(doc.year, doc.quarter) for doc in result}
            assert result_set == expected_set
