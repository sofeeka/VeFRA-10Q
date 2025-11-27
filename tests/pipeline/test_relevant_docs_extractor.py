import pytest

from src.data_models.retrieval import DocumentMetadata, Intent, RelevantDocumentsModel
from src.pipeline.relevant_docs_extractor import get_relevant_docs
from src.utils.exceptions import VeFRA_GenerationError


@pytest.mark.asyncio
async def test_get_relevant_docs_specific_time(mocker):
    mock_output = RelevantDocumentsModel(
        status="success",
        intent=Intent.SPECIFIC_TIME,
        needed_periods=["2023 Q1", "2022 Q2"],
    )
    mock_async_func = mocker.patch(
        "src.pipeline.relevant_docs_extractor.get_output_parsed_for_relevant_document_extraction",
        return_value=mock_output,
    )
    mock_async_func.return_value = mock_output

    result = await get_relevant_docs(question="some query", user_id="test")

    assert result == [
        DocumentMetadata(year="2023", quarter="Q1"),
        DocumentMetadata(year="2022", quarter="Q2"),
    ]


@pytest.mark.asyncio
async def test_get_relevant_docs_wrong_q_fallback(mocker):
    mock_output = RelevantDocumentsModel(
        status="failure", intent=Intent.WRONG_Q_FALLBACK, needed_periods=None
    )
    mock_async_func = mocker.patch(
        "src.pipeline.relevant_docs_extractor.get_output_parsed_for_relevant_document_extraction",
        return_value=mock_output,
    )
    mock_async_func.return_value = mock_output

    with pytest.raises(VeFRA_GenerationError, match="WRONG Q FALLBACK triggered"):
        await get_relevant_docs(question="some query", user_id="test")
