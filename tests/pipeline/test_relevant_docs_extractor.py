import pytest

from src.data_models.retrieval import DocumentMetadata, Intent, RelevantDocumentsModel
from src.pipeline.relevant_docs_extractor import get_relevant_docs
from src.utils.exceptions import VeFRA_GenerationError


def test_get_relevant_docs_specific_time(mocker):
    mock_output = RelevantDocumentsModel(
        status="success",
        intent=Intent.SPECIFIC_TIME,
        needed_periods=["2023 Q1", "2022 Q2"],
    )
    mocker.patch(
        "src.pipeline.relevant_docs_extractor.get_output_parsed_for_relevant_document_extraction",
        return_value=mock_output,
    )

    result = get_relevant_docs(question="some query", user_id="test")

    assert result == [
        DocumentMetadata(year="2023", quarter="Q1"),
        DocumentMetadata(year="2022", quarter="Q2"),
    ]


def test_get_relevant_docs_k10_fallback(mocker):
    mock_output = RelevantDocumentsModel(
        status="failure", intent=Intent.K_10_FALLBACK, needed_periods=None
    )
    mocker.patch(
        "src.pipeline.relevant_docs_extractor.get_output_parsed_for_relevant_document_extraction",
        return_value=mock_output,
    )

    with pytest.raises(VeFRA_GenerationError, match="10 K FALLBACK triggered"):
        get_relevant_docs(question="some query", user_id="test")
