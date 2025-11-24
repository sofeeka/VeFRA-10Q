from unittest.mock import MagicMock

import pytest
from docling.datamodel.document import TextItem
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)

from src.processing.chunking.docling_chunker import VeFRATableSerializer
from src.utils.exceptions import VeFRA_TableExtractionError


def test_vefra_table_serializer_success(
    mock_user_id, temp_data_dir, sample_docling_document
):
    serializer = VeFRATableSerializer(
        user_id=mock_user_id, document_name="sample_doc.pdf"
    )
    table_item = [
        item
        for item, _ in sample_docling_document.iterate_items()
        if not isinstance(item, TextItem)
    ][0]

    mock_doc_serializer = MagicMock(spec=BaseDocSerializer)
    result = serializer.serialize(
        item=table_item,
        doc_serializer=mock_doc_serializer,
        doc=sample_docling_document,
    )

    assert isinstance(result, SerializationResult)

    # 1. Check placeholder format
    placeholder = result.text
    assert "[TABLE_REFERENCE: Table_sample_doc_p-1_n0]" in placeholder
    assert "Headers: Header, Value" in placeholder
    assert "First Column Content: Data, More Data" in placeholder

    # 2. Check that the table file was created
    table_dir = temp_data_dir / "tables" / mock_user_id
    expected_filepath = table_dir / "Table_sample_doc_p-1_n0.md"
    assert expected_filepath.exists()

    # 3. Check file content
    content = expected_filepath.read_text()
    assert content == "| Header | Value |\n|---|---|\n| Data | 123 |"


def test_vefra_table_serializer_export_fails(
    mock_user_id, temp_data_dir, sample_docling_document
):
    serializer = VeFRATableSerializer(
        user_id=mock_user_id, document_name="sample_doc.pdf"
    )
    table_item = [
        item
        for item, _ in sample_docling_document.iterate_items()
        if not isinstance(item, TextItem)
    ][0]
    table_item.export_to_markdown.side_effect = Exception("Markdown export failed")

    mock_doc_serializer = MagicMock(spec=BaseDocSerializer)
    with pytest.raises(VeFRA_TableExtractionError):
        serializer.serialize(
            item=table_item,
            doc_serializer=mock_doc_serializer,
            doc=sample_docling_document,
        )
