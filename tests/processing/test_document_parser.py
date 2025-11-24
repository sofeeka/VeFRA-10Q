from unittest.mock import MagicMock

import pytest
from docling.exceptions import ConversionError

from src.processing.document_parser import DocumentParser
from src.utils.exceptions import VeFRA_DocumentParsingError


def test_parse_document_success(mocker):
    mock_converter = MagicMock()
    mock_result = MagicMock()

    mock_doc = MagicMock()
    mock_doc.name = "Fake DoclingDocument"
    mock_result.document = mock_doc

    mock_converter.convert.return_value = mock_result

    parser = DocumentParser(converter=mock_converter)
    filepath = "dummy/path.pdf"

    result = parser.parse_document(filepath)

    assert result.name == "Fake DoclingDocument"
    mock_converter.convert.assert_called_once()


def test_parse_document_conversion_error(mocker):
    mock_converter = MagicMock()
    mock_converter.convert.side_effect = ConversionError("PDF processing failed")

    parser = DocumentParser(converter=mock_converter)
    filepath = "dummy/path.pdf"

    with pytest.raises(VeFRA_DocumentParsingError, match="Failed to parse document"):
        parser.parse_document(filepath)
