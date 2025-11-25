from unittest.mock import MagicMock

from src.data_models.api import FileUploadModel
from src.pipeline.data_ingestion import ingest_single_document


def test_ingest_single_document_integration(
    mocker, mock_user_id, sample_docling_document
):
    # mock the dependencies that are outside the scope of this integration test
    mock_get_chunker = mocker.patch("src.pipeline.data_ingestion.get_docling_chunker")
    mock_chunker_instance = mock_get_chunker.return_value
    # Simulate the chunker returning a list of text strings
    mock_chunker_instance.chunk.return_value = [
        "This is the first paragraph",
        "This is the second paragraph",
    ]

    mock_get_parser = mocker.patch("src.pipeline.data_ingestion.get_document_parser")
    mock_parser_instance = mock_get_parser.return_value
    mock_parser_instance.parse_document.return_value = sample_docling_document

    mock_db = MagicMock()
    mock_db.user_id = mock_user_id

    # create a fake FileUploadModel
    mock_file_upload = MagicMock(spec=FileUploadModel)
    mock_file_upload.filepath = "fake/path.pdf"

    mock_file = MagicMock()
    mock_file.filename = "2023 Q1 TEST.pdf"
    mock_file_upload.file = mock_file

    mock_file_upload.parsed_year = "2023"
    mock_file_upload.parsed_quarter = "Q1"

    ingest_single_document(model=mock_file_upload, db=mock_db)

    # check that the final step (adding to DB) was called
    mock_db.add_chunks.assert_called_once()

    # check the payload sent to the database
    _, kwargs = mock_db.add_chunks.call_args
    chunk_payloads = kwargs["chunks"]

    assert len(chunk_payloads) > 0
    first_payload = chunk_payloads[0]
    assert first_payload.metadata["year"] == "2023"
    assert first_payload.metadata["quarter"] == "Q1"
    assert "This is the first paragraph" in first_payload.text
