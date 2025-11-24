from unittest.mock import MagicMock

from src.api.app import app
from src.utils.dependency import get_generator


def test_upload_file_e2e_happy_path(client, mocker, tmp_path, mock_user_id):
    # mock the service layer to prevent actual processing
    mocker.patch(
        "src.api.app.process_document_ingestion", new_callable=mocker.AsyncMock
    )

    file_path = tmp_path / "2023 Q1 TEST-USER.pdf"
    file_path.write_text("fake pdf content")

    with open(file_path, "rb") as f:
        response = client.post(
            f"/{mock_user_id}/uploadfile/",
            files={"file": ("2023 Q1 TEST-USER.pdf", f, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["filename"] == "2023 Q1 TEST-USER.pdf"


def test_generate_e2e_happy_path(client, mocker, mock_user_id):
    # mock user existence check
    mocker.patch(
        "src.api.input_validation._user_exists_in_file_system", return_value=True
    )

    # mock the full query answering pipeline for testing the API layer,
    # not the whole RAG logic here
    mocker.patch("src.api.app.answer_query", return_value=("Mocked answer", []))

    # override dependencies for this test
    app.dependency_overrides[get_generator] = lambda: MagicMock()

    response = client.post(f"/{mock_user_id}/generate/", params={"query": "test query"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Mocked answer"

    # cleanup dependency overrides
    app.dependency_overrides = {}
