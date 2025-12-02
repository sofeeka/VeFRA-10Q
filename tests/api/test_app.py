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
