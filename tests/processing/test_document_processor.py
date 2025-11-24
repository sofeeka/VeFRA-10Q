from src.processing.document_processor import process_chunk_after_retrieval


def test_process_chunk_after_retrieval_success(mock_user_id, temp_data_dir):
    # Create a fake table file in our temporary directory
    table_dir = temp_data_dir / "tables" / mock_user_id
    table_file = table_dir / "Table_doc1_p1_n0.md"
    table_content = "| Header | Value |\n|---|---|\n| Data | 123 |"
    table_file.write_text(table_content)

    chunk_with_ref = "Some text before. [TABLE_REFERENCE: Table_doc1_p1_n0]. Contexts: Headers: ... Some text after."

    reconstructed_chunk = process_chunk_after_retrieval(chunk_with_ref, mock_user_id)

    assert f"\n\n{table_content}\n\n" in reconstructed_chunk
    assert "[TABLE_REFERENCE:" not in reconstructed_chunk


def test_process_chunk_with_missing_table(mock_user_id, temp_data_dir):
    chunk_with_bad_ref = "Text with [TABLE_REFERENCE: non_existent_table]."

    reconstructed_chunk = process_chunk_after_retrieval(
        chunk_with_bad_ref, mock_user_id
    )

    assert "[TABLE_NOT_FOUND: non_existent_table]" in reconstructed_chunk
