from pathlib import Path
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import TableItem, TextItem
from fastapi.testclient import TestClient
from openai.types.responses.parsed_response import ParsedResponse
from qdrant_client import QdrantClient  # TODO maybe change to AsyncQdrantClient
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.http import models as rest

from src.api.app import app
from src.data_models.generation import ResponseModel
from src.retrieval.embedding.dense_embedding_model import DenseEmbeddingModel


@pytest.fixture(scope="session")
def test_assets_dir():
    return Path(__file__).parent / "assets"


@pytest.fixture
def mock_user_id():
    return "test-user"


@pytest.fixture
def temp_data_dir(tmp_path, monkeypatch, mock_user_id):
    """
    Creates a temporary data directory for tests and uses monkeypatch
    to redirect all config functions (get_user_..._folder) to use it.
    This isolates filesystem operations during tests.
    """
    test_data_root = tmp_path / "data"
    test_data_root.mkdir()

    # Create user-specific subdirectories
    (test_data_root / "user_sources" / mock_user_id).mkdir(parents=True)
    (test_data_root / "tables" / mock_user_id).mkdir(parents=True)
    (test_data_root / "evaluation_results" / mock_user_id).mkdir(parents=True)

    # Monkeypatch the config functions to point to our temp directory
    monkeypatch.setattr(
        "src.utils.config.USERS_SOURCES_ROOT_FOLDER", test_data_root / "user_sources"
    )
    monkeypatch.setattr("src.utils.config.TABLE_DIR_PATH", test_data_root / "tables")
    monkeypatch.setattr(
        "src.utils.config.EVALUATION_DIR_PATH", test_data_root / "evaluation_results"
    )
    monkeypatch.setattr(
        "src.utils.config.DEFAULT_QDRANT_STORAGE_PATH",
        test_data_root / "qdrant_storage",
    )

    yield test_data_root

    # Teardown: clean up the directory (handled by tmp_path)


@pytest.fixture
def mock_openai_client(mocker):
    """Mocks the OpenAI client to avoid actual API calls."""
    mock_client = mocker.MagicMock()
    # Mock the response generation
    mock_parsed_response = ParsedResponse(
        output_parsed=ResponseModel(response="This is a mock LLM response."),
        http_response=mocker.MagicMock(),
    )
    mock_client.responses.parse.return_value = mock_parsed_response
    return mock_client


@pytest.fixture
def mock_qdrant_client(mocker):
    """Mocks the Qdrant client to avoid creating a real database."""
    return mocker.MagicMock(spec=QdrantClient)


@pytest.fixture
def mock_async_qdrant_client(mocker):
    """Mocks the AsyncQdrantClient."""
    client = mocker.AsyncMock()
    # Mock the return value of an awaited call
    client.upsert.return_value = mocker.MagicMock(status=rest.UpdateStatus.COMPLETED)
    client.search.return_value = [
        ScoredPoint(id="1", version=1, score=0.9, payload={"text": "chunk 1"})
    ]
    return client


@pytest.fixture
def mock_embedding_model(mocker):
    """Mocks the dense embedding model."""
    mock_model = mocker.MagicMock(spec=DenseEmbeddingModel)
    mock_model.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_model.dim = 3
    return mock_model


@pytest.fixture
def sample_docling_document():
    """Creates a sample DoclingDocument for testing purposes."""
    doc = MagicMock(spec=DoclingDocument)
    doc.name = "sample_doc.pdf"

    # Add some text items
    text_item1 = TextItem(
        self_ref="#/texts/1",
        label="text",
        orig="This is the first paragraph.",
        text="This is the first paragraph.",
    )

    text_item2 = TextItem(
        self_ref="#/texts/2",
        label="text",
        orig="This is the second paragraph before the table.",
        text="This is the second paragraph before the table.",
    )

    # Add a table item
    table_item = MagicMock(spec=TableItem)
    table_item.prov = []  # Keep it simple for unit tests
    table_item.export_to_markdown.return_value = (
        "| Header | Value |\n|---|---|\n| Data | 123 |"
    )

    # Mock export_to_dataframe for contextualization
    import pandas as pd

    mock_df = pd.DataFrame({"Header": ["Data", "More Data"], "Value": [123, 456]})
    table_item.export_to_dataframe.return_value = mock_df

    text_item3 = TextItem(
        self_ref="#/texts/3",
        label="text",
        orig="This is text after the table.",
        text="This is text after the table.",
    )

    children = [text_item1, text_item2, table_item, text_item3]

    doc.iterate_items.return_value = [(item, []) for item in children]
    return doc


@pytest.fixture(scope="module")
def client():
    """Provides a FastAPI TestClient for E2E tests."""
    with TestClient(app) as c:
        yield c
