from unittest.mock import MagicMock

import pytest
from qdrant_client.http import models as rest

from src.data_models.retrieval import DocumentMetadata
from src.retrieval.database import UserKnowledgeBase
from src.utils.config import CHUNKS_PER_DOC


@pytest.mark.asyncio
async def test_get_search_results_with_metadata_filter_chunks_per_doc(
    mock_async_qdrant_client,
    mock_dense_embedding_model,
    mock_sparse_embedding_model,
    mock_user_id,
):
    """
    Test that when doc_metadata_filter is provided, we query for each document
    with limit=CHUNKS_PER_DOC.
    """
    db = UserKnowledgeBase(
        user_id=mock_user_id,
        client=mock_async_qdrant_client,
        dense_embedding_model=mock_dense_embedding_model,
        sparse_embedding_model=mock_sparse_embedding_model,
        collection_name="test-collection",
    )

    # Mock embeddings
    mock_dense_embedding_model.embed.return_value = [[0.1, 0.2, 0.3]]

    # Mock sparse embeddings
    from qdrant_client import models

    mock_sparse_embedding_model.embed.return_value = [
        models.SparseVector(indices=[10, 25], values=[0.5, 0.8])
    ]

    # Mock query_points response
    # We expect 2 calls, so we provide 2 responses
    mock_search_result_1 = [
        rest.ScoredPoint(
            id="1", version=1, score=0.9, payload={"text": "chunk 1 doc 1"}
        )
    ]
    mock_search_result_2 = [
        rest.ScoredPoint(
            id="2", version=1, score=0.8, payload={"text": "chunk 1 doc 2"}
        )
    ]

    query_response_1 = type("obj", (object,), {"points": mock_search_result_1})()
    query_response_2 = type("obj", (object,), {"points": mock_search_result_2})()

    mock_async_qdrant_client.query_points.side_effect = [
        query_response_1,
        query_response_2,
    ]

    doc_filter = [
        DocumentMetadata(year="2023", quarter="Q1"),
        DocumentMetadata(year="2022", quarter="Q3"),
    ]

    results = await db.get_search_results(
        query="test query", doc_metadata_filter=doc_filter
    )

    # Verify results are aggregated
    assert len(results) == 2
    assert results[0].payload["text"] == "chunk 1 doc 1"
    assert results[1].payload["text"] == "chunk 1 doc 2"

    # Verify query_points was called twice
    assert mock_async_qdrant_client.query_points.call_count == 2

    # Verify call arguments for the first call
    call_args_1 = mock_async_qdrant_client.query_points.call_args_list[0]
    _, call_kwargs_1 = call_args_1

    assert call_kwargs_1["limit"] == CHUNKS_PER_DOC

    prefetch_1 = call_kwargs_1["prefetch"]
    assert prefetch_1[0].limit == CHUNKS_PER_DOC
    assert prefetch_1[1].limit == CHUNKS_PER_DOC

    # Verify filter for first document
    filter_1 = prefetch_1[0].filter
    assert filter_1.must[1].key == "metadata.year"
    assert filter_1.must[1].match.value == "2023"
    assert filter_1.must[2].key == "metadata.quarter"
    assert filter_1.must[2].match.value == "Q1"

    # Verify call arguments for the second call
    call_args_2 = mock_async_qdrant_client.query_points.call_args_list[1]
    _, call_kwargs_2 = call_args_2

    assert call_kwargs_2["limit"] == CHUNKS_PER_DOC

    prefetch_2 = call_kwargs_2["prefetch"]
    # Verify filter for second document
    filter_2 = prefetch_2[0].filter
    assert filter_2.must[1].key == "metadata.year"
    assert filter_2.must[1].match.value == "2022"
    assert filter_2.must[2].key == "metadata.quarter"
    assert filter_2.must[2].match.value == "Q3"


@pytest.mark.asyncio
async def test_get_search_results_without_metadata_filter(
    mock_async_qdrant_client,
    mock_dense_embedding_model,
    mock_sparse_embedding_model,
    mock_user_id,
):
    """
    Test that when doc_metadata_filter is NOT provided, we use the default logic
    (single query with provided limit).
    """
    db = UserKnowledgeBase(
        user_id=mock_user_id,
        client=mock_async_qdrant_client,
        dense_embedding_model=mock_dense_embedding_model,
        sparse_embedding_model=mock_sparse_embedding_model,
        collection_name="test-collection",
    )

    # Mock embeddings
    mock_dense_embedding_model.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_sparse_embedding_model.embed.return_value = [MagicMock()]  # Mock sparse vector

    # Mock query_points response
    mock_search_result = [
        rest.ScoredPoint(id="1", version=1, score=0.9, payload={"text": "chunk 1"})
    ]
    query_response = type("obj", (object,), {"points": mock_search_result})()
    mock_async_qdrant_client.query_points.return_value = query_response

    results = await db.get_search_results(
        query="test query", limit=5, doc_metadata_filter=None
    )

    assert len(results) == 1
    assert mock_async_qdrant_client.query_points.call_count == 1

    _, call_kwargs = mock_async_qdrant_client.query_points.call_args
    assert call_kwargs["limit"] == 5
