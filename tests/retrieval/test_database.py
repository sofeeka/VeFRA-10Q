import pytest
from qdrant_client.http import models as rest

from src.data_models.retrieval import ChunkPayload
from src.retrieval.database import UserKnowledgeBase


@pytest.mark.asyncio
async def test_add_chunks_success(
    mock_async_qdrant_client,
    mock_dense_embedding_model,
    mock_sparse_embedding_model,
    mock_user_id,
):
    db = UserKnowledgeBase(
        user_id=mock_user_id,
        client=mock_async_qdrant_client,
        dense_embedding_model=mock_dense_embedding_model,
        sparse_embedding_model=mock_sparse_embedding_model,
        collection_name="test-collection",
    )
    chunks = [ChunkPayload(text="chunk 1"), ChunkPayload(text="chunk 2")]

    # Mock dense embeddings
    mock_dense_embedding_model.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    # Mock sparse embeddings
    from qdrant_client import models

    mock_sparse_embedding_model.embed.return_value = [
        models.SparseVector(indices=[10, 25], values=[0.5, 0.8]),
        models.SparseVector(indices=[15, 30], values=[0.6, 0.7]),
    ]

    mock_async_qdrant_client.upsert.return_value = rest.UpdateResult(
        operation_id=0, status=rest.UpdateStatus.COMPLETED
    )

    await db.add_chunks(chunks)

    # Verify both embedding models were called
    mock_dense_embedding_model.embed.assert_called_once_with(["chunk 1", "chunk 2"])
    mock_sparse_embedding_model.embed.assert_called_once_with(["chunk 1", "chunk 2"])
    mock_async_qdrant_client.upsert.assert_awaited_once()

    # Verify the upsert was called with both dense and sparse vectors
    call_args = mock_async_qdrant_client.upsert.call_args
    points = call_args.kwargs["points"]
    assert len(points) == 2
    # Check that each point has both dense and sparse vectors
    from src.utils.config import DENSE_DEFAULT, SPARSE_DEFAULT

    assert DENSE_DEFAULT in points[0].vector
    assert SPARSE_DEFAULT in points[0].vector


@pytest.mark.asyncio
async def test_get_related_chunks_with_filter(
    mock_async_qdrant_client,
    mock_dense_embedding_model,
    mock_sparse_embedding_model,
    mock_user_id,
):
    db = UserKnowledgeBase(
        user_id=mock_user_id,
        client=mock_async_qdrant_client,
        dense_embedding_model=mock_dense_embedding_model,
        sparse_embedding_model=mock_sparse_embedding_model,
        collection_name="test-collection",
    )

    mock_search_result = [
        rest.ScoredPoint(id="1", version=1, score=0.9, payload={"text": "found chunk"})
    ]
    # Mock query_points response for hybrid search
    query_response = type("obj", (object,), {"points": mock_search_result})()
    mock_async_qdrant_client.query_points.return_value = query_response

    from src.data_models.retrieval import DocumentMetadata

    doc_filter = [DocumentMetadata(year="2023", quarter="Q1")]
    points = await db.get_related_chunks(
        query="test query", doc_metadata_filter=doc_filter
    )

    chunks = [point.payload["text"] for point in points]

    assert chunks == ["found chunk"]
    mock_async_qdrant_client.query_points.assert_awaited()
    assert mock_async_qdrant_client.query_points.call_count == 1

    call_args, call_kwargs = mock_async_qdrant_client.query_points.call_args

    # In the new implementation, the filter is passed inside the prefetch objects
    # and NOT as a top-level argument to query_points
    prefetch = call_kwargs["prefetch"]
    assert len(prefetch) == 2  # Should have both dense and sparse prefetch

    # Check filter in the first prefetch (dense)
    query_filter = prefetch[0].filter
    assert query_filter.must[0].key == "user_id"
    assert query_filter.must[1].key == "metadata.year"
    assert query_filter.must[1].match.value == "2023"
    assert query_filter.must[2].key == "metadata.quarter"
    assert query_filter.must[2].match.value == "Q1"

    # Check filter in the second prefetch (sparse)
    query_filter_sparse = prefetch[1].filter
    assert query_filter_sparse.must[0].key == "user_id"
    assert query_filter_sparse.must[1].key == "metadata.year"
    assert query_filter_sparse.must[1].match.value == "2023"


@pytest.mark.asyncio
async def test_hybrid_search_uses_both_embeddings(
    mock_async_qdrant_client,
    mock_dense_embedding_model,
    mock_sparse_embedding_model,
    mock_user_id,
):
    """Test that hybrid search generates both dense and sparse query vectors."""
    db = UserKnowledgeBase(
        user_id=mock_user_id,
        client=mock_async_qdrant_client,
        dense_embedding_model=mock_dense_embedding_model,
        sparse_embedding_model=mock_sparse_embedding_model,
        collection_name="test-collection",
    )

    # Mock embedding responses
    mock_dense_embedding_model.embed.return_value = [[0.1, 0.2, 0.3]]

    from qdrant_client import models

    mock_sparse_embedding_model.embed.return_value = [
        models.SparseVector(indices=[10, 25], values=[0.5, 0.8])
    ]

    await db.get_search_results(query="test query")

    # Verify both embedding models were called for the query
    mock_dense_embedding_model.embed.assert_called_once()
    mock_sparse_embedding_model.embed.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_search_uses_rrf_fusion(
    mock_async_qdrant_client,
    mock_dense_embedding_model,
    mock_sparse_embedding_model,
    mock_user_id,
):
    """Test that hybrid search uses RRF (Reciprocal Rank Fusion) for combining results."""
    db = UserKnowledgeBase(
        user_id=mock_user_id,
        client=mock_async_qdrant_client,
        dense_embedding_model=mock_dense_embedding_model,
        sparse_embedding_model=mock_sparse_embedding_model,
        collection_name="test-collection",
    )

    await db.get_search_results(query="test query")

    # Verify query_points was called with FusionQuery
    call_args, call_kwargs = mock_async_qdrant_client.query_points.call_args
    query_param = call_kwargs["query"]

    # Check that it's using FusionQuery with RRF
    assert hasattr(query_param, "fusion")
    from qdrant_client.http import models as rest

    assert query_param.fusion == rest.Fusion.RRF


@pytest.mark.asyncio
async def test_add_chunks_with_empty_list(
    mock_async_qdrant_client,
    mock_dense_embedding_model,
    mock_sparse_embedding_model,
    mock_user_id,
):
    """Test that add_chunks handles empty chunk list gracefully."""
    db = UserKnowledgeBase(
        user_id=mock_user_id,
        client=mock_async_qdrant_client,
        dense_embedding_model=mock_dense_embedding_model,
        sparse_embedding_model=mock_sparse_embedding_model,
        collection_name="test-collection",
    )

    # Should not raise an error
    await db.add_chunks([])

    # Should not call embedding models or upsert
    mock_dense_embedding_model.embed.assert_not_called()
    mock_sparse_embedding_model.embed.assert_not_called()
    mock_async_qdrant_client.upsert.assert_not_awaited()
