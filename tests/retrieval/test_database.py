import pytest
from qdrant_client.http import models as rest

from src.data_models.retrieval import ChunkPayload
from src.retrieval.database import UserKnowledgeBase


@pytest.mark.asyncio
async def test_add_chunks_success(
    mock_async_qdrant_client, mock_embedding_model, mock_user_id
):
    db = UserKnowledgeBase(
        user_id=mock_user_id,
        client=mock_async_qdrant_client,
        dense_embedding_model=mock_embedding_model,
        collection_name="test-collection",
    )
    chunks = [ChunkPayload(text="chunk 1"), ChunkPayload(text="chunk 2")]

    mock_embedding_model.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_async_qdrant_client.upsert.return_value = rest.UpdateResult(
        operation_id=0, status=rest.UpdateStatus.COMPLETED
    )

    await db.add_chunks(chunks)

    mock_embedding_model.embed.assert_called_once_with(["chunk 1", "chunk 2"])
    mock_async_qdrant_client.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_related_chunks_with_filter(
    mock_async_qdrant_client, mock_embedding_model, mock_user_id
):
    db = UserKnowledgeBase(
        user_id=mock_user_id,
        client=mock_async_qdrant_client,
        dense_embedding_model=mock_embedding_model,
        collection_name="test-collection",
    )

    mock_search_result = [
        rest.ScoredPoint(id="1", version=1, score=0.9, payload={"text": "found chunk"})
    ]
    mock_async_qdrant_client.search.return_value = mock_search_result

    from src.data_models.retrieval import DocumentMetadata

    doc_filter = [DocumentMetadata(year="2023", quarter="Q1")]
    points = await db.get_related_chunks(
        query="test query", doc_metadata_filter=doc_filter
    )

    chunks = [point.payload["text"] for point in points]

    assert chunks == ["found chunk"]
    mock_async_qdrant_client.search.assert_awaited_once()

    call_args, call_kwargs = mock_async_qdrant_client.search.call_args
    query_filter = call_kwargs["query_filter"]

    assert query_filter.must[0].key == "user_id"
    assert query_filter.should is not None
    assert len(query_filter.should) == 1
    assert query_filter.should[0].must[0].key == "metadata.year"
    assert query_filter.should[0].must[0].match.value == "2023"
