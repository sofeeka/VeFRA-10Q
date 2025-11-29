from loguru import logger
from qdrant_client.conversions.common_types import ScoredPoint

from src.data_models.retrieval import DocumentMetadata
from src.debug.debug import DebugData_Chunk, DebugData_Document, debug_manager
from src.generation.async_generator import AsyncGenerator
from src.pipeline.query_expansion import expand_query
from src.pipeline.relevant_docs_extractor import get_relevant_docs
from src.processing.document_processor import process_chunk_after_retrieval
from src.retrieval.database import UserKnowledgeBase
from src.utils.dependency import get_reranking_model
from src.utils.exceptions import VeFRA_GenerationError


async def extract_relevant_docs(user_id: str, query: str) -> list[DocumentMetadata]:
    try:
        logger.info("Extracting relevant document metadata...", query=query)

        relevant_docs_metadata = await get_relevant_docs(
            question=query, user_id=user_id
        )

        if debug_manager.is_enabled():
            for doc in relevant_docs_metadata:
                debug_manager.get_data().add_document(
                    quarter=doc.quarter, year=doc.year
                )

        if not relevant_docs_metadata:
            logger.warning("No documents identified. Searching without filter.")
            return []
        else:
            serializable = [doc.model_dump() for doc in relevant_docs_metadata]
            logger.info(
                f"{len(relevant_docs_metadata)} relevant document metadata extracted.",
                metadata=serializable,
            )
        return relevant_docs_metadata
    except VeFRA_GenerationError as e:
        logger.warning(
            f"Document extraction failed with a generation error: {e.message}"
        )
        return []
    except Exception:
        logger.error(
            "An unexpected error occurred during document metadata extraction. Proceeding without filter."
        )
        return []


async def get_retrieved_points(
    queries: list[str],
    db: UserKnowledgeBase,
    relevant_docs_metadata: list[DocumentMetadata],
) -> list[ScoredPoint]:
    all_retrieved_points: list[ScoredPoint] = []

    for query in queries:
        retrieved_points = await db.get_related_points(
            query=query,
            doc_metadata_filter=relevant_docs_metadata,
        )

        if debug_manager.is_enabled():
            chunks_per_query: list[DebugData_Chunk] = []
            for point in retrieved_points:
                metadata = point.payload.get(
                    "metadata", {"year": "N/A", "quarter": "N/A"}
                )
                year = metadata["year"]
                quarter = metadata["quarter"]
                chunk = DebugData_Chunk(
                    text=point.payload["text"],
                    document=DebugData_Document(year=year, quarter=quarter),
                )
                chunks_per_query.append(chunk)

            debug_manager.get_data().add_expanded_query(
                query=query, chunks=chunks_per_query
            )

        all_retrieved_points.extend(retrieved_points)

    logger.info(
        "Retrieved {chunk_count} chunks from database.",
        chunk_count=len(all_retrieved_points),
        query_count=len(queries),
    )

    if not all_retrieved_points:
        logger.warning(
            "No chunks found after retrieval, possibly due to document filtering.",
            relevant_docs_metadata=relevant_docs_metadata,
        )

        # TODO add debug data before returning
        return (
            "I could not find any information relevant to your question. Try to rephrase it and be more specific with dates and years.",
            [],
        )

    # Filter the chunks
    unique_points_dict = {point.id: point for point in all_retrieved_points}
    unique_retrieved_points = list(unique_points_dict.values())

    return unique_retrieved_points


def rebuild_chunks(
    retrieved_points: list[ScoredPoint],
    user_id: str,
) -> list[str]:
    context_parts = []
    rebuilt_chunks = []

    for _, point in enumerate(retrieved_points):
        payload = point.payload
        raw_chunk_text = payload.get("text", "")
        metadata = payload.get("metadata", {})
        year = metadata.get("year", "N/A")
        quarter = metadata.get("quarter", "N/A")

        rebuilt_chunk = process_chunk_after_retrieval(
            chunk=raw_chunk_text,
            user_id=user_id,
        )

        if debug_manager.is_enabled():
            debug_manager.get_data().add_unique_retrieved_chunk(
                rebuilt_chunk, quarter, year
            )

        rebuilt_chunks.append(rebuilt_chunk)

        context_block = f"""
Document: {year} {quarter}
Content:
{rebuilt_chunk}
"""
        context_parts.append(context_block)

    logger.debug("Reconstructed context parts.", context_parts=context_parts)

    return rebuilt_chunks


def rerank_chunks(
    input_query: str,
    user_id: str,
    chunks: list[str],
) -> list[str]:
    reranker = get_reranking_model()
    try:
        reranked_chunks = reranker.rerank(
            query=input_query,
            chunks=chunks,
            top_n=10,
        )
    except Exception:
        logger.error(
            "Re-ranking failed. Proceeding with original chunk order.",
            user_id=user_id,
            exc_info=True,
        )
        reranked_chunks = chunks[:10]
    return reranked_chunks


def format_for_prompt(ordered_chunks: list[str]) -> str:
    """
    Helper to join chunks into a single string for the LLM prompt.
    Adds clear separators so the LLM knows where one chunk ends.
    """
    return "\n\n---\n\n".join(ordered_chunks)


async def generate_answer(
    user_id: str,
    generator: AsyncGenerator,
    reranked_chunks: list[str],
    user_prompt: str,
) -> tuple[str, list[str]]:
    try:
        parsed_response = await generator.generate_response(prompt=user_prompt)
        response = parsed_response.output_parsed.response
        logger.success("Successfully generated and parsed response from LLM.")

        return response, reranked_chunks
    except Exception:
        logger.error(
            "Failed to generate or parse response from LLM.",
            user_id=user_id,
            exc_info=True,
        )
        raise


async def answer_query(
    input_query: str,
    db: UserKnowledgeBase,
    generator: AsyncGenerator,
    document_extraction: bool = True,
    query_expansion: bool = True,
    reranking: bool = True,
) -> tuple[str, list[str]]:
    """
    Answers a user query based on the documents in the Qdrant database.
    """

    logger.info(
        "Starting query answering pipeline.", query=input_query, user_id=db.user_id
    )

    if document_extraction:
        metadata = await extract_relevant_docs(user_id=db.user_id, query=input_query)
    else:
        metadata = []

    if query_expansion:
        queries: list[str] = await expand_query(query=input_query)
        queries.insert(0, input_query)
    else:
        queries = [input_query]

    all_retrieved_points = await get_retrieved_points(
        queries=queries, db=db, relevant_docs_metadata=metadata
    )

    chunks = rebuild_chunks(retrieved_points=all_retrieved_points, user_id=db.user_id)

    if reranking:
        reranked_chunks = rerank_chunks(
            chunks=chunks,
            input_query=input_query,
            user_id=db.user_id,
        )
    else:
        reranked_chunks = chunks[:10]

    context: str = format_for_prompt(reranked_chunks)

    logger.info("Constructed final context for LLM.", context_length=len(context))

    user_prompt: str = f"""
    Context from 10-Q form:
    ---
    {context}
    ---
    Question: {input_query}
    """

    return await generate_answer(
        user_id=db.user_id,
        generator=generator,
        reranked_chunks=reranked_chunks,
        user_prompt=user_prompt,
    )
