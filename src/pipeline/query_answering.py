from loguru import logger

from src.data_models.retrieval import DocumentMetadata
from src.debug.debug import debug_manager
from src.generation.async_generator import AsyncGenerator
from src.pipeline.query_expansion import expand_query
from src.pipeline.relevant_docs_extractor import get_relevant_docs
from src.processing.document_processor import process_chunk_after_retrieval
from src.retrieval.database import UserKnowledgeBase
from src.utils.dependency import get_reranking_model
from src.utils.exceptions import VeFRA_GenerationError


async def extract_relevant_docs(user_id: str, query: str) -> list[DocumentMetadata]:
    try:
        logger.info("Extracting relevant document metadata from query.")
        relevant_docs_metadata = await get_relevant_docs(
            question=query, user_id=user_id
        )
        if not relevant_docs_metadata:
            logger.warning(
                "No specific documents identified. Searching across all user documents."
            )
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


def format_for_prompt(ordered_chunks: list[str]) -> str:
    """
    Helper to join chunks into a single string for the LLM prompt.
    Adds clear separators so the LLM knows where one chunk ends.
    """
    return "\n\n---\n\n".join(ordered_chunks)


async def answer_query(
    query: str,
    db: UserKnowledgeBase,
    generator: AsyncGenerator,
    debug: bool = False,
) -> tuple[str, list[str]]:
    """
    Answers a user query based on the documents in the Qdrant database.
    """

    logger.info(
        "Starting query answering pipeline.",
        query=query,
        user_id=db.user_id,
    )

    # Extract relevant documents
    relevant_docs_metadata = await extract_relevant_docs(
        user_id=db.user_id,
        query=query,
    )

    if debug_manager.is_enabled():
        debug_data = debug_manager.get_data()
        for doc in relevant_docs_metadata:
            debug_data.add_document(doc.quarter, doc.year)

    # Perform query expansion
    expanded_queries: list[str] = await expand_query(query=query)
    expanded_queries.insert(0, query)

    # Retrieve chunks for each query
    all_retrieved_points = []
    # retrieved_chunks_per_query = {}

    for exp_query in expanded_queries:
        if debug_manager.is_enabled():
            debug_manager.get_data().add_expanded_query(query=exp_query)

        retrieved_points = await db.get_related_chunks(
            query=exp_query,
            doc_metadata_filter=relevant_docs_metadata,
        )

        # if debug:
        #     retrieved_chunks_per_query[exp_query] = [
        #         RetrievedChunkDebug(
        #             id=p.id,
        #             text=p.payload.get("text", ""),
        #             score=p.score,
        #             metadata=p.payload.get("metadata", {}),
        #         )
        #         for p in retrieved_points
        #     ]

        all_retrieved_points.extend(retrieved_points)

    # if debug:
    #     debug_data["retrieved_chunks_per_query"] = retrieved_chunks_per_query

    logger.info(
        "Retrieved {chunk_count} chunks from database.",
        chunk_count=len(all_retrieved_points),
        query_count=len(expanded_queries),
    )

    if not all_retrieved_points:
        logger.warning(
            "No chunks found after retrieval, possibly due to document filtering.",
            relevant_docs_metadata=relevant_docs_metadata,
        )
        if debug:
            debug_data["unique_retrieved_chunks"] = []
            debug_data["reranked_chunks"] = []
            debug_data["final_system_prompt"] = "N/A - No chunks found"
            debug_data["final_user_prompt"] = "N/A - No chunks found"
            debug_data["final_llm_answer"] = (
                "I could not find any information relevant to your question. Try to rephrase it and be more specific with dates and years."
            )

        return (
            "I could not find any information relevant to your question. Try to rephrase it and be more specific with dates and years.",
            [],
        )

    # Filter the chunks
    unique_points_dict = {point.id: point for point in all_retrieved_points}
    unique_retrieved_points = list(unique_points_dict.values())

    # Reconstruct the context
    context_parts = []
    final_rebuilt_chunks = []

    for i, point in enumerate(unique_retrieved_points):
        payload = point.payload
        raw_chunk_text = payload.get("text", "")
        metadata = payload.get("metadata", {})
        year = metadata.get("year", "N/A")
        quarter = metadata.get("quarter", "N/A")

        rebuilt_chunk = process_chunk_after_retrieval(
            chunk=raw_chunk_text,
            user_id=db.user_id,
        )

        if debug_manager.is_enabled():
            debug_manager.get_data().add_chunk(rebuilt_chunk, quarter, year)

        final_rebuilt_chunks.append(rebuilt_chunk)

        context_block = f"""
Document: {year} {quarter}
Content:
{rebuilt_chunk}
"""
        context_parts.append(context_block)

    logger.debug("Reconstructed context parts.", context_parts=context_parts)

    # Re-rank the chunks
    reranker = get_reranking_model()
    try:
        reranked_chunks = reranker.rerank(
            query=query,
            chunks=final_rebuilt_chunks,
            top_n=10,
        )
    except Exception:
        logger.error(
            "Re-ranking failed. Proceeding with original chunk order.",
            user_id=db.user_id,
            exc_info=True,
        )
        reranked_chunks = final_rebuilt_chunks[:10]

    if debug:
        debug_data["reranked_chunks"] = reranked_chunks

    # Format the context
    context: str = format_for_prompt(reranked_chunks)

    logger.info("Constructed final context for LLM.", context_length=len(context))

    user_prompt: str = f"""
    Context from 10-Q form:
    ---
    {context}
    ---
    Question: {query}
    """

    if debug:
        debug_data["final_system_prompt"] = generator.system_prompt
        debug_data["final_user_prompt"] = user_prompt

    # Generate answer
    try:
        parsed_response = await generator.generate_response(prompt=user_prompt)
        response = parsed_response.output_parsed.response
        logger.success("Successfully generated and parsed response from LLM.")

        return response, reranked_chunks
    except Exception:
        logger.error(
            "Failed to generate or parse response from LLM.",
            user_id=db.user_id,
            exc_info=True,
        )
        raise
