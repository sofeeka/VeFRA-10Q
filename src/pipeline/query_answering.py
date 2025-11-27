from loguru import logger

from src.data_models.retrieval import DocumentMetadata
from src.generation.async_generator import AsyncGenerator
from src.pipeline.query_expansion import expand_query
from src.pipeline.relevant_docs_extractor import get_relevant_docs
from src.processing.document_processor import process_chunk_after_retrieval
from src.retrieval.database import UserKnowledgeBase
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


async def answer_query(
    query: str,
    db: UserKnowledgeBase,
    generator: AsyncGenerator,
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

    # Perform query expansion
    expanded_queries: list[str] = await expand_query(query=query)

    # Retrieve chunks for each query
    all_retrieved_points = []

    for query in expanded_queries:
        retrieved_points = await db.get_related_chunks(
            query=query,
            doc_metadata_filter=relevant_docs_metadata,
        )
        all_retrieved_points.extend(retrieved_points)

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
        return (
            "I could not find any information relevant to your question. Try to rephrase it and be more specific with dates and years.",
            [],
        )

    # Filter the chunks
    unique_points_dict = {point.id: point for point in all_retrieved_points}
    all_retrieved_points = list(unique_points_dict.values())

    # Reconstruct the context
    context_parts = []
    final_rebuilt_chunks = []

    for i, point in enumerate(retrieved_points):
        payload = point.payload
        raw_chunk_text = payload.get("text", "")
        metadata = payload.get("metadata", {})
        year = metadata.get("year", "N/A")
        quarter = metadata.get("quarter", "N/A")

        logger.debug(
            f"Point {i}: ",
            payload=payload,
            raw_chunk_text=raw_chunk_text,
            metadata=metadata,
        )
        rebuilt_chunk = process_chunk_after_retrieval(
            chunk=raw_chunk_text,
            user_id=db.user_id,
        )
        final_rebuilt_chunks.append(rebuilt_chunk)

        # Re-rank the chunks
        context_block = f"""
--- START OF CONTEXT CHUNK {i + 1} ---
Source Document: {year} {quarter}

Content:
{rebuilt_chunk}
--- END OF CONTEXT CHUNK {i + 1} ---
"""
        context_parts.append(context_block)

    # (context (rebuilt chunks) + user question -> Generator) + system prompt - > LLM response
    context: str = "\n---\n".join(context_parts)

    logger.info("Constructed final context for LLM.", context_length=len(context))

    user_prompt: str = f"""
    Context from 10-Q form:
    ---
    {context}
    ---
    Question: {query}
    """

    # Generate answer
    try:
        parsed_response = await generator.generate_response(prompt=user_prompt)
        response = parsed_response.output_parsed.response
        logger.success("Successfully generated and parsed response from LLM.")
        return response, final_rebuilt_chunks
    except Exception:
        logger.error(
            "Failed to generate or parse response from LLM.",
            user_id=db.user_id,
            exc_info=True,
        )
        raise
