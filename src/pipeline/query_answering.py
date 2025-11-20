from loguru import logger

from src.generation.generator import Generator
from src.pipeline.relevant_docs_extractor import get_relevant_docs
from src.processing.document_processor import process_chunks_after_retrieval
from src.retrieval.database import UserKnowledgeBase
from src.utils.exceptions import VeFRA_GenerationError


def answer_query(
    query: str, db: UserKnowledgeBase, generator: Generator
) -> tuple[str, list[str]]:
    """
    Answers a user query based on the documents in the Qdrant database.
    """

    logger.info(
        "Starting query answering pipeline.",
        query=query,
        user_id=db.user_id,
    )

    try:
        logger.info("Extracting relevant document metadata from query.")
        relevant_docs_metadata = get_relevant_docs(question=query, user_id=db.user_id)

        serializable = [doc.model_dump() for doc in relevant_docs_metadata]
        logger.info("Relevant document metadata extracted.", metadata=serializable)

        if not relevant_docs_metadata:
            logger.warning(
                "No specific documents identified. Searching across all user documents."
            )

    except VeFRA_GenerationError as e:
        logger.warning(
            f"Document extraction failed with a generation error: {e.message}"
        )
        return e.message, []
    except Exception:
        logger.error(
            "An unexpected error occurred during document metadata extraction. Proceeding without filter."
        )
        relevant_docs_metadata = None  # fallback to searching all documents

    chunks = db.get_related_chunks(
        query=query,
        doc_metadata_filter=relevant_docs_metadata,
    )

    logger.info(
        "Retrieved {chunk_count} chunks from database.", chunk_count=len(chunks)
    )

    if not chunks:
        # NEW: More informative message if no chunks are found after filtering
        logger.warning(
            "No chunks found after retrieval, possibly due to document filtering."
        )
        return (
            "I could not find any information relevant to your question. Try to rephrase it and be more specific with dates and years.",
            [],
        )

    # chunks without tables -> rebuilt chunks
    rebuilt_chunks = process_chunks_after_retrieval(chunks=chunks, user_id=db.user_id)

    # (context (rebuilt chunks) + user question -> Generator) + system prompt - > LLM response
    context: str = "\n---\n".join(rebuilt_chunks)

    logger.info("Constructed final context for LLM.", context_length=len(context))

    user_prompt: str = f"""
    Context from 10-Q form:
    ---
    {context}
    ---
    Question: {query}
    """

    try:
        parsed_response = generator.generate_response(prompt=user_prompt)
        response = parsed_response.output_parsed.response
        logger.success("Successfully generated and parsed response from LLM.")
        return response, rebuilt_chunks
    except Exception:
        logger.error(
            "Failed to generate or parse response from LLM.",
            user_id=db.user_id,
            exc_info=True,
        )
        raise
