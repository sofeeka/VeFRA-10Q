from loguru import logger

from src.generation.generator import Generator
from src.processing.document_processor import process_chunks_after_retrieval
from src.retrieval.database import UserKnowledgeBase


# TODO maybe create a class
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

    chunks = db.get_related_chunks(query=query)
    logger.info(
        "Retrieved {chunk_count} chunks from database.", chunk_count=len(chunks)
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
