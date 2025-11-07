from typing import List, Tuple

from src.generation.generator import Generator
from src.processing.document_processor import process_chunks_after_retrieval
from src.retrieval.database import UserKnowledgeBase


# TODO maybe create a class
def answer_query(
    query: str, db: UserKnowledgeBase, generator: Generator
) -> Tuple[str, List[str]]:
    """
    Answers a user query based on the documents in the Qdrant database.
    """

    # str -> embedding -> chunks without tables
    chunks: List[str] = db.get_related_chunks(query=query)

    # chunks without tables -> rebuilt chunks
    rebuilt_chunks: List[str] = process_chunks_after_retrieval(chunks=chunks)

    # (context (rebuilt chunks) + user question -> Generator) + system prompt - > LLM response
    context: str = "\n---\n".join(rebuilt_chunks)

    user_prompt: str = f"""
    Context from 10-Q form:
    ---
    {context}
    ---
    Question: {query}
    """

    response: str = generator.generate_response(prompt=user_prompt)
    return response, rebuilt_chunks
