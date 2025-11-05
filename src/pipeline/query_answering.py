from src.generation.generator import Generator
from src.processing.document_processor import process_chunks_after_retrieval
from src.retrieval.database import UserKnowledgeBase


def answer_query(query: str, db: UserKnowledgeBase, generator: Generator) -> str:
    """
    Answers a user query based on the documents in the Qdrant database.
    """

    # str -> embedding -> chunks without tables
    chunks = db.get_related_chunks(query=query)

    # chunks without tables -> rebuilt chunks
    rebuilt_chunks = process_chunks_after_retrieval(chunks=chunks)

    # system prompt + query + rebuilt chunks -> LLM response
    context_str = "\n---\n".join(rebuilt_chunks)

    user_prompt = f"""
    Context from 10-Q form:
    ---
    {context_str}
    ---
    Question: {query}
    """

    response = generator.generate_response(user_prompt=user_prompt)
    return response
