import asyncio

from src.pipeline.query_answering import answer_query
from src.retrieval.database import UserKnowledgeBase
from src.scripts.setup_database import setup_database
from src.utils.config import (
    MAIN_RESPONSE_GENERATION_MODEL,
)
from src.utils.dependency import get_async_generator, get_user_knowledge_base


async def main():
    """Asynchronous main function to run the application."""
    await setup_database()

    user_id = "msft"
    db: UserKnowledgeBase = get_user_knowledge_base(user_id=user_id)

    query = "In Q1 2023, how did Microsoft's operating expenses measure up against its revenue?"
    rag_generator = get_async_generator(model=MAIN_RESPONSE_GENERATION_MODEL)

    answer, _ = await answer_query(query=query, db=db, generator=rag_generator)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
