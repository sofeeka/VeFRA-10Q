import asyncio

from src.pipeline.query_answering import QueryAnsweringPipeline
from src.scripts.setup_database import setup_database
from src.utils.config import (
    MAIN_RESPONSE_GENERATION_MODEL,
)
from src.utils.dependency import get_async_generator, get_user_knowledge_base


async def main():
    """Asynchronous main function to run the application."""
    await setup_database()

    user_id = "msft"

    query = "In Q1 2023, how did Microsoft's operating expenses measure up against its revenue?"

    query_answering_pipeline = QueryAnsweringPipeline(
        db=get_user_knowledge_base(user_id=user_id),
        generator=get_async_generator(model=MAIN_RESPONSE_GENERATION_MODEL),
    )

    answer = await query_answering_pipeline.run(query=query)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
