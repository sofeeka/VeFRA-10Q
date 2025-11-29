from loguru import logger

from src.data_models.retrieval import QueryExpansionModel
from src.generation.prompts import (
    QUERY_EXPANSION_SYSTEM_PROMPT,
    QUERY_EXPANSION_USER_PROMPT,
)
from src.utils.config import QUERY_EXPANSION_MODEL
from src.utils.dependency import get_async_generator


async def expand_query(query: str) -> list[str]:
    logger.info(f"Expanding the query: {query}", query=query)

    generator = get_async_generator(
        model=QUERY_EXPANSION_MODEL, system_prompt=QUERY_EXPANSION_SYSTEM_PROMPT
    )

    response = await generator.generate_response(
        prompt=QUERY_EXPANSION_USER_PROMPT.format(user_query=query),
        text_format=QueryExpansionModel,
    )
    logger.debug("Query expansion response received.", response=response.model_dump())

    expanded_queries = response.output_parsed.queries
    logger.debug("Expanded queries extracted.", expanded_queries=expanded_queries)
    return [str(q) for q in expanded_queries]
