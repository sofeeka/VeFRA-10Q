from loguru import logger

from src.data_models.retrieval import QueryExpansionModel
from src.generation.prompts import QUERY_EXPANSION_SYSTEM_PROMPT
from src.utils.config import QUERY_EXPANSION_MODEL
from src.utils.dependency import get_async_generator


async def expand_query(query: str) -> list[str]:
    logger.info("Expanding the query.", query=query)

    generator = get_async_generator(
        model=QUERY_EXPANSION_MODEL, system_prompt=QUERY_EXPANSION_SYSTEM_PROMPT
    )

    response = await generator.generate_response(
        prompt=query, text_format=QueryExpansionModel
    )

    expanded_queries = response.output_parsed.queries

    queries = [query.query for query in expanded_queries]

    return queries
