from loguru import logger

from src.retrieval.database_manager import QdrantCollectionManager
from src.utils.dependency import get_qdrant_collection_manager


def setup_database() -> bool:
    """ """

    logger.info("Setting the database up...")
    manager: QdrantCollectionManager = get_qdrant_collection_manager()

    created = manager.create_collection_if_not_exists()

    if created:
        logger.info("Database is set up.")
    else:
        raise Exception("Could not set up the database.")  # TODO improve
    pass
