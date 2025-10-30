import logging

from src.pipeline.data_ingestion import populate_database_with_docs_in_folder
from src.utils.config import SOURCE_DATA_DIR_PATH, TEST_DATA_DIR_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    populate_database_with_docs_in_folder(SOURCE_DATA_DIR_PATH)
