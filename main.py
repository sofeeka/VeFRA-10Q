import logging

from src.retrieval.embedder import FastEmbedModel
from src.retrieval.database import QdrantDatabase
from src.generation.generator import Generator
from src.pipeline.data_ingestion import populate_database_with_docs_in_folder
from src.pipeline.query_answering import answer_query
from src.utils.config import SOURCE_DATA_DIR_PATH, TEST_DATA_DIR_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":

    db = QdrantDatabase(embedding_model=FastEmbedModel())
    # db.recreate_collection()  # TODO remove or change to create if absent
    # populate_database_with_docs_in_folder(data_dir_path=SOURCE_DATA_DIR_PATH, db=db)

    query = "What significant changes, if any, in accounting practices were reported by NVIDIA in its most recent 10-Q?"
    rag_generator = Generator()

    answer = answer_query(query=query, db=db, generator=rag_generator)
    print(answer)
