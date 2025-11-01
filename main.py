import logging

from src.retrieval.embedder import FastEmbedModel
from src.retrieval.database import UserKnowledgeBase
from src.generation.generator import Generator
from src.pipeline.data_ingestion import populate_database_with_docs_in_folder
from src.pipeline.query_answering import answer_query
from src.utils.config import USER_SOURCE_DATA_DIR_PATH, TEST_DATA_DIR_PATH

from src.dependency import get_user_knowledge_base, get_generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    user_id = 'msft'
    db = get_user_knowledge_base(user_id=user_id)
    # db.recreate_collection()  # TODO remove or change to create if absent

    # question about 2-page testing document
    query = "What are the industry trends right now?"
    rag_generator = get_generator()

    answer = answer_query(query=query, db=db, generator=rag_generator)
    print(answer)
