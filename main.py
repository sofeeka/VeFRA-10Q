import os
import logging

from src.retrieval.embedder import FastEmbedModel
from src.retrieval.database import UserKnowledgeBase
from src.generation.generator import Generator
from src.pipeline.data_ingestion import populate_database_with_docs_in_folder
from src.pipeline.query_answering import answer_query
from src.utils.config import DEFAULT_QDRANT_COLLECTION_NAME, DEFAULT_QDRANT_STORAGE_PATH

from src.dependency import get_user_knowledge_base, get_generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    user_id = 'msft'
    db: UserKnowledgeBase = get_user_knowledge_base(user_id=user_id)

    if not os.path.exists(os.path.join(DEFAULT_QDRANT_STORAGE_PATH, "collection", DEFAULT_QDRANT_COLLECTION_NAME)):
        print("recreating collection")
        # TODO: move to management. user id should not be needed to recreate collection
        db.recreate_collection()

    query = "In Q1 2023, how did Microsoft's operating expenses measure up against its revenue?"
    rag_generator = get_generator()

    answer = answer_query(query=query, db=db, generator=rag_generator)
    print(answer)
