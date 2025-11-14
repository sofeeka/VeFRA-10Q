from loguru import logger
from qdrant_client import models

from src.retrieval.database_manager import QdrantCollectionManager
from src.retrieval.embedding.dense_embedding_model import FastEmbedModel
from src.utils.config import DENSE_DEFAULT
from src.utils.dependency import get_embedder, get_qdrant_collection_manager


def setup_database() -> bool:
    """ """

    logger.info("Setting the database up...")
    manager: QdrantCollectionManager = get_qdrant_collection_manager()
    dense_embedding_model: FastEmbedModel = get_embedder()
    dense_configs = {
        DENSE_DEFAULT: models.VectorParams(
            size=dense_embedding_model.dim,
            distance=models.Distance.COSINE,
        ),
        # "colbert_vectors": models.VectorParams(
        #     size=LATE_INTERACTION_DIM,
        #     distance=models.Distance.DOT,
        #     multivectoring_config=models.MultiVectorConfig(
        #         comparator=models.MultiVectorComparator.MAX_SIM
        #     ),
        # ),
    }

    # sparse_configs = {
    #     "sparse_splade": models.SparseVectorParams(
    #         index=models.SparseIndexParams(
    #             on_disk=True,  # Sparse indexes are often large
    #             full_scan_threshold=1000,
    #         )
    #     )
    # }

    created = manager.create_collection_if_not_exists(
        vectors_config=dense_configs,
        # sparse_vectors_config=sparse_configs,
    )

    if created:
        logger.info("Database is set up.")
    else:
        raise Exception("Could not set up the database.")  # TODO improve
    pass
