from pathlib import Path

from loguru import logger

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent

# data
DATA_DIR_PATH = Path(PROJECT_ROOT_PATH, "data")

TABLE_DIR_PATH = Path(DATA_DIR_PATH, "tables")
USERS_SOURCES_ROOT_FOLDER = Path(DATA_DIR_PATH, "user_sources")
EVALUATION_DIR_PATH = Path(DATA_DIR_PATH, "evaluation_results")


def get_user_sources_folder(user_id: str, create: bool = True) -> Path:
    user_data_dir = USERS_SOURCES_ROOT_FOLDER / user_id
    if create:
        user_data_dir.mkdir(parents=True, exist_ok=True)
    return user_data_dir


def get_user_sources_filepath(user_id: str, filename: str) -> Path:
    return get_user_sources_folder(user_id=user_id) / filename


def get_user_tables_folder(user_id: str) -> Path:
    user_data_dir = TABLE_DIR_PATH / user_id
    user_data_dir.mkdir(parents=True, exist_ok=True)
    return user_data_dir


def get_user_evaluations_folder(user_id: str) -> Path:
    user_data_dir = EVALUATION_DIR_PATH / user_id
    user_data_dir.mkdir(parents=True, exist_ok=True)
    return user_data_dir


# chunking
CHUNKING_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQUENCE_LENGTH = 1024

# embedding
# DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004" # google embedding
DENSE_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DENSE_DEFAULT = "dense_default"
SPARSE_EMBEDDING_MODEL_NAME = "prithivida/Splade_PP_en_v1"
SPARSE_DEFAULT = "sparse_default"


# database
DEFAULT_QDRANT_COLLECTION_NAME = "VeFRA-10Q-Collection"
DEFAULT_QDRANT_STORAGE_PATH = Path(DATA_DIR_PATH, "./qdrant_storage")
DEFAULT_QDRANT_DISTANCE_METRIC = "Cosine"
DEFAULT_SEARCH_K = 10
CHUNKS_PER_DOC = 5

# generation
MAIN_RESPONSE_GENERATION_MODEL = "gpt-5-nano"
CHOOSING_RELEVANT_DOCUMENTS_MODEL = "gpt-4.1-nano"
EVALUATION_MODEL = "gpt-5-nano"
QUERY_EXPANSION_MODEL = "gpt-5-nano"

# reranking
RERANKING_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

# evaluation
EVALUATION_CONCURRENCY_LIMIT = 10
MSFT_FISCAL_MAP = {
    "Q3 2022": "three months ended September 30, 2022",
    "Q1 2023": "three months ended December 31, 2022",
    "Q2 2023": "three months ended March 31, 2023",
    "Q3 2023": "three months ended September 30, 2023",
}
NVDA_FISCAL_MAP = {
    "Q3 2022": "three months ended October 30, 2022",
    "Q1 2023": "three months ended April 30, 2023",
    "Q2 2023": "three months ended July 30, 2023",
    "Q3 2023": "three months ended October 29, 2023",
}


def get_fiscal_map_for_user_id(user_id: str):
    match user_id:
        case "msft":
            return MSFT_FISCAL_MAP
        case "nvda":
            return NVDA_FISCAL_MAP
        case _:
            logger.warning(
                f"User ID {user_id} is not supported, there is no known fiscal calendar mapping for this user. Proceeding without a mapping, the LLM may get confuded."
            )
            return {}


# full
MSFT_BENCHMARK = DATA_DIR_PATH / "msft_benchmark.csv"
NVDA_BENCHMARK = DATA_DIR_PATH / "nvda_benchmark.csv"

# small
# MSFT_BENCHMARK = DATA_DIR_PATH / "msft_benchmark_small.csv"
# NVDA_BENCHMARK = DATA_DIR_PATH / "nvda_benchmark_small.csv"
