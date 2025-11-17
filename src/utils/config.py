from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent

# data
DATA_DIR_PATH = Path(PROJECT_ROOT_PATH, "data")
TABLE_DIR_PATH = Path(DATA_DIR_PATH, "tables")

USERS_SOURCES_ROOT_FOLDER = Path(DATA_DIR_PATH, "user_sources")


def get_user_sources_folder(user_id: str) -> Path:
    user_data_dir = USERS_SOURCES_ROOT_FOLDER / user_id
    user_data_dir.mkdir(parents=True, exist_ok=True)
    return user_data_dir


def get_user_sources_filepath(user_id: str, filename: str) -> Path:
    return get_user_sources_folder(user_id=user_id) / filename


def get_user_tables_folder(user_id: str) -> Path:
    user_data_dir = TABLE_DIR_PATH / user_id
    user_data_dir.mkdir(parents=True, exist_ok=True)
    return user_data_dir


# chunking
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_CHUNKING_STRATEGY = "recursive"

# embedding
# DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004" # google embedding
DENSE_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DENSE_DEFAULT = "dense_default"


# database
DEFAULT_QDRANT_COLLECTION_NAME = "VeFRA-10Q-Collection"
DEFAULT_QDRANT_STORAGE_PATH = Path(DATA_DIR_PATH, "./qdrant_storage")
DEFAULT_QDRANT_DISTANCE_METRIC = "Cosine"
DEFAULT_SEARCH_K = 10

# generation
MAIN_RESPONSE_GENERATION_MODEL = "gpt-5-nano"
CHOOSING_RELEVANT_DOCUMENTS_MODEL = "gpt-4.1-nano"
EVALUATION_MODEL = "gpt-4.1-nano"
