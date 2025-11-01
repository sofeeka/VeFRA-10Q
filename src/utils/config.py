from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent

# data
DATA_DIR_PATH = Path(PROJECT_ROOT_PATH, "data")

USER_SOURCE_DATA_DIR_PATH = Path(DATA_DIR_PATH, "user_sources")
TEST_DATA_DIR_PATH = Path(DATA_DIR_PATH, "testing")
TABLE_DIR_PATH = Path(DATA_DIR_PATH, "tables")


# chunking
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_CHUNKING_STRATEGY = "recursive"

# embedding
# DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004" # google embedding
FAST_EMBED_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# database
DEFAULT_QDRANT_COLLECTION_NAME = "VeFRA-10Q-Collection"
DEFAULT_QDRANT_STORAGE_PATH = Path(DATA_DIR_PATH, "./qdrant_storage")
DEFAULT_QDRANT_DISTANCE_METRIC = "Cosine"
DEFAULT_SEARCH_K = 10

# generation
TESTING_OPENAI_MODEL = "gpt-5-nano"
