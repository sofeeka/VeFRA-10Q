from pathlib import Path


# data
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
DATA_DIR_PATH = Path(PROJECT_ROOT_PATH, "data")
TEST_DATA_DIR_PATH = Path(PROJECT_ROOT_PATH, "testing-data")
TABLE_DIR_PATH = Path(PROJECT_ROOT_PATH, "tables")
