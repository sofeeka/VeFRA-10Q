# %%
import logging
from pathlib import Path
from typing import Union

from docling.document_converter import DocumentConverter
from docling.datamodel.document import TableItem, TextItem
from docling_core.types.doc import DoclingDocument

from utils.config import DATA_DIR_PATH

logger = logging.getLogger(__name__)

converter = DocumentConverter()


def print_items_from_document(document: DoclingDocument):
    """
    Utility function to print all text and table items from a document.
    """
    logger.info(f"Printing items from document {document.name}...")

    for item, _ in document.iterate_items():
        if isinstance(item, TableItem):
            print("--- [TABLE START] ---")
            print(item.export_to_markdown(doc=document))
            print("--- [TABLE END] ---")
        elif isinstance(item, TextItem):
            print(item.text.strip())
        else:
            print(f"Unknown item type: {type(item)}")


def parse_document(file_path: Union[str, Path]) -> DoclingDocument:
    """
        Parses a single document from a given full file path.
    """
    logger.info(f"Parsing document at: {file_path}...")

    result = converter.convert(file_path)
    document = result.document
    return document


def parse_documents_in_directory(directory_path: Union[str, Path] = DATA_DIR_PATH):
    """
    Locates and parses all PDF documents in a given directory.
    """
    logger.info(f"Parsing documents in directory: {directory_path}...")

    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    documents = []
    for file_path in directory_path.glob("*.pdf"):
        document = parse_document(file_path)
        documents.append(document)
    return documents


# %% :
testing_path = Path(DATA_DIR_PATH.parent, "testing-data")
table = parse_document(Path(testing_path, "table.pdf"))
text = parse_document(Path(testing_path, "text.pdf"))
table_text = parse_document(Path(testing_path, "table_text.pdf"))
# %%
print_items_from_document(table)
# %%
print_items_from_document(text)
# %%
print_items_from_document(table_text)
