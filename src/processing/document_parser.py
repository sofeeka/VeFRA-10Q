from loguru import logger
from pathlib import Path
from typing import Union, List

from docling.document_converter import DocumentConverter
from docling.datamodel.document import TableItem, TextItem
from docling_core.types.doc import DoclingDocument


class DocumentParser:
    def __init__(self, converter: DocumentConverter = None):
        if converter is None:
            converter = DocumentConverter()
        self.converter = converter

    def parse_document(self, file_path: Union[str, Path]) -> DoclingDocument:
        """
        Parses a single document from a given full file path.
        """
        logger.info(f"Parsing document at: {file_path}...")

        try:
            result = self.converter.convert(file_path)
            document = result.document
            if not document.name:
                document.name = Path(file_path).name
            return document
        except Exception as e:
            logger.error(f"Failed to parse document at {file_path}: {e}")
            raise e

    def parse_documents_in_directory(self, directory_path: Union[str, Path]) -> List[DoclingDocument]:
        """
        Locates and parses all PDF documents in a given directory.
        """
        logger.info(f"Parsing documents in directory: {directory_path}...")

        if not isinstance(directory_path, Path):
            directory_path = Path(directory_path)

        documents: List[DoclingDocument] = []

        for file_path in directory_path.glob("*.pdf"):
            document: DoclingDocument = self.parse_document(file_path)

            if document is None:
                logger.error(f"Could not parse document at {file_path}")
                continue

            documents.append(document)

        return documents

    @staticmethod  # TODO move to utils.py
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
