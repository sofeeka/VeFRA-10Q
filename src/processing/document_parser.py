import re
import logging
from pathlib import Path
from typing import Union, List, Tuple, Final
from collections import defaultdict

from docling.document_converter import DocumentConverter
from docling.datamodel.document import TableItem, TextItem
from docling_core.types.doc import DoclingDocument

from src.utils.config import SOURCE_DATA_DIR_PATH, TABLE_DIR_PATH

logger = logging.getLogger(__name__)

TABLE_REFERENCE_PATTERN: Final[re.Pattern] = re.compile(
    r"\[TABLE_REFERENCE:\s*([^\]]+)\]"
)


class DocumentParser:
    def __init__(self, converter: DocumentConverter = None):
        if converter is None:
            self.converter = DocumentConverter()

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
            return None

    def parse_documents_in_directory(self, directory_path: Union[str, Path] = SOURCE_DATA_DIR_PATH) -> List[DoclingDocument]:
        """
        Locates and parses all PDF documents in a given directory.
        """
        logger.info(f"Parsing documents in directory: {directory_path}...")

        if not isinstance(directory_path, Path):
            directory_path = Path(directory_path)

        documents: List[DoclingDocument] = []
        for file_path in directory_path.glob("*.pdf"):
            document: DoclingDocument = self.parse_document(file_path)
            documents.append(document)
        return documents

    @staticmethod
    def extract_tables_from_document(document: DoclingDocument, table_dir: Path = TABLE_DIR_PATH) -> str:
        """
        Extracts and processes all table items from a document.
        Saves tables as markdown and replaces them with a reference string.
        """
        logger.info(f"Processing tables in document {document.name}...")

        base_file_name = Path(document.name).stem.replace(
            " ", "_").replace(".", "_")

        page_table_counts = defaultdict(int)

        processed_items = []

        for item, _ in document.iterate_items():
            if isinstance(item, TextItem):
                processed_items.append(item.text.strip())

            elif isinstance(item, TableItem):

                page_numbers = sorted(
                    set(prov.page_no for prov in item.prov if hasattr(prov, "page_no"))
                )

                page_num = -1
                if page_numbers:
                    page_num = page_numbers[0]
                else:
                    logger.warning(
                        f"Could not find page number for a table in {document.name}. Defaulting to -1.")

                table_num_on_page = page_table_counts[page_num]
                page_table_counts[page_num] += 1

                table_id = (
                    f"Table_{base_file_name}_p{page_num}_n{table_num_on_page}"
                )

                try:
                    table_md = item.export_to_markdown(doc=document)
                    table_filepath = table_dir / f"{table_id}.md"

                    with open(table_filepath, "w", encoding="utf-8") as f:
                        f.write(table_md)

                except IOError as e:
                    logger.error(f"Failed to save table {table_id}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to export table {table_id}: {e}")
                    continue

                reference_string = f"\n\n[TABLE_REFERENCE: {table_id}]\n\n"
                processed_items.append(reference_string)

        logger.info(f"Finished processing {document.name}.")
        return "\n\n".join(processed_items)

    @staticmethod
    def insert_tables_into_chunk(chunk_text: str, table_dir: Path = TABLE_DIR_PATH) -> str:
        """
        Reconstructs a text chunk by replacing all table references
        with their actual Markdown content from saved files.
        """

        # nested "replacer" function that re.sub will call for every match
        def _load_table_content(match: re.Match) -> str:
            """
            This is a helper function called by re.sub.
            It receives a match object and returns the replacement string.
            """

            table_id = match.group(1).strip()
            table_filepath = table_dir / f"{table_id}.md"

            try:
                with open(table_filepath, "r", encoding="utf-8") as f:
                    table_md = f.read()
                return f"\n\n{table_md}\n\n"

            except FileNotFoundError:
                logger.warning(f"Could not find table file: {table_filepath}")
                return f"\n\n[TABLE_NOT_FOUND: {table_id}]\n\n"
            except IOError as e:
                logger.error(f"Error reading table file {table_filepath}: {e}")
                return f"\n\n[TABLE_READ_ERROR: {table_id}]\n\n"

        reconstructed_text = TABLE_REFERENCE_PATTERN.sub(
            _load_table_content,
            chunk_text
        )

        return reconstructed_text

    def prepare_document_for_chunking(self, file_path: Union[str, Path]) -> str:
        """
        Runs the full pipeline on a single document:
        parsing, table processing, and reconstruction.
        """
        logger.info(f"Running full pipeline on document: {file_path}...")

        parsed_doc = self.parse_document(file_path)
        if parsed_doc is None:
            logger.error(
                f"Document parsing failed for {file_path}. Aborting pipeline.")
            return ""

        processed_doc: str = self.extract_tables_from_document(parsed_doc)
        return processed_doc

    def prepare_documents_in_directory_for_chunking(self, directory_path: Union[str, Path] = SOURCE_DATA_DIR_PATH) -> List[Tuple[Path, str]]:
        """
        Runs the full pipeline on all documents in a directory.
        Returns a list of tuples containing file paths and their processed text.
        """
        logger.info(
            f"Running full pipeline on documents in directory: {directory_path}...")

        if not isinstance(directory_path, Path):
            directory_path = Path(directory_path)

        processed_documents = []
        for file_path in directory_path.glob("*.pdf"):
            processed_text = self.prepare_document_for_chunking(file_path)
            processed_documents.append((file_path, processed_text))

        return processed_documents

    @staticmethod
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
