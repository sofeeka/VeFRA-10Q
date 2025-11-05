import re
from collections import defaultdict
from pathlib import Path
from typing import Final, List

from docling.datamodel.document import TableItem, TextItem
from docling_core.types.doc import DoclingDocument
from loguru import logger

from src.utils.config import TABLE_DIR_PATH

TABLE_REFERENCE_PATTERN: Final[re.Pattern] = re.compile(
    r"\[TABLE_REFERENCE:\s*([^\]]+)\]"
)

TABLE_DIR_PATH.mkdir(parents=True, exist_ok=True)


def process_document_for_chunking(document: DoclingDocument) -> str:
    """
    Processes a single document to extract text and extract tables preparing it for chunking.
    """
    logger.info(f"Preparing document {document.name} for chunking...")

    processed_doc: str = _extract_tables_from_document(document=document)
    return processed_doc


def process_documents_for_chunking(documents: List[DoclingDocument]) -> List[str]:
    """
    Processes documents to extract text and extract tables preparing them for chunking.
    """
    logger.info(f"Preparing {len(documents)} documents for chunking...")

    processed_docs = [process_document_for_chunking(doc) for doc in documents]
    return processed_docs


def process_chunk_after_retrieval(chunk: str) -> str:
    """
    Processes a retrieved text chunk to insert tables back into the text.
    """
    logger.info(f"Processing a retrieved chunk with length {len(chunk)}...")
    logger.info(f"\n---\n{chunk}\n---\n")

    processed_chunk = _insert_tables_into_chunk(chunk_text=chunk)
    return processed_chunk


def process_chunks_after_retrieval(chunks: List[str]) -> List[str]:
    """
    Processes retrieved text chunks to insert tables back into the text.
    """
    logger.info(f"Processing {len(chunks)} retrieved chunks...")

    processed_chunks = [process_chunk_after_retrieval(chunk) for chunk in chunks]
    return processed_chunks


def _extract_tables_from_document(
    document: DoclingDocument, table_dir: Path = TABLE_DIR_PATH
) -> str:
    """
    Extracts and processes all table items from a document.
    Saves tables as markdown and replaces them with a reference string.
    """
    logger.info(f"Processing tables in document {document.name}...")

    base_file_name = Path(document.name).stem.replace(" ", "_").replace(".", "_")

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
                    f"Could not find page number for a table in {document.name}. Defaulting to -1."
                )

            table_num_on_page = page_table_counts[page_num]
            page_table_counts[page_num] += 1

            table_id = f"Table_{base_file_name}_p{page_num}_n{table_num_on_page}"

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


def _insert_tables_into_chunk(chunk_text: str, table_dir: Path = TABLE_DIR_PATH) -> str:
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

    reconstructed_text = TABLE_REFERENCE_PATTERN.sub(_load_table_content, chunk_text)

    return reconstructed_text
