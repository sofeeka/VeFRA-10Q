from collections import defaultdict
from pathlib import Path
from typing import Any

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseTableSerializer,
    SerializationResult,
)
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import (
    TableItem,
)
from loguru import logger
from transformers import AutoTokenizer

from src.processing.chunking.base_chunker import BaseChunker
from src.utils.config import CHUNKING_EMBEDDING_MODEL, get_user_tables_folder
from src.utils.exceptions import VeFRA_FileIOError, VeFRA_TableExtractionError


class VeFRATableSerializer(BaseTableSerializer):
    def __init__(self, user_id: str, document_name: str):
        self.user_id = user_id
        self.document_name = document_name
        self.table_dir = get_user_tables_folder(user_id=self.user_id)
        self.base_file_name = (
            Path(self.document_name).stem.replace(" ", "_").replace(".", "_")
        )
        self.page_table_counts = defaultdict(int)

    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # 1. Generate a unique table ID
        page_numbers = sorted(
            set(prov.page_no for prov in item.prov if hasattr(prov, "page_no"))
        )
        page_num = page_numbers[0] if page_numbers else -1

        table_num_on_page = self.page_table_counts[page_num]
        self.page_table_counts[page_num] += 1

        table_id = f"Table_{self.base_file_name}_p{page_num}_n{table_num_on_page}"

        # 2. Export and save the table
        try:
            table_md = item.export_to_markdown(doc=doc)
        except Exception as e:
            logger.error(
                "Failed to export table to markdown.",
                table_id=table_id,
                exc_info=True,
            )
            raise VeFRA_TableExtractionError(
                f"Failed to extract table {table_id} from document {self.document_name}"
            ) from e

        try:
            table_filepath = self.table_dir / f"{table_id}.md"
            with open(table_filepath, "w", encoding="utf-8") as f:
                f.write(table_md)
        except OSError as e:
            logger.error(
                "Failed to save table markdown to file.",
                table_id=table_id,
                filepath=str(table_filepath),
                exc_info=True,
            )
            raise VeFRA_FileIOError(
                f"Failed to save table {table_id} from document {self.document_name}"
            ) from e

        # 3. Extract context (headers and first column)
        headers_str = ""
        first_col_str = ""
        try:
            df = item.export_to_dataframe(doc=doc)
            if not df.empty:
                headers = [str(col) for col in df.columns]
                headers_str = ", ".join(headers)

                if len(df.columns) > 0:
                    first_col_values = df.iloc[:, 0].dropna().astype(str).tolist()
                    first_col_str = ", ".join(first_col_values[:5])
        except Exception as e:
            logger.warning(
                f"Could not extract headers/first column for table {table_id}: {e}"
            )

        # 4. Create the contextualized placeholder
        context_parts = []
        if headers_str:
            context_parts.append(f"Headers: {headers_str}")
        if first_col_str:
            context_parts.append(f"First Column Content: {first_col_str}")

        context_str = " \n ".join(context_parts)
        if context_str:
            reference_string = (
                f"\n\n[TABLE_REFERENCE: {table_id}] \n\n {context_str}\n\n"
            )
        else:
            reference_string = f"\n\n[TABLE_REFERENCE: {table_id}]\n\n"

        return SerializationResult(text=reference_string)


class VeFRATableSerializerProvider(ChunkingSerializerProvider):
    def __init__(self, user_id: str):
        self.user_id = user_id

    def get_serializer(self, doc: DoclingDocument):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=VeFRATableSerializer(
                user_id=self.user_id,
                document_name=doc.name,
            ),
        )


class DoclingChunker(BaseChunker):
    def chunk(self, document: DoclingDocument, user_id: str) -> list[str]:
        tokenizer: BaseTokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(CHUNKING_EMBEDDING_MODEL),
        )

        chunker = HybridChunker(
            tokenizer=tokenizer,
            serializer_provider=VeFRATableSerializerProvider(user_id=user_id),
            max_chunk_tokens=400,
            overlap_tokens=80,
        )

        chunk_iter = chunker.chunk(dl_doc=document)
        chunks = [chunk.text for chunk in chunk_iter]

        return chunks
