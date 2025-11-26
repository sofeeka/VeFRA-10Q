from pathlib import Path

import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.exceptions import ConversionError
from docling_core.types.doc import DoclingDocument
from loguru import logger

from src.utils.exceptions import VeFRA_DocumentParsingError


class DocumentParser:
    def __init__(self, converter: DocumentConverter = None):
        if converter is None:
            # configure GPU acceleration if available
            if torch.cuda.is_available():
                logger.info(
                    f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}"
                )
                accelerator_options = AcceleratorOptions(
                    num_threads=8, device=AcceleratorDevice.CUDA
                )
            else:
                logger.info("CUDA is not available. Using CPU.")
                accelerator_options = AcceleratorOptions(
                    num_threads=8, device=AcceleratorDevice.CPU
                )

            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = accelerator_options

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        self.converter = converter

    def parse_document(self, filepath: str | Path) -> DoclingDocument:
        """
        Parses a single document from a given full file path.
        """
        filepath = Path(filepath)
        logger.info(f"Parsing document at: {filepath}...")

        try:
            result = self.converter.convert(filepath)
        except ConversionError as e:
            logger.error(f"Failed to parse document at {filepath}: {e}")
            raise VeFRA_DocumentParsingError(
                f"Failed to parse document at {filepath}: {e}"
            ) from e

        document = result.document
        if not document:
            raise VeFRA_DocumentParsingError(
                f"No parsing errors were raised, but document parsed from {filepath} is empty."
            )

        if not document.name:
            document.name = Path(filepath).name

        logger.info(f"Successfully parsed {filepath}.")
        return document

    def parse_documents_in_directory(
        self, directory_path: str | Path
    ) -> list[DoclingDocument]:
        """
        Locates and parses all PDF documents in a given directory.
        """
        logger.info(f"Parsing documents in directory: {directory_path}...")

        if not isinstance(directory_path, Path):
            directory_path = Path(directory_path)

        documents = []

        for filepath in directory_path.glob("*.pdf"):
            try:
                document = self.parse_document(filepath)
                documents.append(document)
            except VeFRA_DocumentParsingError as e:
                logger.warning(
                    f"Skipping file {filepath.name}, failed to parse: {e.message}"
                )

        logger.info(
            f"Parsing complete. Successfully parsed {len(documents)} documents."
        )
        return documents
