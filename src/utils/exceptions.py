class VeFRAException(Exception):
    """Base exception class for all VeFRA-10Q project errors."""

    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def __str__(self):
        return self.message


# --- API & Validation Errors (Client-side issues: 4xx) ---


class DataValidationError(VeFRAException):
    """For errors in input data validation (e.g., Pydantic models)."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=400, details=details)


class FileUploadError(VeFRAException):
    """Base for file upload related errors."""


class InvalidFileNameError(FileUploadError):
    """Filename does not match the expected format."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=400, details=details)


class UnsupportedFileTypeError(FileUploadError):
    """File type is not supported (e.g., not a PDF)."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=415, details=details)


class FileConflictError(FileUploadError):
    """File with the same name already exists on the server."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=409, details=details)


# --- Core Logic & External Service Errors (Server-side issues: 5xx) ---


class ConfigurationError(VeFRAException):
    """For configuration-related issues, like missing API keys."""


class FileIOError(VeFRAException):
    """For errors related to reading from or writing to the filesystem."""


class ProcessingError(VeFRAException):
    """Base for errors during the document ingestion pipeline."""


class DocumentParsingError(ProcessingError):
    """Failed to parse a document using Docling."""


class TableExtractionError(ProcessingError):
    """Failed to extract or save a table during processing."""


class ChunkingError(ProcessingError):
    """Failed to chunk a document after processing."""


class DatabaseError(VeFRAException):
    """Base for errors related to the vector database (Qdrant)."""


class CollectionSetupError(DatabaseError):
    """Failed to create or configure a Qdrant collection."""


class DataInsertionError(DatabaseError):
    """Failed to insert data (embeddings) into Qdrant."""


class SearchError(DatabaseError):
    """Failed to perform a search query in Qdrant."""


class GenerationError(VeFRAException):
    """For errors related to LLM response generation (e.g., OpenAI API call failed)."""

    def __init__(self, message: str, details: dict = None):
        # 502 Bad Gateway is appropriate when we depend on an external service that fails.
        super().__init__(message, status_code=502, details=details)
