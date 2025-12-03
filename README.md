# VeFRA: Vention Financial Report Analyser

VeFRA-10Q is a Retrieval-Augmented Generation (RAG) system designed to answer questions from financial reports (10-Q). It leverages a vector database for efficient retrieval and Large Language Models (LLMs) for accurate answer generation.

## Architecture & Tech Stack

The system is architected around a modular RAG pipeline exposed via a FastAPI interface.

1.  **API Layer (FastAPI)**: Manages user requests, data validation, and exposes endpoints for document upload, question-answering, and evaluation.
2.  **Ingestion Pipeline**:
    - **Parsing**: Uses `docling` to parse PDF documents, preserving structure and extracting tables.
    - **Chunking**: Employs a hybrid chunker to intelligently segment documents, with special handling for tables.
    - **Embedding**: Generates both dense (`BAAI/bge-small-en-v1.5`) and sparse (`SPLADE`) embeddings for hybrid search.
    - **Indexing**: Stores chunks and their vector representations in a **Qdrant** vector database, indexed by user ID for data isolation.
3.  **Query Answering Pipeline**:
    - **Query Pre-processing**: Checks question validity, expands the query into multiple variations, and extracts metadata (e.g., "Q1 2023") to filter relevant documents.
    - **Retrieval**: Performs a hybrid search in Qdrant using a Reciprocal Rank Fusion (RRF) strategy to combine dense and sparse search results.
    - **Re-ranking**: Uses a Cross-Encoder model (`jina-reranker`) to re-rank the retrieved chunks for maximum relevance to the query.
    - **Generation**: Constructs a detailed prompt with the re-ranked context and uses an **OpenAI** model (e.g., `gpt-5-nano`) to generate a grounded, factual answer.
4.  **Evaluation (LLM as Judge)**:
    - Runs a benchmark dataset of questions against the pipeline.
    - Uses an LLM to score the results on metrics like Answer Correctness, Groundedness, and Context Coverage.

### Prerequisites

-   Python 3.10 or higher
-   An OpenAI API Key

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

For local development and to install the package in editable mode:

```bash
pip install -e .
```

## Configuration

The system configuration is managed in `src/utils/config.py`. This file contains settings for:

*   **Paths**: Data directories, model paths, and output folders.
*   **Models**: Selection of embedding and generation models.
*   **Parameters**: Chunk sizes, overlap, and retrieval settings (e.g., `k` for top-k retrieval).

Ensure you have a `.env` file in the root directory with the necessary API keys (e.g., OpenAI API key) if using external LLM services.

## API Usage

The application exposes a FastAPI interface. To start the server, run:

```bash
uvicorn src.api.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`. On the first run, it will automatically set up the Qdrant vector database in the `data/qdrant_storage` directory.

The API uses `{user_id}` in the path to ensure workspace isolation. A `user_id` can be any alphanumeric string (e.g., `msft`, `team-alpha`).

-   #### Upload Document
    *   `POST /{user_id}/uploadfile/`
    *   Uploads a PDF file for a specific user, triggering the ingestion pipeline. The filename must follow the format `YYYY QN COMPANY.pdf` (e.g., `2023 Q1 MSFT.pdf`). The `COMPANY` part must match the `{user_id}`.
    *   **Body**: `multipart/form-data` with a `file` field.

-   #### Generate Answer
    *   `GET /{user_id}/generate/`
    *   Generates an answer to a query using the user's indexed documents.
    *   **Query Parameter**: `query` (string).

-   #### Debug Answer Generation
    *   `GET /{user_id}/debug_generate/`
    *   Runs the full pipeline for a query and returns a detailed HTML page visualizing each step: query expansion, retrieved chunks, re-ranked chunks, and the final prompt. Invaluable for development and debugging.
    *   **Query Parameter**: `query` (string).

-   #### Evaluate with Default Benchmark
    *   `GET /{user_id}/evaluate/`
    *   Runs the evaluation pipeline for the specified user using the default benchmark file configured in `src/utils/config.py`.

-   #### Evaluate with Custom Benchmark
    *   `POST /{user_id}/evaluate_file/`
    *   Runs the evaluation pipeline using a custom benchmark file uploaded by the user.
    *   **Body**: `multipart/form-data` with a `file` field containing a CSV.

## Evaluation

The system includes a robust evaluation framework to measure performance.

1.  **Benchmark Data Format**: Create a CSV file with the following required columns:
    *   `Question Id`: A unique identifier for the question.
    *   `Question`: The question to ask the system.
    *   `Answer`: The ground-truth answer.
    *   `Context`: A key piece of text from the source document that is *required* to answer the question correctly.

2.  **Execution**:
    *   **Default**: Use the `GET /{user_id}/evaluate/` endpoint to run against the pre-configured benchmark files (`data/msft_benchmark.csv` or `data/nvda_benchmark.csv`).
    *   **Custom**: Upload your own benchmark via the `POST /{user_id}/evaluate_file/` endpoint.

3.  **Output**: The evaluation endpoints return a JSON object with aggregated metrics, including:
    *   `mean_answer_correctness`: How factually correct the generated answer is compared to the ground truth.
    *   `mean_groundedness`: Whether the answer is fully supported by the provided context.
    *   `mean_context_coverage`: Whether the retrieved context was sufficient to answer the question.
    *   `context_recall_hit_score`: A binary score indicating if the required `Context` from the benchmark was present in the retrieved chunks.

## Development

### Setting up the Development Environment

Install the main package (as shown in the Installation section) and the development dependencies:

```bash
pip install -r requirements-dev.txt
```

### Running Tests

The project uses `pytest`. To run the full test suite:

```bash
pytest
```

This will automatically discover and run all tests in the `tests/` directory.
