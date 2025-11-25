# VeFRA-10Q: Vention Financial Report Analyser

VeFRA-10Q is a Retrieval-Augmented Generation (RAG) system designed to answer questions from financial reports (10-Q). It leverages a vector database for efficient retrieval and Large Language Models (LLMs) for accurate answer generation.

## Architecture

The system consists of three main components:

1.  **Ingestion Pipeline**: Processes PDF documents, extracts text, chunks it, and embeds it into a vector database.
2.  **Retrieval**: Uses Qdrant as a vector database to store and retrieve relevant document chunks based on semantic similarity.
3.  **Generation**: Utilizes LLMs to generate answers based on the retrieved context and the user's query.

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

### Endpoints

*   **Upload Document**
    *   `POST /{user_id}/uploadfile/`
    *   Uploads a PDF file for a specific user, triggering the ingestion pipeline.

*   **Generate Answer**
    *   `POST /{user_id}/generate/`
    *   Generates an answer to a query using the user's knowledge base.
    *   Query parameter: `query` (string)

*   **Evaluate System**
    *   `POST /{user_id}/evaluate/`
    *   Runs the evaluation pipeline for the specified user.

## Development

To install development dependencies (including testing tools):

```bash
pip install -r requirements-dev.txt
```

To run the test suite:

```bash
pytest
```

## Evaluation

To evaluate the system's performance, you need a benchmark dataset.

1.  **Data Setup**: Place your benchmark CSV file in the `data/` directory (e.g., `data/my_benchmark.csv`).
2.  **Format**: The CSV file must contain the following columns:
    *   `Question Id`
    *   `Question`
    *   `Ground Truth Answer`
3.  **Configuration**: Update `src/utils/config.py` to point to your benchmark file. The `MSFT_BENCHMARK` will be used when running the evaluation with `user_id` msft, `NVDA_BENCHMARK` will be used with `user_id` nvda.
4.  **Execution**: You can trigger evaluation via the API endpoint `/{user_id}/evaluate/`.
