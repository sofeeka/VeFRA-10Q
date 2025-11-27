from qdrant_client import models

from src.retrieval.embedding.sparse_embedding_model import SparseEmbeddingModel


class TestSparseEmbeddingModel:
    """Tests for the SparseEmbeddingModel (SPLADE) implementation."""

    def test_sparse_model_initialization(self, mocker):
        """Test that the sparse embedding model initializes correctly."""
        # Mock the transformers components
        mock_tokenizer = mocker.MagicMock()
        mock_model = mocker.MagicMock()

        mocker.patch(
            "src.retrieval.embedding.sparse_embedding_model.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        )
        mocker.patch(
            "src.retrieval.embedding.sparse_embedding_model.AutoModelForMaskedLM.from_pretrained",
            return_value=mock_model,
        )

        model = SparseEmbeddingModel(model_name="test-model")

        assert model.tokenizer == mock_tokenizer
        assert model.model == mock_model
        mock_model.eval.assert_called_once()

    def test_sparse_embedding_output_format(self, mocker):
        """Test that sparse embeddings return correct SparseVector format."""
        # Mock the transformers components
        mock_tokenizer = mocker.MagicMock()
        mock_model = mocker.MagicMock()

        mocker.patch(
            "src.retrieval.embedding.sparse_embedding_model.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        )
        mocker.patch(
            "src.retrieval.embedding.sparse_embedding_model.AutoModelForMaskedLM.from_pretrained",
            return_value=mock_model,
        )

        # Mock the tokenizer and model output
        import torch

        mock_tokens = mocker.MagicMock()
        mock_tokens.input_ids = torch.tensor([[1, 2, 3]])
        mock_tokens.attention_mask = torch.tensor([[1, 1, 1]])
        mock_tokenizer.return_value = mock_tokens

        mock_output = mocker.MagicMock()
        # Create a simple logits tensor
        mock_output.logits = torch.randn(1, 3, 100)
        mock_model.return_value = mock_output

        model = SparseEmbeddingModel(model_name="test-model")
        result = model.embed(["test text"])

        # Verify output is a list of SparseVector objects
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], models.SparseVector)
        assert hasattr(result[0], "indices")
        assert hasattr(result[0], "values")
        assert isinstance(result[0].indices, list)
        assert isinstance(result[0].values, list)

    def test_sparse_embedding_multiple_texts(self, mocker):
        """Test that sparse embeddings handle multiple texts correctly."""
        # Mock the transformers components
        mock_tokenizer = mocker.MagicMock()
        mock_model = mocker.MagicMock()

        mocker.patch(
            "src.retrieval.embedding.sparse_embedding_model.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        )
        mocker.patch(
            "src.retrieval.embedding.sparse_embedding_model.AutoModelForMaskedLM.from_pretrained",
            return_value=mock_model,
        )

        # Mock the tokenizer and model output
        import torch

        mock_tokens = mocker.MagicMock()
        mock_tokens.input_ids = torch.tensor([[1, 2, 3]])
        mock_tokens.attention_mask = torch.tensor([[1, 1, 1]])
        mock_tokenizer.return_value = mock_tokens

        mock_output = mocker.MagicMock()
        mock_output.logits = torch.randn(1, 3, 100)
        mock_model.return_value = mock_output

        model = SparseEmbeddingModel(model_name="test-model")
        texts = ["text 1", "text 2", "text 3"]
        result = model.embed(texts)

        # Verify we get one SparseVector per input text
        assert len(result) == len(texts)
        for vec in result:
            assert isinstance(vec, models.SparseVector)

    def test_sparse_model_dim_property(self, mocker):
        """Test that the dim property returns vocab size."""
        mock_tokenizer = mocker.MagicMock()
        mock_model = mocker.MagicMock()
        mock_model.config.vocab_size = 30522

        mocker.patch(
            "src.retrieval.embedding.sparse_embedding_model.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        )
        mocker.patch(
            "src.retrieval.embedding.sparse_embedding_model.AutoModelForMaskedLM.from_pretrained",
            return_value=mock_model,
        )

        model = SparseEmbeddingModel(model_name="test-model")

        assert model.dim == 30522
