import threading

from openai.types.responses.parsed_response import ParsedResponse


class TokenCounter:
    """
    Thread-safe token counter for tracking LLM token usage.

    Tracks input tokens, output tokens, and reasoning tokens (for models like o1/o3)
    across multiple concurrent LLM calls.
    """

    def __init__(self):
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._reasoning_tokens: int = 0
        self._lock = threading.Lock()
        self._input_price: float = 0.0
        self._output_price: float = 0.0

    @staticmethod
    def get_input_token_price_per_million(model: str) -> float:
        match model:
            case "gpt-5-nano":
                return 0.05
            case "gpt-4.1-nano":
                return 0.10

    @staticmethod
    def get_output_token_price_per_million(model: str) -> float:
        match model:
            case "gpt-5-nano":
                return 0.40
            case "gpt-4.1-nano":
                return 0.40

    def add_from_response(self, response: ParsedResponse, model: str) -> None:
        """
        Extract and add token usage from an OpenAI API response.
        """
        if not hasattr(response, "usage") or response.usage is None:
            return

        usage = response.usage

        with self._lock:
            input_tokens = (
                getattr(usage, "prompt_tokens", None)
                or getattr(usage, "input_tokens", None)
                or 0
            )
            if input_tokens:
                self._input_tokens += input_tokens
                self._input_price += (
                    input_tokens
                    * self.get_input_token_price_per_million(model=model)
                    / 1_000_000
                )

            output_tokens = (
                getattr(usage, "completion_tokens", None)
                or getattr(usage, "output_tokens", None)
                or 0
            )
            if output_tokens:
                self._output_tokens += output_tokens
                self._output_price += (
                    output_tokens
                    * self.get_output_token_price_per_million(model=model)
                    / 1_000_000
                )

            if (
                hasattr(usage, "completion_tokens_details")
                and usage.completion_tokens_details
            ):
                details = usage.completion_tokens_details
                reasoning_tokens = getattr(details, "reasoning_tokens", None) or 0
                if reasoning_tokens:
                    self._reasoning_tokens += reasoning_tokens

    def get_counts(self) -> dict[str, int]:
        """
        Get current token counts in a thread-safe manner.

        Returns:
            Dictionary with input_tokens, output_tokens, reasoning_tokens, and total_tokens
        """
        with self._lock:
            return {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "reasoning_tokens": self._reasoning_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
            }

    def reset(self) -> None:
        """Reset all token counts to zero."""
        with self._lock:
            self._input_tokens = 0
            self._output_tokens = 0
            self._reasoning_tokens = 0

    def __str__(self) -> str:
        """String representation of token counts."""
        counts = self.get_counts()
        return (
            f"TokenCounter(input={counts['input_tokens']}, "
            f"output={counts['output_tokens']}, "
            f"reasoning={counts['reasoning_tokens']}, "
            f"total={counts['total_tokens']})"
        )
