from loguru import logger
from openai import OpenAI, RateLimitError
from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.data_models.generation import ResponseModel
from src.generation.base_generator import BaseGenerator


class Generator(BaseGenerator):
    def __init__(
        self,
        model: str,
        system_prompt: str,
        client: OpenAI,
    ):
        logger.info("Initializing Generator with OpenAI API.")
        self.model = model
        self.system_prompt = system_prompt
        self.client = client

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry_error_callback=lambda retry_state: logger.warning(
            f"OpenAI API call failed after {retry_state.attempt_number} attempts."
        ),
        retry=retry_if_exception_type(RateLimitError),
    )
    def generate_response(
        self,
        prompt: str,
        text_format: BaseModel = ResponseModel,
    ) -> ParsedResponse:
        """
        Generates a response to a user query using RAG.
        """

        self._log_generation_start(prompt=prompt)

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            text_format=text_format,
        )
        return response

        # return f"""Response generation stub:
        # User prompt: {user_prompt} \n ----------
        # System prompt: {self.system_prompt}"""
