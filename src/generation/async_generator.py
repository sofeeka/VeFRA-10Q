from loguru import logger
from openai import AsyncOpenAI, RateLimitError
from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..data_models.generation import ResponseModel
from .base_generator import BaseGenerator


class AsyncGenerator(BaseGenerator):
    def __init__(
        self,
        model: str,
        system_prompt: str,
        client: AsyncOpenAI,
    ):
        logger.info("Initializing Async Generator with AsyncOpenAI API.")
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
    async def generate_response(
        self,
        prompt: str,
        text_format: BaseModel = ResponseModel,
        reasoning_effort: str = "medium",
    ) -> ParsedResponse:
        """
        Asynchronously generates a response to a user query using RAG.
        """

        self._log_generation_start(prompt=prompt)

        response = await self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            text_format=text_format,
            # this adds ability to configure reasoning effort for gpt-5 models (gpt-4 and lower do not accept this parameter)
            reasoning={"effort": reasoning_effort} if "5" in self.model else {},
        )
        return response

        # return f"""Response generation stub:
        # User prompt: {user_prompt} \n ----------
        # System prompt: {self.system_prompt}"""
