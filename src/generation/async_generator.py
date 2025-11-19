from loguru import logger
from openai import AsyncOpenAI
from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel

from src.data_models.generation import ResponseModel
from src.generation.base_generator import BaseGenerator


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

    async def generate_response(
        self,
        prompt: str,
        text_format: BaseModel = ResponseModel,
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
        )
        return response

        # return f"""Response generation stub:
        # User prompt: {user_prompt} \n ----------
        # System prompt: {self.system_prompt}"""
