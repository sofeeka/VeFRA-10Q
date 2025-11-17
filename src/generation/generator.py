from loguru import logger
from openai import OpenAI
from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel

from src.data_models.generation import ResponseModel


class Generator:
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

    def generate_response(
        self,
        prompt: str,
        text_format: BaseModel = ResponseModel,
    ) -> ParsedResponse:
        """
        Generates a response to a user query using RAG.
        """
        logger.info("Generating response using OpenAI API.")
        logger.info(f"User prompt: {prompt}")
        logger.info(f"System prompt: {self.system_prompt}")

        response: ParsedResponse = self.client.responses.parse(
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
