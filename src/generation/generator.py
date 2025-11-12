from loguru import logger
from openai import OpenAI
from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel


class ResponseModel(BaseModel):
    response: str


class Generator:
    def __init__(self, model: str, system_prompt: str, client: OpenAI):
        logger.info("Initializing Generator with OpenAI API.")
        self.model = model
        self.system_prompt = system_prompt
        self.client = client

    def generate_response(self, user_prompt: str) -> str:
        """
        Generates a response to a user query using RAG.
        """
        logger.info("Generating response using OpenAI API.")
        logger.info(f"User prompt: {user_prompt}")
        logger.info(f"System prompt: {self.system_prompt}")

        response: ParsedResponse = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=ResponseModel,
        )

        return response.output_parsed.response

        # return f"""Response generation stub:
        # User prompt: {user_prompt} \n ----------
        # System prompt: {self.system_prompt}"""
