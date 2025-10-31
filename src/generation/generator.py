import os
import logging
from openai import OpenAI

from src.generation.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        logger.info("Initializing Generator with OpenAI API.")

        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, user_prompt: str) -> str:
        """
        Generates a response to a user query using RAG.
        """
        logger.info("Generating response using OpenAI API.")
        logger.info(f"User prompt: {user_prompt}")
        logger.info(f"System prompt: {self.system_prompt}")

        # TODO implement actual call to OpenAI API
        # response = self.client.chat.completions.create(
        #     model="gpt-4-turbo",  # TODO change to small models for testing
        #     messages=[
        #         {"role": "system", "content": self.system_prompt},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     temperature=0.2,
        # )

        # return response.choices[0].message.content
        return f"""Response generation stub:
        User prompt: {user_prompt} \n ----------
        System prompt: {self.system_prompt}"""
