from abc import ABC, abstractmethod

from loguru import logger
from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel

from ..data_models.generation import ResponseModel


class BaseGenerator(ABC):
    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        text_format: BaseModel = ResponseModel,
    ) -> ParsedResponse: ...

    def _log_generation_start(self, prompt):
        logger.info(
            "Generating response from LLM.",
            model=self.model,
            prompt_length=len(prompt),
        )

        logger.debug(f"Full user prompt:\n{prompt}")
        logger.debug(f"Full system prompt:\n{self.system_prompt}")
