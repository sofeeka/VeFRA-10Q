import os
from functools import lru_cache

import dotenv

from src.utils.exceptions import VeFRA_ConfigurationError

dotenv.load_dotenv()


@lru_cache()
def get_openai_api_key():
    try:
        openai_api_key = os.environ["OPENAI_API_KEY"]
        return openai_api_key
    except KeyError:
        raise VeFRA_ConfigurationError("OPENAI_API_KEY environment variable not set.")


@lru_cache()
def get_google_api_key():
    try:
        google_api_key = os.environ["GOOGLE_API_KEY"]
        return google_api_key
    except KeyError:
        raise VeFRA_ConfigurationError("GOOGLE_API_KEY environment variable not set.")
