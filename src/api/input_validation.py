import re

from fastapi import Depends, HTTPException, Path

from src.utils.config import get_user_sources_folder

VALID_USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


async def validate_user_id(
    user_id: str = Path(
        ...,
        min_length=2,
        max_length=10,
        description="The unique identifier for the user.",
    ),
) -> str:
    """
    FastAPI Dependency to validate and normalize a user_id from a URL path.
    """

    if not VALID_USER_ID_PATTERN.match(user_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid user_id format '{user_id}'. "
            "Only alphanumeric characters, hyphens, and underscores are allowed.",
        )

    normalized_user_id = user_id.lower()
    return normalized_user_id


async def get_existing_user(user_id: str = Depends(validate_user_id)) -> str:
    """
    Dependency that validates user_id format and ensures the user exists.
    """
    if not _user_exists_in_file_system(user_id=user_id):
        raise HTTPException(
            status_code=404, detail=f"User with id '{user_id}' not found."
        )
    return user_id


def _user_exists_in_file_system(user_id: str) -> bool:
    """
    Checks if a user exists in the system.
    For now, "existence" is defined as having an initialized data directory.
    """
    user_dir = get_user_sources_folder(user_id=user_id, create=False)
    return user_dir.is_dir()
