import re
from pathlib import Path
from typing import Optional, Union

from fastapi import HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel, field_validator, model_validator

from src.utils.config import get_user_sources_file_path


# TODO improve error handling, maybe use loguru for catching raising exceptions
class FileUploadModel(BaseModel):
    file: UploadFile
    user_id: str

    filepath: Optional[Union[Path, str]]
    parsed_year: Optional[str] = None
    parsed_quarter: Optional[str] = None
    parsed_company: Optional[str] = None

    @field_validator("user_id", mode="before")
    @classmethod
    def validate_user_id(cls, user_id: str) -> str:
        """
        Validates the user_id for invalid characters and normalizes it.
        """

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required.")

        if ".." in user_id or "/" in user_id or "\\" in user_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid user_id format. Contains illegal characters.",
            )

        return user_id.lower()

    @field_validator("file", mode="before")
    @classmethod
    def validate_file(cls, file: UploadFile):
        """
        Validates the uploaded file.
        1. It must have a filename.
        2. It must have a .pdf extension.
        """

        if not file:
            raise HTTPException(status_code=400, detail="File is required.")

        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="File has no filename.")

        # file extension validation
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only .pdf files are accepted.",
            )

        return file

    @model_validator(mode="after")
    def check_for_file_conflict(self) -> "FileUploadModel":
        # TODO maybe get rid of user id == company name check
        """
        Validates the model as a whole after user_id and file were validated.

        1. File must be unique (it cannot be present already)
        2. The filename must be in format "YYY QN COMPANY.pdf"
        3. Company matches user_id

        If valid, extracts year, quarter and company from
        """

        filename = self.file.filename

        # pattern match check
        pattern = re.compile(r"^(\d{4}) (Q[1-4]) ([\w\s.-]+?)\.pdf$", re.IGNORECASE)
        match = pattern.match(filename)

        if not match:
            logger.warning(
                f"Invalid filename format for user {self.user_id}: {filename}"
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid filename format. "
                "Expected 'YYYY QN COMPANY.pdf' (e.g., '2022 Q1 MSFT.pdf').",
            )

        year = match.group(1)
        quarter = match.group(2)
        company = match.group(3).strip()

        # TODO maybe get rid of this check
        # company name matches user_id validator
        if company.lower() != self.user_id:
            logger.warning(
                f"Filename company '{company}' does not match user_id '{self.user_id}'"
            )
            raise HTTPException(
                status_code=400,  # 400 Bad Request is appropriate
                detail=f"Company name in filename ('{company}') "
                f"does not match your user ID.",
            )

        self.parsed_year = year
        self.parsed_quarter = quarter
        self.parsed_company = company

        permanent_file_path = get_user_sources_file_path(
            user_id=self.user_id, filename=filename
        )

        # file conflict validation
        if permanent_file_path.exists():
            logger.warning(f"File conflict: {permanent_file_path} already exists.")
            raise HTTPException(
                status_code=409,  # 409 Conflict
                detail=f"File '{self.file.filename}' already exists. "
                "Please rename the file or delete the existing one first.",
            )

        self.filepath = permanent_file_path
        return self
