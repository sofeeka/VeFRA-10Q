import re
from pathlib import Path

from fastapi import UploadFile
from loguru import logger
from pydantic import BaseModel, field_validator, model_validator

from src.utils.config import get_user_sources_filepath
from src.utils.exceptions import (
    VeFRA_DataValidationError,
    VeFRA_FileConflictError,
    VeFRA_InvalidFileNameError,
    VeFRA_UnsupportedFileTypeError,
)


class FileUploadModel(BaseModel):
    file: UploadFile
    user_id: str

    filepath: Path | str | None = None
    parsed_year: str | None = None
    parsed_quarter: str | None = None
    parsed_company: str | None = None

    @field_validator("user_id", mode="before")
    @classmethod
    def validate_user_id_format(cls, value: str) -> str:
        """
        Validates the format of the user_id.
        """
        if not (2 <= len(value) <= 10):
            raise VeFRA_DataValidationError(
                message=f"Invalid user_id '{value}'. ID must be between 2 and 10 characters long."
            )
        return value

    @field_validator("file", mode="before")
    @classmethod
    def validate_file(cls, file: UploadFile):
        """
        Validates the uploaded file.
        1. It must have a filename.
        2. It must have a .pdf extension.
        """

        filename = file.filename
        if not filename:
            raise VeFRA_DataValidationError("File has no filename.")

        # file extension validation
        if not filename.lower().endswith(".pdf"):
            raise VeFRA_UnsupportedFileTypeError(
                "Invalid file type. Only .pdf files are accepted.",
            )

        return file

    @model_validator(mode="after")
    def check_for_file_conflict(self) -> "FileUploadModel":
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
            raise VeFRA_InvalidFileNameError(
                "Invalid filename format. "
                "Expected 'YYYY QN COMPANY.pdf' (e.g., '2022 Q1 MSFT.pdf').",
            )

        year = match.group(1)
        quarter = match.group(2)
        company = match.group(3).strip()

        # company name matches user_id validator
        if company.lower() != self.user_id:
            logger.warning(
                f"Filename company '{company}' does not match user_id '{self.user_id}'"
            )
            raise VeFRA_DataValidationError(
                f"Company name in filename ('{company}') "
                f"does not match your user ID {self.user_id}",
            )

        self.parsed_year = year
        self.parsed_quarter = quarter
        self.parsed_company = company

        permanent_filepath = get_user_sources_filepath(
            user_id=self.user_id, filename=filename
        )

        # file conflict validation
        if permanent_filepath.exists():
            logger.warning(f"File conflict: {permanent_filepath} already exists.")
            raise VeFRA_FileConflictError(
                f"File '{self.file.filename}' already exists. "
                "Please rename the file or delete the existing one first.",
            )

        self.filepath = permanent_filepath
        return self
