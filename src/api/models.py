from fastapi import HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel, field_validator, model_validator

from src.utils.config import USER_SOURCE_DATA_DIR_PATH


class FileUploadModel(BaseModel):
    file: UploadFile
    user_id: str

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

    @field_validator("file")
    @classmethod
    def validate_file(cls, file: UploadFile):
        """
        Validates the uploaded file.
        """

        if not file:
            raise HTTPException(status_code=400, detail="File is required.")

        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="File has no filename.")

        # filename extension validation
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only .pdf files are accepted.",
            )

        return file

    @model_validator(mode="after")
    def check_for_file_conflict(self) -> "FileUploadModel":
        """
        Validates if the file already exists at the destination path.
        """

        if not self.file or not self.user_id:
            return self

        # TODO move to a path manager or something similar,
        # this path is hard coded in two places already
        user_data_dir = USER_SOURCE_DATA_DIR_PATH / self.user_id
        user_data_dir.mkdir(parents=True, exist_ok=True)
        permanent_file_path = user_data_dir / self.file.filename

        # file conflict validation
        if permanent_file_path.exists():
            logger.warning(f"File conflict: {permanent_file_path} already exists.")
            raise HTTPException(
                status_code=409,  # 409 Conflict
                detail=f"File '{self.file.filename}' already exists. "
                "Please rename the file or delete the existing one first.",
            )

        return self
