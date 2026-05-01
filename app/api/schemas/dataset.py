from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DatasetUploadResponse(BaseModel):
    dataset_id: UUID
    name: str
    row_count: int
    column_count: int
    target_column: str
    date_column: str | None
    class_balance: dict[str, float]
    created_at: datetime


class DatasetInfo(DatasetUploadResponse):
    description: str | None
    file_format: str
    content_hash: str
    validation_report: dict | None
