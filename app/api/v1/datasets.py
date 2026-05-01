import hashlib
import os
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.dataset import DatasetInfo, DatasetUploadResponse
from app.config import settings
from app.dependencies import get_db, verify_api_key
from app.exceptions import NotFoundError

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetUploadResponse, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    target_column: str = Form(...),
    description: str = Form(default=""),
    date_column: str = Form(default=""),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    os.makedirs(settings.dataset_upload_path, exist_ok=True)
    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    file_ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else "csv"
    file_path = os.path.join(settings.dataset_upload_path, f"{content_hash}.{file_ext}")

    with open(file_path, "wb") as f:
        f.write(content)

    if file_ext == "parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    class_balance = df[target_column].value_counts(normalize=True).to_dict()
    class_balance = {str(k): round(float(v), 4) for k, v in class_balance.items()}

    from app.db.models.dataset import Dataset
    dataset = Dataset(
        name=name,
        description=description or None,
        file_path=file_path,
        file_format=file_ext,
        row_count=len(df),
        column_count=len(df.columns),
        target_column=target_column,
        date_column=date_column or None,
        class_balance=class_balance,
        schema_json={col: str(dtype) for col, dtype in df.dtypes.items()},
        content_hash=content_hash,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    return DatasetUploadResponse(
        dataset_id=dataset.id,
        name=dataset.name,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        target_column=dataset.target_column,
        date_column=dataset.date_column,
        class_balance=class_balance,
        created_at=dataset.created_at,
    )


@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    from app.db.models.dataset import Dataset
    from sqlalchemy import select

    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.is_deleted.is_(False))
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise NotFoundError(f"Dataset {dataset_id} not found")

    return DatasetInfo(
        dataset_id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        target_column=dataset.target_column,
        date_column=dataset.date_column,
        class_balance=dataset.class_balance or {},
        file_format=dataset.file_format,
        content_hash=dataset.content_hash,
        validation_report=dataset.validation_report,
        created_at=dataset.created_at,
    )
