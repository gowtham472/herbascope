from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.dependencies import InferenceDep, SettingsDep
from app.schemas.analysis import AnalysisResponse, ErrorResponse
from app.services.inference_service import UploadRejectedError

router = APIRouter(tags=["analysis"])


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Run the complete screening pipeline on one microscopic image",
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        422: {"model": ErrorResponse, "description": "Not a readable image"},
        503: {"model": ErrorResponse, "description": "Model artifacts not loaded"},
    },
)
def analyze(
    image: Annotated[UploadFile, File(description="Microscopic sample image (PNG, JPEG, WEBP, TIFF or BMP)")],
    service: InferenceDep,
    settings: SettingsDep,
) -> AnalysisResponse:
    # Read at most one byte past the limit so oversized uploads are rejected without buffering them fully.
    data = image.file.read(settings.max_upload_bytes + 1)
    try:
        return service.analyze(image.filename or "upload", image.content_type, data)
    except UploadRejectedError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
