from __future__ import annotations

from fastapi import APIRouter, Request

from app import API_VERSION
from app.dependencies import SettingsDep
from app.schemas.analysis import Health
from app.services.inference_service import to_model_info

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health, summary="API, model and index health")
def health(request: Request, settings: SettingsDep) -> Health:
    service = request.app.state.inference_service
    if service is None:
        return Health(
            status="degraded",
            api_version=API_VERSION,
            model_loaded=False,
            index_loaded=False,
            reference_count=0,
            max_upload_mb=settings.max_upload_mb,
            model=None,
            detail=request.app.state.load_error,
        )
    info = service.pipeline.model_info
    return Health(
        status="ok",
        api_version=API_VERSION,
        model_loaded=True,
        index_loaded=True,
        reference_count=info.reference_count,
        max_upload_mb=settings.max_upload_mb,
        model=to_model_info(info),
        detail=None,
    )
