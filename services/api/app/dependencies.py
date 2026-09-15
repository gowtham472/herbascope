"""Request-scoped access to the singletons created once at startup (see app.main)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from app.config import Settings
from app.services.analysis_store import AnalysisStore
from app.services.inference_service import InferenceService
from app.services.reference_service import ReferenceService


def _models_unavailable(request: Request) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=f"Model artifacts are not loaded: {request.app.state.load_error}. "
        "Run `python -m scripts.run_pipeline` and restart the API.",
    )


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> AnalysisStore:
    return request.app.state.store


def get_inference_service(request: Request) -> InferenceService:
    service = request.app.state.inference_service
    if service is None:
        raise _models_unavailable(request)
    return service


def get_reference_service(request: Request) -> ReferenceService:
    service = request.app.state.reference_service
    if service is None:
        raise _models_unavailable(request)
    return service


SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreDep = Annotated[AnalysisStore, Depends(get_store)]
InferenceDep = Annotated[InferenceService, Depends(get_inference_service)]
ReferenceDep = Annotated[ReferenceService, Depends(get_reference_service)]
