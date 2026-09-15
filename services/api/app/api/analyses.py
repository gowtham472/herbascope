from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse

from app.dependencies import StoreDep
from app.schemas.analysis import AnalysisList, AnalysisResponse, ErrorResponse

router = APIRouter(prefix="/analyses", tags=["analysis history"])

AnalysisId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$", description="Server-generated analysis id")]
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Unknown analysis id"}}


@router.get("", response_model=AnalysisList, summary="List recent analyses, newest first")
def list_analyses(store: StoreDep, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> AnalysisList:
    return AnalysisList(items=store.list_recent(limit))


@router.get(
    "/{analysis_id}", response_model=AnalysisResponse, responses=NOT_FOUND, summary="Get one analysis"
)
def get_analysis(analysis_id: AnalysisId, store: StoreDep) -> AnalysisResponse:
    analysis = store.get(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return analysis


@router.get("/{analysis_id}/image", response_class=FileResponse, responses=NOT_FOUND, summary="Sample image")
def get_analysis_image(analysis_id: AnalysisId, store: StoreDep) -> FileResponse:
    path = store.sample_path(analysis_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Image for analysis {analysis_id} not found")
    return FileResponse(path, media_type="image/png")
