from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import FileResponse

from app.dependencies import ReferenceDep
from app.schemas.analysis import ErrorResponse, Reference

router = APIRouter(prefix="/reference", tags=["reference library"])

ReferenceId = Annotated[str, Path(pattern=r"^REF\d{4,}$", description="Reference id, e.g. REF0001")]
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Unknown reference id"}}


@router.get("/{reference_id}", response_model=Reference, responses=NOT_FOUND, summary="Reference metadata")
def get_reference(reference_id: ReferenceId, references: ReferenceDep) -> Reference:
    reference = references.get(reference_id)
    if reference is None:
        raise HTTPException(status_code=404, detail=f"Reference {reference_id} not found")
    return reference


@router.get(
    "/{reference_id}/image", response_class=FileResponse, responses=NOT_FOUND, summary="Reference image"
)
def get_reference_image(reference_id: ReferenceId, references: ReferenceDep) -> FileResponse:
    path = references.image_path(reference_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Image for reference {reference_id} not found")
    return FileResponse(path)
