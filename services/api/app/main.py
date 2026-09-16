"""FastAPI application factory. Model artifacts are loaded exactly once, at startup.

Run from the repository root:
    python -m uvicorn app.main:app --app-dir services/api --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import API_VERSION
from app.api import analyses, analyze, health, reference
from app.config import Settings
from app.core.logging import configure_logging, get_logger
from app.services.analysis_store import AnalysisStore
from app.services.inference_service import InferenceService
from app.services.reference_service import ReferenceService
from ml.inference.screening_pipeline import ArtifactMismatchError, ArtifactPaths, ScreeningPipeline

API_PREFIX = "/api/v1"


def create_app(settings: Settings | None = None, pipeline: ScreeningPipeline | None = None) -> FastAPI:
    """Build the app. Tests inject a pipeline; production loads it from MODEL_DIR at startup."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        logger = get_logger()
        app.state.settings = settings
        app.state.store = AnalysisStore(settings.database_path, settings.sample_dir)
        app.state.inference_service = None
        app.state.reference_service = None
        app.state.load_error = None
        loaded = pipeline
        if loaded is None:
            try:
                loaded = ScreeningPipeline.load(ArtifactPaths.from_manifest(settings.model_dir))
            except (FileNotFoundError, ArtifactMismatchError) as exc:
                app.state.load_error = str(exc)
                logger.warning("starting in degraded mode: %s", exc)
        if loaded is not None:
            app.state.inference_service = InferenceService(loaded, app.state.store, settings.max_upload_bytes)
            app.state.reference_service = ReferenceService(loaded.index, settings.data_dir)
            logger.info(
                "models loaded encoder=%s classifier=%s index=%s references=%d",
                loaded.model_info.encoder,
                loaded.model_info.classifier_version,
                loaded.model_info.index_version,
                loaded.model_info.reference_count,
            )
        yield

    app = FastAPI(
        title="HerbaScope API",
        version=API_VERSION,
        description="Evidence-driven preliminary visual screening of microscopic medicinal-plant material.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    for router in (analyze.router, analyses.router, reference.router, health.router):
        app.include_router(router, prefix=API_PREFIX)
    return app


app = create_app()
