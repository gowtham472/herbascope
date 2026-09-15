from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.helpers import build_synthetic_pipeline


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    return build_synthetic_pipeline(tmp_path_factory.mktemp("synthetic-data"))


@pytest.fixture
def settings(tmp_path, synthetic):
    """Isolated settings: synthetic reference images, fresh runtime state, no real artifacts."""
    return Settings(
        _env_file=None,
        model_dir=tmp_path / "models",
        data_dir=synthetic.data_dir,
        app_state_dir=tmp_path / "state",
        max_upload_mb=1,
    )


@pytest.fixture
def client(settings, synthetic):
    with TestClient(create_app(settings, pipeline=synthetic.pipeline)) as test_client:
        yield test_client


@pytest.fixture
def degraded_client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client
