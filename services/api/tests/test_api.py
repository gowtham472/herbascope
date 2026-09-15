from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from ml import paths
from ml.training import datasets as ds
from tests.helpers import png_bytes, texture

API = "/api/v1"


def _upload(client: TestClient, data: bytes, content_type: str = "image/png", name: str = "sample.png"):
    return client.post(f"{API}/analyze", files={"image": (name, data, content_type)})


def test_health_reports_loaded_models(client):
    body = client.get(f"{API}/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] and body["index_loaded"]
    assert body["reference_count"] == 8
    assert body["max_upload_mb"] == 1
    assert body["model"]["classes"] == ["alpha", "beta"]


def test_analyze_returns_complete_frozen_schema(client):
    response = _upload(client, png_bytes(texture("alpha", 900)))
    assert response.status_code == 200
    body = response.json()
    for block in ("prediction", "retrieval", "unknown", "evidence", "decision", "model"):
        assert block in body
    assert set(body["prediction"]) >= {"class_name", "confidence", "top_k"}
    assert set(body["retrieval"]) >= {"matches", "top_similarity", "retrieved_class"}
    assert set(body["unknown"]) >= {"risk", "distance", "status"}
    assert set(body["evidence"]) >= {"agreement", "strength", "quality"}
    assert set(body["decision"]) >= {"status", "reason", "policy_version"}
    assert set(body["model"]) >= {"encoder", "classifier_version", "index_version"}
    assert body["prediction"]["class_name"] == "alpha"
    assert body["decision"]["status"] == "PRELIMINARY_PASS"
    assert body["decision"]["thresholds"]["min_classifier_confidence"] == 0.6
    assert body["sample"]["image_url"] == f"/analyses/{body['id']}/image"
    assert body["retrieval"]["matches"][0]["image_url"].startswith("/reference/REF")
    assert body["disclaimer"].startswith("HerbaScope X provides preliminary visual screening support")
    assert any("same visual representation" in item for item in body["limitations"])


def test_out_of_distribution_upload_is_unknown(client):
    body = _upload(client, png_bytes(texture("ood", 901))).json()
    assert body["unknown"]["status"] == "UNKNOWN"
    assert body["decision"]["status"] == "UNKNOWN"


def test_analysis_is_persisted_and_listed(client):
    created = _upload(client, png_bytes(texture("beta", 902)), name="beta.png").json()
    fetched = client.get(f"{API}/analyses/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == created
    items = client.get(f"{API}/analyses").json()["items"]
    assert items[0]["id"] == created["id"]
    assert items[0]["filename"] == "beta.png"
    assert items[0]["decision_status"] == created["decision"]["status"]
    image = client.get(f"{API}{created['sample']['image_url']}")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(image.content)).size == (256, 256)


def test_reference_metadata_and_image(client):
    body = client.get(f"{API}/reference/REF0001").json()
    assert body["class_name"] == "alpha"
    assert body["index_version"] == "index-test"
    image = client.get(f"{API}{body['image_url']}")
    assert image.status_code == 200
    assert client.get(f"{API}/reference/REF9999").status_code == 404


@pytest.mark.parametrize(
    ("data", "content_type", "status"),
    [
        (b"%PDF-1.7 not an image", "image/png", 422),
        (b"", "image/png", 422),
        (b"plain text", "text/plain", 415),
        (b"\x00" * (1024 * 1024 + 10), "image/png", 413),
    ],
    ids=["not-an-image", "empty-file", "wrong-media-type", "too-large"],
)
def test_invalid_uploads_are_rejected(client, data, content_type, status):
    response = _upload(client, data, content_type)
    assert response.status_code == status
    assert response.json()["detail"]
    assert client.get(f"{API}/analyses").json()["items"] == []


def test_malformed_ids_are_rejected(client):
    assert client.get(f"{API}/analyses/not-a-valid-id").status_code == 422
    assert client.get(f"{API}/analyses/{'0' * 32}").status_code == 404
    assert client.get(f"{API}/reference/..%2F..%2Fsecret").status_code in {404, 422}


def test_degraded_mode_without_artifacts(degraded_client):
    health = degraded_client.get(f"{API}/health").json()
    assert health["status"] == "degraded"
    assert "Missing model artifacts" in health["detail"]
    response = _upload(degraded_client, png_bytes(texture("alpha", 0)))
    assert response.status_code == 503
    assert "run_pipeline" in response.json()["detail"]
    assert degraded_client.get(f"{API}/reference/REF0001").status_code == 503
    assert degraded_client.get(f"{API}/analyses").json() == {"items": []}


def test_cors_allows_configured_frontend_origin(client):
    response = client.options(
        f"{API}/analyze",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


REAL_SETTINGS = Settings(_env_file=None)


@pytest.mark.artifacts
@pytest.mark.skipif(
    bool(REAL_SETTINGS.artifacts.missing()) or not ds.split_path(ds.TEST).is_file(),
    reason="real model artifacts not built; run `python -m scripts.run_pipeline`",
)
def test_real_artifacts_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(paths.REPO_ROOT)
    settings = Settings(_env_file=None, app_state_dir=tmp_path / "state")
    known_path = ds.load_split(ds.TEST).image_paths()[0]
    ood_path = ds.load_split(ds.OOD_EVALUATION).image_paths()[0]
    with TestClient(create_app(settings)) as client:
        assert client.get(f"{API}/health").json()["model"]["encoder"] == "DINOv2 ViT-S/14"
        known = _upload(client, known_path.read_bytes(), name=known_path.name).json()
        ood = _upload(client, ood_path.read_bytes(), name=ood_path.name).json()
        assert known["prediction"]["class_name"] in known["model"]["classes"]
        assert len(known["retrieval"]["matches"]) == 5
        assert client.get(f"{API}{known['retrieval']['matches'][0]['image_url']}").status_code == 200
        assert ood["decision"]["status"] == "UNKNOWN"
