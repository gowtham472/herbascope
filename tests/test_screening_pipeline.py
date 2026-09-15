import copy
import json
from dataclasses import replace

import pytest

from ml import paths
from ml.decision.decision_engine import PRELIMINARY_PASS, UNKNOWN_DECISION
from ml.inference.screening_pipeline import (
    ArtifactMismatchError,
    ArtifactPaths,
    ScreeningPipeline,
    file_sha256,
)
from ml.preprocessing.image_io import decode_image, load_image_file
from ml.training import datasets as ds
from tests.helpers import build_synthetic_pipeline, png_bytes, texture


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return build_synthetic_pipeline(tmp_path_factory.mktemp("data"))


def test_known_texture_passes_end_to_end(setup):
    result = setup.pipeline.analyze_image(decode_image(png_bytes(texture("alpha", 500))))
    assert result.evidence.prediction.class_name == "alpha"
    assert result.decision.status == PRELIMINARY_PASS
    assert result.explanation.startswith("The visual evidence supports a preliminary screening pass.")


def test_out_of_distribution_image_is_unknown(setup):
    result = setup.pipeline.analyze_image(decode_image(png_bytes(texture("ood", 500))))
    assert result.evidence.unknown.status == "UNKNOWN"
    assert result.decision.status == UNKNOWN_DECISION


def test_batch_and_single_analysis_agree(setup):
    decoded = decode_image(png_bytes(texture("beta", 7)))
    single = setup.pipeline.analyze_image(decoded)
    embedding = setup.pipeline.embed(decoded)
    batch = setup.pipeline.analyze_embeddings(
        embedding[None, :], [setup.pipeline.assess_image_quality(decoded)]
    )[0]
    assert single == batch


def test_mismatched_fingerprints_are_rejected(setup):
    p = setup.pipeline
    other_index = copy.copy(p.index)
    other_index.fingerprint = "other-space"
    with pytest.raises(ArtifactMismatchError, match="fingerprints differ"):
        ScreeningPipeline(
            p.encoder, p.classifier, other_index, p.unknown_calibration, p.quality_bounds, p.policy
        )


def test_calibration_for_another_index_is_rejected(setup):
    p = setup.pipeline
    stale = replace(p.unknown_calibration, index_version="index-old")
    with pytest.raises(ArtifactMismatchError, match="fitted on index"):
        ScreeningPipeline(p.encoder, p.classifier, p.index, stale, p.quality_bounds, p.policy)


def test_missing_manifest_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        ArtifactPaths.from_manifest(tmp_path)


def _write_release(model_dir, roles):
    artifacts = {}
    for role in roles:
        path = model_dir / "files" / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode())
        artifacts[role] = {"path": f"files/{role}.bin", "sha256": file_sha256(path)}
    (model_dir / "manifest.json").write_text(json.dumps({"release": "test", "artifacts": artifacts}))


RELEASE_ROLES = [
    "encoder_settings",
    "classifier_model",
    "reference_index",
    "reference_metadata",
    "unknown_calibration",
    "quality_bounds",
    "decision_policy",
]


def test_manifest_locates_verified_artifacts(tmp_path):
    _write_release(tmp_path, RELEASE_ROLES)
    artifacts = ArtifactPaths.from_manifest(tmp_path)
    assert artifacts.index_path == tmp_path / "files" / "reference_index.bin"
    assert artifacts.classifier_dir == tmp_path / "files"


def test_manifest_rejects_modified_artifacts(tmp_path):
    _write_release(tmp_path, RELEASE_ROLES)
    (tmp_path / "files" / "decision_policy.bin").write_bytes(b"edited by hand")
    with pytest.raises(ArtifactMismatchError, match="decision_policy"):
        ArtifactPaths.from_manifest(tmp_path)


def test_manifest_reports_missing_files(tmp_path):
    _write_release(tmp_path, RELEASE_ROLES)
    (tmp_path / "files" / "reference_index.bin").unlink()
    with pytest.raises(FileNotFoundError, match="reference_index"):
        ArtifactPaths.from_manifest(tmp_path)


needs_artifacts = pytest.mark.skipif(
    not paths.MANIFEST_PATH.is_file() or not ds.split_path(ds.TEST).is_file(),
    reason="real model artifacts not built; run `python -m scripts.run_pipeline`",
)


@pytest.mark.artifacts
@needs_artifacts
def test_real_pipeline_screens_real_images():
    pipeline = ScreeningPipeline.load(ArtifactPaths.from_manifest(paths.MODELS_DIR))
    known = ds.load_split(ds.TEST)
    ood = ds.load_split(ds.OOD_EVALUATION)
    known_result = pipeline.analyze_image(load_image_file(known.image_paths()[0]))
    ood_result = pipeline.analyze_image(load_image_file(ood.image_paths()[0]))
    assert known_result.evidence.prediction.class_name in pipeline.classifier.classes
    assert known_result.decision.status in {"PRELIMINARY_PASS", "REVIEW_REQUIRED", "UNKNOWN"}
    assert ood_result.decision.status == UNKNOWN_DECISION
