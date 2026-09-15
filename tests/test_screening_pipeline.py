import copy
from dataclasses import replace

import pytest

from ml import paths
from ml.decision.decision_engine import PRELIMINARY_PASS, UNKNOWN_DECISION
from ml.inference.screening_pipeline import ArtifactMismatchError, ArtifactPaths, ScreeningPipeline
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


def test_load_reports_missing_artifacts(tmp_path):
    artifacts = ArtifactPaths.from_layout(tmp_path, tmp_path / "references.faiss", tmp_path / "meta.json")
    with pytest.raises(FileNotFoundError, match="Missing model artifacts"):
        ScreeningPipeline.load(artifacts)


REAL_ARTIFACTS = ArtifactPaths.from_layout(
    paths.MODELS_DIR, paths.INDEXES_DIR / "references.faiss", paths.INDEXES_DIR / "reference_metadata.json"
)
needs_artifacts = pytest.mark.skipif(
    bool(REAL_ARTIFACTS.missing()) or not ds.split_path(ds.TEST).is_file(),
    reason="real model artifacts not built; run `python -m scripts.run_pipeline`",
)


@pytest.mark.artifacts
@needs_artifacts
def test_real_pipeline_screens_real_images():
    pipeline = ScreeningPipeline.load(REAL_ARTIFACTS)
    known = ds.load_split(ds.TEST)
    ood = ds.load_split(ds.OOD_EVALUATION)
    known_result = pipeline.analyze_image(load_image_file(known.image_paths()[0]))
    ood_result = pipeline.analyze_image(load_image_file(ood.image_paths()[0]))
    assert known_result.evidence.prediction.class_name in pipeline.classifier.classes
    assert known_result.decision.status in {"PRELIMINARY_PASS", "REVIEW_REQUIRED", "UNKNOWN"}
    assert ood_result.decision.status == UNKNOWN_DECISION
