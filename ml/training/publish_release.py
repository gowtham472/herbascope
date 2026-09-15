"""Publish the trained artifacts as one model release.

Loads the complete ScreeningPipeline from the artifacts named by the pipeline configuration
(which verifies embedding fingerprints, index versions and class sets), then writes
models/manifest.json listing every artifact with its SHA-256. The API and evaluation load
models only through this manifest, so a release is swapped atomically by replacing it.

Output: models/manifest.json
Usage:  python -m ml.training.publish_release
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime

from ml import paths
from ml.inference.screening_pipeline import ArtifactPaths, ScreeningPipeline, file_sha256
from ml.pipeline_config import load_pipeline_config


def configured_artifacts() -> ArtifactPaths:
    config = load_pipeline_config()
    return ArtifactPaths(
        model_dir=paths.MODELS_DIR,
        encoder_settings_path=paths.model_config_path(config.encoder.version),
        classifier_dir=paths.CLASSIFIERS_DIR,
        index_path=paths.INDEXES_DIR / "references.faiss",
        index_metadata_path=paths.INDEXES_DIR / "reference_metadata.json",
        unknown_calibration_path=paths.CLASSIFIERS_DIR / "calibration.json",
        quality_bounds_path=paths.model_config_path(config.quality.version),
        decision_policy_path=paths.model_config_path(config.decision.version),
    )


def main() -> int:
    config = load_pipeline_config()
    artifacts = configured_artifacts()
    pipeline = ScreeningPipeline.load(artifacts)
    info = asdict(pipeline.model_info)
    info["classes"] = list(info["classes"])
    manifest = {
        "release": config.version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": info,
        "artifacts": {
            role: {"path": path.relative_to(paths.MODELS_DIR).as_posix(), "sha256": file_sha256(path)}
            for role, path in artifacts.files().items()
        },
    }
    paths.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"release": manifest["release"], "model": info}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
