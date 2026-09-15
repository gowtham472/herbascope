"""Calibrate image-quality bounds from the Mikrobat training images.

Objective: flag an image as DEGRADED when a quality signal is more extreme than the
configured lower/upper percentile of the reference-library training images, i.e. when it is
atypical compared with the material the model was built from. ``min_side`` is technical
(the DINOv2 pretraining resolution) and is copied from the config rather than calibrated.

Output: models/configs/<quality version>.json
Usage:  python -m ml.training.calibrate_quality
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np

from ml import paths
from ml.pipeline_config import load_pipeline_config
from ml.preprocessing.image_io import load_image_file
from ml.preprocessing.quality import QualityBounds, detail_coverage, measure_quality
from ml.training import datasets as ds


def main() -> int:
    config = load_pipeline_config()
    q = config.quality
    train = ds.load_split(ds.TRAIN)
    measurements = [measure_quality(load_image_file(path).image, q.tile_grid) for path in train.image_paths()]
    tiles = np.concatenate([np.asarray(m.tile_sharpness) for m in measurements])
    tile_floor = float(np.percentile(tiles, q.tile_floor_percentile))
    sharpness = np.array([m.sharpness for m in measurements])
    brightness = np.array([m.brightness for m in measurements])
    clipped = np.array([m.clipped_fraction for m in measurements])
    coverage = np.array([detail_coverage(m, tile_floor) for m in measurements])

    bounds = QualityBounds(
        version=q.version,
        tile_grid=q.tile_grid,
        min_side=q.min_side,
        min_sharpness=float(np.percentile(sharpness, q.lower_percentile)),
        min_brightness=float(np.percentile(brightness, q.lower_percentile)),
        max_brightness=float(np.percentile(brightness, q.upper_percentile)),
        max_clipped_fraction=float(np.percentile(clipped, q.upper_percentile)),
        tile_detail_floor=tile_floor,
        min_detail_coverage=float(np.percentile(coverage, q.lower_percentile)),
    )
    payload = {
        "version": q.version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "calibrated_on": {"dataset": "Mikrobat", "split": ds.TRAIN, "images": len(measurements)},
        "objective": (
            f"flag signals below percentile {q.lower_percentile:g} or above percentile {q.upper_percentile:g} "
            "of the reference-library training images; min_side is the DINOv2 pretraining resolution"
        ),
        "bounds": {k: v for k, v in asdict(bounds).items() if k != "version"},
        "training_distribution": {
            name: {p: float(np.percentile(values, p)) for p in (1, 5, 50, 95, 99)}
            for name, values in {
                "sharpness": sharpness,
                "brightness": brightness,
                "clipped_fraction": clipped,
                "detail_coverage": coverage,
            }.items()
        },
    }
    target = paths.model_config_path(q.version)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["bounds"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
