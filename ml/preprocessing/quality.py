"""Image-quality signals for microscopy screening.

Quality never changes species identity. It feeds evidence strength and can only move a
decision from PRELIMINARY_PASS to REVIEW_REQUIRED.

Every bound except ``min_side`` is calibrated from the Mikrobat training images by
ml/training/calibrate_quality.py. ``min_side`` equals the DINOv2 input size: a smaller
image has to be upsampled, so the encoder sees interpolated rather than recorded detail.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ml.preprocessing.transforms import MODEL_INPUT_SIZE, to_grayscale

ACCEPTABLE = "ACCEPTABLE"
DEGRADED = "DEGRADED"


@dataclass(frozen=True)
class QualityMeasurements:
    width: int
    height: int
    sharpness: float  # variance of the Laplacian (focus)
    brightness: float  # mean intensity in [0, 1] (exposure level)
    clipped_fraction: float  # share of pixels at pure black or pure white (saturation)
    tile_sharpness: tuple[float, ...]  # per-tile Laplacian variance (specimen visibility)


@dataclass(frozen=True)
class QualityBounds:
    version: str
    tile_grid: int
    min_side: int
    min_sharpness: float
    min_brightness: float
    max_brightness: float
    max_clipped_fraction: float
    tile_detail_floor: float
    min_detail_coverage: float

    @classmethod
    def load(cls, path: Path) -> QualityBounds:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload["bounds"], version=payload["version"])


@dataclass(frozen=True)
class QualityCheck:
    name: str
    label: str
    passed: bool
    value: float
    bound: float
    comparison: str  # ">=" or "<="


@dataclass(frozen=True)
class QualityAssessment:
    status: str
    checks: tuple[QualityCheck, ...]
    version: str

    @property
    def passed_fraction(self) -> float:
        return sum(check.passed for check in self.checks) / len(self.checks)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "version": self.version,
            "passed_fraction": self.passed_fraction,
            "checks": [asdict(check) for check in self.checks],
        }


def analysis_canvas(image: Image.Image) -> np.ndarray:
    """Grayscale uint8 array whose short side equals the encoder input size.

    Measuring at a fixed scale keeps sharpness comparable between the 300 px reference
    micrographs and arbitrarily sized uploads.
    """
    gray = to_grayscale(image)
    scale = MODEL_INPUT_SIZE / min(gray.size)
    size = (max(1, round(gray.width * scale)), max(1, round(gray.height * scale)))
    return np.asarray(gray.resize(size, Image.Resampling.BICUBIC), dtype=np.uint8)


def _laplacian_variance(pixels: np.ndarray) -> float:
    return float(cv2.Laplacian(pixels, cv2.CV_64F).var())


def measure_quality(image: Image.Image, tile_grid: int) -> QualityMeasurements:
    canvas = analysis_canvas(image)
    rows = np.array_split(np.arange(canvas.shape[0]), tile_grid)
    cols = np.array_split(np.arange(canvas.shape[1]), tile_grid)
    tiles = tuple(_laplacian_variance(canvas[np.ix_(r, c)]) for r in rows for c in cols if r.size and c.size)
    clipped = np.count_nonzero((canvas == 0) | (canvas == 255)) / canvas.size
    return QualityMeasurements(
        width=image.width,
        height=image.height,
        sharpness=_laplacian_variance(canvas),
        brightness=float(canvas.mean() / 255.0),
        clipped_fraction=float(clipped),
        tile_sharpness=tiles,
    )


def detail_coverage(measurements: QualityMeasurements, tile_detail_floor: float) -> float:
    tiles = np.asarray(measurements.tile_sharpness)
    return float(np.mean(tiles >= tile_detail_floor))


def assess_quality(measurements: QualityMeasurements, bounds: QualityBounds) -> QualityAssessment:
    coverage = detail_coverage(measurements, bounds.tile_detail_floor)
    short_side = min(measurements.width, measurements.height)
    checks = (
        QualityCheck(
            "resolution",
            "Resolution (short side, px)",
            short_side >= bounds.min_side,
            float(short_side),
            float(bounds.min_side),
            ">=",
        ),
        QualityCheck(
            "focus",
            "Focus (Laplacian variance)",
            measurements.sharpness >= bounds.min_sharpness,
            measurements.sharpness,
            bounds.min_sharpness,
            ">=",
        ),
        QualityCheck(
            "underexposure",
            "Brightness (not too dark)",
            measurements.brightness >= bounds.min_brightness,
            measurements.brightness,
            bounds.min_brightness,
            ">=",
        ),
        QualityCheck(
            "overexposure",
            "Brightness (not too bright)",
            measurements.brightness <= bounds.max_brightness,
            measurements.brightness,
            bounds.max_brightness,
            "<=",
        ),
        QualityCheck(
            "saturation",
            "Clipped pixels (fraction)",
            measurements.clipped_fraction <= bounds.max_clipped_fraction,
            measurements.clipped_fraction,
            bounds.max_clipped_fraction,
            "<=",
        ),
        QualityCheck(
            "specimen_visibility",
            "Detail coverage (fraction of tiles)",
            coverage >= bounds.min_detail_coverage,
            coverage,
            bounds.min_detail_coverage,
            ">=",
        ),
    )
    status = ACCEPTABLE if all(check.passed for check in checks) else DEGRADED
    return QualityAssessment(status=status, checks=checks, version=bounds.version)
