import numpy as np
from PIL import Image, ImageFilter

from ml.preprocessing.quality import ACCEPTABLE, DEGRADED, QualityBounds, assess_quality, measure_quality
from tests.helpers import texture

BOUNDS = QualityBounds(
    version="quality-test",
    tile_grid=8,
    min_side=224,
    min_sharpness=50.0,
    min_brightness=0.2,
    max_brightness=0.8,
    max_clipped_fraction=0.01,
    tile_detail_floor=20.0,
    min_detail_coverage=0.5,
)


def _failed(image: Image.Image) -> set[str]:
    assessment = assess_quality(measure_quality(image, BOUNDS.tile_grid), BOUNDS)
    return {check.name for check in assessment.checks if not check.passed}


def test_sharp_well_exposed_texture_is_acceptable():
    assessment = assess_quality(measure_quality(texture("beta", 0), BOUNDS.tile_grid), BOUNDS)
    assert assessment.status == ACCEPTABLE
    assert assessment.passed_fraction == 1.0


def test_blurred_image_fails_focus_and_visibility():
    blurred = texture("beta", 0).filter(ImageFilter.GaussianBlur(radius=12))
    assert {"focus", "specimen_visibility"} <= _failed(blurred)


def test_small_image_fails_resolution():
    assert "resolution" in _failed(texture("beta", 0, size=120))


def test_exposure_checks():
    dark = Image.fromarray((np.asarray(texture("beta", 0)) * 0.15).astype(np.uint8))
    bright = Image.fromarray(np.full((256, 256), 250, dtype=np.uint8))
    assert "underexposure" in _failed(dark)
    assert "overexposure" in _failed(bright)


def test_saturated_image_fails_clipping():
    saturated = Image.fromarray(np.where(np.asarray(texture("beta", 0)) > 128, 255, 0).astype(np.uint8))
    assert "saturation" in _failed(saturated)


def test_degraded_status_when_any_check_fails():
    assessment = assess_quality(measure_quality(texture("beta", 0, size=120), BOUNDS.tile_grid), BOUNDS)
    assert assessment.status == DEGRADED
    assert 0 < assessment.passed_fraction < 1


def test_measurement_uses_encoder_scale_for_any_input_size():
    small = measure_quality(texture("alpha", 0, size=256), BOUNDS.tile_grid)
    large = measure_quality(texture("alpha", 0, size=256).resize((768, 768)), BOUNDS.tile_grid)
    assert (large.width, large.height) == (768, 768)
    assert len(small.tile_sharpness) == len(large.tile_sharpness) == 64
