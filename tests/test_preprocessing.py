import io

import numpy as np
import pytest
from PIL import Image

from ml.preprocessing.image_io import InvalidImageError, decode_image
from ml.preprocessing.transforms import (
    DIHEDRAL_TRANSFORMS,
    dihedral_view,
    pad_to_square,
    to_grayscale,
    to_model_input,
)
from tests.helpers import png_bytes, texture


def _encode(image: Image.Image, fmt: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "TIFF", "BMP"])
def test_decode_accepts_supported_formats(fmt):
    decoded = decode_image(_encode(texture("alpha", 0).convert("RGB"), fmt))
    assert decoded.format == fmt
    assert (decoded.width, decoded.height) == (256, 256)


def test_decode_rejects_empty_bytes():
    with pytest.raises(InvalidImageError, match="empty"):
        decode_image(b"")


def test_decode_rejects_non_image_bytes():
    with pytest.raises(InvalidImageError, match="not a readable image"):
        decode_image(b"%PDF-1.7 definitely not an image")


def test_decode_rejects_unsupported_format():
    with pytest.raises(InvalidImageError, match="Unsupported image format"):
        decode_image(_encode(texture("alpha", 0), "GIF"))


def test_decode_rejects_truncated_image():
    data = png_bytes(texture("alpha", 0))
    with pytest.raises(InvalidImageError):
        decode_image(data[: len(data) // 2])


def test_decode_applies_exif_orientation():
    image = Image.new("RGB", (40, 20), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    decoded = decode_image(buffer.getvalue())
    assert (decoded.width, decoded.height) == (20, 40)


def test_to_grayscale_keeps_luminance():
    rgb = Image.new("RGB", (4, 4), (200, 200, 200))
    assert to_grayscale(rgb).mode == "L"
    assert np.asarray(to_grayscale(rgb))[0, 0] == 200


def test_pad_to_square_preserves_content_without_cropping():
    wide = Image.fromarray(np.full((10, 30), 90, dtype=np.uint8), mode="L")
    square = pad_to_square(wide)
    assert square.size == (30, 30)
    pixels = np.asarray(square)
    assert (pixels[10:20, :] == 90).all()  # original content fully kept in the middle band
    assert (pixels == 90).all()  # padding uses the mean intensity


@pytest.mark.parametrize("size", [224, 308, 448])
def test_to_model_input_shape_and_normalisation(size):
    array = to_model_input(texture("beta", 1, size=300), size)
    assert array.shape == (3, size, size)
    assert array.dtype == np.float32
    assert np.allclose(array[0] * 0.229 + 0.485, array[1] * 0.224 + 0.456, atol=1e-5)


def test_to_model_input_rejects_sizes_that_are_not_patch_multiples():
    with pytest.raises(ValueError, match="multiple"):
        to_model_input(texture("beta", 1), 300)


def test_to_model_input_is_deterministic_and_colour_invariant():
    gray = texture("alpha", 3)
    rgb = gray.convert("RGB")
    assert np.array_equal(to_model_input(gray, 224), to_model_input(rgb, 224))


def test_dihedral_views_are_the_eight_distinct_symmetries_of_the_square():
    asymmetric = Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4) * 16, mode="L")
    views = [np.asarray(dihedral_view(asymmetric, v)).tobytes() for v in range(len(DIHEDRAL_TRANSFORMS))]
    assert len(DIHEDRAL_TRANSFORMS) == 8
    assert len(set(views)) == 8
    assert views[0] == np.asarray(asymmetric).tobytes()
