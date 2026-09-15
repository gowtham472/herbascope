"""Decode and validate untrusted image bytes (uploads and dataset files alike)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = frozenset({"PNG", "JPEG", "WEBP", "TIFF", "BMP"})
# Memory guard, not a quality rule: 50 MP is far above any microscope camera frame we expect.
MAX_PIXELS = 50_000_000


class InvalidImageError(ValueError):
    """Raised when bytes are not a decodable, supported, non-empty image."""


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image
    format: str
    width: int
    height: int


def decode_image(data: bytes) -> DecodedImage:
    if not data:
        raise InvalidImageError("The file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = probe.format
            width, height = probe.size
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError("The file is not a readable image.") from exc
    if image_format not in ALLOWED_FORMATS:
        raise InvalidImageError(
            f"Unsupported image format {image_format!r}; use one of {sorted(ALLOWED_FORMATS)}."
        )
    if width <= 0 or height <= 0:
        raise InvalidImageError("The image has no pixels.")
    if width * height > MAX_PIXELS:
        raise InvalidImageError(f"The image exceeds the {MAX_PIXELS:,}-pixel limit.")
    try:
        # verify() leaves the parser unusable, so decode again from the start.
        image = Image.open(io.BytesIO(data))
        image.load()
    except (OSError, SyntaxError) as exc:
        raise InvalidImageError("The image data is truncated or corrupted.") from exc
    oriented = ImageOps.exif_transpose(image)
    return DecodedImage(image=oriented, format=image_format, width=oriented.width, height=oriented.height)


def load_image_file(path: Path) -> DecodedImage:
    return decode_image(path.read_bytes())
