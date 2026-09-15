"""Deterministic model-input transform shared by training, evaluation and the API.

Policy (docs/Architecture_Decisions.md, ADR-005):
  1. Convert to grayscale. Mikrobat is distributed as grayscale micrographs, so colour
     carries no reference information and must not become a shortcut signal.
  2. Pad to a square with the image's mean intensity instead of cropping. Microscopy
     evidence can sit anywhere in the frame, so nothing is cut away.
  3. Resize to the configured encoder input size (a multiple of the 14 px ViT patch) and
     apply ImageNet normalisation (DINOv2 pretraining).

Dihedral views (ADR-020): a micrograph has no canonical orientation, so the eight rotations
and reflections of the square are equally valid observations of the same specimen. They are
used as training augmentation and averaged at inference (test-time augmentation).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

PREPROCESSING_VERSION = "preprocess-v1"
PATCH_SIZE = 14
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

# The dihedral group of the square: identity, three rotations, and their mirror images.
_T = Image.Transpose
DIHEDRAL_TRANSFORMS: tuple[tuple[_T, ...], ...] = (
    (),
    (_T.ROTATE_90,),
    (_T.ROTATE_180,),
    (_T.ROTATE_270,),
    (_T.FLIP_LEFT_RIGHT,),
    (_T.FLIP_LEFT_RIGHT, _T.ROTATE_90),
    (_T.FLIP_LEFT_RIGHT, _T.ROTATE_180),
    (_T.FLIP_LEFT_RIGHT, _T.ROTATE_270),
)


def to_grayscale(image: Image.Image) -> Image.Image:
    return image if image.mode == "L" else image.convert("L")


def pad_to_square(gray: Image.Image) -> Image.Image:
    width, height = gray.size
    if width == height:
        return gray
    side = max(width, height)
    fill = round(float(np.asarray(gray, dtype=np.float32).mean()))
    canvas = Image.new("L", (side, side), color=fill)
    canvas.paste(gray, ((side - width) // 2, (side - height) // 2))
    return canvas


def dihedral_view(image: Image.Image, view: int) -> Image.Image:
    for operation in DIHEDRAL_TRANSFORMS[view]:
        image = image.transpose(operation)
    return image


def to_model_input(image: Image.Image, input_size: int) -> np.ndarray:
    """Return a float32 array shaped (3, input_size, input_size) ready for the encoder."""
    if input_size % PATCH_SIZE:
        raise ValueError(f"input size {input_size} is not a multiple of the {PATCH_SIZE} px patch size")
    square = pad_to_square(to_grayscale(image))
    resized = square.resize((input_size, input_size), Image.Resampling.BICUBIC)
    channel = np.asarray(resized, dtype=np.float32) / 255.0
    stacked = np.repeat(channel[None, :, :], 3, axis=0)
    return (stacked - _IMAGENET_MEAN) / _IMAGENET_STD
