"""Deterministic model-input transform shared by training, evaluation and the API.

Policy (docs/Architecture_Decisions.md, ADR-005):
  1. Convert to grayscale. Mikrobat is distributed as grayscale micrographs, so colour
     carries no reference information and must not become a shortcut signal.
  2. Pad to a square with the image's mean intensity instead of cropping. Microscopy
     evidence can sit anywhere in the frame, so nothing is cut away.
  3. Resize to the DINOv2 input size and apply ImageNet normalisation (DINOv2 pretraining).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

PREPROCESSING_VERSION = "preprocess-v1"
MODEL_INPUT_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


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


def to_model_input(image: Image.Image) -> np.ndarray:
    """Return a float32 array shaped (3, 224, 224) ready for the encoder."""
    square = pad_to_square(to_grayscale(image))
    resized = square.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BICUBIC)
    channel = np.asarray(resized, dtype=np.float32) / 255.0
    stacked = np.repeat(channel[None, :, :], 3, axis=0)
    return (stacked - _IMAGENET_MEAN) / _IMAGENET_STD
