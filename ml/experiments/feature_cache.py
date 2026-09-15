"""Cache raw DINOv2 token features per backbone, resolution, split and dihedral view.

Every experiment configuration that shares a backbone and input size reuses one encoding pass:
pooling and the number of views are applied afterwards with the production functions
(ml.encoders.dinov2_encoder.pool_features / l2_normalize), so experiments measure exactly the
embeddings the production pipeline would compute.

Layout: models/experiments/features/<backbone dir>_<input size>/<split>.npz
        (cls [views, N, hidden], patch_mean [views, N, hidden], cls_layers [views, N, blocks,
        hidden], image_ids, revision, seconds). Caches written before per-block CLS tokens were
        recorded remain valid for recipes that do not need them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml import paths
from ml.encoders.dinov2_encoder import Dinov2Encoder, ViewTokens, ensure_backbone, l2_normalize, pool_features
from ml.pipeline_config import Backbone
from ml.preprocessing.image_io import load_image_file
from ml.training import datasets as ds

FEATURES_DIR = paths.MODELS_DIR / "experiments" / "features"


@dataclass(frozen=True)
class TokenFeatures:
    tokens: ViewTokens  # leading axes (views, N)
    image_ids: list[str]
    seconds_per_view_image: float

    def view_embeddings(self, pooling: str, views: int) -> np.ndarray:
        layers = self.tokens.cls_layers
        subset = ViewTokens(
            self.tokens.cls[:views],
            self.tokens.patch_mean[:views],
            None if layers is None else layers[:views],
        )
        return pool_features(subset, pooling)

    def embeddings(self, pooling: str, views: int) -> np.ndarray:
        return l2_normalize(self.view_embeddings(pooling, views).mean(axis=0))


def _cache_path(backbone: Backbone, input_size: int, split: str) -> Path:
    return FEATURES_DIR / f"{backbone.local_dir}_{input_size}" / f"{split}.npz"


def load_features(
    backbone: Backbone, input_size: int, split: str, views: int, batch_size: int, need_layers: bool
) -> TokenFeatures:
    """Return cached token features, encoding (and downloading the backbone) only when needed."""
    frame = ds.load_split(split)
    image_ids = frame.frame["image_id"].tolist()
    location = _cache_path(backbone, input_size, split)
    if location.is_file():
        with np.load(location, allow_pickle=False) as cached:
            has_layers = "cls_layers" in cached.files
            if (
                cached["image_ids"].tolist() == image_ids
                and str(cached["revision"]) == backbone.revision
                and cached["cls"].shape[0] >= views
                and (has_layers or not need_layers)
            ):
                tokens = ViewTokens(
                    cached["cls"], cached["patch_mean"], cached["cls_layers"] if has_layers else None
                )
                return TokenFeatures(tokens, image_ids, float(cached["seconds"]))

    weights_dir = paths.PRETRAINED_DIR / backbone.local_dir
    ensure_backbone(backbone.name, backbone.hub_id, backbone.revision, backbone.license, weights_dir)
    encoder = Dinov2Encoder(weights_dir, input_size, "cls", 1, batch_size)
    images = [load_image_file(path).image for path in frame.image_paths()]
    started = time.perf_counter()
    per_view = [encoder.token_features(images, view) for view in range(views)]
    seconds = (time.perf_counter() - started) / (views * len(images))
    tokens = ViewTokens(
        np.stack([view.cls for view in per_view]),
        np.stack([view.patch_mean for view in per_view]),
        np.stack([view.cls_layers for view in per_view]),
    )
    location.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        location,
        cls=tokens.cls,
        patch_mean=tokens.patch_mean,
        cls_layers=tokens.cls_layers,
        image_ids=np.array(image_ids, dtype=np.str_),
        revision=np.array(backbone.revision),
        seconds=np.array(seconds),
    )
    return TokenFeatures(tokens, image_ids, seconds)
