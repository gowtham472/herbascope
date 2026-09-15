"""Encode every split with the configured frozen DINOv2 recipe and cache the vectors.

Writes the encoder settings artifact (models/configs/<encoder version>.json) that the API
uses to rebuild the identical encoder. A split is re-encoded only when its image list or the
embedding fingerprint changed.

Output: data/embeddings/<split>.npz  (embeddings [N, D], view_embeddings [views, N, D],
        image_ids, fingerprint)
Usage:  python -m ml.training.extract_embeddings
"""

from __future__ import annotations

import sys
import time

import numpy as np

from ml import paths
from ml.encoders.dinov2_encoder import Dinov2Encoder, EncoderSettings, l2_normalize
from ml.pipeline_config import load_pipeline_config
from ml.preprocessing.image_io import load_image_file
from ml.training import datasets as ds


def _is_current(name: str, image_ids: list[str], fingerprint: str) -> bool:
    location = ds.embeddings_path(name)
    if not location.is_file():
        return False
    with np.load(location, allow_pickle=False) as cached:
        return (
            "view_embeddings" in cached.files
            and cached["image_ids"].tolist() == image_ids
            and str(cached["fingerprint"]) == fingerprint
        )


def main() -> int:
    config = load_pipeline_config().encoder
    settings = EncoderSettings(
        version=config.version,
        weights_dir=config.backbone.local_dir,
        input_size=config.input_size,
        pooling=config.pooling,
        views=config.views,
        batch_size=config.batch_size,
    )
    encoder = Dinov2Encoder.from_settings(paths.MODELS_DIR, settings)
    settings.save(paths.model_config_path(settings.version))
    paths.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[encoder] {encoder.fingerprint}")
    for name in ds.ALL_SPLITS:
        split = ds.load_split(name)
        image_ids = split.frame["image_id"].tolist()
        if _is_current(name, image_ids, encoder.fingerprint):
            print(f"[{name}] {len(image_ids)} embeddings already current")
            continue
        started = time.perf_counter()
        images = [load_image_file(path).image for path in split.image_paths()]
        view_vectors = encoder.view_embeddings(images)
        np.savez(
            ds.embeddings_path(name),
            embeddings=l2_normalize(view_vectors.mean(axis=0)),
            view_embeddings=view_vectors,
            image_ids=np.array(image_ids, dtype=np.str_),
            fingerprint=np.array(encoder.fingerprint),
        )
        print(
            f"[{name}] encoded {len(images)} images x {encoder.views} views in {time.perf_counter() - started:.1f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
