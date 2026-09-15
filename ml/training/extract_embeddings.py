"""Encode every split with frozen DINOv2 ViT-S/14 and cache the vectors.

A split is re-encoded only when its image list or the embedding fingerprint changed.

Output: data/embeddings/<split>.npz  (embeddings float32 [N, D], image_ids, fingerprint)
Usage:  python -m ml.training.extract_embeddings
"""

from __future__ import annotations

import sys
import time

import numpy as np

from ml import paths
from ml.encoders.dinov2_encoder import Dinov2Encoder
from ml.pipeline_config import load_pipeline_config
from ml.preprocessing.image_io import load_image_file
from ml.preprocessing.transforms import to_model_input
from ml.training import datasets as ds


def _is_current(name: str, image_ids: list[str], fingerprint: str) -> bool:
    location = ds.embeddings_path(name)
    if not location.is_file():
        return False
    with np.load(location, allow_pickle=False) as cached:
        return cached["image_ids"].tolist() == image_ids and str(cached["fingerprint"]) == fingerprint


def main() -> int:
    config = load_pipeline_config()
    encoder = Dinov2Encoder(paths.PRETRAINED_DIR / config.encoder.local_dir)
    paths.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ds.ALL_SPLITS:
        split = ds.load_split(name)
        image_ids = split.frame["image_id"].tolist()
        if _is_current(name, image_ids, encoder.fingerprint):
            print(f"[{name}] {len(image_ids)} embeddings already current")
            continue
        started = time.perf_counter()
        batches = []
        files = split.image_paths()
        for start in range(0, len(files), config.encoder.batch_size):
            chunk = files[start : start + config.encoder.batch_size]
            batch = np.stack([to_model_input(load_image_file(path).image) for path in chunk])
            batches.append(encoder.encode(batch))
        vectors = np.concatenate(batches) if batches else np.zeros((0, encoder.dimension), np.float32)
        np.savez(
            ds.embeddings_path(name),
            embeddings=vectors,
            image_ids=np.array(image_ids, dtype=np.str_),
            fingerprint=np.array(encoder.fingerprint),
        )
        print(f"[{name}] encoded {len(files)} images in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
