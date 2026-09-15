"""DINOv2 ViT-S/14 frozen feature extractor.

The embedding is the layer-normalised CLS token (``pooler_output``), L2-normalised so that
inner product equals cosine similarity for both the classifier and FAISS retrieval.

The weights directory is self-describing: ``scripts.download_assets`` writes a SOURCE.json
receipt (model name, hub revision) next to the weights, so the API can load the encoder
from MODEL_DIR alone without reading the training configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from ml.preprocessing.transforms import PREPROCESSING_VERSION


def download_pretrained(hub_id: str, revision: str, target: Path) -> None:
    """Fetch pinned weights once so every later load is offline (local_files_only)."""
    model = AutoModel.from_pretrained(hub_id, revision=revision)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)


def embedding_fingerprint(encoder_name: str, revision: str, dimension: int) -> str:
    """Identity of the embedding space; every downstream artifact must record the same value."""
    return f"{encoder_name}@{revision[:12]}|{PREPROCESSING_VERSION}|dim={dimension}"


class Dinov2Encoder:
    def __init__(self, model_dir: Path) -> None:
        receipt = model_dir / "SOURCE.json"
        if not receipt.is_file() or not (model_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"DINOv2 weights not found in {model_dir}; run `python -m scripts.download_assets`."
            )
        source = json.loads(receipt.read_text(encoding="utf-8"))
        self.name: str = source["model"]
        self._model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
        self.dimension: int = int(self._model.config.hidden_size)
        self.fingerprint = embedding_fingerprint(self.name, source["revision"], self.dimension)

    @torch.inference_mode()
    def encode(self, batch: np.ndarray) -> np.ndarray:
        """Encode a (N, 3, 224, 224) float32 batch into (N, D) unit-length float32 vectors."""
        outputs = self._model(pixel_values=torch.from_numpy(np.ascontiguousarray(batch)))
        features = outputs.pooler_output.float().numpy()
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return (features / np.clip(norms, 1e-12, None)).astype(np.float32)
