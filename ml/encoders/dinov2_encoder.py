"""Frozen DINOv2 feature extractor with configurable resolution, pooling and dihedral views.

Embedding recipe (all parts recorded in the embedding fingerprint):
  * input_size  encoder input resolution, a multiple of the 14 px patch (DINOv2 interpolates
                its position embeddings, so resolutions above 224 are supported)
  * pooling     "cls"            layer-normalised CLS token (``pooler_output``)
                "cls_patchmean"  CLS token concatenated with the mean of the final-layer
                                 patch tokens, each L2-normalised (patch tokens carry local
                                 texture detail)
                "cls_last4"      layer-normalised CLS tokens of the last four transformer
                                 blocks, each L2-normalised and concatenated (the DINOv2
                                 multi-block linear-probe recipe; earlier blocks retain
                                 mid-level structure such as cell outlines)
                "cls_last4_patchmean"  cls_last4 plus the mean final-layer patch token as a
                                 fifth, equally weighted part (texture statistics of the
                                 whole field)
  * views       number of dihedral views (1 = no test-time augmentation, up to 8)

The final embedding is the L2-normalised mean of the per-view embeddings, so inner product
equals cosine similarity for both the classifier and FAISS retrieval.

The weights directory is self-describing (SOURCE.json receipt with model name and revision),
and the chosen recipe is saved as an encoder settings artifact, so the API reconstructs the
exact encoder from MODEL_DIR alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from ml.preprocessing.transforms import (
    DIHEDRAL_TRANSFORMS,
    PREPROCESSING_VERSION,
    dihedral_view,
    to_model_input,
)

POOLINGS = ("cls", "cls_patchmean", "cls_last4", "cls_last4_patchmean")
BLOCK_POOLINGS = ("cls_last4", "cls_last4_patchmean")  # need per-block CLS tokens
LAST_BLOCKS = 4
_POOLING_WIDTH = {
    "cls": 1,
    "cls_patchmean": 2,
    "cls_last4": LAST_BLOCKS,
    "cls_last4_patchmean": LAST_BLOCKS + 1,
}


def ensure_backbone(name: str, hub_id: str, revision: str, license_name: str, target: Path) -> bool:
    """Cache pinned weights once so every later load is offline; returns True if downloaded.

    Writes the SOURCE.json receipt that Dinov2Encoder reads for the model name and revision.
    """
    receipt = target / "SOURCE.json"
    if receipt.is_file() and json.loads(receipt.read_text(encoding="utf-8")).get("revision") == revision:
        return False
    model = AutoModel.from_pretrained(hub_id, revision=revision)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    receipt.write_text(
        json.dumps(
            {
                "model": name,
                "hub_id": hub_id,
                "revision": revision,
                "license": license_name,
                "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return True


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return (features / np.clip(norms, 1e-12, None)).astype(np.float32)


@dataclass(frozen=True)
class ViewTokens:
    """Raw token features for one or more views; leading axes are shared by all fields."""

    cls: np.ndarray  # (..., hidden) layer-normalised CLS token of the final block
    patch_mean: np.ndarray  # (..., hidden) mean final-block patch token
    cls_layers: np.ndarray | None  # (..., LAST_BLOCKS, hidden) layer-normalised CLS of the last blocks


def _concatenate_normalised(parts: list[np.ndarray]) -> np.ndarray:
    """L2-normalise each part, concatenate, and normalise again so every part weighs equally."""
    return l2_normalize(np.concatenate([l2_normalize(part) for part in parts], axis=-1))


def pool_features(tokens: ViewTokens, pooling: str) -> np.ndarray:
    """Combine raw token features into unit-length embeddings."""
    if pooling == "cls":
        return l2_normalize(tokens.cls)
    if pooling == "cls_patchmean":
        return _concatenate_normalised([tokens.cls, tokens.patch_mean])
    if pooling in BLOCK_POOLINGS:
        if tokens.cls_layers is None:
            raise ValueError(f"{pooling} pooling needs per-block CLS tokens")
        parts = [tokens.cls_layers[..., block, :] for block in range(tokens.cls_layers.shape[-2])]
        if pooling == "cls_last4_patchmean":
            parts.append(tokens.patch_mean)
        return _concatenate_normalised(parts)
    raise ValueError(f"unknown pooling {pooling!r}; expected one of {POOLINGS}")


@dataclass(frozen=True)
class EncoderSettings:
    version: str
    weights_dir: str  # directory name under MODEL_DIR/pretrained
    input_size: int
    pooling: str
    views: int
    batch_size: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EncoderSettings:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__})


class Dinov2Encoder:
    def __init__(self, weights_dir: Path, input_size: int, pooling: str, views: int, batch_size: int) -> None:
        receipt = weights_dir / "SOURCE.json"
        if not receipt.is_file() or not (weights_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"DINOv2 weights not found in {weights_dir}; run `python -m scripts.download_assets`."
            )
        if pooling not in POOLINGS:
            raise ValueError(f"unknown pooling {pooling!r}; expected one of {POOLINGS}")
        if not 1 <= views <= len(DIHEDRAL_TRANSFORMS):
            raise ValueError(f"views must be between 1 and {len(DIHEDRAL_TRANSFORMS)}")
        source = json.loads(receipt.read_text(encoding="utf-8"))
        self.name: str = source["model"]
        self.input_size = input_size
        self.pooling = pooling
        self.views = views
        self.batch_size = batch_size
        self._model = AutoModel.from_pretrained(weights_dir, local_files_only=True).eval()
        hidden = int(self._model.config.hidden_size)
        self.dimension = hidden * _POOLING_WIDTH[pooling]
        self.fingerprint = (
            f"{self.name}@{source['revision'][:12]}|{PREPROCESSING_VERSION}|size={input_size}"
            f"|pool={pooling}|views={views}|dim={self.dimension}"
        )

    @classmethod
    def from_settings(cls, model_dir: Path, settings: EncoderSettings) -> Dinov2Encoder:
        return cls(
            model_dir / "pretrained" / settings.weights_dir,
            settings.input_size,
            settings.pooling,
            settings.views,
            settings.batch_size,
        )

    @torch.inference_mode()
    def token_features(self, images: list[Image.Image], view: int) -> ViewTokens:
        """Raw token features of every image for one dihedral view (leading axis N)."""
        cls_parts, patch_parts, layer_parts = [], [], []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            batch = np.stack([to_model_input(dihedral_view(image, view), self.input_size) for image in chunk])
            outputs = self._model(pixel_values=torch.from_numpy(batch), output_hidden_states=True)
            cls_parts.append(outputs.pooler_output.float().numpy())
            patch_parts.append(outputs.last_hidden_state[:, 1:, :].mean(dim=1).float().numpy())
            # hidden_states are pre-normalisation block outputs; apply the final layer norm to each,
            # exactly as pooler_output does for the last block.
            blocks = [self._model.layernorm(hidden)[:, 0] for hidden in outputs.hidden_states[-LAST_BLOCKS:]]
            layer_parts.append(torch.stack(blocks, dim=1).float().numpy())
        return ViewTokens(np.concatenate(cls_parts), np.concatenate(patch_parts), np.concatenate(layer_parts))

    def view_embeddings(self, images: list[Image.Image]) -> np.ndarray:
        """Unit-length embeddings per dihedral view, shaped (views, N, dimension)."""
        return np.stack(
            [pool_features(self.token_features(images, view), self.pooling) for view in range(self.views)]
        )

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Final (N, dimension) embeddings: L2-normalised mean over the configured views."""
        return l2_normalize(self.view_embeddings(images).mean(axis=0))
