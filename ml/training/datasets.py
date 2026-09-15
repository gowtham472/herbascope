"""Split-file access shared by every training and evaluation step."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml import paths

# Split roles. Only TRAIN feeds the classifier and the reference library.
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
HELDOUT_KNOWN = "heldout_known"  # held-out known-material fragment groups (never "unknown species")
AMBIGUITY_PROBE = "ambiguity_probe"  # byte-identical images filed under more than one class
OOD_CALIBRATION = "ood_calibration"  # DIMPSAR far-OOD subset used to calibrate
OOD_EVALUATION = "ood_evaluation"  # class-disjoint DIMPSAR far-OOD subset used to evaluate

ALL_SPLITS = (TRAIN, VALIDATION, TEST, HELDOUT_KNOWN, AMBIGUITY_PROBE, OOD_CALIBRATION, OOD_EVALUATION)

SPLIT_COLUMNS = ["image_id", "dataset", "class_name", "fragment_type", "group_id", "sha256", "path"]

OOD_CALIBRATION_LIMITATION = (
    "Because Mikrobat contains only two species, true held-out-species OOD calibration was not possible. "
    "HerbaScope X therefore uses held-out Mikrobat specimen/fragment groups to characterize within-distribution "
    "variability and DIMPSAR field-leaf imagery as a far-OOD negative set. The unknown threshold is calibrated "
    "against these distributions and evaluated separately for known-material retention and OOD rejection."
)


@dataclass(frozen=True)
class SplitFrame:
    name: str
    frame: pd.DataFrame

    def image_paths(self) -> list[Path]:
        return [paths.DATA_DIR / relative for relative in self.frame["path"]]


def split_path(name: str) -> Path:
    return paths.SPLITS_DIR / f"{name}.csv"


def load_split(name: str) -> SplitFrame:
    location = split_path(name)
    if not location.is_file():
        raise FileNotFoundError(f"{location} not found; run `python -m ml.training.prepare_dataset` first.")
    return SplitFrame(name, pd.read_csv(location, dtype=str, keep_default_na=False))


def embeddings_path(name: str) -> Path:
    return paths.EMBEDDINGS_DIR / f"{name}.npz"


@dataclass(frozen=True)
class SplitEmbeddings:
    split: SplitFrame
    vectors: np.ndarray  # (N, D) final embeddings (mean over dihedral views)
    view_vectors: np.ndarray  # (views, N, D) per-view embeddings, used for training augmentation
    fingerprint: str


def load_embeddings(name: str) -> SplitEmbeddings:
    """Load cached embeddings and verify they align row-for-row with the split file."""
    split = load_split(name)
    location = embeddings_path(name)
    if not location.is_file():
        raise FileNotFoundError(
            f"{location} not found; run `python -m ml.training.extract_embeddings` first."
        )
    with np.load(location, allow_pickle=False) as payload:
        ids = payload["image_ids"].tolist()
        vectors = payload["embeddings"]
        view_vectors = payload["view_embeddings"]
        fingerprint = str(payload["fingerprint"])
    if ids != split.frame["image_id"].tolist():
        raise ValueError(f"{location} is stale: its image ids do not match {split_path(name)}")
    return SplitEmbeddings(split, vectors, view_vectors, fingerprint)
