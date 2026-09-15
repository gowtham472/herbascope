"""Canonical repository-relative locations used by the offline pipeline.

The API does not import these defaults directly; it resolves the same layout from
environment variables (see services/api/app/config.py) so artifacts stay replaceable.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "pipeline-v1.json"

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCES_DIR = DATA_DIR / "references"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SPLITS_DIR = DATA_DIR / "splits"
METADATA_DIR = DATA_DIR / "metadata"

MODELS_DIR = REPO_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
CLASSIFIERS_DIR = MODELS_DIR / "classifiers"
INDEXES_DIR = MODELS_DIR / "indexes"
MODEL_CONFIGS_DIR = MODELS_DIR / "configs"
MODEL_REPORTS_DIR = MODELS_DIR / "reports"

DOCS_REPORTS_DIR = REPO_ROOT / "docs" / "reports"
