"""Typed loader for ml/configs/pipeline.json — the single source of every pipeline choice.

Nothing in this file is a calibrated threshold. The config holds pinned sources, split
policy, the embedding recipe and the *objectives* that calibration scripts optimise;
calibrated values are written to models/ by those scripts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml import paths


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MikrobatSource(_Strict):
    name: str
    repository: str
    commit: str
    reported_images: int
    license_note: str

    @property
    def archive_url(self) -> str:
        owner_repo = self.repository.removeprefix("https://github.com/")
        return f"https://codeload.github.com/{owner_repo}/zip/{self.commit}"


class DimpsarSource(_Strict):
    name: str
    repository: str
    revision: str
    file: str
    sha256: str
    license: str
    reported_images: int

    @property
    def file_url(self) -> str:
        return f"{self.repository}/resolve/{self.revision}/{self.file}"


class Sources(_Strict):
    mikrobat: MikrobatSource
    dimpsar: DimpsarSource


class Backbone(_Strict):
    """A pinned pretrained DINOv2 checkpoint."""

    name: str
    hub_id: str
    revision: str
    local_dir: str
    license: str


class EncoderConfig(_Strict):
    version: str
    backbone: Backbone
    input_size: int = Field(gt=0, multiple_of=14)
    pooling: Literal["cls", "cls_patchmean"]
    views: int = Field(ge=1, le=8)
    batch_size: int = Field(gt=0)


class MikrobatCuration(_Strict):
    fragment_type_aliases: dict[str, str]
    near_duplicate_null_percentile: float = Field(gt=0, lt=100)
    heldout_min_images: int = Field(gt=0)


class SplitRatios(_Strict):
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitRatios:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        return self


class DimpsarOodConfig(_Strict):
    images_per_class: int = Field(gt=0)
    calibration_class_fraction: float = Field(gt=0, lt=1)


class ClassifierConfig(_Strict):
    version: str
    train_on_views: bool  # train on every dihedral view of each image (augmentation)
    c_grid: list[float] = Field(min_length=1)
    class_weight: str
    max_iter: int = Field(gt=0)


class ReferenceConfig(_Strict):
    version: str
    per_class: int = Field(gt=0)
    top_k: int = Field(gt=0)


class QualityConfig(_Strict):
    version: str
    min_side: int = Field(gt=0)
    lower_percentile: float = Field(ge=0, le=100)
    upper_percentile: float = Field(ge=0, le=100)
    tile_grid: int = Field(gt=1)
    tile_floor_percentile: float = Field(ge=0, le=100)


class UnknownConfig(_Strict):
    version: str
    target_known_acceptance: float = Field(gt=0, le=1)


class DecisionConfig(_Strict):
    version: str
    target_known_retention: float = Field(gt=0, le=1)
    target_selective_accuracy: float = Field(gt=0, le=1)


class PipelineConfig(_Strict):
    version: str
    seed: int
    sources: Sources
    encoder: EncoderConfig
    mikrobat_curation: MikrobatCuration
    splits: SplitRatios
    dimpsar_ood: DimpsarOodConfig
    classifier: ClassifierConfig
    references: ReferenceConfig
    quality: QualityConfig
    unknown: UnknownConfig
    decision: DecisionConfig


@lru_cache(maxsize=4)
def load_pipeline_config(path: Path = paths.CONFIG_PATH) -> PipelineConfig:
    return PipelineConfig.model_validate_json(path.read_text(encoding="utf-8"))
