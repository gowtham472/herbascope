from typing import get_args

import numpy as np
import pytest

from ml.encoders.dinov2_encoder import (
    BLOCK_POOLINGS,
    LAST_BLOCKS,
    POOLINGS,
    EncoderSettings,
    ViewTokens,
    l2_normalize,
    pool_features,
)
from ml.pipeline_config import Pooling
from ml.training.train_classifier import training_rows


def _tokens(cls, patch, layers=None) -> ViewTokens:
    return ViewTokens(np.array(cls, dtype=np.float32), np.array(patch, dtype=np.float32), layers)


def test_cls_pooling_is_unit_length_cls_token():
    assert np.allclose(pool_features(_tokens([[3.0, 4.0]], [[1.0, 0.0]]), "cls"), [[0.6, 0.8]])


def test_cls_patchmean_pooling_weights_both_parts_equally():
    pooled = pool_features(_tokens([[10.0, 0.0]], [[0.0, 0.5]]), "cls_patchmean")
    assert pooled.shape == (1, 4)
    assert np.allclose(pooled, [[1, 0, 0, 1]] / np.sqrt(2))
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0)


def test_cls_last4_pooling_concatenates_normalised_blocks():
    layers = np.zeros((2, 3, LAST_BLOCKS, 2), dtype=np.float32)  # (views, N, blocks, hidden)
    for block in range(LAST_BLOCKS):
        layers[..., block, block % 2] = 10.0 * (block + 1)
    pooled = pool_features(_tokens(np.ones((2, 3, 2)), np.ones((2, 3, 2)), layers), "cls_last4")
    assert pooled.shape == (2, 3, 2 * LAST_BLOCKS)
    assert np.allclose(pooled[0, 0], np.array([1, 0, 0, 1, 1, 0, 0, 1]) / 2)


def test_cls_last4_patchmean_pooling_adds_patch_mean_as_equal_fifth_part():
    layers = np.zeros((1, LAST_BLOCKS, 2), dtype=np.float32)  # (N, blocks, hidden)
    layers[..., 0] = 5.0
    pooled = pool_features(_tokens([[1.0, 0.0]], [[0.0, 3.0]], layers), "cls_last4_patchmean")
    assert pooled.shape == (1, 2 * (LAST_BLOCKS + 1))
    assert np.allclose(pooled[0], np.array([1, 0] * LAST_BLOCKS + [0, 1]) / np.sqrt(LAST_BLOCKS + 1))


@pytest.mark.parametrize("pooling", BLOCK_POOLINGS)
def test_block_poolings_require_block_tokens(pooling):
    with pytest.raises(ValueError, match="per-block"):
        pool_features(_tokens([[1.0, 0.0]], [[1.0, 0.0]]), pooling)


def test_encoder_poolings_match_the_config_schema():
    assert get_args(Pooling) == POOLINGS
    assert set(BLOCK_POOLINGS) <= set(POOLINGS)


def test_pooling_rejects_unknown_recipe():
    with pytest.raises(ValueError, match="unknown pooling"):
        pool_features(_tokens([[1.0, 1.0]], [[1.0, 1.0]]), "max")


def test_l2_normalize_handles_view_axis():
    views = np.random.default_rng(0).normal(size=(8, 5, 16))
    assert np.allclose(np.linalg.norm(l2_normalize(views), axis=-1), 1.0)


def test_encoder_settings_round_trip(tmp_path):
    settings = EncoderSettings("encoder-v9", "dinov2_vitb14", 308, "cls_patchmean", 8, 16)
    settings.save(tmp_path / "encoder.json")
    assert EncoderSettings.load(tmp_path / "encoder.json") == settings


VIEW_VECTORS = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)


def test_training_rows_without_augmentation_use_normalised_view_average():
    x, y = training_rows(VIEW_VECTORS, np.array([0, 1, 0]), on_views=False)
    assert x.shape == (3, 2) and y.tolist() == [0, 1, 0]
    assert np.allclose(x, l2_normalize(VIEW_VECTORS.mean(axis=0)))


def test_training_rows_with_augmentation_keep_labels_aligned_to_views():
    x, y = training_rows(VIEW_VECTORS, np.array([0, 1, 0]), on_views=True)
    assert x.shape == (12, 2)
    assert y.tolist() == [0, 1, 0] * 4
    assert np.array_equal(x[3], VIEW_VECTORS[1, 0])
