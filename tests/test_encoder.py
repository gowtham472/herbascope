import numpy as np
import pytest

from ml.encoders.dinov2_encoder import EncoderSettings, l2_normalize, pool_features
from ml.training.train_classifier import training_rows


def test_cls_pooling_is_unit_length_cls_token():
    cls = np.array([[3.0, 4.0]], dtype=np.float32)
    patch = np.array([[1.0, 0.0]], dtype=np.float32)
    assert np.allclose(pool_features(cls, patch, "cls"), [[0.6, 0.8]])


def test_cls_patchmean_pooling_weights_both_parts_equally():
    cls = np.array([[10.0, 0.0]], dtype=np.float32)
    patch = np.array([[0.0, 0.5]], dtype=np.float32)
    pooled = pool_features(cls, patch, "cls_patchmean")
    assert pooled.shape == (1, 4)
    assert np.allclose(pooled, [[1, 0, 0, 1]] / np.sqrt(2))
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0)


def test_pooling_rejects_unknown_recipe():
    with pytest.raises(ValueError, match="unknown pooling"):
        pool_features(np.ones((1, 2)), np.ones((1, 2)), "max")


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
