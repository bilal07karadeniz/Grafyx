"""Tests for the shared ML inference module."""
import numpy as np
import pytest
from pathlib import Path

from grafyx.ml_inference import MLPModel, get_model, _models


def _make_weights(tmp_path, **arrays):
    """Save arrays as an npz file and return the path."""
    path = tmp_path / "test_weights.npz"
    np.savez(path, **arrays)
    return path


class TestMLPModel:
    def test_predict_matches_manual(self, tmp_path):
        """Create a tiny 2-layer MLP with known weights and verify output."""
        W0 = np.array([[0.5, -0.3], [0.2, 0.8]], dtype=np.float32)  # 2x2
        b0 = np.array([0.1, -0.1], dtype=np.float32)
        W1 = np.array([[0.4], [-0.6]], dtype=np.float32)  # 2x1
        b1 = np.array([0.0], dtype=np.float32)

        path = _make_weights(tmp_path, W0=W0, b0=b0, W1=W1, b1=b1)
        model = MLPModel(path)
        features = np.array([1.0, 0.5], dtype=np.float32)
        result = model.predict(features)

        # Manual computation:
        # h = relu([1.0, 0.5] @ [[0.5,-0.3],[0.2,0.8]] + [0.1,-0.1])
        #   = relu([0.5+0.1+0.1, -0.3+0.4-0.1]) = relu([0.7, 0.0]) = [0.7, 0.0]
        # logit = [0.7, 0.0] @ [[0.4],[-0.6]] + [0.0] = [0.28]
        # sigmoid(0.28) ≈ 0.5695
        assert 0.0 < result < 1.0
        assert abs(result - 0.5695) < 0.01

    def test_predict_three_layers(self, tmp_path):
        """Test with a 3-layer (2 hidden + output) MLP."""
        W0 = np.eye(2, dtype=np.float32)
        b0 = np.zeros(2, dtype=np.float32)
        W1 = np.eye(2, dtype=np.float32)
        b1 = np.zeros(2, dtype=np.float32)
        W2 = np.ones((2, 1), dtype=np.float32)
        b2 = np.zeros(1, dtype=np.float32)

        path = _make_weights(tmp_path, W0=W0, b0=b0, W1=W1, b1=b1, W2=W2, b2=b2)
        model = MLPModel(path)
        features = np.array([1.0, 1.0], dtype=np.float32)
        result = model.predict(features)

        # h0 = relu([1,1] @ I + 0) = [1,1]
        # h1 = relu([1,1] @ I + 0) = [1,1]
        # logit = [1,1] @ [[1],[1]] + 0 = 2.0
        # sigmoid(2.0) ≈ 0.8808
        assert abs(result - 0.8808) < 0.01

    def test_predict_batch(self, tmp_path):
        """Test batch prediction with multiple inputs."""
        W0 = np.array([[1.0], [1.0]], dtype=np.float32)  # 2x1
        b0 = np.array([0.0], dtype=np.float32)

        path = _make_weights(tmp_path, W0=W0, b0=b0)
        model = MLPModel(path)

        batch = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [-5.0, -5.0],
            [5.0, 5.0],
        ], dtype=np.float32)
        results = model.predict_batch(batch)

        assert results.shape == (4,)
        assert abs(results[0] - 0.5) < 0.01  # sigmoid(0) = 0.5
        assert results[1] > 0.8  # sigmoid(2) ≈ 0.88
        assert results[2] < 0.01  # sigmoid(-10) ≈ 0
        assert results[3] > 0.99  # sigmoid(10) ≈ 1

    def test_predict_batch_single_item(self, tmp_path):
        """Batch with a single item should work and match predict()."""
        W0 = np.array([[0.5], [-0.5]], dtype=np.float32)
        b0 = np.array([0.1], dtype=np.float32)

        path = _make_weights(tmp_path, W0=W0, b0=b0)
        model = MLPModel(path)

        features = np.array([1.0, 2.0], dtype=np.float32)
        single = model.predict(features)
        batch = model.predict_batch(features.reshape(1, -1))

        assert abs(single - batch[0]) < 1e-6

    def test_sigmoid_output_bounds(self, tmp_path):
        """Output should always be in (0, 1)."""
        W0 = np.array([[100.0]], dtype=np.float32)
        b0 = np.array([0.0], dtype=np.float32)

        path = _make_weights(tmp_path, W0=W0, b0=b0)
        model = MLPModel(path)

        # Very large positive input
        result_pos = model.predict(np.array([100.0], dtype=np.float32))
        assert 0.0 < result_pos <= 1.0

        # Very large negative input
        result_neg = model.predict(np.array([-100.0], dtype=np.float32))
        assert 0.0 <= result_neg < 1.0

    def test_relu_clips_negatives(self, tmp_path):
        """Hidden layers should apply ReLU (clip negatives to 0)."""
        W0 = np.array([[-1.0]], dtype=np.float32)  # 1x1, will negate input
        b0 = np.array([0.0], dtype=np.float32)
        W1 = np.array([[1.0]], dtype=np.float32)  # 1x1
        b1 = np.array([0.0], dtype=np.float32)

        path = _make_weights(tmp_path, W0=W0, b0=b0, W1=W1, b1=b1)
        model = MLPModel(path)

        # Input 5.0 -> h = relu(-5.0) = 0.0 -> logit = 0.0 -> sigmoid(0) = 0.5
        result = model.predict(np.array([5.0], dtype=np.float32))
        assert abs(result - 0.5) < 0.01

    def test_is_available_true(self, tmp_path):
        """is_available should be True when weights file exists."""
        path = _make_weights(tmp_path, W0=np.eye(2, dtype=np.float32),
                             b0=np.zeros(2, dtype=np.float32))
        model = MLPModel(path)
        assert model.is_available is True

    def test_is_available_false(self, tmp_path):
        """is_available should be False when weights file doesn't exist."""
        model = MLPModel(tmp_path / "nonexistent.npz")
        assert model.is_available is False

    def test_ensure_loaded_raises_for_missing_file(self, tmp_path):
        """_ensure_loaded should raise FileNotFoundError for missing weights."""
        model = MLPModel(tmp_path / "nonexistent.npz")
        with pytest.raises(FileNotFoundError, match="Model weights not found"):
            model._ensure_loaded()

    def test_lazy_loading(self, tmp_path):
        """Model should not load weights until first prediction."""
        # Single layer: 2->1 so predict() returns a scalar
        W0 = np.ones((2, 1), dtype=np.float32)
        b0 = np.zeros(1, dtype=np.float32)
        path = _make_weights(tmp_path, W0=W0, b0=b0)
        model = MLPModel(path)
        assert model._loaded is False
        model.predict(np.array([1.0, 1.0], dtype=np.float32))
        assert model._loaded is True

    def test_loads_only_once(self, tmp_path):
        """Multiple predictions should not re-load weights."""
        W0 = np.array([[1.0]], dtype=np.float32)
        b0 = np.array([0.0], dtype=np.float32)
        path = _make_weights(tmp_path, W0=W0, b0=b0)
        model = MLPModel(path)

        r1 = model.predict(np.array([1.0], dtype=np.float32))
        layers_id = id(model._layers)
        r2 = model.predict(np.array([2.0], dtype=np.float32))

        assert id(model._layers) == layers_id  # Same object, not reloaded


class TestGetModel:
    def test_returns_none_for_nonexistent(self):
        """get_model should return None when weights don't exist."""
        # Clear singleton cache to avoid stale entries
        _models.pop("definitely_nonexistent_model_xyz", None)
        result = get_model("definitely_nonexistent_model_xyz")
        assert result is None

    def test_returns_model_for_existing(self):
        """get_model should return MLPModel for known existing weights."""
        # The 'relevance' model should exist in grafyx/search/model/
        _models.pop("relevance", None)
        model = get_model("relevance")
        if model is not None:
            assert isinstance(model, MLPModel)
            assert model.is_available

    def test_caches_model_instances(self):
        """get_model should return the same instance on repeated calls."""
        _models.pop("definitely_nonexistent_cache_test", None)
        get_model("definitely_nonexistent_cache_test")
        # Even though it returned None, the MLPModel is cached
        assert "definitely_nonexistent_cache_test" in _models
