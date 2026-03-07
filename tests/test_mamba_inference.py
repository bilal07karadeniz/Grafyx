"""Tests for Mamba and Attention block numpy inference."""
import numpy as np
import pytest
from grafyx.search._mamba import (
    selective_scan, MambaBlock, AttentionBlock,
    _layer_norm, _softplus, _silu, _softmax,
)


class TestHelpers:
    def test_layer_norm_zero_mean_unit_var(self):
        x = np.random.randn(5, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        out = _layer_norm(x, w, b)
        # Each row should have ~zero mean and ~unit var
        assert abs(out.mean(axis=-1)).max() < 1e-5
        assert abs(out.var(axis=-1) - 1.0).max() < 1e-4

    def test_softplus_positive(self):
        x = np.array([-10, -1, 0, 1, 10], dtype=np.float32)
        out = _softplus(x)
        assert (out > 0).all()

    def test_silu_zero_at_large_negative(self):
        x = np.array([-20], dtype=np.float32)
        out = _silu(x)
        assert abs(out[0]) < 0.01

    def test_softmax_sums_to_one(self):
        x = np.random.randn(3, 5).astype(np.float32)
        out = _softmax(x)
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-5)

    def test_softmax_all_positive(self):
        x = np.array([[-100, 0, 100]], dtype=np.float32)
        out = _softmax(x)
        assert (out >= 0).all()


class TestSelectiveScan:
    def test_output_shape(self):
        seq_len, d_inner, d_state = 10, 8, 4
        x = np.random.randn(seq_len, d_inner).astype(np.float32)
        A = np.random.randn(d_inner, d_state).astype(np.float32) * 0.1
        B = np.random.randn(seq_len, d_state).astype(np.float32)
        C = np.random.randn(seq_len, d_state).astype(np.float32)
        D = np.random.randn(d_inner).astype(np.float32)
        delta = np.abs(np.random.randn(seq_len, d_inner).astype(np.float32)) * 0.1

        y = selective_scan(x, A, B, C, D, delta)
        assert y.shape == (seq_len, d_inner)

    def test_deterministic(self):
        np.random.seed(42)
        seq_len, d_inner, d_state = 5, 4, 2
        x = np.random.randn(seq_len, d_inner).astype(np.float32)
        A = -np.abs(np.random.randn(d_inner, d_state).astype(np.float32))
        B = np.random.randn(seq_len, d_state).astype(np.float32)
        C = np.random.randn(seq_len, d_state).astype(np.float32)
        D = np.ones(d_inner, dtype=np.float32)
        delta = np.abs(np.random.randn(seq_len, d_inner).astype(np.float32)) * 0.1

        y1 = selective_scan(x, A, B, C, D, delta)
        y2 = selective_scan(x, A, B, C, D, delta)
        np.testing.assert_array_equal(y1, y2)

    def test_zero_input_zero_output(self):
        seq_len, d_inner, d_state = 5, 4, 2
        x = np.zeros((seq_len, d_inner), dtype=np.float32)
        A = -np.ones((d_inner, d_state), dtype=np.float32)
        B = np.zeros((seq_len, d_state), dtype=np.float32)
        C = np.zeros((seq_len, d_state), dtype=np.float32)
        D = np.zeros(d_inner, dtype=np.float32)
        delta = np.ones((seq_len, d_inner), dtype=np.float32) * 0.1

        y = selective_scan(x, A, B, C, D, delta)
        np.testing.assert_allclose(y, 0.0, atol=1e-7)


def _make_mamba_weights(d_model=16, d_inner=8, d_state=4):
    """Create random MambaBlock weights for testing."""
    np.random.seed(123)
    s = 0.1  # scale for stability
    return {
        "in_proj_w": np.random.randn(d_model, 2 * d_inner).astype(np.float32) * s,
        "in_proj_b": np.zeros(2 * d_inner, dtype=np.float32),
        "A_log": np.random.randn(d_inner, d_state).astype(np.float32) * s,
        "D": np.ones(d_inner, dtype=np.float32),
        "dt_proj_w": np.random.randn(d_inner, d_inner).astype(np.float32) * s,
        "dt_proj_b": np.zeros(d_inner, dtype=np.float32),
        "B_proj_w": np.random.randn(d_inner, d_state).astype(np.float32) * s,
        "C_proj_w": np.random.randn(d_inner, d_state).astype(np.float32) * s,
        "out_proj_w": np.random.randn(d_inner, d_model).astype(np.float32) * s,
        "out_proj_b": np.zeros(d_model, dtype=np.float32),
        "norm_w": np.ones(d_model, dtype=np.float32),
        "norm_b": np.zeros(d_model, dtype=np.float32),
    }


def _make_attention_weights(d_model=12, n_heads=3, d_ffn=24):
    """Create random AttentionBlock weights."""
    np.random.seed(456)
    s = 0.1
    return {
        "qkv_w": np.random.randn(d_model, 3 * d_model).astype(np.float32) * s,
        "qkv_b": np.zeros(3 * d_model, dtype=np.float32),
        "out_w": np.random.randn(d_model, d_model).astype(np.float32) * s,
        "out_b": np.zeros(d_model, dtype=np.float32),
        "ffn_w1": np.random.randn(d_model, d_ffn).astype(np.float32) * s,
        "ffn_b1": np.zeros(d_ffn, dtype=np.float32),
        "ffn_w2": np.random.randn(d_ffn, d_model).astype(np.float32) * s,
        "ffn_b2": np.zeros(d_model, dtype=np.float32),
        "norm1_w": np.ones(d_model, dtype=np.float32),
        "norm1_b": np.zeros(d_model, dtype=np.float32),
        "norm2_w": np.ones(d_model, dtype=np.float32),
        "norm2_b": np.zeros(d_model, dtype=np.float32),
        "n_heads": n_heads,
    }


class TestMambaBlock:
    def test_output_shape(self):
        w = _make_mamba_weights(d_model=16, d_inner=8, d_state=4)
        block = MambaBlock(w)
        x = np.random.randn(10, 16).astype(np.float32) * 0.1
        y = block(x)
        assert y.shape == (10, 16)

    def test_residual_connection(self):
        """Output should be close to input for small weights (residual dominant)."""
        w = _make_mamba_weights()
        # Make weights very small
        for k in w:
            if isinstance(w[k], np.ndarray) and "norm" not in k and k != "D":
                w[k] *= 0.001
        block = MambaBlock(w)
        x = np.random.randn(5, 16).astype(np.float32)
        y = block(x)
        # With tiny weights, output ~ input (residual)
        np.testing.assert_allclose(y, x, atol=0.5)

    def test_deterministic(self):
        w = _make_mamba_weights()
        block = MambaBlock(w)
        x = np.random.randn(5, 16).astype(np.float32) * 0.1
        y1 = block(x)
        y2 = block(x)
        np.testing.assert_array_equal(y1, y2)


class TestAttentionBlock:
    def test_output_shape(self):
        w = _make_attention_weights(d_model=12, n_heads=3)
        block = AttentionBlock(w)
        x = np.random.randn(8, 12).astype(np.float32) * 0.1
        y = block(x)
        assert y.shape == (8, 12)

    def test_deterministic(self):
        w = _make_attention_weights()
        block = AttentionBlock(w)
        x = np.random.randn(6, 12).astype(np.float32) * 0.1
        y1 = block(x)
        y2 = block(x)
        np.testing.assert_array_equal(y1, y2)

    def test_residual_connection(self):
        w = _make_attention_weights()
        for k in w:
            if isinstance(w[k], np.ndarray) and "norm" not in k:
                w[k] *= 0.001
        block = AttentionBlock(w)
        x = np.random.randn(4, 12).astype(np.float32)
        y = block(x)
        np.testing.assert_allclose(y, x, atol=0.5)
