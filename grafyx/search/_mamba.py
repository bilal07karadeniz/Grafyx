"""Mamba (Selective State Space Model) inference in numpy/cupy.

Implements the core building blocks for M5 (bi-encoder) and M6 (cross-encoder):
- selective_scan: The Mamba SSM core — O(n) sequential state update
- MambaBlock: Full Mamba layer (norm -> in_proj -> SSM -> gate -> out_proj + residual)
- AttentionBlock: Standard self-attention (for hybrid Mamba+Attention models)
- Helper functions: layer_norm, softplus, silu, softmax
"""
import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np


def selective_scan(x, A, B, C, D, delta):
    """Mamba selective scan. Pure numpy, no dependencies.

    Args:
        x: (seq_len, d_inner) — input sequence
        A: (d_inner, d_state) — state transition (log space, already negated)
        B: (seq_len, d_state) — input-dependent input matrix
        C: (seq_len, d_state) — input-dependent output matrix
        D: (d_inner,) — skip connection
        delta: (seq_len, d_inner) — input-dependent step size

    Returns:
        y: (seq_len, d_inner) — output sequence
    """
    seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # Discretize A and B
    dA = xp.exp(delta[:, :, None] * A[None, :, :])  # (seq_len, d_inner, d_state)
    dB = delta[:, :, None] * B[:, None, :]  # (seq_len, d_inner, d_state)

    h = xp.zeros((d_inner, d_state), dtype=x.dtype)
    ys = xp.zeros_like(x)

    for t in range(seq_len):
        h = dA[t] * h + dB[t] * x[t, :, None]
        ys[t] = (h * C[t, None, :]).sum(axis=-1) + D * x[t]

    return ys


class MambaBlock:
    """Single Mamba block with in_proj, SSM, out_proj, and LayerNorm."""

    def __init__(self, weights: dict):
        self.in_proj_w = xp.asarray(weights["in_proj_w"])  # (d_model, 2*d_inner)
        self.in_proj_b = xp.asarray(weights["in_proj_b"])
        self.A_log = xp.asarray(weights["A_log"])  # (d_inner, d_state)
        self.D = xp.asarray(weights["D"])  # (d_inner,)
        self.dt_proj_w = xp.asarray(weights["dt_proj_w"])
        self.dt_proj_b = xp.asarray(weights["dt_proj_b"])
        self.B_proj_w = xp.asarray(weights["B_proj_w"])
        self.C_proj_w = xp.asarray(weights["C_proj_w"])
        self.out_proj_w = xp.asarray(weights["out_proj_w"])
        self.out_proj_b = xp.asarray(weights["out_proj_b"])
        self.norm_w = xp.asarray(weights["norm_w"])
        self.norm_b = xp.asarray(weights["norm_b"])

    def __call__(self, x):
        """Forward pass. x: (seq_len, d_model) -> (seq_len, d_model)."""
        residual = x
        x = _layer_norm(x, self.norm_w, self.norm_b)

        # In projection -> split into x and z (gate)
        xz = x @ self.in_proj_w + self.in_proj_b
        d_inner = xz.shape[-1] // 2
        x_inner, z = xz[:, :d_inner], xz[:, d_inner:]

        # Compute input-dependent B, C, delta
        B = x_inner @ self.B_proj_w
        C = x_inner @ self.C_proj_w
        delta = _softplus(x_inner @ self.dt_proj_w + self.dt_proj_b)

        # SSM
        A = -xp.exp(self.A_log)
        y = selective_scan(x_inner, A, B, C, self.D, delta)

        # Gate and output projection
        y = y * _silu(z)
        y = y @ self.out_proj_w + self.out_proj_b

        return y + residual


class AttentionBlock:
    """Standard multi-head self-attention block for Mamba+Attention hybrid."""

    def __init__(self, weights: dict):
        self.qkv_w = xp.asarray(weights["qkv_w"])  # (d_model, 3*d_model)
        self.qkv_b = xp.asarray(weights["qkv_b"])
        self.out_w = xp.asarray(weights["out_w"])
        self.out_b = xp.asarray(weights["out_b"])
        self.ffn_w1 = xp.asarray(weights["ffn_w1"])
        self.ffn_b1 = xp.asarray(weights["ffn_b1"])
        self.ffn_w2 = xp.asarray(weights["ffn_w2"])
        self.ffn_b2 = xp.asarray(weights["ffn_b2"])
        self.norm1_w = xp.asarray(weights["norm1_w"])
        self.norm1_b = xp.asarray(weights["norm1_b"])
        self.norm2_w = xp.asarray(weights["norm2_w"])
        self.norm2_b = xp.asarray(weights["norm2_b"])
        self.n_heads = weights.get("n_heads", 6)

    def __call__(self, x):
        """x: (seq_len, d_model) -> (seq_len, d_model)."""
        # Self-attention with residual
        residual = x
        x = _layer_norm(x, self.norm1_w, self.norm1_b)
        qkv = x @ self.qkv_w + self.qkv_b
        d = x.shape[-1]
        q, k, v = qkv[:, :d], qkv[:, d:2*d], qkv[:, 2*d:]

        # Multi-head attention
        head_dim = d // self.n_heads
        seq_len = x.shape[0]
        q = q.reshape(seq_len, self.n_heads, head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, self.n_heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, self.n_heads, head_dim).transpose(1, 0, 2)

        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(head_dim)
        attn = _softmax(scores)
        out = (attn @ v).transpose(1, 0, 2).reshape(seq_len, d)
        x = residual + out @ self.out_w + self.out_b

        # FFN with residual
        residual = x
        x = _layer_norm(x, self.norm2_w, self.norm2_b)
        x = _silu(x @ self.ffn_w1 + self.ffn_b1) @ self.ffn_w2 + self.ffn_b2
        return x + residual


def _layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return w * (x - mean) / xp.sqrt(var + eps) + b


def _softplus(x):
    return xp.log1p(xp.exp(xp.clip(x, -20, 20)))


def _silu(x):
    return x / (1.0 + xp.exp(-xp.clip(x, -20, 20)))


def _softmax(x):
    e = xp.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
