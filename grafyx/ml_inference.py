"""Shared ML model infrastructure. Lazy loading, GPU detection, numpy inference."""
import numpy as np
from pathlib import Path
from typing import Any

_MODEL_DIR = Path(__file__).parent / "search" / "model"


def _get_backend():
    """Auto-detect numpy (CPU) or cupy (GPU)."""
    try:
        import cupy as cp
        return cp
    except ImportError:
        return np


xp = _get_backend()


class MLPModel:
    """Generic MLP with numpy/cupy inference. Supports any layer count."""

    def __init__(self, weights_path: str | Path):
        self._weights_path = Path(weights_path)
        self._layers: list[tuple[Any, Any]] = []  # [(weight, bias), ...]
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not self._weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self._weights_path}")
        data = np.load(self._weights_path)
        i = 0
        while f"W{i}" in data:
            W = xp.asarray(data[f"W{i}"])
            b = xp.asarray(data[f"b{i}"])
            self._layers.append((W, b))
            i += 1
        self._loaded = True

    def predict(self, features: np.ndarray) -> float:
        """Forward pass. Returns sigmoid output."""
        self._ensure_loaded()
        x = xp.asarray(features, dtype=xp.float32)
        for i, (W, b) in enumerate(self._layers):
            x = x @ W + b
            if i < len(self._layers) - 1:  # ReLU for hidden layers
                x = xp.maximum(x, 0)
        # Sigmoid for output
        logit = float(x.item() if hasattr(x, 'item') else x)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))

    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Batch forward pass. features_batch shape: (N, D). Returns (N,) sigmoid outputs."""
        self._ensure_loaded()
        x = xp.asarray(features_batch, dtype=xp.float32)
        for i, (W, b) in enumerate(self._layers):
            x = x @ W + b
            if i < len(self._layers) - 1:
                x = xp.maximum(x, 0)
        # Sigmoid
        logits = x.flatten()
        if hasattr(logits, 'get'):  # cupy -> numpy
            logits = logits.get()
        else:
            logits = np.asarray(logits)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

    @property
    def is_available(self) -> bool:
        return self._weights_path.exists()


# Lazy singletons
_models: dict[str, MLPModel] = {}


def get_model(name: str) -> MLPModel | None:
    """Get a named model, or None if weights don't exist."""
    if name not in _models:
        path = _MODEL_DIR / f"{name}_weights.npz"
        model = MLPModel(path)
        _models[name] = model
    m = _models[name]
    return m if m.is_available else None
