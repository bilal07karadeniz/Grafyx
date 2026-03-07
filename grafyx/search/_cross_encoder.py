"""Cross-Encoder Reranker -- M6 Mamba+Attention hybrid inference.

Reranks top-15 search candidates using full query-code interaction.
Falls back gracefully when model weights are not available.
"""
import logging
import numpy as np
from pathlib import Path

from grafyx.search._mamba import MambaBlock, AttentionBlock, _layer_norm
from grafyx.search._tokenizer import CodeTokenizer

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent / "model"


class CrossEncoder:
    """Mamba+Attention cross-encoder for search result reranking.

    Input: [CLS] query [SEP] code_name code_doc [SEP]
    Output: relevance score in [0, 1]
    """

    def __init__(self):
        self._loaded = False
        self._mamba_blocks: list[MambaBlock] = []
        self._attn_blocks: list[AttentionBlock] = []
        self._token_embedding: np.ndarray | None = None
        self._pos_embedding: np.ndarray | None = None
        self._final_norm_w: np.ndarray | None = None
        self._final_norm_b: np.ndarray | None = None
        self._head_layers: list[tuple[np.ndarray, np.ndarray]] = []
        self._tokenizer = CodeTokenizer()

    def _ensure_loaded(self):
        if self._loaded:
            return
        weights_path = _MODEL_DIR / "cross_encoder_weights.npz"
        if not weights_path.exists():
            raise FileNotFoundError("Cross-encoder weights not found")

        data = np.load(weights_path)
        self._token_embedding = data["token_embedding"]
        self._pos_embedding = data["pos_embedding"]

        # Load Mamba blocks
        i = 0
        while f"mamba_{i}_norm.weight" in data:
            prefix = f"mamba_{i}_"
            block_weights = {
                "norm_w": data[f"{prefix}norm.weight"],
                "norm_b": data[f"{prefix}norm.bias"],
                "in_proj_w": data.get(
                    f"{prefix}mamba.in_proj.weight", np.zeros((1, 1))
                ).T,
                "in_proj_b": data.get(
                    f"{prefix}mamba.in_proj.bias", np.zeros(1)
                ),
                "A_log": data.get(
                    f"{prefix}mamba.A_log", np.zeros((1, 1))
                ),
                "D": data.get(f"{prefix}mamba.D", np.ones(1)),
                "dt_proj_w": data.get(
                    f"{prefix}mamba.dt_proj.weight", np.zeros((1, 1))
                ).T,
                "dt_proj_b": data.get(
                    f"{prefix}mamba.dt_proj.bias", np.zeros(1)
                ),
                "B_proj_w": data.get(
                    f"{prefix}mamba.B_proj.weight", np.zeros((1, 1))
                ).T,
                "C_proj_w": data.get(
                    f"{prefix}mamba.C_proj.weight", np.zeros((1, 1))
                ).T,
                "out_proj_w": data.get(
                    f"{prefix}mamba.out_proj.weight", np.zeros((1, 1))
                ).T,
                "out_proj_b": data.get(
                    f"{prefix}mamba.out_proj.bias", np.zeros(1)
                ),
            }
            self._mamba_blocks.append(MambaBlock(block_weights))
            i += 1

        # Load Attention blocks
        i = 0
        while f"attn_{i}_self_attn.in_proj_weight" in data:
            prefix = f"attn_{i}_"
            attn_weights = {
                "qkv_w": data[f"{prefix}self_attn.in_proj_weight"].T,
                "qkv_b": data[f"{prefix}self_attn.in_proj_bias"],
                "out_w": data[f"{prefix}self_attn.out_proj.weight"].T,
                "out_b": data[f"{prefix}self_attn.out_proj.bias"],
                "ffn_w1": data[f"{prefix}linear1.weight"].T,
                "ffn_b1": data[f"{prefix}linear1.bias"],
                "ffn_w2": data[f"{prefix}linear2.weight"].T,
                "ffn_b2": data[f"{prefix}linear2.bias"],
                "norm1_w": data[f"{prefix}norm1.weight"],
                "norm1_b": data[f"{prefix}norm1.bias"],
                "norm2_w": data[f"{prefix}norm2.weight"],
                "norm2_b": data[f"{prefix}norm2.bias"],
                "n_heads": 6,
            }
            self._attn_blocks.append(AttentionBlock(attn_weights))
            i += 1

        self._final_norm_w = data["final_norm_w"]
        self._final_norm_b = data["final_norm_b"]

        # Head layers
        head_idx = 0
        while f"head_{head_idx}.weight" in data:
            W = data[f"head_{head_idx}.weight"]  # Pre-transposed
            b = data[f"head_{head_idx}.bias"]
            self._head_layers.append((W, b))
            head_idx += 1

        self._loaded = True

    def score(self, query: str, code_text: str, max_length: int = 512) -> float:
        """Score relevance of (query, code_text) pair.

        Returns:
            Score in [0, 1], where 1.0 = highly relevant.
        """
        self._ensure_loaded()

        # Combine with separator
        combined = f"{query} [SEP] {code_text}"
        token_ids = self._tokenizer.encode(combined, max_length=max_length)

        # Token + positional embedding
        x = self._token_embedding[token_ids] + self._pos_embedding[:len(token_ids)]

        # Mamba layers
        for block in self._mamba_blocks:
            x = block(x)

        # Attention layers
        for block in self._attn_blocks:
            x = block(x)

        # Final norm
        x = _layer_norm(x, self._final_norm_w, self._final_norm_b)

        # CLS pooling (first token)
        cls = x[0]

        # Classification head
        for i, (W, b) in enumerate(self._head_layers):
            cls = cls @ W + b
            if i < len(self._head_layers) - 1:
                cls = np.maximum(cls, 0)  # ReLU

        # Sigmoid
        logit = float(cls.item() if hasattr(cls, 'item') else cls)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20))))

    def rerank(self, query: str, candidates: list[dict], top_k: int = 15) -> list[dict]:
        """Rerank search candidates by cross-encoder score.

        Args:
            query: Search query
            candidates: List of search result dicts (must have 'name' and optionally 'context')
            top_k: Number of results to return

        Returns:
            Reranked list of candidate dicts with updated scores.
        """
        scored = []
        for cand in candidates[:top_k]:
            code_text = f"{cand.get('name', '')} {cand.get('context', '')}"
            score = self.score(query, code_text)
            cand_copy = dict(cand)
            cand_copy["cross_encoder_score"] = score
            scored.append((score, cand_copy))

        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored]

    @property
    def is_available(self) -> bool:
        return (
            (_MODEL_DIR / "cross_encoder_weights.npz").exists()
            and self._tokenizer.is_available
        )


# Lazy singleton
_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder | None:
    """Get cross-encoder singleton, or None if not available."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder()
    return _cross_encoder if _cross_encoder.is_available else None
