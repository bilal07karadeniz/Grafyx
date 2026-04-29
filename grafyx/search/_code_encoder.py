"""Code Search Encoder -- M5 Mamba bi-encoder inference.

Pre-computes embeddings for all codebase symbols at initialization,
then answers semantic queries via cosine similarity in ~10ms.
Falls back gracefully when model weights are not available.
"""
import logging
import numpy as np
from pathlib import Path

from grafyx.search._mamba import MambaBlock, FeedForwardBlock, _layer_norm
from grafyx.search._tokenizer import CodeTokenizer

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent / "model"


class CodeEncoder:
    """Mamba bi-encoder for semantic code search.

    Usage:
        encoder = CodeEncoder()
        if encoder.is_available:
            encoder.build_index(symbols)  # Pre-compute at init
            results = encoder.search("handle user login", top_k=50)
    """

    def __init__(self):
        self._loaded = False
        self._blocks: list[MambaBlock] = []
        self._token_embedding: np.ndarray | None = None
        self._pos_embedding: np.ndarray | None = None
        self._final_norm_w: np.ndarray | None = None
        self._final_norm_b: np.ndarray | None = None
        self._projection_w: np.ndarray | None = None
        self._projection_b: np.ndarray | None = None
        self._tokenizer = CodeTokenizer()
        self._embeddings: np.ndarray | None = None  # Pre-computed code embeddings
        self._embedding_index: list[dict] = []  # Symbol info for each embedding

    def _ensure_loaded(self):
        if self._loaded:
            return
        weights_path = _MODEL_DIR / "code_encoder_weights.npz"
        if not weights_path.exists():
            raise FileNotFoundError("Code encoder weights not found")

        data = np.load(weights_path)

        # Load embeddings
        self._token_embedding = data["token_embedding"]
        self._pos_embedding = data["pos_embedding"]

        # Load sequence blocks (auto-detect Mamba vs FeedForward)
        i = 0
        while f"layer_{i}_norm.weight" in data:
            prefix = f"layer_{i}_"
            is_ff = f"{prefix}ff.w1" in data

            if is_ff:
                block_weights = {
                    "norm_w": data[f"{prefix}norm.weight"],
                    "norm_b": data[f"{prefix}norm.bias"],
                    "ff_w1": data[f"{prefix}ff.w1"],
                    "ff_b1": data[f"{prefix}ff.b1"],
                    "ff_w2": data[f"{prefix}ff.w2"],
                    "ff_b2": data[f"{prefix}ff.b2"],
                }
                self._blocks.append(FeedForwardBlock(block_weights))
            else:
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
                self._blocks.append(MambaBlock(block_weights))
            i += 1

        # Final norm + projection
        self._final_norm_w = data["final_norm_w"]
        self._final_norm_b = data["final_norm_b"]
        self._projection_w = data["projection_w"]  # Already pre-transposed
        self._projection_b = data["projection_b"]

        self._loaded = True

    def encode(self, text: str, max_length: int = 0) -> np.ndarray:
        """Encode text to embedding vector.

        Returns:
            L2-normalized embedding vector (embed_dim,).
        """
        self._ensure_loaded()
        # Auto-detect max_length from position embedding size
        if max_length <= 0:
            max_length = self._pos_embedding.shape[0]
        token_ids = self._tokenizer.encode(text, max_length=max_length)

        # Token + positional embedding
        x = self._token_embedding[token_ids] + self._pos_embedding[:len(token_ids)]

        # Forward through Mamba blocks
        for block in self._blocks:
            x = block(x)

        # Final norm
        x = _layer_norm(x, self._final_norm_w, self._final_norm_b)

        # Mean pooling (exclude padding: token_id != 0)
        mask = np.array(token_ids) != 0
        mask_expanded = mask[:, None].astype(np.float32)
        pooled = (x * mask_expanded).sum(axis=0) / max(mask.sum(), 1)

        # Projection
        emb = pooled @ self._projection_w + self._projection_b

        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        return emb.astype(np.float32)

    def build_index(self, symbols: list[dict]):
        """Pre-compute embeddings for all symbols. Call once at init.

        Args:
            symbols: List of dicts with keys: name, docstring, file, kind
        """
        self._ensure_loaded()
        embeddings = []
        self._embedding_index = []

        for sym in symbols:
            name = sym.get("name", "")
            doc = sym.get("docstring", "") or ""
            text = f"{name} {doc[:200]}".strip()
            if not text:
                continue
            emb = self.encode(text)
            embeddings.append(emb)
            self._embedding_index.append(sym)

        if embeddings:
            self._embeddings = np.stack(embeddings)
        else:
            self._embeddings = None

        logger.info("Code encoder: indexed %d symbols", len(embeddings))

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, str, float]]:
        """Semantic search: encode query, cosine sim against index.

        Returns:
            List of (name, file_path, score) tuples sorted by score desc.
        """
        if self._embeddings is None or len(self._embedding_index) == 0:
            return []
        q_emb = self.encode(query)
        sims = self._embeddings @ q_emb  # (n_symbols,)
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for i in top_idx:
            sym = self._embedding_index[i]
            score = float(sims[i])
            if score > 0.1:  # Minimum threshold
                results.append((sym.get("name", ""), sym.get("file", ""), score))
        return results

    @property
    def is_available(self) -> bool:
        return (
            (_MODEL_DIR / "code_encoder_weights.npz").exists()
            and self._tokenizer.is_available
        )


# Lazy singleton
_encoder: CodeEncoder | None = None


def get_code_encoder() -> CodeEncoder | None:
    """Get code encoder singleton, or None if not available."""
    global _encoder
    if _encoder is None:
        _encoder = CodeEncoder()
    return _encoder if _encoder.is_available else None
