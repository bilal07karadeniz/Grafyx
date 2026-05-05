"""Optional embedding-based semantic search for Grafyx.

This module provides dense vector similarity search using fastembed (ONNX
Runtime, ~150 MB).  It is entirely optional -- if fastembed is not installed,
all methods gracefully degrade to returning empty results.

Why embeddings?
    Keyword search finds "send_message" for the query "send message" but fails
    for conceptual queries like "notification system" or "how does authentication
    work".  Embeddings encode semantic meaning into dense vectors, so
    semantically similar concepts (even with different words) produce similar
    vectors and high cosine similarity scores.

Architecture:
    - Two embedding levels: function-level and file-level.
    - Function-level: each function gets a compact text document combining its
      name, first docstring line, and rare source tokens.  Good for finding
      specific functions.
    - File-level: each file gets a document aggregating all its function names,
      class names, docstrings, and source tokens.  Good for finding relevant
      files when the user describes a feature area.
    - Embeddings are cached to disk (keyed by a fingerprint hash of function
      signatures) so subsequent startups are instant.
    - The index is built lazily in a background thread on first use, so the
      first search call returns token-based results immediately while embeddings
      are being computed.

Model choice:
    BAAI/bge-small-en-v1.5 is used by default.  It's a compact model (~33M
    params) that runs on CPU via ONNX Runtime (~150 MB).  This avoids the
    ~2 GB PyTorch dependency.  The model produces normalized embeddings, so
    cosine similarity reduces to a simple dot product.

Cache invalidation:
    The fingerprint is a SHA-256 hash of all function names, paths, and class
    names.  When the codebase changes (functions added/removed/renamed), the
    fingerprint changes and embeddings are rebuilt.  Function body changes
    WITHOUT signature changes do NOT invalidate the cache -- this is a
    deliberate tradeoff for faster startup at the cost of slightly stale
    embeddings.
"""

import logging
import os

from grafyx.graph import CodebaseGraph
from grafyx.search._tokens import _tokenize_source

logger = logging.getLogger(__name__)

# --- Optional Dependency Check ---
# fastembed is not a required dependency.  The _HAS_EMBEDDINGS flag is
# checked by CodeSearcher to decide whether to instantiate EmbeddingSearcher.
try:
    from fastembed import TextEmbedding  # type: ignore[import-untyped]
    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False


# --- Encoder registry ---
# Each entry maps a Grafyx-internal id to (a) the model name fastembed
# expects, (b) the query prefix the model was trained with (empty for
# jina-v2), and (c) optional custom-model registration metadata for
# fastembed.
ENCODER_REGISTRY = {
    "jina-v2": {
        "id": "jina-v2",
        "model_name": "jinaai/jina-embeddings-v2-base-code",
        "query_prefix": "",
        "needs_custom_registration": False,
        "dim": 768,
    },
    "coderankembed": {
        "id": "coderankembed",
        "model_name": "Bilal7Dev/grafyx-coderankembed-onnx",
        "query_prefix": "Represent this query for searching relevant code: ",
        "needs_custom_registration": True,
        "dim": 768,
        "hf_repo": "Bilal7Dev/grafyx-coderankembed-onnx",
        "model_file": "onnx/model.onnx",
    },
}

# jina-v2 is shippable today (Apache-2.0, fastembed-native). CodeRankEmbed
# becomes the default once the head-to-head benchmark in docs/benchmarks/0.2.0/
# confirms it beats jina-v2 by >=3 nDCG@10 points.
DEFAULT_ENCODER = "jina-v2"


def resolve_model_config(env_var: str = "GRAFYX_ENCODER") -> dict:
    """Look up the encoder config from the GRAFYX_ENCODER env var (or default).

    Returns a fresh dict copy so callers can mutate without affecting the
    registry.
    """
    chosen = os.environ.get(env_var, DEFAULT_ENCODER).strip().lower()
    if chosen not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder {chosen!r}. "
            f"Expected one of: {sorted(ENCODER_REGISTRY)}"
        )
    return dict(ENCODER_REGISTRY[chosen])


_CUSTOM_REGISTERED: set[str] = set()


def _register_custom_model_once(cfg: dict) -> None:
    """Register a custom fastembed model if not already registered.

    fastembed's add_custom_model is per-process global state; calling it
    twice raises. We guard with a module-level set.
    """
    if not cfg.get("needs_custom_registration"):
        return
    if cfg["model_name"] in _CUSTOM_REGISTERED:
        return
    if not _HAS_EMBEDDINGS:
        return
    try:
        from fastembed import TextEmbedding  # type: ignore[import-untyped]
        from fastembed.common.model_description import (  # type: ignore[import-untyped]
            ModelSource,
            PoolingType,
        )
    except ImportError:
        # Older/different fastembed builds expose model_description elsewhere.
        # Fall through silently — search() will surface a clear failure when
        # the model is actually requested.
        logger.debug("fastembed model_description API not found; skipping custom registration")
        return

    try:
        TextEmbedding.add_custom_model(
            model=cfg["model_name"],
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf=cfg["hf_repo"]),
            dim=cfg["dim"],
            model_file=cfg["model_file"],
        )
    except Exception as exc:  # fastembed raises if already registered or API mismatch
        logger.debug("fastembed.add_custom_model failed: %s", exc)
        return
    _CUSTOM_REGISTERED.add(cfg["model_name"])


class EmbeddingSearcher:
    """Semantic search using dense vector embeddings (optional).

    Requires ``fastembed`` to be installed (``pip install fastembed``).
    Uses ONNX Runtime under the hood (~150 MB), NOT PyTorch (~2 GB).

    Embeddings are cached to disk so that subsequent startups are instant.
    The index is built lazily in a background thread on first use, so the
    first search call returns token-based results immediately while
    embeddings are being computed.

    Typical lifecycle:
        1. CodeSearcher creates EmbeddingSearcher(graph)
        2. Calls build_in_background() -- starts a daemon thread
        3. First search() call may return [] while building
        4. Subsequent calls return embedding-based results
        5. On next startup, cache hit -> instant ready state

    Thread safety:
        _ready and _building flags are set atomically (GIL-protected bool
        assignments).  The build() method is safe to call from a background
        thread because it only writes to self._embeddings and self._metadata
        once the full computation is complete.
    """

    def __init__(
        self,
        graph: CodebaseGraph,
        cache_dir: str = "~/.grafyx/embeddings",
        model_id: str | None = None,
    ):
        from pathlib import Path

        self._graph = graph
        # Pull config from the registry (env-var-driven by default).
        self._cfg = (
            resolve_model_config()
            if model_id is None
            else dict(ENCODER_REGISTRY[model_id])
        )
        _register_custom_model_once(self._cfg)
        self._model_name = self._cfg["model_name"]
        self._query_prefix = self._cfg["query_prefix"]
        # Cache per-encoder so switching encoders doesn't blow away the
        # other's cached vectors.
        self._cache_dir = Path(os.path.expanduser(cache_dir)) / self._cfg["id"]
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # --- Function-level embedding state ---
        self._embeddings = None  # numpy array (N, dim) or None
        self._metadata: list[dict] = []  # [{name, file, class_name}]
        self._ready = False       # True when function embeddings are usable
        self._building = False    # True during background build (prevents double-build)
        self._fingerprint = ""    # Cache key derived from function signatures
        self._build_error: str | None = None  # Set when build() fails permanently

        # --- File-level embedding state ---
        self._file_embeddings = None  # numpy array (M, dim) or None
        self._file_metadata: list[dict] = []  # [{file}]
        self._file_ready = False  # True when file embeddings are usable

    # --- Availability Check ---

    @staticmethod
    def available() -> bool:
        """Check if fastembed is installed."""
        return _HAS_EMBEDDINGS

    # --- Fingerprinting ---

    def _compute_fingerprint(self) -> str:
        """Create a stable hash of the codebase function signatures.

        This serves as the cache key.  If the fingerprint matches a cached
        file on disk, we skip the expensive embedding computation.

        Note: only function names/paths/class names are hashed, NOT source
        code.  This means body-only changes don't invalidate the cache.
        This is intentional -- embedding the source body would make the
        fingerprint too volatile for minor edits.
        """
        import hashlib

        h = hashlib.sha256()
        for name, fpath, _source, cls_name in self._graph.iter_functions_with_source():
            h.update(f"{name}:{fpath}:{cls_name}\n".encode())
        return h.hexdigest()[:16]

    # --- Document Building ---

    def _build_documents(self) -> list[str]:
        """Create text documents for each function to embed.

        Each document is a compact representation of a function (~256 chars max)
        combining:
        - Display name: "ClassName.method_name" or just "function_name"
        - Short file path: just the filename for brevity
        - First docstring line: the most informative summary
        - Rare source tokens (up to 20): implementation-level keywords that
          aren't in the name or docstring

        The 256-char cap matches the sweet spot for the embedding model --
        longer documents don't significantly improve retrieval quality but
        slow down embedding computation.
        """
        docs: list[str] = []
        self._metadata = []
        for name, fpath, source, cls_name in self._graph.iter_functions_with_source():
            if not name:
                continue
            # Build concise document: name + first docstring sentence + rare source tokens
            short_path = fpath.rsplit("/", 1)[-1] if "/" in fpath else fpath
            # Extract docstring first line
            doc_line = ""
            if source:
                for line in source.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith(('"""', "'''", '"', "'")):
                        doc_line = stripped.strip("\"'").strip()[:100]
                        break
            # Rare source tokens (first 20 unique, excluding the function name
            # itself since it's already in the document)
            source_toks = []
            seen = set()
            for tok in _tokenize_source(source or ""):
                if tok not in seen and tok != name.lower():
                    seen.add(tok)
                    source_toks.append(tok)
                    if len(source_toks) >= 20:
                        break

            display = f"{cls_name}.{name}" if cls_name else name
            doc = f"{display} in {short_path}"
            if doc_line:
                doc += f": {doc_line}"
            if source_toks:
                doc += f". {' '.join(source_toks)}"
            docs.append(doc[:256])  # Cap at model sweet spot
            self._metadata.append({
                "name": name,
                "file": fpath,
                "class_name": cls_name,
            })
        return docs

    def _build_file_documents(self) -> list[str]:
        """Create text documents for each file to embed.

        Aggregates function names, class names, docstrings, and key source
        tokens per file into a single document for file-level semantic search.
        This provides a broader view than function-level embeddings -- useful
        for queries like "authentication module" or "database layer" that
        describe a feature area rather than a specific function.

        Each file document includes:
        - Short path (last 3 components): "backend/auth/views.py"
        - Function/method names (up to 15): the file's API surface
        - Class names: the file's type definitions
        - First docstring lines (up to 5): conceptual summary
        - Rare source tokens (up to 20): implementation keywords

        The 512-char cap is higher than function documents because file-level
        context benefits from more information density.
        """
        # --- Group function data by file ---
        file_data: dict[str, dict] = {}  # file -> {names, docs, tokens}
        for name, fpath, source, cls_name in self._graph.iter_functions_with_source():
            if not name or not fpath:
                continue
            if fpath not in file_data:
                file_data[fpath] = {"names": [], "docs": [], "tokens": set()}
            display = f"{cls_name}.{name}" if cls_name else name
            file_data[fpath]["names"].append(display)
            # Extract docstring first line
            if source:
                for line in source.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith(('"""', "'''", '"', "'")):
                        doc_line = stripped.strip("\"'").strip()[:60]
                        if doc_line:
                            file_data[fpath]["docs"].append(doc_line)
                        break
                # Rare source tokens
                for tok in _tokenize_source(source):
                    file_data[fpath]["tokens"].add(tok)

        # --- Add class info to file documents ---
        # Classes are important structural elements that should be represented
        # in the file's embedding document even if they have no methods.
        all_classes = self._graph.get_all_classes(max_results=500)
        for cls_dict in all_classes:
            cpath = cls_dict.get("file", "")
            cname = cls_dict.get("name", "")
            if not cpath or not cname:
                continue
            if cpath not in file_data:
                file_data[cpath] = {"names": [], "docs": [], "tokens": set()}
            file_data[cpath]["names"].append(cname)
            doc = cls_dict.get("docstring", "")
            if doc:
                first_line = doc.split("\n")[0].strip()[:60]
                if first_line:
                    file_data[cpath]["docs"].append(first_line)

        # --- Build final document strings ---
        docs: list[str] = []
        self._file_metadata = []
        for fpath, data in file_data.items():
            short_path = "/".join(fpath.replace("\\", "/").split("/")[-3:])
            names_str = ", ".join(data["names"][:15])
            docs_str = ". ".join(data["docs"][:5])
            tokens = sorted(data["tokens"])[:20]
            tokens_str = " ".join(tokens)
            doc = f"{short_path}: {names_str}"
            if docs_str:
                doc += f". {docs_str}"
            if tokens_str:
                doc += f". {tokens_str}"
            docs.append(doc[:512])  # More context per file than per function
            self._file_metadata.append({"file": fpath})
        return docs

    # --- Index Building ---

    def build(self) -> None:
        """Build or load the embedding index (function + file level).

        Attempts to load cached embeddings from disk first.  If cache miss,
        builds fresh embeddings using the fastembed model and saves them.

        Cache files per fingerprint:
        - {fp}_vectors.npy + {fp}_meta.json    (function-level)
        - {fp}_file_vectors.npy + {fp}_file_meta.json (file-level)

        Both levels are built independently -- if one is cached and the other
        isn't, only the missing one is rebuilt.
        """
        if not _HAS_EMBEDDINGS:
            return

        import json
        import os

        try:
            import numpy as np
        except ImportError:
            logger.debug("numpy not available, skipping embedding build")
            return

        fp = self._compute_fingerprint()
        self._fingerprint = fp
        logger.info("Grafyx embedding index: fingerprint %s (model=%s)", fp[:8], self._cfg["id"])

        # --- Try loading function-level embeddings from cache ---
        cache_file = self._cache_dir / f"{fp}_vectors.npy"
        meta_file = self._cache_dir / f"{fp}_meta.json"

        func_loaded = False
        if cache_file.exists() and meta_file.exists():
            try:
                self._embeddings = np.load(str(cache_file))
                with open(str(meta_file), "r") as f:
                    self._metadata = json.load(f)
                self._ready = True
                func_loaded = True
                logger.info("Embedding index: loaded %d function vectors from cache", len(self._metadata))
            except Exception:
                pass

        # --- Try loading file-level embeddings from cache ---
        file_cache = self._cache_dir / f"{fp}_file_vectors.npy"
        file_meta = self._cache_dir / f"{fp}_file_meta.json"

        file_loaded = False
        if file_cache.exists() and file_meta.exists():
            try:
                self._file_embeddings = np.load(str(file_cache))
                with open(str(file_meta), "r") as f:
                    self._file_metadata = json.load(f)
                self._file_ready = True
                file_loaded = True
                logger.info("Embedding index: loaded %d file vectors from cache", len(self._file_metadata))
            except Exception:
                pass

        if func_loaded and file_loaded:
            logger.info("Embedding index: fully loaded from cache — ready")
            return

        # --- Build fresh embeddings for missing levels ---
        self._building = True
        self._build_error = None
        try:
            logger.info("Embedding index: loading model %r (first use may download ~150 MB)…", self._model_name)
            model = TextEmbedding(self._model_name)
            logger.info("Embedding index: model loaded — building vectors")

            if not func_loaded:
                docs = self._build_documents()
                if docs:
                    logger.info("Embedding index: embedding %d function documents…", len(docs))
                    vectors = list(model.embed(docs))
                    self._embeddings = np.array(vectors, dtype=np.float32)
                    np.save(str(cache_file), self._embeddings)
                    with open(str(meta_file), "w") as f:
                        json.dump(self._metadata, f)
                    self._ready = True
                    logger.info("Embedding index: %d function vectors built and cached", len(docs))

            if not file_loaded:
                file_docs = self._build_file_documents()
                if file_docs:
                    logger.info("Embedding index: embedding %d file documents…", len(file_docs))
                    file_vectors = list(model.embed(file_docs))
                    self._file_embeddings = np.array(file_vectors, dtype=np.float32)
                    np.save(str(file_cache), self._file_embeddings)
                    with open(str(file_meta), "w") as f:
                        json.dump(self._file_metadata, f)
                    self._file_ready = True
                    logger.info("Embedding index: %d file vectors built and cached — ready", len(file_docs))
        except Exception as e:
            self._build_error = str(e)
            logger.error("Embedding index build failed: %s", e)
        finally:
            self._building = False

    def build_in_background(self) -> None:
        """Start building embeddings in a background daemon thread.

        Safe to call multiple times -- subsequent calls are no-ops if
        embeddings are already ready or already being built.  The thread
        is daemonic so it won't prevent process shutdown.
        """
        import threading

        if self._ready or self._building or not _HAS_EMBEDDINGS:
            return
        t = threading.Thread(target=self.build, daemon=True)
        t.start()

    # --- Search Methods ---

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[str, str, str, float]]:
        """Search by embedding similarity at the function level.

        Embeds the query string using the same model, then computes cosine
        similarity (dot product, since vectors are pre-normalized by fastembed)
        against all function embeddings.

        Results below 0.15 similarity are filtered out as noise -- cosine
        similarity of 0.15 for normalized embeddings indicates very weak
        semantic overlap.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.

        Returns:
            List of (name, file_path, class_name, score) sorted by score
            descending.  Score is cosine similarity clamped to [0, 1].
        """
        if not self._ready or self._embeddings is None:
            return []

        try:
            import numpy as np
        except ImportError:
            return []

        try:
            model = TextEmbedding(self._model_name)
            full_query = (self._query_prefix or "") + query
            q_vectors = list(model.embed([full_query]))
            q_vec = np.array(q_vectors[0], dtype=np.float32)

            # Cosine similarity via dot product (embeddings are pre-normalized
            # by the encoder model -- jina-v2 and CodeRankEmbed both emit
            # unit-norm vectors).
            similarities = self._embeddings @ q_vec
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                # 0.15 threshold filters out noise -- below this, the semantic
                # overlap is too weak to be meaningful
                if score < 0.15:
                    break
                meta = self._metadata[idx]
                results.append((
                    meta["name"],
                    meta["file"],
                    meta.get("class_name", ""),
                    min(1.0, score),
                ))
            return results
        except Exception as e:
            logger.error("Embedding search failed: %s", e)
            return []

    def search_files(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search files by embedding similarity at the file level.

        Similar to search() but uses file-level embeddings.  The higher
        threshold (0.20 vs 0.15) reflects that file documents are longer and
        noisier, so a higher similarity is needed for meaningful matches.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.

        Returns:
            List of (file_path, score) sorted by score descending.
        """
        if not self._file_ready or self._file_embeddings is None:
            return []

        try:
            import numpy as np
        except ImportError:
            return []

        try:
            model = TextEmbedding(self._model_name)
            full_query = (self._query_prefix or "") + query
            q_vectors = list(model.embed([full_query]))
            q_vec = np.array(q_vectors[0], dtype=np.float32)

            similarities = self._file_embeddings @ q_vec
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                # Higher threshold for file-level search because file documents
                # are noisier (aggregated from many functions)
                if score < 0.20:
                    break
                meta = self._file_metadata[idx]
                results.append((meta["file"], min(1.0, score)))
            return results
        except Exception as e:
            logger.error("File embedding search failed: %s", e)
            return []
