"""ML-based relevance scorer for search queries.

Self-contained numpy-only scorer. Replaces the hand-tuned _score_match()
heuristic with a learned MLP that scores (query, symbol) relevance from
42 features (v2): token overlaps (exact/stem/substring) for name, path,
docstring, and source code, plus structural, IDF, character similarity,
IDF-aware mismatch signals, and 9 new structural/graph features.

The model is stable because it optimizes over all training patterns
simultaneously, unlike hand-tuned weights that break when adjusted.

v2 adds: is_dunder, is_init_file, is_method, is_class,
receiver_call_ratio, source_token_entropy, source_unique_token_ratio,
embedding_similarity, caller_count_normalized.
Falls back to v1 (33 features) if v2 weights not found.
"""

import os
import re

import numpy as np

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# ── Token utilities (matching Grafyx's split_tokens + stem_match) ──

_TEST_DIRS = frozenset({"test", "tests", "load_tests", "__tests__",
                        "spec", "specs", "__mocks__"})


def _split_tokens(text: str) -> list[str]:
    if not text:
        return []
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    parts = re.split(r'[^a-zA-Z0-9]+', s.lower())
    return [p for p in parts if len(p) >= 2]


def _stem_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 4:
        return False
    if longer.startswith(shorter):
        return True
    min_prefix = max(4, len(shorter) - 1)
    if min_prefix <= len(longer) and longer[:min_prefix] == shorter[:min_prefix]:
        return True
    if len(shorter) >= 7 and len(longer) >= 7:
        shared = 0
        for c1, c2 in zip(shorter, longer):
            if c1 == c2:
                shared += 1
            else:
                break
        if shared >= 5:
            return True
    return False


def _char_bigrams(text: str) -> set[str]:
    t = re.sub(r'[^a-z]', '', text.lower())
    if len(t) < 2:
        return set()
    return {t[i:i+2] for i in range(len(t) - 1)}


# ── Feature extraction (42 features, v2) ─────────────────────────

_FEATURE_COUNT = 42


def _extract_features(
    query_tokens: list[str],
    query_lower: str,
    name: str,
    docstring: str,
    file_path: str,
    token_weights: dict[str, float] | None = None,
    source_tokens: set[str] | None = None,
    # New v2 params:
    is_dunder: bool = False,
    is_init_file: bool = False,
    is_method: bool = False,
    is_class: bool = False,
    receiver_call_ratio: float = 0.0,
    source_token_entropy: float = 0.0,
    source_unique_token_ratio: float = 0.0,
    embedding_similarity: float = 0.0,
    caller_count_normalized: float = 0.0,
) -> np.ndarray:
    vec = np.zeros(_FEATURE_COUNT, dtype=np.float32)
    n_qt = len(query_tokens)
    if n_qt == 0:
        return vec

    name_lower = name.lower() if name else ""
    name_tokens = _split_tokens(name) if name else []
    name_token_set = set(name_tokens)
    doc_lower = str(docstring or "").lower()
    doc_words = set(re.findall(r'[a-z]{3,}', doc_lower)) if doc_lower else set()

    def w(tok: str) -> float:
        if token_weights is None:
            return 1.0
        return token_weights.get(tok, 0.5)

    total_weight = sum(w(qt) for qt in query_tokens) or 1.0

    # Name signals (0-6)
    exact_hits = stem_hits = substr_hits = 0
    weighted_exact = 0.0
    unmatched_idfs: list[float] = []
    for qt in query_tokens:
        if qt in name_token_set:
            exact_hits += 1
            weighted_exact += w(qt)
        elif any(_stem_match(qt, nt) for nt in name_tokens):
            stem_hits += 1
        elif any(qt in nt or nt in qt for nt in name_tokens):
            substr_hits += 1
        else:
            unmatched_idfs.append(w(qt))

    vec[0] = exact_hits / n_qt
    vec[1] = (exact_hits + stem_hits) / n_qt
    vec[2] = (exact_hits + stem_hits + substr_hits) / n_qt
    if name_tokens:
        covered = sum(1 for nt in name_tokens
                      if nt in set(query_tokens) or
                      any(_stem_match(nt, qt) for qt in query_tokens))
        vec[3] = covered / len(name_tokens)
    vec[4] = 1.0 if query_lower and query_lower in name_lower else 0.0
    qt_set = set(query_tokens)
    union = qt_set | name_token_set
    inter = qt_set & name_token_set
    vec[5] = len(inter) / len(union) if union else 0.0
    vec[6] = weighted_exact / total_weight

    # Path signals (7-11)
    if file_path:
        path_norm = file_path.replace("\\", "/")
        path_tokens = set(_split_tokens(path_norm))
        parts = path_norm.split("/")
        filename_tokens = set(_split_tokens(parts[-1] if parts else ""))
        dir_tokens: set[str] = set()
        for d in parts[:-1]:
            dir_tokens.update(_split_tokens(d))

        path_exact = sum(1 for qt in query_tokens if qt in path_tokens)
        path_stem = sum(1 for qt in query_tokens
                        if qt not in path_tokens and
                        any(_stem_match(qt, pt) for pt in path_tokens))
        vec[7] = (path_exact + path_stem) / n_qt
        vec[8] = path_stem / n_qt if path_stem else 0.0
        vec[9] = sum(1 for qt in query_tokens if qt in filename_tokens) / n_qt
        vec[10] = sum(1 for qt in query_tokens if qt in dir_tokens) / n_qt
        path_parts_lower = [p.lower() for p in parts]
        vec[11] = 1.0 if any(p in _TEST_DIRS or p.startswith("test_")
                             for p in path_parts_lower) else 0.0

    # Docstring signals (12-14)
    if doc_words:
        doc_exact = sum(1 for qt in query_tokens if qt in doc_words)
        doc_stem = sum(1 for qt in query_tokens
                       if qt not in doc_words and
                       any(_stem_match(qt, dw) for dw in doc_words))
        vec[12] = (doc_exact + doc_stem) / n_qt
        vec[13] = doc_stem / n_qt if doc_stem else 0.0
    vec[14] = 1.0 if doc_lower.strip() else 0.0

    # Query properties (15-17)
    vec[15] = min(1.0, n_qt / 5.0)
    vec[16] = min(1.0, (sum(len(t) for t in query_tokens) / n_qt) / 10.0)
    vec[17] = 1.0 if '_' in query_lower else 0.0

    # Symbol properties (18-20)
    vec[18] = min(1.0, len(name_tokens) / 5.0) if name_tokens else 0.0
    if name_tokens:
        vec[19] = min(1.0, (sum(len(t) for t in name_tokens) / len(name_tokens)) / 10.0)
    vec[20] = 1.0 if name and any(c.isupper() for c in name) else 0.0

    # IDF context (21-23)
    if token_weights and query_tokens:
        idf_vals = [w(qt) for qt in query_tokens]
        vec[21] = sum(idf_vals) / len(idf_vals)
        vec[22] = min(idf_vals)
        vec[23] = max(idf_vals)
    else:
        vec[21] = vec[22] = vec[23] = 0.5

    # Aggregate overlap signals (24-26)
    vec[24] = 1.0 if vec[0] > 0 or vec[1] > 0 or vec[2] > 0 else 0.0
    vec[25] = 1.0 if vec[7] > 0 or vec[9] > 0 or vec[10] > 0 else 0.0
    vec[26] = vec[0] + vec[1] + vec[3] + vec[7] + vec[9] + vec[12]

    # Source token signals (27-29)
    if source_tokens:
        src_exact = 0
        src_stem = 0
        for qt in query_tokens:
            if qt in source_tokens:
                src_exact += 1
            elif any(_stem_match(qt, st) for st in source_tokens if len(st) >= 4):
                src_stem += 1
        vec[27] = (src_exact + src_stem) / n_qt
        vec[28] = src_stem / n_qt if src_stem else 0.0
        vec[29] = 1.0 if src_exact > 0 or src_stem > 0 else 0.0

    # Character bigram similarity (30)
    if name:
        q_bigrams = _char_bigrams(query_lower)
        n_bigrams = _char_bigrams(name_lower)
        if q_bigrams and n_bigrams:
            bg_inter = len(q_bigrams & n_bigrams)
            bg_union = len(q_bigrams | n_bigrams)
            vec[30] = bg_inter / bg_union if bg_union else 0.0

    # IDF-aware mismatch signals (31-32)
    if unmatched_idfs:
        vec[31] = max(unmatched_idfs)
        vec[32] = sum(unmatched_idfs) / total_weight

    # Update aggregate with source overlap
    if vec[29] > 0:
        vec[26] += vec[27]

    # New v2 structural features (33-41)
    vec[33] = float(is_dunder)
    vec[34] = float(is_init_file)
    vec[35] = float(is_method)
    vec[36] = float(is_class)
    vec[37] = min(receiver_call_ratio, 1.0)
    vec[38] = min(source_token_entropy / 5.0, 1.0)
    vec[39] = source_unique_token_ratio
    vec[40] = embedding_similarity
    vec[41] = min(caller_count_normalized, 1.0)

    return vec


# ── Scorer class ───────────────────────────────────────────────────

class RelevanceScorer:
    """Fast numpy-only relevance scorer.

    Replaces _score_match() with a learned MLP. Supports v2 (42 features)
    with graceful fallback to v1 (33 features) if v2 weights not found.
    """

    def __init__(self, model_dir: str | None = None):
        model_dir = model_dir or _MODEL_DIR
        # Try v2 weights first, fall back to v1
        v2_path = os.path.join(model_dir, "relevance_weights_v2.npz")
        v1_path = os.path.join(model_dir, "relevance_weights.npz")

        if os.path.exists(v2_path):
            weights_path = v2_path
            self._version = 2
        else:
            weights_path = v1_path
            self._version = 1

        weights = np.load(weights_path)
        # Support both old (w1,b1,w2,b2,w3,b3) and new (W0,b0,W1,b1,...) naming
        if "W0" in weights:
            self._layers: list[tuple[np.ndarray, np.ndarray]] = []
            i = 0
            while f"W{i}" in weights:
                self._layers.append((
                    np.ascontiguousarray(weights[f"W{i}"]),
                    weights[f"b{i}"],
                ))
                i += 1
        else:
            self._layers = [
                (np.ascontiguousarray(weights["w1"].T), weights["b1"]),
                (np.ascontiguousarray(weights["w2"].T), weights["b2"]),
                (np.ascontiguousarray(weights["w3"].T), weights["b3"]),
            ]

    def score(
        self,
        query_tokens: list[str],
        query_lower: str,
        name: str,
        docstring: str,
        file_path: str,
        token_weights: dict[str, float] | None = None,
        source_tokens: set[str] | None = None,
        # New v2 params:
        is_dunder: bool = False,
        is_init_file: bool = False,
        is_method: bool = False,
        is_class: bool = False,
        receiver_call_ratio: float = 0.0,
        source_token_entropy: float = 0.0,
        source_unique_token_ratio: float = 0.0,
        embedding_similarity: float = 0.0,
        caller_count_normalized: float = 0.0,
    ) -> float:
        """Score relevance of a symbol to a query.

        Same signature as _score_match() for drop-in replacement,
        plus optional source_tokens for function body matching
        and v2 structural/graph features.

        Returns:
            Score in [0.0, 1.0], where 1.0 = highly relevant.
        """
        if not query_tokens:
            return 0.0

        x = _extract_features(
            query_tokens, query_lower, name, docstring,
            file_path, token_weights, source_tokens,
            is_dunder=is_dunder, is_init_file=is_init_file,
            is_method=is_method, is_class=is_class,
            receiver_call_ratio=receiver_call_ratio,
            source_token_entropy=source_token_entropy,
            source_unique_token_ratio=source_unique_token_ratio,
            embedding_similarity=embedding_similarity,
            caller_count_normalized=caller_count_normalized,
        )

        # No token overlap across name, path, doc, or source → not relevant.
        if x[26] == 0.0:
            return 0.0

        # If using v1 model, truncate features to 33
        if self._version == 1:
            x = x[:33]

        # Forward pass through layers
        for i, (W, b) in enumerate(self._layers):
            x = x @ W + b
            if i < len(self._layers) - 1:
                x = np.maximum(0, x)

        logit = float(x.item() if hasattr(x, 'item') else x)
        logit = np.clip(logit, -500, 500)
        return float(1.0 / (1.0 + np.exp(-logit)))


# Lazy singleton
_scorer: RelevanceScorer | None = None


def ml_score_match(
    query_tokens: list[str],
    query_lower: str,
    name: str,
    docstring: str,
    file_path: str,
    token_weights: dict[str, float] | None = None,
    source_tokens: set[str] | None = None,
    # New v2 params:
    is_dunder: bool = False,
    is_init_file: bool = False,
    is_method: bool = False,
    is_class: bool = False,
    receiver_call_ratio: float = 0.0,
    source_token_entropy: float = 0.0,
    source_unique_token_ratio: float = 0.0,
    embedding_similarity: float = 0.0,
    caller_count_normalized: float = 0.0,
) -> float:
    """Drop-in replacement for ScoringMixin._score_match().

    Loads the ML model on first call. Applies test file penalty
    and soft cap to match the heuristic's output characteristics.
    """
    global _scorer
    if _scorer is None:
        _scorer = RelevanceScorer()

    raw = _scorer.score(
        query_tokens, query_lower, name, docstring,
        file_path, token_weights, source_tokens,
        is_dunder=is_dunder, is_init_file=is_init_file,
        is_method=is_method, is_class=is_class,
        receiver_call_ratio=receiver_call_ratio,
        source_token_entropy=source_token_entropy,
        source_unique_token_ratio=source_unique_token_ratio,
        embedding_similarity=embedding_similarity,
        caller_count_normalized=caller_count_normalized,
    )

    # Gibberish token penalty — when >50% of query tokens have no
    # symbol presence (IDF weight <= 0.3), dampen scores proportionally.
    if token_weights and query_tokens:
        gibberish_count = sum(
            1 for qt in query_tokens if token_weights.get(qt, 1.0) <= 0.3
        )
        gibberish_ratio = gibberish_count / len(query_tokens)
        if gibberish_ratio > 0.5:
            raw *= (1.0 - gibberish_ratio * 0.6)

    # Test file penalty (same as heuristic)
    if file_path:
        path_norm = file_path.replace("\\", "/").lower()
        path_parts = path_norm.split("/")
        if any(p in _TEST_DIRS or p.startswith("test_") for p in path_parts):
            raw *= 0.70

    # Soft cap (same as heuristic)
    if raw <= 0.85:
        return raw
    excess = raw - 0.85
    return 0.85 + 0.15 * excess / (0.15 + excess)
