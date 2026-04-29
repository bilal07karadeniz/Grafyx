"""ML-based gibberish detector for search queries.

Self-contained numpy-only classifier. Loads pre-trained weights from
grafyx/search/model/ and classifies queries as real or gibberish based
on character bigram patterns and structural meta-features.

The model generalizes to unseen vocabulary because it uses ALL possible
character bigrams (841 dims) rather than a fixed word vocabulary.
"""

import json
import math
import os
import re

import numpy as np

# Characters used in bigram features. Space is included because
# multi-word queries have space transitions. ^/$ are start/end markers.
_BIGRAM_CHARSET = "abcdefghijklmnopqrstuvwxyz ^$"
_BIGRAM_VOCAB_SIZE = len(_BIGRAM_CHARSET) ** 2  # 29^2 = 841
_META_FEATURE_COUNT = 8

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def _build_bigram_index() -> dict[str, int]:
    idx = 0
    index = {}
    for c1 in _BIGRAM_CHARSET:
        for c2 in _BIGRAM_CHARSET:
            index[c1 + c2] = idx
            idx += 1
    return index


_BIGRAM_INDEX = _build_bigram_index()


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_bigrams(text: str) -> list[str]:
    text = f"^{_normalize_text(text)}$"
    return [text[i:i+2] for i in range(len(text) - 1)]


def _compute_meta_features(text: str, bigram_logprobs: dict[str, float] | None) -> np.ndarray:
    clean = re.sub(r'[^a-zA-Z0-9_ ]', '', text).strip()
    alpha_only = re.sub(r'[^a-zA-Z]', '', text.lower())
    length = len(clean)

    # 1. Bigram transition score (English-likeness)
    if bigram_logprobs is not None:
        norm_text = _normalize_text(text)
        if len(norm_text) < 2:
            bigram_score = 0.5
        else:
            padded = f"^{norm_text}$"
            default_lp = bigram_logprobs.get("__default__", -15.0)
            total_lp = sum(bigram_logprobs.get(padded[i:i+2], default_lp)
                           for i in range(len(padded) - 1))
            avg_lp = total_lp / (len(padded) - 1)
            bigram_score = max(0.0, min(1.0, (avg_lp + 10.0) / 7.0))
    else:
        bigram_score = 0.5

    # 2. Character entropy
    char_counts: dict[str, int] = {}
    for c in alpha_only:
        char_counts[c] = char_counts.get(c, 0) + 1
    if char_counts:
        total_chars = sum(char_counts.values())
        entropy = -sum((c / total_chars) * math.log2(c / total_chars)
                        for c in char_counts.values())
        norm_entropy = entropy / 4.7
    else:
        norm_entropy = 0.5

    # 3. Vowel ratio
    vowel_count = sum(1 for c in alpha_only if c in 'aeiou')
    vowel_ratio = vowel_count / max(1, len(alpha_only))

    # 4. Max consonant cluster length
    clusters = re.findall(r'[^aeiou\s]+', alpha_only) if alpha_only else []
    max_cluster = max((len(c) for c in clusters), default=0)
    norm_cluster = min(1.0, max_cluster / 6.0)

    # 5. Normalized length
    norm_len = min(1.0, math.log(length + 1) / math.log(50))

    # 6-7. Structure indicators
    has_underscore = 1.0 if '_' in text else 0.0
    has_upper = 1.0 if any(c.isupper() for c in text) else 0.0

    # 8. Word count
    norm_word_count = min(1.0, len(clean.split()) / 5.0)

    return np.array([
        bigram_score, norm_entropy, vowel_ratio, norm_cluster,
        norm_len, has_underscore, has_upper, norm_word_count,
    ], dtype=np.float32)


def _text_to_features(text: str, bigram_logprobs: dict[str, float] | None) -> np.ndarray:
    total_size = _BIGRAM_VOCAB_SIZE + _META_FEATURE_COUNT
    vec = np.zeros(total_size, dtype=np.float32)

    bigrams = _extract_bigrams(text)
    if bigrams:
        for bg in bigrams:
            idx = _BIGRAM_INDEX.get(bg)
            if idx is not None:
                vec[idx] += 1.0
        bg_section = vec[:_BIGRAM_VOCAB_SIZE]
        norm = np.linalg.norm(bg_section)
        if norm > 0:
            bg_section /= norm
            vec[:_BIGRAM_VOCAB_SIZE] = bg_section

    vec[_BIGRAM_VOCAB_SIZE:] = _compute_meta_features(text, bigram_logprobs) * 5.0
    return vec


class GibberishClassifier:
    """Fast numpy-only gibberish detector.

    Usage:
        clf = GibberishClassifier()
        is_real, confidence = clf.predict("authenticate user")  # (True, 0.98)
        is_real, confidence = clf.predict("xyzzy foobar")       # (False, 0.03)
    """

    def __init__(self, model_dir: str | None = None):
        model_dir = model_dir or _MODEL_DIR
        bigram_path = os.path.join(model_dir, "bigram_stats.json")
        weights_path = os.path.join(model_dir, "gibberish_weights.npz")

        with open(bigram_path) as f:
            self.bigram_logprobs: dict[str, float] = json.load(f)

        weights = np.load(weights_path)
        self.w1T = np.ascontiguousarray(weights["w1"].T)
        self.b1 = weights["b1"]
        self.w2T = np.ascontiguousarray(weights["w2"].T)
        self.b2 = weights["b2"]
        self.w3T = np.ascontiguousarray(weights["w3"].T)
        self.b3 = weights["b3"]

    def predict(self, query: str, threshold: float = 0.3,
                min_length: int = 6) -> tuple[bool, float]:
        """Classify a query as real or gibberish.

        Returns:
            (is_real, confidence): confidence 0.0-1.0, 1.0 = definitely real.
        """
        clean = re.sub(r'[^a-z0-9 ]', '', query.lower().strip())
        clean = re.sub(r'\s+', ' ', clean).strip()
        if len(clean) < min_length:
            return True, 1.0

        x = _text_to_features(query, self.bigram_logprobs)

        x = np.maximum(0, x @ self.w1T + self.b1)
        x = np.maximum(0, x @ self.w2T + self.b2)
        logit = (x @ self.w3T + self.b3).item()
        logit = np.clip(logit, -500, 500)
        confidence = 1.0 / (1.0 + np.exp(-logit))

        return confidence > threshold, float(confidence)


# Lazy singleton — loaded on first use
_classifier: GibberishClassifier | None = None


def is_gibberish(query: str) -> bool:
    """Check if a search query is gibberish.

    Uses a pre-trained character bigram MLP. Returns True if the query
    appears to be gibberish (random characters, keyboard mashing, etc.).

    Short queries (< 6 chars) always return False (not gibberish).
    """
    global _classifier
    if _classifier is None:
        _classifier = GibberishClassifier()
    is_real, _ = _classifier.predict(query)
    return not is_real
