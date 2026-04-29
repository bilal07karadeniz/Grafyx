"""Feature extraction for the gibberish detector.

Two-layer feature architecture:
1. **Character bigrams (ALL possible)** — universal coverage, every char pair
   represented. This is the primary generalizing feature. ~841 dimensions.
2. **Top-N character trigrams** — captures longer patterns seen in training.
   Helps with known vocabulary. ~2000 dimensions.
3. **Meta-features** — structural signals (length, vowel ratio, etc). 8 dims.

The bigram features are the key to generalization: since ALL possible character
bigrams are in the feature space, the model has signal for ANY word, even ones
never seen during training. The model learns character transition patterns
(e.g., "th" is common, "qx" is rare) that generalize to all English text.
"""

import json
import math
import re
import numpy as np


# Characters used in bigram features. Space is included because
# multi-word queries have space transitions. ^/$ are start/end markers.
BIGRAM_CHARSET = "abcdefghijklmnopqrstuvwxyz ^$"
BIGRAM_VOCAB_SIZE = len(BIGRAM_CHARSET) ** 2  # 29^2 = 841


def _build_bigram_index() -> dict[str, int]:
    """Build mapping from all possible char bigrams to feature indices."""
    idx = 0
    index = {}
    for c1 in BIGRAM_CHARSET:
        for c2 in BIGRAM_CHARSET:
            index[c1 + c2] = idx
            idx += 1
    return index

# Pre-built at module level — shared across all calls
_BIGRAM_INDEX = _build_bigram_index()


def _normalize_text(text: str) -> str:
    """Normalize text for feature extraction."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_trigrams(text: str) -> list[str]:
    """Extract character trigrams from normalized text with markers."""
    text = f"^{_normalize_text(text)}$"
    return [text[i:i+3] for i in range(len(text) - 2)]


def _extract_bigrams(text: str) -> list[str]:
    """Extract character bigrams from normalized text with markers."""
    text = f"^{_normalize_text(text)}$"
    return [text[i:i+2] for i in range(len(text) - 1)]


def build_vocabulary(texts: list[str], max_vocab: int = 2000) -> dict[str, int]:
    """Build trigram vocabulary from training texts. Top max_vocab by frequency."""
    counts: dict[str, int] = {}
    for text in texts:
        for tri in extract_trigrams(text):
            counts[tri] = counts.get(tri, 0) + 1
    sorted_trigrams = sorted(counts.items(), key=lambda x: -x[1])
    return {tri: idx for idx, (tri, _) in enumerate(sorted_trigrams[:max_vocab])}


def build_bigram_stats(real_texts: list[str]) -> dict[str, float]:
    """Build character bigram log-probabilities from real query texts."""
    bigram_counts: dict[str, int] = {}
    total = 0
    for text in real_texts:
        for bg in _extract_bigrams(text):
            bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
            total += 1

    vocab_size = len(bigram_counts) + 1
    bigram_logprobs = {}
    for bigram, count in bigram_counts.items():
        bigram_logprobs[bigram] = math.log((count + 1) / (total + vocab_size))
    bigram_logprobs["__default__"] = math.log(1 / (total + vocab_size))
    return bigram_logprobs


def _compute_meta_features(text: str, bigram_logprobs: dict[str, float] | None) -> np.ndarray:
    """Compute 8 meta-features for structural signals."""
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


# Total meta-features count
META_FEATURE_COUNT = 8


def text_to_features(
    text: str,
    trigram_vocab: dict[str, int] | None = None,
    bigram_logprobs: dict[str, float] | None = None,
) -> np.ndarray:
    """Convert text to feature vector: bigrams + meta.

    Layout: [841 bigram features | 8 meta features]

    Trigram vocab is accepted for API compatibility but NOT used.
    The model relies purely on character bigrams (universal coverage)
    and meta-features for generalization.
    """
    total_size = BIGRAM_VOCAB_SIZE + META_FEATURE_COUNT
    vec = np.zeros(total_size, dtype=np.float32)

    # --- Character bigram features (universal coverage) ---
    bigrams = _extract_bigrams(text)
    if bigrams:
        for bg in bigrams:
            idx = _BIGRAM_INDEX.get(bg)
            if idx is not None:
                vec[idx] += 1.0
        # L2 normalize
        bg_section = vec[:BIGRAM_VOCAB_SIZE]
        norm = np.linalg.norm(bg_section)
        if norm > 0:
            bg_section /= norm
            vec[:BIGRAM_VOCAB_SIZE] = bg_section

    # --- Meta-features (scaled for influence) ---
    vec[BIGRAM_VOCAB_SIZE:] = _compute_meta_features(text, bigram_logprobs) * 5.0

    return vec


def batch_to_features(
    texts: list[str],
    trigram_vocab: dict[str, int] | None = None,
    bigram_logprobs: dict[str, float] | None = None,
) -> np.ndarray:
    """Convert batch of texts to feature matrix."""
    return np.stack([text_to_features(t, trigram_vocab, bigram_logprobs)
                     for t in texts])


def get_feature_size(trigram_vocab_size: int = 0) -> int:
    """Get total feature vector size. Trigram vocab is ignored (bigrams only)."""
    return BIGRAM_VOCAB_SIZE + META_FEATURE_COUNT


def save_vocabulary(vocab: dict[str, int], path: str) -> None:
    with open(path, "w") as f:
        json.dump(vocab, f)

def load_vocabulary(path: str) -> dict[str, int]:
    with open(path) as f:
        return json.load(f)

def save_bigram_stats(bigram_logprobs: dict[str, float], path: str) -> None:
    with open(path, "w") as f:
        json.dump(bigram_logprobs, f)

def load_bigram_stats(path: str) -> dict[str, float]:
    with open(path) as f:
        return json.load(f)
