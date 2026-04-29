"""Numpy-only inference for the gibberish detector.

No PyTorch dependency — loads weights from .npz and vocabulary from .json.
"""

import os
import numpy as np

from features import text_to_features, load_vocabulary, load_bigram_stats


class GibberishClassifier:
    """Fast numpy-only gibberish detector.

    Usage:
        clf = GibberishClassifier("model/")
        is_real, confidence = clf.predict("authenticate user")  # (True, 0.98)
        is_real, confidence = clf.predict("xyzzy foobar")       # (False, 0.03)
    """

    def __init__(self, model_dir: str):
        vocab_path = os.path.join(model_dir, "vocab.json")
        weights_path = os.path.join(model_dir, "gibberish_weights.npz")
        bigram_path = os.path.join(model_dir, "bigram_stats.json")

        self.trigram_vocab = load_vocabulary(vocab_path)
        self.bigram_logprobs = load_bigram_stats(bigram_path)

        weights = np.load(weights_path)
        # Pre-transpose for faster inference
        self.w1T = np.ascontiguousarray(weights["w1"].T)
        self.b1 = weights["b1"]
        self.w2T = np.ascontiguousarray(weights["w2"].T)
        self.b2 = weights["b2"]
        self.w3T = np.ascontiguousarray(weights["w3"].T)
        self.b3 = weights["b3"]

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> float:
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, query: str, threshold: float = 0.5,
                min_length: int = 6) -> tuple[bool, float]:
        """Classify a query as real or gibberish.

        Args:
            query: The search query to classify.
            threshold: Confidence threshold. Default 0.5.
                For production, 0.3 is recommended.
            min_length: Queries with cleaned length < this bypass the model
                and are always classified as real. Short queries don't have
                enough character data for reliable classification.
                Default 6 (handles 3-5 char terms like "zig", "nim", "bun").

        Returns:
            (is_real, confidence): confidence 0.0-1.0, 1.0 = definitely real.
        """
        import re
        clean = re.sub(r'[^a-z0-9 ]', '', query.lower().strip())
        clean = re.sub(r'\s+', ' ', clean).strip()
        if len(clean) < min_length:
            return True, 1.0  # Too short for model — let search engine handle

        x = text_to_features(query, self.trigram_vocab, self.bigram_logprobs)

        x = self._relu(x @ self.w1T + self.b1)
        x = self._relu(x @ self.w2T + self.b2)
        logit = (x @ self.w3T + self.b3).item()
        confidence = self._sigmoid(logit)

        return confidence > threshold, confidence

    def predict_batch(self, queries: list[str]) -> list[tuple[bool, float]]:
        return [self.predict(q) for q in queries]
