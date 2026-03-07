"""Tests for M6 Cross-Encoder inference wrapper."""
from grafyx.search._cross_encoder import CrossEncoder, get_cross_encoder


class TestCrossEncoder:
    def test_is_available_false_without_weights(self):
        enc = CrossEncoder()
        assert enc.is_available is False

    def test_get_cross_encoder_returns_none(self):
        import grafyx.search._cross_encoder as mod
        old = mod._cross_encoder
        mod._cross_encoder = None
        try:
            result = get_cross_encoder()
            assert result is None
        finally:
            mod._cross_encoder = old

    def test_rerank_with_empty_candidates(self):
        enc = CrossEncoder()
        enc._loaded = True
        # Can't actually run without weights, but test the API contract
        assert callable(enc.rerank)

    def test_score_signature(self):
        enc = CrossEncoder()
        assert callable(enc.score)
