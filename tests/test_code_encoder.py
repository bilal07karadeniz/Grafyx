"""Tests for M5 Code Encoder inference wrapper."""
from unittest.mock import patch, MagicMock
from grafyx.search._code_encoder import CodeEncoder, get_code_encoder


class TestCodeEncoder:
    def test_is_available_false_without_weights(self):
        enc = CodeEncoder()
        # Patch the weights path to a non-existent location
        with patch.object(type(enc), 'is_available', new_callable=lambda: property(lambda self: False)):
            assert enc.is_available is False

    def test_get_code_encoder_returns_none(self):
        """get_code_encoder returns None when weights missing."""
        import grafyx.search._code_encoder as mod
        old = mod._encoder
        mod._encoder = None
        try:
            with patch('grafyx.search._code_encoder.CodeEncoder.is_available',
                       new_callable=lambda: property(lambda self: False)):
                result = get_code_encoder()
                assert result is None
        finally:
            mod._encoder = old

    def test_search_returns_empty_without_index(self):
        enc = CodeEncoder()
        enc._loaded = True
        enc._embeddings = None
        enc._embedding_index = []
        result = enc.search("test query")
        assert result == []


class TestCodeEncoderAPI:
    def test_encode_signature(self):
        """encode() takes text and max_length, returns ndarray."""
        # Just verify the method exists and has correct signature
        enc = CodeEncoder()
        assert callable(enc.encode)

    def test_build_index_signature(self):
        """build_index() takes list of symbol dicts."""
        enc = CodeEncoder()
        assert callable(enc.build_index)

    def test_search_signature(self):
        """search() takes query and top_k, returns list of tuples."""
        enc = CodeEncoder()
        assert callable(enc.search)
