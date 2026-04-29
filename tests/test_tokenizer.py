"""Tests for pure Python BPE tokenizer."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from grafyx.search._tokenizer import CodeTokenizer, get_tokenizer


@pytest.fixture
def mini_tokenizer(tmp_path):
    """Create a minimal tokenizer with a tiny vocabulary."""
    # Simple vocab: individual chars + some common merges
    vocab = {
        "[PAD]": 0, "[UNK]": 1, "[SEP]": 2, "[CLS]": 3,
        "g": 4, "e": 5, "t": 6, "c": 7, "u": 8, "r": 9,
        "n": 10, "h": 11, "a": 12, "l": 13, "d": 14,
        "s": 15, "o": 16, "i": 17, "p": 18, "m": 19,
        "f": 20, "w": 21, "b": 22, "v": 23, "y": 24,
        "k": 25, "x": 26, "j": 27, "z": 28, "q": 29,
        "_": 30,
        "ge": 31, "get": 32, "us": 33, "er": 34, "user": 35,
        "ha": 36, "han": 37, "nd": 38, "le": 39, "lo": 40,
        "gi": 41, "in": 42,
    }
    merges = [
        ["g", "e"], ["ge", "t"], ["u", "s"], ["e", "r"], ["us", "er"],
        ["h", "a"], ["ha", "n"], ["n", "d"], ["l", "e"], ["l", "o"],
        ["g", "i"], ["i", "n"],
    ]

    (tmp_path / "bpe_vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "bpe_merges.json").write_text(json.dumps(merges))

    tokenizer = CodeTokenizer()
    # Directly load the mini vocab/merges (bypass file loading)
    tokenizer._merges = [tuple(m) for m in merges]
    tokenizer._vocab = vocab
    tokenizer._id_to_token = {v: k for k, v in vocab.items()}
    tokenizer._loaded = True

    return tokenizer


class TestCodeTokenizer:
    def test_encode_returns_list_of_ints(self, mini_tokenizer):
        result = mini_tokenizer.encode("get user")
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_encode_starts_with_cls(self, mini_tokenizer):
        result = mini_tokenizer.encode("get")
        assert result[0] == 3  # CLS token

    def test_encode_max_length_respected(self, mini_tokenizer):
        result = mini_tokenizer.encode("get user handle login", max_length=10)
        assert len(result) == 10

    def test_encode_pads_short_input(self, mini_tokenizer):
        result = mini_tokenizer.encode("get", max_length=10)
        assert len(result) == 10
        # Last elements should be padding
        assert result[-1] == 0  # PAD

    def test_unknown_tokens_get_unk_id(self, mini_tokenizer):
        # Numbers aren't in our mini vocab
        result = mini_tokenizer.encode("123")
        # Should have CLS + unk tokens + padding
        assert 1 in result  # UNK id

    def test_bpe_merges_applied(self, mini_tokenizer):
        result = mini_tokenizer.encode("get", max_length=10)
        # "get" should be merged: g->ge->get (id 32)
        assert result[1] == 32  # "get" merged token

    def test_multi_word_merges(self, mini_tokenizer):
        result = mini_tokenizer.encode("get user", max_length=10)
        # CLS=3, "get"=32, "user"=35, then padding
        assert result[0] == 3
        assert result[1] == 32  # "get"
        assert result[2] == 35  # "user"
        assert result[3] == 0   # PAD

    def test_decode_roundtrip(self, mini_tokenizer):
        ids = mini_tokenizer.encode("get", max_length=10)
        text = mini_tokenizer.decode(ids)
        assert "get" in text

    def test_decode_skips_special_tokens(self, mini_tokenizer):
        # CLS, "get", SEP, PAD, PAD
        ids = [3, 32, 2, 0, 0]
        text = mini_tokenizer.decode(ids)
        assert text == "get"
        assert "[CLS]" not in text
        assert "[PAD]" not in text

    def test_vocab_size(self, mini_tokenizer):
        assert mini_tokenizer.vocab_size == 43  # Our mini vocab

    def test_default_max_length(self, mini_tokenizer):
        result = mini_tokenizer.encode("get")
        assert len(result) == 256  # Default max_length

    def test_punctuation_split(self, mini_tokenizer):
        # Punctuation should be split into separate tokens
        result = mini_tokenizer.encode("get.user", max_length=10)
        # "get" -> 32, "." -> UNK (1), "user" -> 35
        assert result[0] == 3   # CLS
        assert result[1] == 32  # "get"
        assert result[2] == 1   # "." is UNK
        assert result[3] == 35  # "user"


class TestCodeTokenizerFileLoading:
    def test_ensure_loaded_raises_when_files_missing(self):
        tok = CodeTokenizer()
        with patch("grafyx.search._tokenizer._MODEL_DIR", Path("/nonexistent/path")):
            with pytest.raises(FileNotFoundError, match="BPE tokenizer files not found"):
                tok._ensure_loaded()

    def test_ensure_loaded_from_json(self, tmp_path):
        """Test that _ensure_loaded correctly reads JSON files."""
        vocab = {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2, "[CLS]": 3, "a": 4}
        merges = [["a", "b"]]
        (tmp_path / "bpe_vocab.json").write_text(json.dumps(vocab))
        (tmp_path / "bpe_merges.json").write_text(json.dumps(merges))

        tok = CodeTokenizer()
        with patch("grafyx.search._tokenizer._MODEL_DIR", tmp_path):
            tok._ensure_loaded()
        assert tok._loaded is True
        assert tok._vocab == vocab
        assert tok._merges == [("a", "b")]
        assert tok._id_to_token[4] == "a"

    def test_ensure_loaded_idempotent(self, tmp_path):
        """Calling _ensure_loaded twice should not re-read files."""
        vocab = {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2, "[CLS]": 3}
        merges = []
        (tmp_path / "bpe_vocab.json").write_text(json.dumps(vocab))
        (tmp_path / "bpe_merges.json").write_text(json.dumps(merges))

        tok = CodeTokenizer()
        with patch("grafyx.search._tokenizer._MODEL_DIR", tmp_path):
            tok._ensure_loaded()
            # Mutate after first load
            tok._vocab["extra"] = 999
            tok._ensure_loaded()
            # Should still have the mutation (not re-loaded)
            assert "extra" in tok._vocab


class TestGetTokenizer:
    def test_returns_none_when_files_missing(self):
        """get_tokenizer() should return None when BPE files don't exist."""
        import grafyx.search._tokenizer as mod
        old = mod._tokenizer
        mod._tokenizer = None
        try:
            with patch("grafyx.search._tokenizer._MODEL_DIR", Path("/nonexistent")):
                result = get_tokenizer()
                assert result is None
        finally:
            mod._tokenizer = old

    def test_returns_tokenizer_when_files_exist(self, tmp_path):
        """get_tokenizer() returns CodeTokenizer when files exist."""
        import grafyx.search._tokenizer as mod
        vocab = {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2, "[CLS]": 3}
        (tmp_path / "bpe_vocab.json").write_text(json.dumps(vocab))
        (tmp_path / "bpe_merges.json").write_text(json.dumps([]))

        old = mod._tokenizer
        mod._tokenizer = None
        try:
            with patch("grafyx.search._tokenizer._MODEL_DIR", tmp_path):
                result = get_tokenizer()
                assert isinstance(result, CodeTokenizer)
        finally:
            mod._tokenizer = old

    def test_is_available_false_without_files(self):
        tok = CodeTokenizer()
        with patch("grafyx.search._tokenizer._MODEL_DIR", Path("/nonexistent")):
            assert tok.is_available is False

    def test_is_available_true_with_files(self, tmp_path):
        (tmp_path / "bpe_vocab.json").write_text("{}")
        (tmp_path / "bpe_merges.json").write_text("[]")
        tok = CodeTokenizer()
        with patch("grafyx.search._tokenizer._MODEL_DIR", tmp_path):
            assert tok.is_available is True
