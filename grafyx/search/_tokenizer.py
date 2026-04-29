"""Pure Python BPE tokenizer for code search. No runtime dependencies beyond json."""
import json
import re
from pathlib import Path

_MODEL_DIR = Path(__file__).parent / "model"


class CodeTokenizer:
    """BPE tokenizer for code and natural language queries.

    Loads merge rules and vocabulary from JSON files. No sentencepiece
    or other tokenizer libraries needed at runtime.
    """

    def __init__(self):
        self._merges: list[tuple[str, str]] = []
        self._vocab: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._loaded = False
        self.pad_id = 0
        self.unk_id = 1
        self.sep_id = 2
        self.cls_id = 3

    def _ensure_loaded(self):
        if self._loaded:
            return
        merges_path = _MODEL_DIR / "bpe_merges.json"
        vocab_path = _MODEL_DIR / "bpe_vocab.json"
        if not merges_path.exists() or not vocab_path.exists():
            raise FileNotFoundError("BPE tokenizer files not found")
        self._merges = [tuple(m) for m in json.loads(merges_path.read_text(encoding="utf-8"))]
        self._vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._loaded = True

    def encode(self, text: str, max_length: int = 256) -> list[int]:
        """Encode text to token IDs with BPE.

        Pre-tokenizes on whitespace and punctuation, then applies BPE merges.
        Output starts with CLS token and is padded/truncated to max_length.
        """
        self._ensure_loaded()
        # Pre-tokenize: split on whitespace and punctuation, lowercase
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        all_ids = [self.cls_id]
        for word in words:
            chars = list(word)
            # Apply BPE merges greedily
            for a, b in self._merges:
                i = 0
                while i < len(chars) - 1:
                    if chars[i] == a and chars[i + 1] == b:
                        chars[i] = a + b
                        del chars[i + 1]
                    else:
                        i += 1
            for token in chars:
                all_ids.append(self._vocab.get(token, self.unk_id))
            if len(all_ids) >= max_length - 1:
                break
        # Truncate and pad
        all_ids = all_ids[:max_length]
        while len(all_ids) < max_length:
            all_ids.append(self.pad_id)
        return all_ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        self._ensure_loaded()
        tokens = []
        for id_ in ids:
            if id_ in (self.pad_id, self.cls_id, self.sep_id):
                continue
            token = self._id_to_token.get(id_, "<unk>")
            tokens.append(token)
        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        self._ensure_loaded()
        return len(self._vocab)

    @property
    def is_available(self) -> bool:
        return (
            (_MODEL_DIR / "bpe_merges.json").exists()
            and (_MODEL_DIR / "bpe_vocab.json").exists()
        )


# Lazy singleton
_tokenizer: CodeTokenizer | None = None


def get_tokenizer() -> CodeTokenizer | None:
    """Get the tokenizer singleton, or None if files don't exist."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = CodeTokenizer()
    return _tokenizer if _tokenizer.is_available else None
