"""15-feature extraction for Source Token Filter.

Given a (token, function) pair, extracts 15 features that indicate
whether the token is semantically relevant to the function (should
be indexed for search) vs noise (import names, string literals,
__getattr__ bodies, comments).

Features:
   0: token_in_function_name        (bool)
   1: token_in_docstring            (bool)
   2: token_in_param_names          (bool)
   3: token_in_decorator            (bool)
   4: token_in_import_statement     (bool)
   5: token_in_string_literal       (bool)
   6: token_in_comment              (bool)
   7: token_in_getattr_body         (bool)
   8: token_frequency_in_source     (normalized, count / total_tokens)
   9: token_length                  (normalized, len / 20)
  10: is_stop_word                  (bool)
  11: is_keyword                    (bool)
  12: source_line_count             (normalized, lines / 200)
  13: token_position_normalized     (first occurrence line / total lines)
  14: token_is_identifier           (bool, matches [a-zA-Z_]\w*)
"""

import keyword
import re
import numpy as np

FEATURE_COUNT = 15

# Common stop words that appear in code but carry no search value
STOP_WORDS = frozenset({
    "the", "is", "in", "at", "of", "and", "or", "not", "for", "to",
    "if", "else", "this", "that", "with", "from", "by", "as", "on",
    "all", "are", "be", "been", "has", "have", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can",
    "an", "no", "so", "it", "its", "was", "were",
    # Common code noise
    "none", "true", "false", "self", "cls", "args", "kwargs",
    "return", "pass", "break", "continue", "raise", "yield",
    "import", "from", "class", "def",
})

# All Python keywords
PYTHON_KEYWORDS = frozenset(keyword.kwlist)

# Identifier pattern
_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_]\w*$')


def _split_tokens(text: str) -> list[str]:
    if not text:
        return []
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    parts = re.split(r'[^a-zA-Z0-9]+', s.lower())
    return [p for p in parts if len(p) >= 2]


def _find_first_line(token: str, lines: list[str]) -> int | None:
    """Find the 0-indexed line number of first occurrence of token."""
    token_lower = token.lower()
    for i, line in enumerate(lines):
        if token_lower in line.lower():
            return i
    return None


def _is_in_string_context(token: str, source: str) -> bool:
    """Check if all occurrences of token are inside string literals.

    Uses a simple heuristic: check if the token appears between
    quote delimiters. Not AST-level, but good enough for training.
    """
    # Find all string spans using regex
    string_pattern = re.compile(
        r'"""[\s\S]*?"""|'
        r"'''[\s\S]*?'''|"
        r'"[^"\n]*"|'
        r"'[^'\n]*'"
    )
    string_spans = [(m.start(), m.end()) for m in string_pattern.finditer(source)]
    if not string_spans:
        return False

    # Check if any occurrence of token is outside strings
    token_lower = token.lower()
    source_lower = source.lower()
    pos = 0
    while True:
        idx = source_lower.find(token_lower, pos)
        if idx == -1:
            break
        in_string = any(start <= idx < end for start, end in string_spans)
        if not in_string:
            return False
        pos = idx + 1
    return True


def _is_in_comment_context(token: str, lines: list[str]) -> bool:
    """Check if all occurrences of token are in comment lines."""
    token_lower = token.lower()
    found_any = False
    for line in lines:
        line_lower = line.lower()
        if token_lower not in line_lower:
            continue
        found_any = True
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            return False
    return found_any


def _is_in_import_context(token: str, lines: list[str]) -> bool:
    """Check if all occurrences of token are in import statements."""
    token_lower = token.lower()
    found_any = False
    for line in lines:
        line_lower = line.lower()
        if token_lower not in line_lower:
            continue
        found_any = True
        stripped = line.strip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            return False
    return found_any


def _is_in_getattr_body(token: str, source: str) -> bool:
    """Check if token appears in a __getattr__ method body."""
    # Find __getattr__ def blocks
    pattern = re.compile(
        r'def\s+__getattr__\s*\([^)]*\)\s*(?:->.*?)?:\s*\n((?:\s+.+\n)*)',
        re.MULTILINE,
    )
    for match in pattern.finditer(source):
        body = match.group(1)
        if token.lower() in body.lower():
            return True
    return False


def extract_features(
    token: str,
    function_name: str,
    docstring: str,
    param_names: list[str],
    decorator_names: list[str],
    source_code: str,
) -> np.ndarray:
    """Extract 15 features for a (token, function) pair."""
    vec = np.zeros(FEATURE_COUNT, dtype=np.float32)
    token_lower = token.lower()
    lines = source_code.split("\n") if source_code else []
    all_source_tokens = _split_tokens(source_code)
    total_tokens = len(all_source_tokens) or 1

    # 0: token_in_function_name (bool)
    name_tokens = set(_split_tokens(function_name))
    vec[0] = 1.0 if token_lower in name_tokens else 0.0

    # 1: token_in_docstring (bool)
    doc_tokens = set(_split_tokens(docstring)) if docstring else set()
    vec[1] = 1.0 if token_lower in doc_tokens else 0.0

    # 2: token_in_param_names (bool)
    param_token_set: set[str] = set()
    for p in param_names:
        param_token_set.update(_split_tokens(p))
    vec[2] = 1.0 if token_lower in param_token_set else 0.0

    # 3: token_in_decorator (bool)
    decorator_token_set: set[str] = set()
    for d in decorator_names:
        decorator_token_set.update(_split_tokens(d))
    vec[3] = 1.0 if token_lower in decorator_token_set else 0.0

    # 4: token_in_import_statement (bool)
    vec[4] = 1.0 if _is_in_import_context(token, lines) else 0.0

    # 5: token_in_string_literal (bool)
    vec[5] = 1.0 if _is_in_string_context(token, source_code) else 0.0

    # 6: token_in_comment (bool)
    vec[6] = 1.0 if _is_in_comment_context(token, lines) else 0.0

    # 7: token_in_getattr_body (bool)
    vec[7] = 1.0 if _is_in_getattr_body(token, source_code) else 0.0

    # 8: token_frequency_in_source (normalized)
    token_count = sum(1 for t in all_source_tokens if t == token_lower)
    vec[8] = min(1.0, token_count / total_tokens)

    # 9: token_length (normalized)
    vec[9] = min(1.0, len(token) / 20.0)

    # 10: is_stop_word (bool)
    vec[10] = 1.0 if token_lower in STOP_WORDS else 0.0

    # 11: is_keyword (bool)
    vec[11] = 1.0 if token in PYTHON_KEYWORDS else 0.0

    # 12: source_line_count (normalized)
    vec[12] = min(1.0, len(lines) / 200.0)

    # 13: token_position_normalized (first occurrence line / total lines)
    first_line = _find_first_line(token, lines)
    if first_line is not None and lines:
        vec[13] = first_line / max(1, len(lines))

    # 14: token_is_identifier (bool)
    vec[14] = 1.0 if _IDENTIFIER_RE.match(token) else 0.0

    return vec
