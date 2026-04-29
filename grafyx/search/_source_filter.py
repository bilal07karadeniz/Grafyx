"""Source Token Filter -- M3 integration.

Predicts P(token is semantically relevant to function) for each source token.
Prevents contamination from __getattr__ bodies, string literals, and imports
from inflating search scores.

When M3 model weights are available (source_filter_weights.npz), each query
token found in a function's source code is scored for semantic relevance.
Tokens that appear only in import statements, string literals, or __getattr__
bodies get lower weights.

When model weights are NOT available, all tokens get weight 1.0 (no filtering).
This ensures the search pipeline works identically to before M3 was added.

Public API:
    filter_source_tokens(query_tokens, function_name, function_source,
                         function_file, function_docstring) -> {token: score}
"""

import re

import numpy as np

from grafyx.ml_inference import get_model

FEATURE_COUNT = 15

# Python keywords for feature extraction
_PYTHON_KEYWORDS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
})

_STOP_WORDS = frozenset({
    "the", "is", "at", "which", "on", "in", "to", "for", "with", "and",
    "or", "not", "of", "by", "from", "that", "this", "it", "an", "be",
    "as", "do", "if", "no", "so", "up", "but", "out", "its", "all",
})


def _extract_token_features(
    token: str,
    function_name: str,
    function_source: str,
    function_file: str,
    function_docstring: str = "",
) -> np.ndarray:
    """Extract 15 features for a (token, function) pair.

    Features capture WHERE in the function's source the token appears,
    which is a strong signal for semantic relevance:

    - Tokens in the function name or parameters are almost always relevant.
    - Tokens only in import statements or string literals are usually noise.
    - Tokens in __getattr__ bodies are contamination from dynamic dispatch.

    Feature vector:
        [0]  token_in_function_name
        [1]  token_in_docstring
        [2]  token_in_param_names
        [3]  token_in_decorator
        [4]  token_in_import_statement
        [5]  token_in_string_literal
        [6]  token_in_comment
        [7]  token_in_getattr_body
        [8]  token_frequency_in_source (normalized)
        [9]  token_length (normalized)
        [10] is_stop_word
        [11] is_keyword
        [12] source_line_count (normalized)
        [13] token_position_normalized (first occurrence / total lines)
        [14] token_is_identifier
    """
    vec = np.zeros(FEATURE_COUNT, dtype=np.float32)
    token_lower = token.lower()
    source_lower = function_source.lower()

    # Feature 0: token_in_function_name
    name_lower = function_name.lower()
    name_parts = set(
        re.split(
            r'[^a-zA-Z0-9]+',
            re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', function_name).lower(),
        )
    )
    name_parts.discard("")
    vec[0] = float(token_lower in name_parts or token_lower in name_lower)

    # Feature 1: token_in_docstring
    doc_lower = function_docstring.lower() if function_docstring else ""
    vec[1] = float(token_lower in doc_lower) if doc_lower else 0.0

    # Feature 2: token_in_param_names
    param_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', function_source)
    if param_match:
        params_str = param_match.group(1).lower()
        vec[2] = float(token_lower in params_str)

    # Feature 3: token_in_decorator
    decorator_lines = [
        line.strip()
        for line in function_source.split('\n')
        if line.strip().startswith('@')
    ]
    vec[3] = float(any(token_lower in d.lower() for d in decorator_lines))

    # Feature 4: token_in_import_statement
    import_lines = [
        line.strip()
        for line in function_source.split('\n')
        if line.strip().startswith('import ') or line.strip().startswith('from ')
    ]
    vec[4] = float(any(token_lower in line.lower() for line in import_lines))

    # Feature 5: token_in_string_literal
    strings = re.findall(
        r'(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"]*"|\'[^\']*\')',
        function_source,
    )
    vec[5] = float(any(token_lower in s.lower() for s in strings))

    # Feature 6: token_in_comment
    comment_lines = [
        line.strip()
        for line in function_source.split('\n')
        if '#' in line
    ]
    comments = [line[line.index('#'):] for line in comment_lines]
    vec[6] = float(any(token_lower in c.lower() for c in comments))

    # Feature 7: token_in_getattr_body
    vec[7] = float('__getattr__' in function_source and token_lower in source_lower)

    # Feature 8: token_frequency_in_source (normalized)
    count = source_lower.count(token_lower)
    total_tokens = len(re.findall(r'\w+', function_source))
    vec[8] = min(count / max(total_tokens, 1) * 10.0, 1.0)

    # Feature 9: token_length (normalized)
    vec[9] = min(len(token) / 20.0, 1.0)

    # Feature 10: is_stop_word
    vec[10] = float(token_lower in _STOP_WORDS)

    # Feature 11: is_keyword
    vec[11] = float(token in _PYTHON_KEYWORDS or token_lower in _PYTHON_KEYWORDS)

    # Feature 12: source_line_count (normalized)
    line_count = len(function_source.split('\n'))
    vec[12] = min(line_count / 100.0, 1.0)

    # Feature 13: token_position_normalized (first occurrence / total lines)
    lines = source_lower.split('\n')
    for i, line in enumerate(lines):
        if token_lower in line:
            vec[13] = (i + 1) / max(len(lines), 1)
            break

    # Feature 14: token_is_identifier
    vec[14] = float(bool(re.match(r'^[a-zA-Z_]\w*$', token)))

    return vec


def filter_source_tokens(
    query_tokens: list[str],
    function_name: str,
    function_source: str,
    function_file: str,
    function_docstring: str = "",
) -> dict[str, float]:
    """Return quality-filtered source token weights.

    For each query token found in the function's source, predict
    P(semantically relevant) using the M3 model. Tokens NOT found
    in source get weight 0.0.

    Args:
        query_tokens: Lowercased tokens from the user's query.
        function_name: Name of the function being scored.
        function_source: Full source code of the function.
        function_file: File path where the function is defined.
        function_docstring: Docstring of the function (optional).

    Returns:
        Dict mapping each query token to its relevance weight in [0, 1].
        When M3 model is not available, all source-present tokens get 1.0.
    """
    model = get_model("source_token_filter")
    if model is None:
        # Fallback: all tokens equally weighted (no filtering)
        return {t: 1.0 for t in query_tokens}

    results: dict[str, float] = {}
    for token in query_tokens:
        if token.lower() not in function_source.lower():
            results[token] = 0.0
            continue
        features = _extract_token_features(
            token, function_name, function_source,
            function_file, function_docstring,
        )
        results[token] = model.predict(features)
    return results
