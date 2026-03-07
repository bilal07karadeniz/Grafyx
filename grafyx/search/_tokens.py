"""Token extraction, stopword sets, and search result type for code search.

This module is the shared foundation for all search engines in the package.
It provides:

- **SearchResult**: dataclass representing a single search hit with score.
- **_tokenize_source()**: extracts meaningful tokens from raw function source
  code, used by both the source-token index (_source_index.py) and the
  embedding document builder (_embeddings.py).
- **Stopword sets**: English stop words and language-keyword stop words that
  are filtered out to reduce noise.
- **Framework name sets**: frontend/backend framework names used by the
  searcher to apply language bias (boost TypeScript results for React queries,
  Python results for FastAPI queries, etc.).
- **Compiled regex patterns**: for identifier extraction, string literal
  extraction, and word extraction from strings.

Design decision: tokenization is intentionally language-agnostic.  The same
regex patterns and stopword union work for Python, TypeScript, JavaScript,
and Go.  Language-specific keywords (e.g., "const", "let") are included in
_CODE_STOPWORDS so they get filtered regardless of which language is being
indexed.
"""

import re
from dataclasses import dataclass

from grafyx.utils import split_tokens

# --- Stopword Sets ---
# These sets filter out tokens that are too common to provide useful search
# signal.  Two separate sets because they serve different contexts:
# _STOP_WORDS -> filtered from the USER's query tokens
# _CODE_STOPWORDS -> filtered from SOURCE CODE tokens during indexing

# Common English stop words that cause false positives when they match
# substrings of identifiers (e.g., "for" in "format", "or" in "error").
# Filtered from query tokens during search.
_STOP_WORDS: set[str] = {
    "a", "an", "the", "for", "to", "in", "of", "on", "at", "by",
    "is", "it", "or", "and", "as", "if", "do", "so", "no", "up",
    "be", "we", "he", "me", "my", "us", "am",
}

# Code keywords / builtins that are too common to be useful as search tokens
# when scanning function source code.  Language-agnostic union covering Python,
# TypeScript/JavaScript, and common low-signal identifiers that appear in nearly
# every function (e.g., "self", "return", "args").
_CODE_STOPWORDS: set[str] = {
    # Python keywords
    "return", "self", "cls", "def", "class", "elif",
    "while", "try", "except", "finally", "with", "import",
    "from", "pass", "break", "continue", "yield", "raise", "lambda",
    "none", "true", "false", "not",
    # TypeScript / JavaScript keywords
    "const", "let", "var", "function", "new", "this",
    "typeof", "instanceof", "void", "null", "undefined", "async",
    "await", "export", "default", "extends", "implements",
    # Common low-signal identifiers -- these appear in so many functions
    # that matching them provides almost no discriminative value.
    "print", "log", "error", "string", "number", "boolean",
    "type", "interface", "enum", "dict", "list", "tuple",
    "int", "float", "str", "bool", "any", "object",
    "get", "set", "has", "can", "key", "val", "value",
    "args", "kwargs", "param", "params", "result", "data",
}

# --- Compiled Regex Patterns ---
# These are compiled once at module load time and reused across all calls
# to _tokenize_source().

# Matches identifiers (variable names, function calls, class names) that are
# at least 3 characters long.  The 3-char minimum filters out noise like
# loop variables ("i", "j") and common abbreviations ("ok", "fn").
_IDENT_RE = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]{2,}')

# Matches the contents of single- or double-quoted string literals, capturing
# only the inner text (at least 3 chars).  This extracts meaningful strings
# like route paths ("/api/admin"), error messages, and config keys.
_STRING_RE = re.compile(r'''["']([^"']{3,})["']''')

# Extracts alphabetic words (3+ chars) from string literal contents.
# Applied to the captured groups from _STRING_RE to break multi-word strings
# into individual searchable tokens.
_WORD_RE = re.compile(r'[a-zA-Z]{3,}')


# --- Source Code Tokenizer ---

def _tokenize_source(source: str) -> list[str]:
    """Extract meaningful tokens from function source code.

    This is the core tokenization function used by both the source-token
    inverted index and the embedding document builder.  It performs three
    extraction passes:

    1. **Identifier extraction**: finds all identifiers >= 3 chars in the raw
       source using _IDENT_RE, then splits camelCase/snake_case via
       split_tokens() (e.g., "getUserProfile" -> ["get", "user", "profile"]).

    2. **String literal extraction**: finds quoted strings via _STRING_RE,
       then extracts individual words via _WORD_RE.  This captures semantically
       rich tokens from route paths, error messages, and config values that
       would otherwise be invisible to name-only search.

    3. **Stopword filtering**: removes language keywords, low-signal identifiers,
       English stop words, and any token shorter than 3 chars.

    Args:
        source: Raw function/method source code as a string.

    Returns:
        List of lowercase tokens suitable for indexing.  May contain duplicates
        (the caller decides whether to deduplicate).
    """
    if not source:
        return []
    # 1. Extract identifiers (variable names, function calls, etc.)
    identifiers = _IDENT_RE.findall(source)
    # 2. Extract string literal contents (route paths, error messages, etc.)
    strings = _STRING_RE.findall(source)
    # 3. Split camelCase / snake_case via split_tokens, lowercase
    tokens: list[str] = []
    for ident in identifiers:
        tokens.extend(split_tokens(ident))
    for s in strings:
        tokens.extend(_WORD_RE.findall(s.lower()))
    # 4. Filter stopwords, code keywords, and short tokens
    return [t for t in tokens if len(t) >= 3 and t not in _CODE_STOPWORDS and t not in _STOP_WORDS]


# --- Framework Name Sets (for language bias) ---
# When a query mentions a frontend framework, TypeScript/JS results get boosted;
# when it mentions a backend framework, Python results get boosted.  This is a
# simple heuristic that improves relevance for polyglot codebases where the same
# concept (e.g., "store") exists in both frontend and backend code.

_FRONTEND_FRAMEWORKS: set[str] = {
    "react", "vue", "angular", "svelte", "next", "nuxt", "remix",
    "zustand", "redux", "mobx", "jotai", "recoil", "pinia",
    "tailwind", "chakra", "mui", "shadcn",
}
_BACKEND_FRAMEWORKS: set[str] = {
    "django", "flask", "fastapi", "celery", "sqlalchemy",
    "express", "nest", "koa", "hono",
}


# --- Search Result Dataclass ---

@dataclass
class SearchResult:
    """A single search result with relevance score.

    Used internally by all search engines and the merge logic.  The final
    output to the MCP client is a plain dict (converted in searcher.py),
    but SearchResult provides type safety and attribute access during the
    scoring/merging pipeline.

    Attributes:
        name:      Symbol or file name (e.g., "get_user", "auth.py").
        kind:      One of "function", "class", or "file".
        file_path: Absolute or relative path to the file containing the symbol.
        score:     Relevance score in [0.0, 1.0].  Higher is more relevant.
                   Scores above 0.85 are soft-capped to preserve ordering
                   among strong matches.
        context:   Human-readable context string (signature, base classes, or
                   function/class count for files).
        language:  Programming language (e.g., "python", "typescript").
    """

    name: str
    kind: str  # "function", "class", "file"
    file_path: str
    score: float  # 0.0 to 1.0
    context: str  # signature or first line of docstring
    language: str
