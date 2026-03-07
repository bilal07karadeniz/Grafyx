"""Grafyx code search package -- fuzzy, token-based, and semantic search over codebases.

This package implements the search subsystem for Grafyx.  It provides three
search engines that work together:

1. **Keyword scoring** (ScoringMixin in _scoring.py):
   IDF-weighted token matching against symbol names, docstrings, and file paths.
   Handles morphological variants via stem matching ("send" <-> "sending").

2. **Source-token index** (SourceIndexMixin in _source_index.py):
   Inverted index over function body tokens.  Finds functions by implementation
   details even when names don't match (e.g., "jwt" -> get_current_user).

3. **Embedding search** (EmbeddingSearcher in _embeddings.py):
   Optional dense vector similarity using fastembed.  Provides semantic matching
   for conceptual queries ("authentication flow") that keyword search misses.

All three engines are orchestrated by CodeSearcher (searcher.py), which merges
their results with diversity guarantees (_merge.py) so no single result kind
(function, class, file) dominates the output.

Architecture overview::

    CodeSearcher (searcher.py)
        |-- inherits ScoringMixin (_scoring.py)      -- relevance scoring
        |-- inherits SourceIndexMixin (_source_index.py) -- source body search
        |-- owns EmbeddingSearcher (_embeddings.py)  -- semantic search
        |-- uses _merge_results (_merge.py)          -- diversity merging
        |-- uses tokens/stopwords (_tokens.py)       -- shared tokenization

Re-exports the public API so existing ``from grafyx.search import ...``
statements continue to work unchanged.
"""

from grafyx.search._embeddings import EmbeddingSearcher, _HAS_EMBEDDINGS
from grafyx.search._tokens import SearchResult, _tokenize_source
from grafyx.search.searcher import CodeSearcher

__all__ = [
    "CodeSearcher",
    "SearchResult",
    "EmbeddingSearcher",
    "_tokenize_source",
    "_HAS_EMBEDDINGS",
]
