"""Source-token inverted index mixin for implementation-level code search.

This module provides SourceIndexMixin, which builds and queries an inverted
index over function source code tokens.  This enables searching by
implementation details -- finding functions that use specific libraries,
call specific APIs, or contain specific patterns -- even when the function's
NAME doesn't match the query.

Example use case:
    Query "JWT authentication" should find ``get_current_user()`` even though
    its name says nothing about JWT.  The source index finds it because
    the function body contains ``jwt.decode()``.

How it works:
    1. On first search, _ensure_source_index() iterates all functions via
       graph.iter_functions_with_source(), tokenizes each function body
       using _tokenize_source(), and builds:
       - An inverted index: token -> set of function indices
       - A per-function token set: function index -> set of tokens
       - A per-file aggregate token set: path suffix -> set of all tokens
         from all functions in that file

    2. Tokens appearing in >30% of all functions are pruned from the search
       index (they're too common to be discriminative, e.g., "data", "error").
       However, per-file aggregate tokens are stored PRE-pruning so that
       combined path+source scoring in search_files() can still use them.

    3. _source_search() queries the inverted index with IDF-weighted tokens
       and returns scored results.  _source_score_for() scores a specific
       function (used for blending source evidence into primary scoring).

This module is mixed into CodeSearcher via Python's MRO.  It accesses shared
state through self: self._graph, self._source_index, self._stem_match(), etc.
"""

from __future__ import annotations

from grafyx.search._tokens import _tokenize_source


class SourceIndexMixin:
    """Mixin providing source-token inverted index methods for CodeSearcher.

    This mixin is designed to be inherited alongside ScoringMixin by CodeSearcher.
    It relies on:
      - self._graph: CodebaseGraph instance (for iter_functions_with_source)
      - self._source_symbols, self._source_index, etc.: lazily-built index state
      - self._stem_match(): from ScoringMixin (resolved via MRO)

    The index is built lazily on first access and cached for the lifetime of
    the CodeSearcher instance.  It is NOT automatically invalidated when the
    graph changes -- the caller must create a new CodeSearcher if the graph
    is rebuilt.

    Data structures built by _ensure_source_index():
        _source_symbols: list[(name, file_path, class_name)] -- ordered by index
        _source_index: dict[token, set[int]] -- inverted index (pruned)
        _source_symbol_tokens: dict[int, set[str]] -- per-function token sets
        _source_symbol_lookup: dict[(name, file), int] -- reverse lookup
        _file_source_tokens: dict[path_suffix, set[str]] -- per-file tokens (pre-pruning)
    """

    # --- Index Construction ---

    def _ensure_source_index(self) -> None:
        """Build the inverted source-token index if not already built.

        Single pass over all functions/methods via graph.iter_functions_with_source().
        For each function, tokenizes the source code and maps each unique token
        to the set of symbol indices that contain it.  Tokens appearing in >30%
        of all functions are pruned (too common to be useful for discriminating
        between functions).

        Also builds auxiliary data structures:
        - _source_symbol_tokens: per-function token sets (for _source_score_for)
        - _source_symbol_lookup: (name, file) -> index (for fast single-function lookup)
        - _file_source_tokens: per-file aggregate tokens indexed by path suffixes
          at multiple depths for robust suffix-based path resolution
        """
        if self._source_index is not None:
            return

        symbols: list[tuple[str, str, str]] = []  # (name, file, class_name)
        raw_index: dict[str, set[int]] = {}
        symbol_tokens: dict[int, set[str]] = {}  # per-function token sets

        for name, fpath, source, cls_name in self._graph.iter_functions_with_source():
            if not source:
                continue
            idx = len(symbols)
            symbols.append((name, fpath, cls_name))
            # Deduplicate tokens per function -- we care about presence, not frequency
            seen_tokens: set[str] = set()
            for tok in _tokenize_source(source):
                if tok not in seen_tokens:
                    seen_tokens.add(tok)
                    if tok not in raw_index:
                        raw_index[tok] = set()
                    raw_index[tok].add(idx)
            symbol_tokens[idx] = seen_tokens

        # --- Prune overly common tokens ---
        # Tokens appearing in >30% of functions are too common to be useful
        # for search discrimination.  "config" appearing in 40% of functions
        # would match almost everything and dilute the results.
        if symbols:
            threshold = len(symbols) * 0.3
            self._source_index = {
                tok: indices
                for tok, indices in raw_index.items()
                if len(indices) <= threshold
            }
        else:
            self._source_index = {}
        self._source_symbols = symbols
        self._source_symbol_tokens = symbol_tokens

        # --- Reverse lookup for fast per-function scoring ---
        # Used by _source_score_for() to quickly check if a specific function
        # is in the index without scanning the full _source_symbols list.
        self._source_symbol_lookup: dict[tuple[str, str], int] = {
            (sym[0], sym[1]): idx for idx, sym in enumerate(symbols)
        }

        # --- Per-file aggregate source tokens (pre-pruning) ---
        # Stores ALL source tokens from all functions in each file, keyed by
        # path suffixes at multiple depths so suffix-based path resolution works.
        #
        # Why pre-pruning?  The source search index prunes tokens appearing in
        # >30% of functions (e.g., "admin" if many functions handle admin routes).
        # But for FILE-level search, knowing that a file's functions collectively
        # use "admin" is valuable context.  So we store the full token set here.
        #
        # Why multiple suffix depths?  Function paths from graph-sitter may be
        # relative ("tests/search.py") while file paths from get_all_files() may
        # be absolute ("/mnt/c/.../tests/search.py").  Indexing by 1-5 suffix
        # components ensures either form can find the tokens.
        file_source_toks: dict[str, set[str]] = {}
        for idx, (name, fpath, _cls) in enumerate(symbols):
            if not fpath:
                continue
            fpath_norm = fpath.replace("\\", "/")
            toks = symbol_tokens.get(idx, set())
            if not toks:
                continue
            # Index by descending suffix depths so lookup works for any path format
            parts = fpath_norm.split("/")
            for depth in range(1, min(len(parts) + 1, 6)):
                sfx = "/".join(parts[-depth:])
                if sfx not in file_source_toks:
                    file_source_toks[sfx] = set()
                file_source_toks[sfx].update(toks)
        self._file_source_tokens = file_source_toks

    # --- Source Token Search ---

    def _source_search(
        self,
        query_tokens: list[str],
        token_weights: dict[str, float],
        max_results: int = 20,
    ) -> list[tuple[str, str, str, float]]:
        """Search the source token index for functions matching query tokens.

        Uses the inverted index to find functions whose source code contains
        the query tokens.  Scoring is based on IDF-weighted token coverage:
        how many query tokens appear in the function's source, weighted by
        their IDF importance.

        Falls back to stem matching if exact token lookup fails -- e.g.,
        "authenticate" in the query will match "authentication" in the source
        via _stem_match().  Only the first stem match is used per token to
        prevent combinatorial explosion.

        Args:
            query_tokens: Lowercased, split tokens from the user's query.
            token_weights: IDF weights from _compute_idf_weights().
            max_results: Maximum number of results to return.

        Returns:
            List of (name, file_path, class_name, score) sorted by score
            descending.  Scores are capped at 0.70 because source-only
            evidence is less reliable than name matches (a function using
            "jwt" internally is relevant but less certainly THE result the
            user wants compared to one named "decode_jwt").
        """
        self._ensure_source_index()
        if not self._source_index or not self._source_symbols:
            return []

        # Gather candidate indices with weighted scores
        candidate_scores: dict[int, float] = {}
        total_weight = sum(token_weights.get(t, 1.0) for t in query_tokens)
        if total_weight == 0:
            return []

        matched_weight_per_candidate: dict[int, float] = {}
        for tok in query_tokens:
            w = token_weights.get(tok, 1.0)
            indices = self._source_index.get(tok)
            if indices:
                # Exact token match in source
                for idx in indices:
                    matched_weight_per_candidate[idx] = (
                        matched_weight_per_candidate.get(idx, 0.0) + w
                    )
            # Stem fallback: try matching against index keys when exact match fails.
            # 0.7x weight because stem matches are less precise than exact matches.
            else:
                for index_tok, indices in self._source_index.items():
                    if self._stem_match(tok, index_tok):
                        for idx in indices:
                            matched_weight_per_candidate[idx] = (
                                matched_weight_per_candidate.get(idx, 0.0) + w * 0.7
                            )
                        break  # Take first stem match only to avoid explosion

        # Score = matched_weight / total_weight, capped at 0.70.
        # Minimum threshold of 0.25 filters out very weak matches.
        results: list[tuple[str, str, str, float]] = []
        for idx, matched_w in matched_weight_per_candidate.items():
            score = min(0.70, matched_w / total_weight)
            if score > 0.25:
                name, fpath, cls_name = self._source_symbols[idx]
                results.append((name, fpath, cls_name, score))

        results.sort(key=lambda x: x[3], reverse=True)
        return results[:max_results]

    # --- Single-Function Source Scoring ---

    def _source_score_for(
        self,
        name: str,
        file_path: str,
        query_tokens: list[str],
        token_weights: dict[str, float],
    ) -> float:
        """Compute source token overlap score for a specific function.

        This is used to BLEND source evidence into the primary keyword score
        during the main search loop.  For example, if get_current_user() has
        a moderate keyword score (name doesn't mention "jwt") but its source
        contains "jwt" and "decode", this method returns a positive score that
        gets blended in to boost the result.

        Unlike _source_search() which scans the full index, this method does
        a fast O(1) lookup for a specific (name, file_path) pair.

        Substring containment (qt in ft or ft in qt) is also checked with a
        0.5x weight discount for tokens >= 4 chars.  This catches partial
        matches like "auth" in "authenticate".

        Returns:
            Score in [0.0, 1.0].  Returns 0.0 if the function isn't in the
            source index or has no matching tokens.
        """
        if self._source_index is None:
            return 0.0
        lookup = getattr(self, "_source_symbol_lookup", None)
        if lookup is None:
            return 0.0
        idx = lookup.get((name, file_path))
        if idx is None:
            return 0.0
        func_tokens = self._source_symbol_tokens.get(idx, set())
        if not func_tokens:
            return 0.0

        total_weight = sum(token_weights.get(qt, 1.0) for qt in query_tokens) or 1.0
        hit_weight = 0.0
        for qt in query_tokens:
            if qt in func_tokens:
                hit_weight += token_weights.get(qt, 1.0)
            # Substring containment for longer tokens (>= 4 chars to avoid noise)
            elif any(qt in ft or ft in qt for ft in func_tokens if len(ft) >= 4):
                hit_weight += token_weights.get(qt, 1.0) * 0.5
        return hit_weight / total_weight
