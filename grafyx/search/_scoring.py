"""Scoring mixin for fuzzy code search -- the core relevance engine.

This module contains ScoringMixin, which provides the three fundamental
scoring methods used by CodeSearcher:

- **_stem_match()**: morphological variant detection ("send" <-> "sending").
- **_compute_idf_weights()**: inverse document frequency weights so rare tokens
  (e.g., "heartbeat") outweigh common ones (e.g., "store").
- **_score_match()**: the main scoring function that combines three clean
  signals with fixed weights -- no multiplicative adjustments.

Scoring philosophy:
    The scorer is designed for natural language queries against code symbols.
    It uses three weighted signals (name 0.55, path 0.25, docstring 0.20)
    combined additively with NO multiplicative boosts or penalties (except
    a test file penalty).  The soft cap at 0.85 compresses high scores to
    preserve ordering among strong matches without collapsing them all to 1.0.

Signal weights:
    - name_score  * 0.55  (symbol name match -- strongest signal)
    - path_score  * 0.25  (file path match -- structural context)
    - doc_score   * 0.20  (docstring match -- conceptual context)

This module is mixed into CodeSearcher via Python's MRO (Method Resolution Order).
It accesses ``self._graph`` and other shared state through the ``self`` reference.
"""

import math
import re

from grafyx.utils import split_tokens

from grafyx.search._tokens import _STOP_WORDS

# Test-related directory names. Files under these directories receive a
# scoring penalty (0.85x) so production code ranks above test code.
_TEST_DIRS = frozenset({"test", "tests", "load_tests", "__tests__",
                        "spec", "specs", "__mocks__"})


class ScoringMixin:
    """Mixin providing relevance scoring methods for CodeSearcher.

    This is not a standalone class -- it is designed to be mixed into
    CodeSearcher alongside SourceIndexMixin.  It accesses shared state
    through self (e.g., self._graph).

    Methods:
        _stem_match(a, b) -> bool: Check morphological variant match.
        _compute_idf_weights(tokens, funcs, classes) -> dict: IDF weights.
        _score_match(tokens, query, name, doc, path, weights) -> float: Score a symbol.
    """

    # --- Stem Matching ---

    @staticmethod
    def _stem_match(a: str, b: str) -> bool:
        """Check if two tokens share a stem, handling common morphological variants.

        This is a lightweight alternative to a full stemmer (like Porter or
        Snowball).  It uses prefix-based heuristics that work well for code
        identifiers, which tend to be English words or abbreviations.

        Two tokens match if:
        1. One is a prefix of the other (min 4 chars), OR
        2. They share a common prefix of length >= max(4, len(shorter)-1),
           which handles -e dropping: store/storing, retrieve/retrieving
        3. For longer words (both >= 7 chars), a shared prefix of >= 5 chars
           suffices -- handles Latin-root variants like permission/permitted,
           execution/executor.

        The 4-char minimum prevents false positives like "log" matching "logging"
        (3 chars is too short, "log" could be coincidental).

        Examples:
            _stem_match("send", "sending") -> True   (prefix)
            _stem_match("store", "storing") -> True   (shared prefix "stor")
            _stem_match("retrieve", "retrieving") -> True (shared "retriev")
            _stem_match("conversation", "conversations") -> True (prefix)
            _stem_match("verify", "verification") -> True (shared "verif")
            _stem_match("permission", "permitted") -> True (shared "permi", both >=7)
            _stem_match("execution", "executor") -> True (shared "execut", both >=7)
            _stem_match("log", "logging") -> False (prefix too short)
        """
        if not a or not b:
            return False
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(shorter) < 4:
            return False
        # Length guard: words differing by 4+ chars aren't morphological
        # variants.  Blocks "embed"→"embedding" (diff=4), allows
        # "send"→"sending" (diff=3).  The relaxed Latin-root check
        # below still handles long-word pairs like "permission"→"permitted".
        if len(longer) - len(shorter) > 3:
            pass
        elif longer.startswith(shorter):
            # Direct prefix: "send" is a prefix of "sending"
            return True
        else:
            # Shared prefix for -e dropping: "store"→"storing" share "stor"
            min_prefix = max(4, len(shorter) - 1)
            if min_prefix <= len(longer):
                if longer[:min_prefix] == shorter[:min_prefix]:
                    return True
        # Relaxed prefix for longer words with shared Latin roots:
        # "permission" <-> "permitted" (shared "permi", 5 chars)
        # "execution" <-> "executor" (shared "execut", 6 chars)
        # Only applies when both words are >=7 chars to avoid false positives
        # on shorter words where 5-char prefixes are less meaningful.
        if len(shorter) >= 7 and len(longer) >= 7:
            shared = 0
            for c1, c2 in zip(shorter, longer):
                if c1 == c2:
                    shared += 1
                else:
                    break
            if shared >= 5:
                return True
        return False

    # --- IDF Weight Computation ---

    def _compute_idf_weights(
        self,
        query_tokens: list[str],
        all_functions: list[dict],
        all_classes: list[dict],
    ) -> dict[str, float]:
        """Compute IDF weights for query tokens based on symbol frequency.

        IDF (Inverse Document Frequency) gives higher weight to rare tokens and
        lower weight to common ones.  "heartbeat" appearing in 2 functions gets
        a much higher weight than "store" appearing in 80 functions.  This
        prevents common words from dominating search results.

        The formula is: idf = log(N / df) / log(N), normalized to [0.5, 1.0].
        - The floor of 0.5 ensures common tokens still contribute meaningful
          weight (they shouldn't be ignored entirely -- "store" in a query
          about "store management" is still relevant).
        - Unknown tokens (df=0) are checked against the source token index:
          if found in source code (e.g., "jwt", "rtc"), they get weight 0.6
          (meaningful technical vocabulary).  If not found anywhere, they get
          weight 0.3 (true gibberish).

        Args:
            query_tokens: Lowercased, split tokens from the user's query.
            all_functions: List of function dicts from the graph.
            all_classes: List of class dicts from the graph.

        Returns:
            Dict mapping each query token to its IDF weight in [0.3, 1.0].
        """
        # Count how many symbols contain each query token (document frequency)
        token_df: dict[str, int] = {}
        total_symbols = 0
        for func_dict in all_functions:
            total_symbols += 1
            name_toks = set(split_tokens(func_dict.get("name", "")))
            doc_words = set(re.findall(r'[a-z]{3,}', (func_dict.get("docstring") or "").lower()))
            combined_toks = name_toks | doc_words
            for qt in query_tokens:
                if qt in combined_toks:
                    token_df[qt] = token_df.get(qt, 0) + 1
                # Also count stem matches so morphological variants contribute
                # to document frequency (e.g., "sending" should count symbols
                # with "send" in their name)
                elif any(self._stem_match(qt, nt) for nt in name_toks):
                    token_df[qt] = token_df.get(qt, 0) + 1
        for cls_dict in all_classes:
            total_symbols += 1
            name_toks = set(split_tokens(cls_dict.get("name", "")))
            doc_words = set(re.findall(r'[a-z]{3,}', (cls_dict.get("docstring") or "").lower()))
            combined_toks = name_toks | doc_words
            for qt in query_tokens:
                if qt in combined_toks:
                    token_df[qt] = token_df.get(qt, 0) + 1
                elif any(self._stem_match(qt, nt) for nt in name_toks):
                    token_df[qt] = token_df.get(qt, 0) + 1

        # Convert document frequencies to IDF weights
        token_weights: dict[str, float] = {}
        for qt in query_tokens:
            df = token_df.get(qt, 0)
            if total_symbols > 0 and df > 0:
                # Normalized IDF: log(N/df) / log(N) gives values in roughly [0, 1].
                # Using max(N, 2) in denominator to avoid log(1)=0 division.
                raw_idf = math.log(total_symbols / df) / math.log(max(total_symbols, 2))
                # Floor at 0.5 so common tokens keep meaningful weight
                token_weights[qt] = max(0.5, raw_idf)
            else:
                # Check source index before marking as gibberish.
                # Tokens like "jwt", "rtc", "daily" don't appear in symbol
                # names but DO appear in source code — they're meaningful
                # technical vocabulary, not gibberish.
                source_index = getattr(self, '_source_index', None)
                if source_index is not None and qt in source_index:
                    token_weights[qt] = 0.6  # Source-present — meaningful
                else:
                    token_weights[qt] = 0.3  # True gibberish
        return token_weights

    # --- Main Scoring Function ---

    def _score_match(
        self,
        query_tokens: list[str],
        query_lower: str,
        name: str,
        docstring: str,
        file_path: str,
        token_weights: dict[str, float] | None = None,
    ) -> float:
        """Score a symbol's relevance to a query.

        Three clean signals with fixed weights, no multiplicative adjustments:
        - name_score * 0.55  (symbol name match -- strongest signal)
        - doc_score  * 0.20  (docstring match -- conceptual context)
        - path_score * 0.25  (file path match -- structural context)
        """
        name_lower = name.lower() if name else ""
        doc_lower = str(docstring or "").lower()
        name_tokens = split_tokens(name) if name else []
        name_token_set = set(name_tokens)

        doc_words = set(re.findall(r'[a-z]{3,}', doc_lower)) if doc_lower else set()
        path_tokens = set(split_tokens(file_path)) if file_path else set()

        def w(token: str) -> float:
            if token_weights is None:
                return 1.0
            return token_weights.get(token, 1.0)

        total_weight = sum(w(qt) for qt in query_tokens) if query_tokens else 1.0

        # --- Name Scoring: max(exact_overlap, containment) ---
        exact_hit_weight = 0.0
        exact_hits = 0
        exact_overlap = 0.0
        if query_tokens and name_tokens:
            for qt in query_tokens:
                if qt in name_token_set:
                    exact_hits += 1
                    exact_hit_weight += w(qt)
            if exact_hits > 0:
                exact_overlap = exact_hit_weight / total_weight

        containment = 0.0
        if query_tokens and name_tokens:
            matched_weight = 0.0
            for qt in query_tokens:
                if qt in name_token_set:
                    matched_weight += w(qt)
                elif any(qt in nt or nt in qt for nt in name_tokens):
                    matched_weight += w(qt) * 0.6
                elif any(self._stem_match(qt, nt) for nt in name_tokens):
                    matched_weight += w(qt) * 0.5
            containment = matched_weight / total_weight

        name_score = max(exact_overlap, containment)

        # Keyword floor
        if exact_hits > 0 and name_tokens:
            best_hit_idf = max(w(qt) for qt in query_tokens if qt in name_token_set)
            name_coverage = exact_hits / len(name_tokens)
            keyword_floor = best_hit_idf * max(0.35, name_coverage)
            name_score = max(name_score, keyword_floor)

        # Full query as substring of name
        if query_lower and query_lower in name_lower:
            name_score = max(name_score, 0.90)

        # --- Docstring Scoring (own doc only, NO file context) ---
        doc_score = 0.0
        if doc_lower and query_tokens:
            doc_hit_weight = 0.0
            for t in query_tokens:
                if t in doc_lower:
                    doc_hit_weight += w(t)
                elif any(self._stem_match(t, dw) for dw in doc_words):
                    doc_hit_weight += w(t) * 0.7
            doc_score = doc_hit_weight / total_weight

        # --- Path Scoring ---
        path_score = 0.0
        if file_path and query_tokens:
            if path_tokens:
                match_weight = 0.0
                for t in query_tokens:
                    if t in path_tokens:
                        match_weight += w(t)
                    elif any(self._stem_match(t, p) for p in path_tokens):
                        match_weight += w(t) * 0.5
                path_score = match_weight / total_weight
                dir_parts = set(file_path.replace("\\", "/").split("/")[:-1])
                dir_tokens: set[str] = set()
                for d in dir_parts:
                    dir_tokens.update(split_tokens(d))
                if dir_tokens:
                    dir_weight = 0.0
                    for t in query_tokens:
                        if t in dir_tokens:
                            dir_weight += w(t)
                        elif any(self._stem_match(t, dt) for dt in dir_tokens):
                            dir_weight += w(t) * 0.5
                    if dir_weight > 0:
                        path_score = max(path_score, (dir_weight / total_weight) * 1.3)

        # --- Weighted combination (NO multiplicative adjustments) ---
        final = name_score * 0.55 + doc_score * 0.20 + path_score * 0.25

        # --- Gibberish token penalty ---
        # When >50% of query tokens have no symbol presence (IDF weight <= 0.3),
        # the query is likely gibberish or irrelevant. Dampen scores proportionally
        # to prevent false positives from spurious path/docstring matches.
        if token_weights and query_tokens:
            gibberish_count = sum(
                1 for qt in query_tokens if token_weights.get(qt, 1.0) <= 0.3
            )
            gibberish_ratio = gibberish_count / len(query_tokens)
            if gibberish_ratio > 0.5:
                final *= (1.0 - gibberish_ratio * 0.5)

        # --- Test file penalty (0.70x, was 0.85x) ---
        if file_path:
            path_norm = file_path.replace("\\", "/").lower()
            path_parts = path_norm.split("/")
            if any(p in _TEST_DIRS or p.startswith("test_") for p in path_parts):
                final *= 0.70

        # --- Soft cap ---
        if final <= 0.85:
            return final
        excess = final - 0.85
        return 0.85 + 0.15 * excess / (0.15 + excess)
