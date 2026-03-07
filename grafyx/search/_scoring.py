"""Scoring mixin for fuzzy code search -- the core relevance engine.

This module contains ScoringMixin, which provides the three fundamental
scoring methods used by CodeSearcher:

- **_stem_match()**: morphological variant detection ("send" <-> "sending").
- **_compute_idf_weights()**: inverse document frequency weights so rare tokens
  (e.g., "heartbeat") outweigh common ones (e.g., "store").
- **_score_match()**: the main scoring function that combines name, docstring,
  and path signals with IDF weighting, stem matching, compound term handling,
  and concept coverage bonuses.

Scoring philosophy:
    The scorer is designed for natural language queries against code symbols.
    It prioritizes RECALL over PRECISION for multi-word queries -- it's better
    to surface a relevant result with a moderate score than to miss it entirely.
    The soft cap at 0.85 compresses high scores to preserve ordering among
    strong matches without collapsing them all to 1.0.

Signal priority (highest to lowest):
    1. Exact token overlap in symbol name (IDF-weighted)
    2. Stem/substring matches in symbol name
    3. Full query as substring of name (very strong)
    4. Docstring matches (exact substring > stem)
    5. Path/directory matches
    6. SequenceMatcher ratio (purely a tiebreaker, heavily capped)

Multiplicative adjustments:
    - Name boost: amplifies score when name tokens match with sufficient IDF weight
    - Docstring boost: 1.2x when docstring covers >= 50% of query weight
    - Adjacent pair bonus: rewards compound term matches ("access" + "control")
    - Orphan penalty: penalizes isolated token matches without adjacent support
    - Concept coverage bonus: rewards matching many UNIQUE query tokens across
      all signal sources

This module is mixed into CodeSearcher via Python's MRO (Method Resolution Order).
It accesses ``self._graph`` and other shared state through the ``self`` reference.
"""

import difflib
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
        # Direct prefix check: "send" is a prefix of "sending"
        if longer.startswith(shorter):
            return True
        # Shared prefix check -- handles -e dropping (store->storing).
        # "store" and "storing" share prefix "stor" (len 4 = max(4, 5-1)).
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
        - Unknown tokens (df=0) get weight 1.0 -- if a token doesn't appear
          in any symbol, it might be a novel concept the user is searching for,
          so we don't penalize it.

        Args:
            query_tokens: Lowercased, split tokens from the user's query.
            all_functions: List of function dicts from the graph.
            all_classes: List of class dicts from the graph.

        Returns:
            Dict mapping each query token to its IDF weight in [0.5, 1.0].
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
                token_weights[qt] = 1.0  # Unknown tokens get full weight
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
        """Score a symbol's relevance to a query using IDF-weighted tokens
        with stem matching for morphological variants.

        This is the central scoring function called for every candidate symbol
        (function, class, or file) during search.  It produces a score in
        [0.0, ~1.0] (soft-capped above 0.85) based on:

        Signals (in priority order):
        1. Name token matches (exact > stem > substring)
        2. Docstring/file-context matches (exact substring > stem)
        3. Path matches (including directory-level matches)

        Stem matching handles morphological variants:
        "sending" <-> "send", "storing" <-> "store",
        "conversations" <-> "conversation", "retrieving" <-> "retrieve"

        The score is composed of three weighted components:
        - name_score * 0.45  (symbol name is the strongest signal)
        - doc_score  * 0.30  (docstring provides conceptual context)
        - path_score * 0.25  (file path provides structural context)

        These base scores are then adjusted by multiplicative boosts/penalties.

        Args:
            query_tokens: Lowercased, split tokens from the user's query.
            query_lower: The full query string, lowercased.
            name: Symbol name (function name, class name, or filename).
            docstring: Symbol's docstring (may include file context for functions).
            file_path: Path to the file containing the symbol.
            token_weights: Optional IDF weights from _compute_idf_weights().

        Returns:
            Relevance score in [0.0, ~1.0].  Scores above 0.85 are compressed
            into [0.85, 1.0) via soft cap.
        """
        name_lower = name.lower() if name else ""
        doc_lower = str(docstring or "").lower()
        name_tokens = split_tokens(name) if name else []
        name_token_set = set(name_tokens)

        # Precompute word sets for reuse across scoring sections
        doc_words = set(re.findall(r'[a-z]{3,}', doc_lower)) if doc_lower else set()
        path_tokens = set(split_tokens(file_path)) if file_path else set()

        # Helper: get IDF weight for a token (1.0 if no weights provided)
        def w(token: str) -> float:
            if token_weights is None:
                return 1.0
            return token_weights.get(token, 1.0)

        total_weight = sum(w(qt) for qt in query_tokens) if query_tokens else 1.0

        # --- Name Scoring ---
        # The name is the most important signal.  Four sub-scores are computed
        # and the maximum is taken as the final name_score.

        # 1. Exact token overlap (strongest signal) -- IDF-weighted.
        #    "heartbeat" in query matching "heartbeat" in name_tokens scores
        #    very high because "heartbeat" has high IDF weight.
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

        # 2. Containment + stem: query token as substring/stem of name token.
        #    "auth" in "authenticate" scores 0.6x (substring containment).
        #    "send" matching "sending" scores 0.5x (stem match).
        containment = 0.0
        if query_tokens and name_tokens:
            matched_weight = 0.0
            for qt in query_tokens:
                if qt in name_token_set:
                    matched_weight += w(qt)
                elif any(qt in nt or nt in qt for nt in name_tokens):
                    matched_weight += w(qt) * 0.6  # Substring containment
                elif any(self._stem_match(qt, nt) for nt in name_tokens):
                    matched_weight += w(qt) * 0.5  # Stem match
            containment = matched_weight / total_weight

        # 3. Per-token substring in full name string -- with stem fallback.
        #    Catches matches that token-level comparison misses, e.g.,
        #    "db" in "db_connection" when "db" was split out differently.
        per_token_in_name = 0.0
        if query_tokens and name_lower:
            hit_weight = 0.0
            for t in query_tokens:
                if t in name_lower:
                    hit_weight += w(t)
                elif any(self._stem_match(t, nt) for nt in name_tokens):
                    hit_weight += w(t) * 0.5
            per_token_in_name = hit_weight / total_weight

        # 4. SequenceMatcher -- heavily capped at 0.15, purely a tiebreaker.
        #    Useful for typo tolerance ("get_usr" vs "get_user") but too noisy
        #    to be given significant weight.
        seq_score = 0.0
        if name_lower:
            seq_score = min(
                0.15,
                difflib.SequenceMatcher(None, query_lower, name_lower).ratio(),
            )

        name_score = max(exact_overlap, containment, per_token_in_name * 0.85, seq_score)

        # Exact keyword floor: when rare query tokens match name tokens,
        # the score should reflect the match quality, not just query coverage.
        # Example: query "SSE streaming heartbeat" on function "_with_heartbeat"
        # -- "heartbeat" has high IDF but only covers 1/3 of query tokens,
        # so exact_overlap would be low.  The floor ensures the score reflects
        # that a rare, valuable token matched.
        if exact_hits > 0 and name_tokens:
            best_hit_idf = max(w(qt) for qt in query_tokens if qt in name_token_set)
            # name_coverage = fraction of name tokens that the query covers
            name_coverage = exact_hits / len(name_tokens)
            keyword_floor = best_hit_idf * max(0.35, name_coverage)
            name_score = max(name_score, keyword_floor)

        # Full query as substring of name -> very strong signal.
        # "send_message" contains "send_message" -> 0.90 minimum.
        if query_lower and query_lower in name_lower:
            name_score = max(name_score, 0.90)

        # --- Docstring Scoring ---
        # Docstring provides conceptual context.  Stem matching is applied so
        # "sending" in the query matches "send" in the docstring.
        doc_score = 0.0
        if doc_lower and query_tokens:
            doc_hit_weight = 0.0
            for t in query_tokens:
                if t in doc_lower:
                    doc_hit_weight += w(t)  # Exact substring in doc
                elif any(self._stem_match(t, dw) for dw in doc_words):
                    doc_hit_weight += w(t) * 0.7  # Stem match in doc (weaker)
            doc_score = doc_hit_weight / total_weight

        # --- Path Scoring ---
        # File path provides structural context.  A query about "admin" matching
        # a file in the "admin/" directory is a strong signal.
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
                # Directory-level matches get a 1.3x boost because they indicate
                # structural relevance (the file lives in a relevant module).
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

        # --- Weighted Combination ---
        # Name is the strongest signal (0.45), docstring second (0.30),
        # path third (0.25).  Docstring weight was raised from 0.20 to 0.30
        # so that file-context matches (sibling function names, module
        # docstrings) can meaningfully influence scores.
        final = name_score * 0.45 + doc_score * 0.30 + path_score * 0.25

        # --- Multiplicative Adjustments ---

        # Name boost: if the symbol name contains query tokens, multiply the
        # final score.  Threshold scales inversely with query length: longer
        # queries naturally have lower per-token hit ratios.  A 4-token query
        # matching 1 name token gives ratio ~0.22, which could never reach a
        # fixed 0.30 threshold.  Scaling ensures the boost triggers appropriately.
        if exact_hits > 0 and query_tokens:
            weighted_hit_ratio = exact_hit_weight / total_weight
            boost_threshold = max(0.15, 0.60 / len(query_tokens))
            if weighted_hit_ratio >= boost_threshold:
                final *= (1.0 + weighted_hit_ratio * 1.0)

        # Docstring boost: when docstring matches >=50% of query weight,
        # apply a 1.2x boost.  This helps symbols whose docstrings describe
        # the concept even when their names don't directly match (e.g.,
        # "send_message" with docstring "Send a notification to the user"
        # for query "sending notifications").
        if doc_score >= 0.5 and query_tokens:
            final *= 1.2

        # --- Adjacent Pair Bonus / Orphan Penalty ---
        # For multi-word queries, compound term handling:
        # - If consecutive query tokens BOTH match somewhere -> pair bonus
        #   (indicates a compound term like "access control" or "user auth")
        # - If a token matches but its adjacent partner(s) DON'T -> orphan
        #   penalty (likely false positive, e.g., "control" in music_control
        #   when querying "access control")
        if len(query_tokens) >= 2 and (name_tokens or doc_lower):
            all_words = name_token_set | doc_words | path_tokens

            def _token_hits(t: str) -> bool:
                return (t in all_words
                        or any(self._stem_match(t, aw) for aw in all_words))

            # Build a cache of which query tokens hit any word in the symbol
            pairs_matched = 0
            total_pairs = len(query_tokens) - 1
            token_hit_cache: dict[str, bool] = {}
            for qt in query_tokens:
                token_hit_cache[qt] = _token_hits(qt)

            # Count how many adjacent token pairs both match
            for i in range(total_pairs):
                t1, t2 = query_tokens[i], query_tokens[i + 1]
                if token_hit_cache[t1] and token_hit_cache[t2]:
                    pairs_matched += 1

            # Pair bonus: up to 1.5x when all adjacent pairs match
            if pairs_matched > 0:
                pair_ratio = pairs_matched / total_pairs
                final *= (1.0 + pair_ratio * 0.5)

            # Orphan penalty: tokens that match but whose adjacent partners
            # ALL miss.  Example: "control" in music_control with query
            # "access control user verification" -> "access" misses, so
            # "control" is an orphan -- penalize by up to 0.5x.
            orphan_penalty_weight = 0.0
            for i, qt in enumerate(query_tokens):
                if not token_hit_cache[qt]:
                    continue
                # Gather adjacent partners (left and right neighbors)
                partners = []
                if i > 0:
                    partners.append(query_tokens[i - 1])
                if i < len(query_tokens) - 1:
                    partners.append(query_tokens[i + 1])
                if partners and not any(token_hit_cache[p] for p in partners):
                    orphan_penalty_weight += w(qt)

            if orphan_penalty_weight > 0:
                # Scale penalty by orphan weight fraction.  max(0.5, ...) ensures
                # we never penalize more than 50% -- the match is still partially
                # relevant.
                penalty = orphan_penalty_weight / total_weight
                final *= max(0.5, 1.0 - penalty * 0.4)

        # --- Concept Coverage Bonus ---
        # For queries with 3+ tokens, reward results that match more UNIQUE
        # query tokens across name/doc/path.  A result matching 4/5 query
        # tokens across different fields is more likely truly relevant than
        # one matching 1 token very strongly.
        if len(query_tokens) >= 3:
            covered = set()
            for qt in query_tokens:
                if qt in name_token_set or any(self._stem_match(qt, nt) for nt in name_tokens):
                    covered.add(qt)
                elif qt in doc_words or any(self._stem_match(qt, dw) for dw in doc_words):
                    covered.add(qt)
                elif qt in path_tokens or any(self._stem_match(qt, p) for p in path_tokens):
                    covered.add(qt)
            coverage = len(covered) / len(query_tokens)
            # Only activate when >= 50% of tokens are covered (avoids boosting
            # weak matches).  Up to 1.3x bonus at 100% coverage.
            if coverage >= 0.5:
                final *= (1.0 + coverage * 0.3)

        # --- Test File Penalty ---
        # Production code should rank above test code for discovery queries.
        # Files in test/tests/load_tests directories or with test_ prefix are
        # penalized 0.85x to push them below equivalent production matches.
        if file_path:
            path_norm = file_path.replace("\\", "/").lower()
            path_parts = path_norm.split("/")
            if any(p in _TEST_DIRS or p.startswith("test_") for p in path_parts):
                final *= 0.85

        # --- Soft Cap ---
        # Scores below 0.85 are unchanged.  Scores above are compressed into
        # [0.85, 1.0) using a diminishing returns formula.  This preserves
        # ordering among strong matches without collapsing them all to 1.0.
        # Formula: 0.85 + 0.15 * excess / (0.15 + excess)
        # At final=1.0: result is ~0.925.  At final=2.0: result is ~0.95.
        if final <= 0.85:
            return final
        excess = final - 0.85
        return 0.85 + 0.15 * excess / (0.15 + excess)
