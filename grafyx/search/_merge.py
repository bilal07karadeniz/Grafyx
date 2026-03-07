"""Result merging with diversity guarantees for code search.

This module contains the standalone _merge_results() function that combines
search results from three sources (functions, classes, files) into a single
ranked list with diversity constraints.

Why diversity matters:
    Without diversity guarantees, a codebase with 500 functions and 20 classes
    would produce search results dominated by function matches.  If a user
    queries "authentication", they want to see the AuthService class, the
    auth.py file, AND the relevant functions -- not just 10 functions.

    The merge strategy reserves minimum quotas for each kind, then fills
    remaining slots with the best results regardless of kind.  This ensures
    every relevant kind gets representation while still ranking by score.

Merge strategy:
    1. Reserve quota slots: functions (max_results/3), classes (max_results/3),
       files (max_results/5).  Minimum 3, 3, 2 respectively.
    2. Fill reserved slots from each kind's top results (deduplicated).
    3. Fill remaining slots from ALL kinds by score descending (best-first).
    4. Final sort by score descending.

When a kind_filter is active (e.g., searching only functions), diversity is
skipped and results are simply sorted by score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grafyx.search._tokens import SearchResult


def _merge_results(
    func_results: list[SearchResult],
    class_results: list[SearchResult],
    file_results: list[SearchResult],
    max_results: int,
    kind_filter: str | None,
) -> list[SearchResult]:
    """Merge results from different kinds with diversity guarantees.

    When kind_filter is set, only one list will be non-empty, so we just
    sort and truncate.  When kind_filter is None, we apply the quota-based
    diversity strategy described in the module docstring.

    Args:
        func_results: Scored function/method search results.
        class_results: Scored class search results.
        file_results: Scored file search results.
        max_results: Maximum number of results to return.
        kind_filter: If set ("function", "class", "file"), skip diversity.

    Returns:
        Merged and sorted list of SearchResult, capped at max_results.
    """
    # --- Single-kind mode: no diversity needed ---
    if kind_filter is not None:
        merged = func_results or class_results or file_results
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:max_results]

    # --- Multi-kind mode: apply diversity quotas ---
    # Pre-sort each list by score so we pick the best from each kind
    func_results.sort(key=lambda r: r.score, reverse=True)
    class_results.sort(key=lambda r: r.score, reverse=True)
    file_results.sort(key=lambda r: r.score, reverse=True)

    # Reserve minimum slots per kind.  The floor values (3, 3, 2) ensure
    # each kind gets meaningful representation even for small max_results.
    func_quota = max(3, max_results // 3)
    class_quota = max(3, max_results // 3)
    file_quota = max(2, max_results // 5)

    # Phase 1: Fill reserved quota slots from each kind (deduplicated)
    reserved = set()
    merged = []
    for r in (func_results[:func_quota] + class_results[:class_quota]
              + file_results[:file_quota]):
        key = (r.name, r.kind, r.file_path)
        if key not in reserved:
            reserved.add(key)
            merged.append(r)

    # Phase 2: Fill remaining slots from ALL kinds by score (best-first).
    # This ensures that high-scoring results of any kind aren't excluded
    # just because their kind's quota was filled.
    remaining = max_results - len(merged)
    if remaining > 0:
        all_remaining = []
        for r in func_results + class_results + file_results:
            key = (r.name, r.kind, r.file_path)
            if key not in reserved:
                all_remaining.append(r)
                reserved.add(key)
        all_remaining.sort(key=lambda r: r.score, reverse=True)
        merged.extend(all_remaining[:remaining])

    # Final sort so the caller sees results in score order
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged
