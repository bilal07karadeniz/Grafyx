# Test Report Fixes Design

Date: 2026-02-26
Status: Approved

## Context

Comprehensive testing of Grafyx against a 504-file fullstack monorepo (Python + TypeScript) revealed 5 issues. Overall score: 8.8/10. This design addresses all 5.

## Issue 1: cross_file_usages inflation (HIGH)

**Root cause**: Strategy 2/3 in `_tools_introspection.py` append files with empty `lines: []`. Cross-language fallback in `_indexes.py` resolves Python imports to TypeScript files.

**Fix**:
- Add `if lines:` guard before appending in both Strategy 2 (line 518) and Strategy 3 (line 565)
- Restrict cross-language import resolution to same language family (JS<->TS ok, Python<->TS blocked)

## Issue 2: find_related_code low recall (HIGH)

**Root cause**: Graph expansion capped at 5 slots, source blending only fires for weak scores, no directory affinity, 1-hop imports only.

**Fix** (tune existing knobs):
1. Expand from top 8 results (was 5) for callers, top 5 (was 3) for imports/co-location
2. Always blend source tokens (mild boost for strong matches, full blend for weak)
3. Add directory affinity expansion (0.45x score for co-located files)
4. Add 2-hop import expansion (0.35x score, capped at 3 additions)

## Issue 3: Module dependency misattribution (MEDIUM)

**Root cause**: `_forward_import_index` includes function-body lazy imports, creating ghost module edges.

**Fix**: Build parallel `_top_level_forward_import_index` during index construction by checking import line numbers against function line ranges. `get_module_dependencies()` uses top-level-only index.

## Issue 4: get_conventions counts (MEDIUM)

**Root cause**: Sampling cap of 500, import name counting, PascalCase regex misses `_Private`.

**Fix**:
- Fetch with max_results=5000 for true total, sample first 500 for patterns
- Deduplicate imports by (file, module)
- Strip leading underscores before PascalCase check

## Issue 5: Dynamic dispatch blindness (MEDIUM)

**Root cause**: No pattern matching for Celery `.delay()`/`.apply_async()`. No inference for untyped instance method calls.

**Fix**:
- Pass 4: Celery task detection (collect @task functions, regex scan for `.delay()`/`.apply_async()`)
- Pass 5: Unique-method heuristic (if `method` exists in exactly 1 class, infer target from `var.method()`)

## Implementation Order

Issues 1, 2, 4 are independent (parallel). Issues 3, 5 both touch `_indexes.py` (sequential, 5 before 3).
