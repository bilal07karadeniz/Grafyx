"""Main CodeSearcher class -- orchestrates multi-engine fuzzy code search.

This is the entry point for all search operations in Grafyx.  CodeSearcher
inherits from ScoringMixin (keyword relevance scoring) and SourceIndexMixin
(source-token inverted index), and owns an EmbeddingSearcher instance for
optional semantic search.

Search pipeline overview (for CodeSearcher.search()):

    1. **Tokenize & weigh**: split query into tokens, compute IDF weights.
    2. **Build context**: class docstrings for method enrichment.
    3. **Score all symbols**: iterate all functions, classes, and files,
       scoring each against the query using _score_match().
    4. **Source token blending**: for functions with weak keyword scores,
       check if their source code contains query tokens and blend in.
    5. **Diversity merge**: combine function/class/file results with
       quotas to prevent one kind from dominating.
    6. **Dual-engine augmentation**: run source-token search and embedding
       search in parallel, merge their results (max score wins per result).
    7. **Graph expansion**: expand results by following callers, imports,
       and co-located symbols to surface related code.
    8. **Confidence flagging**: mark all results as low_confidence when the
       best score is below 0.55.

Search pipeline overview (for CodeSearcher.search_files()):

    1. **Tokenize & weigh**: same as search().
    2. **Score symbols, group by file**: score all functions/classes and
       group them by file path.
    3. **Score filenames/paths directly**: catch files matching by directory
       or filename even when individual symbols don't match.
    4. **Path+source combined admission**: add files matching >=2 distinct
       query tokens across path + aggregate source tokens.
    5. **Dual-engine augmentation**: source-token + embedding search,
       with max-score-wins per file.
    6. **Graph expansion**: add files that import top results.
    7. **Path-relevance penalty**: penalize files admitted only by semantic
       embedding with no path token overlap.
    8. **Confidence flagging**: same as search().
"""

from __future__ import annotations

import logging
import math

from grafyx.graph import CodebaseGraph
from grafyx.utils import split_tokens

from grafyx.search._embeddings import EmbeddingSearcher
from grafyx.search._gibberish import is_gibberish
from grafyx.search._merge import _merge_results
from grafyx.search._relevance import ml_score_match
from grafyx.search._scoring import ScoringMixin
from grafyx.search._source_index import SourceIndexMixin
from grafyx.search._tokens import (
    SearchResult,
    _BACKEND_FRAMEWORKS,
    _FRONTEND_FRAMEWORKS,
    _STOP_WORDS,
    _tokenize_source,
)

logger = logging.getLogger(__name__)


def _log2_1p(x: float) -> float:
    """Compute log2(1 + x) without importing math in the hot loop."""
    return math.log2(1 + x)


class CodeSearcher(ScoringMixin, SourceIndexMixin):
    """Performs fuzzy text search across the codebase graph.

    Inherits:
        ScoringMixin: provides _score_match(), _stem_match(), _compute_idf_weights()
        SourceIndexMixin: provides _ensure_source_index(), _source_search(), _source_score_for()

    Owns:
        EmbeddingSearcher: optional dense vector search (lazy-initialized)

    Public methods:
        search(query, max_results, kind_filter) -> list[dict]:
            Search symbols (functions, classes, files) by fuzzy matching.
        search_files(description, max_results) -> list[dict]:
            Find files related to a description by scoring symbols inside them.

    Thread safety:
        CodeSearcher is NOT thread-safe.  Each MCP server request should use
        a shared CodeSearcher instance (writes are idempotent lazy-init).
    """

    def __init__(self, graph: CodebaseGraph):
        self._graph = graph
        # Source token index -- built lazily on first search call.
        # _source_symbols: list of (name, file_path, class_name)
        # _source_index: token -> set of indices into _source_symbols
        self._source_symbols: list[tuple[str, str, str]] | None = None
        self._source_index: dict[str, set[int]] | None = None
        # Optional embedding searcher -- initialized lazily on first search
        self._embedding_searcher: EmbeddingSearcher | None = None
        self._embedding_init_done = False

    # --- Embedding Support ---

    def _ensure_embeddings(self) -> None:
        """Lazily initialize and start background embedding build.

        Called once per CodeSearcher lifetime.  If fastembed is installed,
        creates an EmbeddingSearcher and kicks off a background thread to
        build/load embeddings.  If fastembed is not installed, this is a no-op.
        """
        if self._embedding_init_done:
            return
        self._embedding_init_done = True
        if EmbeddingSearcher.available():
            self._embedding_searcher = EmbeddingSearcher(self._graph)
            self._embedding_searcher.build_in_background()

    def _embedding_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[str, str, str, float]]:
        """Search using embeddings if available and ready.

        Returns empty list if embeddings haven't finished building yet or
        fastembed is not installed.  This graceful degradation means the
        caller never needs to check -- it just gets fewer results.
        """
        if self._embedding_searcher and self._embedding_searcher._ready:
            return self._embedding_searcher.search(query, top_k=top_k)
        return []

    def _embedding_search_files(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search files using file-level embeddings if available and ready."""
        if self._embedding_searcher and self._embedding_searcher._file_ready:
            return self._embedding_searcher.search_files(query, top_k=top_k)
        return []

    @property
    def degraded(self) -> bool:
        """True when the encoder is not available (offline, missing fastembed,
        or build still pending). Callers can use this to surface a hint that
        results came from token search alone.
        """
        return (
            self._embedding_searcher is None
            or not self._embedding_searcher._ready
        )

    @property
    def encoder_meta(self) -> dict:
        """Identify the encoder *actually* used to score this response.

        Returns ``{"model": "tokens", ...}`` when the embedding searcher is
        absent or its index is not ready — i.e. when ``degraded`` is True.
        Reporting the configured encoder id in that state is misleading
        because the response was scored without it.
        """
        if self._embedding_searcher is None:
            return {"model": "tokens", "version": "", "configured": "none"}

        cfg = getattr(self._embedding_searcher, "_cfg", None)
        configured_id = (
            cfg["id"] if cfg
            else getattr(self._embedding_searcher, "_model_name", "unknown")
        )
        configured_version = cfg.get("model_name", "") if cfg else ""

        if not self._embedding_searcher._ready:
            return {
                "model": "tokens",
                "version": "",
                "configured": configured_id,
            }

        return {
            "model": configured_id,
            "version": configured_version,
            "configured": configured_id,
        }

    def wait_for_index_ready(self, timeout: float = 600.0) -> bool:
        """Block until the embedding index has finished building.

        Used by the benchmark harness to ensure a warm index before timing
        queries. Returns False on timeout, True on ready (or no embeddings
        configured, in which case there is nothing to wait for).
        """
        import time as _time
        self._ensure_embeddings()
        if self._embedding_searcher is None:
            return True
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            if self._embedding_searcher._ready:
                return True
            if getattr(self._embedding_searcher, "_build_error", None):
                return False  # Permanent failure; don't keep polling
            _time.sleep(0.5)
        return False

    # --- Main Symbol Search ---

    def search(
        self,
        query: str,
        max_results: int = 10,
        kind_filter: str | None = None,
    ) -> list[dict]:
        """Search symbols by fuzzy matching query against names, docstrings, file paths.

        This is the primary search method, used by the MCP tool `find_related_code`.
        Returns list of dicts sorted by score descending, capped at max_results.
        Results are diversified: each kind (function, class, file) gets minimum
        representation so one kind cannot completely dominate.

        Args:
            query: Natural language search query (e.g., "JWT authentication",
                   "database connection pool").
            max_results: Maximum number of results to return.
            kind_filter: Optional filter to restrict results to one kind
                         ("function", "class", or "file").

        Returns:
            List of dicts with keys: name, kind, file_path, score, context,
            language.  May include "low_confidence": True when best score < 0.55.
        """
        # Trigger lazy init of source index + embeddings
        self._ensure_source_index()
        self._ensure_embeddings()

        query_tokens = [t for t in split_tokens(query) if t not in _STOP_WORDS]
        query_lower = query.lower()

        if not query_tokens:
            return []

        # --- Pre-fetch all symbols ---
        # max_results=2000 for functions: with 367 top-level functions, 500
        # only left 133 slots for class methods, cutting off most of them.
        # _check_tool_permission and similar methods were never searched.
        all_functions = self._graph.get_all_functions(max_results=2000, include_methods=True)
        all_classes = self._graph.get_all_classes(max_results=500)
        all_files = self._graph.get_all_files(max_results=500)

        # --- Build IDF weights ---
        token_weights = self._compute_idf_weights(query_tokens, all_functions, all_classes)

        # --- Build class docstring index (for method enrichment) ---
        # When scoring a method like UserService.authenticate(), the class
        # docstring ("Service for user management") provides conceptual
        # context that helps match queries like "user management".
        class_docstrings: dict[str, str] = {}  # class_name -> class docstring
        for cls_dict in all_classes:
            cls_name = cls_dict.get("name", "")
            doc = cls_dict.get("docstring", "")
            if cls_name and doc:
                class_docstrings[cls_name] = doc

        # --- Caller index for graph-boosted scoring ---
        caller_index = getattr(self._graph, "_caller_index", {})

        # --- Score all symbols, collecting results per kind ---
        func_results: list[SearchResult] = []
        class_results: list[SearchResult] = []
        file_results: list[SearchResult] = []

        # --- Score functions and methods ---
        if kind_filter is None or kind_filter == "function":
            for func_dict in all_functions:
                name = func_dict.get("name", "")
                func_file = func_dict.get("file", "")
                # Build enriched docstring for scoring:
                # 1. Function's own docstring
                # 2. Parent class docstring (if method) -- targeted, not sprayed
                own_doc = func_dict.get("docstring") or ""
                cls_name = func_dict.get("class_name", "")
                if cls_name:
                    cls_doc = class_docstrings.get(cls_name, "")
                    if cls_doc:
                        own_doc = f"{own_doc} {cls_doc}".strip()
                # Get source tokens for this function (if available)
                src_tokens = self._get_source_tokens(name, func_file)

                score = ml_score_match(
                    query_tokens, query_lower, name,
                    own_doc,
                    func_file,
                    token_weights=token_weights,
                    source_tokens=src_tokens,
                    is_dunder=name.startswith("__") and name.endswith("__"),
                    is_init_file="__init__" in func_file,
                    is_method=bool(cls_name),
                    is_class=False,
                )

                # Graph-boosted scoring: caller count bonus
                # Functions called by many others are structurally important
                caller_count = len(caller_index.get(name, []))
                if caller_count > 0 and score > 0.15:
                    caller_bonus = min(0.12, 0.02 * _log2_1p(caller_count))
                    score = min(score + caller_bonus, 1.0)

                # Threshold: only include results with meaningful relevance
                if score > 0.12:
                    # Add class context for methods to help the user identify them
                    context_str = func_dict.get("signature", "")
                    if cls_name:
                        method_name = context_str.split("(")[0].split()[-1] if "(" in context_str else name
                        context_str = f"{cls_name}.{method_name}  [{context_str}]"
                    func_results.append(SearchResult(
                        name=name,
                        kind="function",
                        file_path=func_file,
                        score=score,
                        context=context_str,
                        language=func_dict.get("language", ""),
                    ))

        # --- Score classes ---
        if kind_filter is None or kind_filter == "class":
            for cls_dict in all_classes:
                name = cls_dict.get("name", "")
                cls_file = cls_dict.get("file", "")
                own_doc = cls_dict.get("docstring") or ""
                # Don't include file context for classes -- in files with many
                # type definitions (e.g., types.py), sibling class names inflate
                # scores for every class, making SkillExecutionResult match
                # "permission" because SkillPermissions is a sibling.
                score = ml_score_match(
                    query_tokens, query_lower, name,
                    own_doc,
                    cls_file,
                    token_weights=token_weights,
                    is_dunder=name.startswith("__") and name.endswith("__"),
                    is_init_file="__init__" in cls_file,
                    is_method=False,
                    is_class=True,
                )
                if score > 0.12:
                    bases = cls_dict.get("base_classes", [])
                    base_str = ", ".join(str(b) for b in bases) if bases else ""
                    ctx = f"class {name}" + (f"({base_str})" if base_str else "")
                    class_results.append(SearchResult(
                        name=name,
                        kind="class",
                        file_path=cls_file,
                        score=score,
                        context=ctx,
                        language=cls_dict.get("language", ""),
                    ))

        # --- Score files ---
        if kind_filter is None or kind_filter == "file":
            for file_dict in all_files:
                path = file_dict.get("path", "")
                filename = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if path else ""
                file_doc = file_dict.get("docstring", "") or ""
                score = ml_score_match(query_tokens, query_lower, filename,
                                          file_doc, path,
                                          token_weights=token_weights)
                if score > 0.12:
                    fcount = file_dict.get("function_count", 0)
                    cc = file_dict.get("class_count", 0)
                    file_results.append(SearchResult(
                        name=filename,
                        kind="file",
                        file_path=path,
                        # 0.95x multiplier slightly deprioritizes file results vs
                        # symbol results at the same score, because a matching
                        # function is usually more actionable than a matching file
                        score=score * 0.95,
                        context=f"{fcount} functions, {cc} classes",
                        language=file_dict.get("language", ""),
                    ))

        # --- Diversity merge ---
        # Use a larger internal limit so augmentation engines (source tokens,
        # embeddings) have headroom to contribute.  Final truncation to
        # max_results happens at the end.
        internal_limit = max(max_results * 3, 30)
        merged = _merge_results(
            func_results, class_results, file_results,
            internal_limit, kind_filter,
        )

        # --- Dual-engine: embedding + source always-on ---
        # Embeddings provide semantic matching for queries that keyword
        # scoring can't handle ("JWT authentication" -> get_current_user).
        # Source tokens provide implementation-level coverage.
        # Per-result: max(keyword_score, embedding_score) wins automatically.
        if kind_filter is None or kind_filter == "function":
            result_keys_src = {(r.name, r.file_path) for r in merged}

            # Source token hits -- always search, not just as fallback.
            # These catch functions by implementation that keyword scoring missed.
            source_hits = self._source_search(query_tokens, token_weights, max_results=20)
            for s_name, s_file, s_cls, s_score in source_hits:
                if (s_name, s_file) not in result_keys_src:
                    result_keys_src.add((s_name, s_file))
                    display = f"{s_cls}.{s_name}" if s_cls else s_name
                    func_results.append(SearchResult(
                        name=s_name,
                        kind="function",
                        file_path=s_file,
                        score=s_score,
                        context=f"{display}  [source match]",
                        language="",
                    ))

            # Embedding search -- dual engine, no artificial score caps.
            # Embedding scores and keyword scores are on the same [0, 1] scale,
            # so max-score-wins comparison is meaningful.
            emb_hits = self._embedding_search(query, top_k=20)
            if emb_hits:
                # Build score map for fast lookup: (name, file) -> (score, class_name)
                emb_map: dict[tuple[str, str], tuple[float, str]] = {}
                for e_name, e_file, e_cls, e_score in emb_hits:
                    if e_score > 0.20:
                        key = (e_name, e_file)
                        if key not in emb_map or e_score > emb_map[key][0]:
                            emb_map[key] = (e_score, e_cls)

                # Upgrade existing results: take max(keyword_score, embedding_score).
                # This ensures embedding evidence never LOWERS a result's score.
                for r in merged:
                    if r.kind == "function":
                        key = (r.name, r.file_path)
                        if key in emb_map:
                            e_score, _ = emb_map.pop(key)
                            if e_score > r.score:
                                r.score = e_score
                # Also upgrade func_results for re-merge consistency
                for r in func_results:
                    key = (r.name, r.file_path)
                    if key in emb_map:
                        e_score, _ = emb_map.pop(key)
                        if e_score > r.score:
                            r.score = e_score

                # Add new embedding-only hits that weren't found by keyword search.
                # 0.35 threshold because embedding-only results lack keyword
                # corroboration and are more likely to be false positives.
                for (e_name, e_file), (e_score, e_cls) in emb_map.items():
                    if e_score > 0.30 and (e_name, e_file) not in result_keys_src:
                        result_keys_src.add((e_name, e_file))
                        display = f"{e_cls}.{e_name}" if e_cls else e_name
                        func_results.append(SearchResult(
                            name=e_name,
                            kind="function",
                            file_path=e_file,
                            score=e_score,
                            context=f"{display}  [semantic match]",
                            language="",
                        ))

            # Re-sort and re-merge with all engines' results
            func_results.sort(key=lambda r: r.score, reverse=True)
            merged = _merge_results(
                func_results, class_results, file_results,
                internal_limit, kind_filter,
            )

        # --- Graph Expansion ---
        # Three sources of expansion surface related code that the user
        # likely wants to see alongside the primary results:
        # 1. Caller expansion -- callers of top function results
        # 2. Import expansion -- files that import top result files
        # 3. Co-location -- other symbols in the same files as top results
        expansion_slots = min(max_results // 2, max(0, max_results - 3))
        if kind_filter is None and expansion_slots > 0 and len(merged) > max_results - expansion_slots:
            merged = merged[:max_results - expansion_slots]
        if kind_filter is None and len(merged) < max_results:
            result_keys = {(r.name, r.kind, r.file_path) for r in merged}
            expanded: list[SearchResult] = []

            # --- Source 1: Caller expansion ---
            # If get_current_user() is a top result, show functions that CALL it
            # (e.g., route handlers) because they're part of the same feature.
            # Score is 0.55x the called function's score (callers are one step
            # removed from the query).
            for r in merged[:15]:
                if r.kind != "function" or not r.file_path:
                    continue
                callers = caller_index.get(r.name, [])
                for caller in callers:
                    c_name = caller.get("name", "")
                    c_file = caller.get("file", "")
                    c_cls = caller.get("class", "")
                    if not c_name or not c_file:
                        continue
                    # Skip same-file callers (co-location handles those)
                    if c_file == r.file_path:
                        continue
                    key = (c_name, "function", c_file)
                    if key not in result_keys:
                        result_keys.add(key)
                        display = f"{c_cls}.{c_name}" if c_cls else c_name
                        expanded.append(SearchResult(
                            name=c_name,
                            kind="function",
                            file_path=c_file,
                            score=r.score * 0.55,
                            context=f"{display}  [calls {r.name}]",
                            language="",
                        ))

            # --- Source 2: Import expansion ---
            # Files that import top result files are likely consumers of the
            # feature being searched for.  Score is 0.50x the imported file's
            # score (one degree of separation).
            import_index = getattr(self._graph, "_import_index", {})
            top_files_for_import: set[str] = set()
            for r in merged[:5]:
                if r.file_path:
                    top_files_for_import.add(r.file_path)
            seen_import_files: set[str] = set()
            for tf in top_files_for_import:
                importers = self._graph.get_importers(tf) if hasattr(self._graph, "get_importers") else []
                seed_score = max(
                    (r.score for r in merged[:5] if r.file_path == tf),
                    default=0.3,
                )
                for imp_file in importers:
                    if imp_file in top_files_for_import or imp_file in seen_import_files:
                        continue
                    seen_import_files.add(imp_file)
                    fname = imp_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                    key = (fname, "file", imp_file)
                    if key not in result_keys:
                        result_keys.add(key)
                        expanded.append(SearchResult(
                            name=fname,
                            kind="file",
                            file_path=imp_file,
                            score=seed_score * 0.50,
                            context=f"imports {tf.rsplit('/', 1)[-1]}",
                            language="",
                        ))

            # --- Source 2b: 2-hop import expansion ---
            # Follow importers one more hop: if file A imports file B (top
            # result), and file C imports file A, add C as a potential
            # consumer. Capped at 3 additions with lower score (0.35x).
            second_hop_added = 0
            for imp_file in list(seen_import_files):
                if second_hop_added >= 3:
                    break
                importers_2 = self._graph.get_importers(imp_file) if hasattr(self._graph, "get_importers") else []
                for imp2_file in importers_2:
                    if second_hop_added >= 3:
                        break
                    if imp2_file in top_files_for_import or imp2_file in seen_import_files:
                        continue
                    fname = imp2_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                    key = (fname, "file", imp2_file)
                    if key not in result_keys:
                        hop1_name = imp_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                        result_keys.add(key)
                        expanded.append(SearchResult(
                            name=fname,
                            kind="file",
                            file_path=imp2_file,
                            score=0.35,
                            context=f"2-hop import via {hop1_name}",
                            language="",
                        ))
                        second_hop_added += 1

            # --- Source 3: Co-location ---
            # Other symbols in the same files as top results are likely part
            # of the same module/feature.  Score is 0.6x the top result's
            # score in that file.  Capped at 2 expansions per file to prevent
            # a single large file from dominating all expansion slots.
            top_files = set()
            for r in merged[:5]:
                if r.file_path and r.kind in ("function", "class"):
                    top_files.add(r.file_path)
            if top_files:
                coloc_per_file: dict[str, int] = {}
                for func_dict in all_functions:
                    fpath = func_dict.get("file", "")
                    fname = func_dict.get("name", "")
                    key = (fname, "function", fpath)
                    if fpath in top_files and key not in result_keys:
                        if coloc_per_file.get(fpath, 0) >= 2:
                            continue
                        parent_score = max(
                            (r.score for r in merged[:5] if r.file_path == fpath),
                            default=0.1,
                        )
                        ctx = func_dict.get("signature", "")
                        cls_name = func_dict.get("class_name", "")
                        if cls_name:
                            method_name = ctx.split("(")[0].split()[-1] if "(" in ctx else fname
                            ctx = f"{cls_name}.{method_name}  [{ctx}]"
                        expanded.append(SearchResult(
                            name=fname,
                            kind="function",
                            file_path=fpath,
                            score=parent_score * 0.6,
                            context=ctx,
                            language=func_dict.get("language", ""),
                        ))
                        result_keys.add(key)
                        coloc_per_file[fpath] = coloc_per_file.get(fpath, 0) + 1

            # --- Source 4: Directory affinity ---
            # Files in the same directory as top results are likely part of
            # the same feature/module. Score is 0.45x the top result's
            # score in that directory.
            top_dirs: dict[str, float] = {}  # dir_path -> best score
            for r in merged[:5]:
                fp = (r.file_path or "").replace("\\", "/")
                if "/" in fp:
                    dir_path = fp.rsplit("/", 1)[0]
                    if dir_path not in top_dirs or r.score > top_dirs[dir_path]:
                        top_dirs[dir_path] = r.score

            if top_dirs:
                for func_dict in all_functions:
                    fpath = (func_dict.get("file", "") or "").replace("\\", "/")
                    fname = func_dict.get("name", "")
                    if not fpath or not fname or "/" not in fpath:
                        continue
                    fdir = fpath.rsplit("/", 1)[0]
                    if fdir in top_dirs:
                        key = (fname, "function", fpath)
                        if key not in result_keys:
                            ctx = func_dict.get("signature", "") or fname
                            cls_name = func_dict.get("class_name", "")
                            if cls_name:
                                method_name = ctx.split("(")[0].split()[-1] if "(" in ctx else fname
                                ctx = f"{cls_name}.{method_name}  [{ctx}]"
                            expanded.append(SearchResult(
                                name=fname,
                                kind="function",
                                file_path=fpath,
                                score=top_dirs[fdir] * 0.45,
                                context=f"{ctx}  [same dir as match]",
                                language=func_dict.get("language", ""),
                            ))
                            result_keys.add(key)

            expanded.sort(key=lambda r: r.score, reverse=True)
            slots = max_results - len(merged)
            merged.extend(expanded[:slots])

        # --- Format output ---
        results = [
            {
                "name": r.name,
                "kind": r.kind,
                "file_path": r.file_path,
                "score": round(r.score, 3),
                "context": r.context,
                "language": r.language,
            }
            for r in merged[:max_results]
        ]

        # --- Gibberish gate ---
        # Two-layer defence: ML model first (character bigram patterns),
        # then heuristic fallback (symbol name / source token matching).
        best_score = results[0]["score"] if results else 0.0
        if results:
            query_is_gibberish = is_gibberish(query)
            if query_is_gibberish:
                # ML model says gibberish — but give the heuristic a chance
                # to override if the query matches actual codebase symbols.
                any_exact_name_hit = False
                all_name_tokens: set[str] = set()
                for fd in all_functions:
                    all_name_tokens.update(split_tokens(fd.get("name", "")))
                for cd in all_classes:
                    all_name_tokens.update(split_tokens(cd.get("name", "")))
                for qt in query_tokens:
                    if qt in all_name_tokens:
                        any_exact_name_hit = True
                        break
                    if any(self._stem_match(qt, nt) for nt in all_name_tokens):
                        any_exact_name_hit = True
                        break
                if not any_exact_name_hit:
                    source_idx = getattr(self, '_source_index', None) or {}
                    # When the ML model says gibberish, require >=2 distinct
                    # query tokens to hit the source index before letting the
                    # results through. A single coincidental match (e.g.,
                    # "foobar" appearing in test fixtures) is too weak to
                    # override the gibberish detection — that produced the
                    # FastAPI v0.2.4 audit failure where "xyzzy foobar qlrmph"
                    # returned high-confidence results from test_*_foobar.
                    source_hits = sum(1 for qt in query_tokens if qt in source_idx)
                    if source_hits < 2:
                        for r in results:
                            r["score"] = min(r["score"], 0.35)
                        best_score = results[0]["score"] if results else 0.0

        # --- Confidence flagging ---
        # When the best score is below 0.55, ALL results are flagged as
        # low_confidence.  This signals to the MCP client that the search
        # may not have found what the user was looking for.
        if results and best_score < 0.55:
            for r in results:
                r["low_confidence"] = True
        return results

    # --- Path Normalization Utility ---

    @staticmethod
    def _path_suffix(p: str, n: int = 3) -> str:
        """Get the last n components of a normalized path for robust matching.

        Different graph-sitter APIs may return absolute vs relative paths
        or mirror vs translated paths.  Using the path suffix as a lookup
        key avoids mismatches caused by differing prefixes.

        Examples:
            _path_suffix("/mnt/c/project/src/auth/views.py") -> "src/auth/views.py"
            _path_suffix("auth/views.py") -> "auth/views.py"  (fewer than n components)
            _path_suffix("views.py", n=1) -> "views.py"
        """
        p = p.replace("\\", "/").rstrip("/")
        parts = p.split("/")
        return "/".join(parts[-n:]) if len(parts) >= n else p

    # --- File Search ---

    def search_files(
        self,
        description: str,
        max_results: int = 5,
    ) -> list[dict]:
        """Find files related to a description by scoring symbols inside them.

        Groups function/class matches by file and returns the top files with
        their relevant symbols.  Also scores file names/paths directly so
        files matching by directory (e.g., frontend/src/stores/) appear even
        if their individual symbols don't match.

        This method is used by the MCP tool `find_related_files`.

        Args:
            description: Natural language description of what to find
                         (e.g., "admin analytics endpoints").
            max_results: Maximum number of files to return.

        Returns:
            List of dicts: {file, score, language, relevant_symbols: [{name, kind, score}]}
            May include "low_confidence": True when best score < 0.55.
        """
        # Trigger lazy init of source index + embeddings
        self._ensure_source_index()
        self._ensure_embeddings()

        query_tokens = [t for t in split_tokens(description) if t not in _STOP_WORDS]
        query_lower = description.lower()

        if not query_tokens:
            return []

        all_functions = self._graph.get_all_functions(max_results=2000, include_methods=True)
        all_classes = self._graph.get_all_classes(max_results=500)
        all_files = self._graph.get_all_files(max_results=500)

        # Build IDF weights so common words ("store", "data") are downweighted
        token_weights = self._compute_idf_weights(query_tokens, all_functions, all_classes)

        # --- Build path normalization index ---
        # Function paths may be relative ("tests/search.py") while file paths
        # are absolute ("/mnt/c/.../tests/search.py").  Using path suffixes as
        # a bridge ensures symbols and files map to the same dict keys.
        suffix_to_full: dict[str, str] = {}
        for file_dict in all_files:
            path = file_dict.get("path", "").replace("\\", "/")
            if path:
                # Store at multiple suffix depths so both short relative paths
                # (e.g., "tests/search.py") and long absolute paths resolve
                # to the same canonical form.
                for depth in (3, 2, 1):
                    suffix = self._path_suffix(path, n=depth)
                    if suffix not in suffix_to_full:
                        suffix_to_full[suffix] = path

        def _resolve_path(p: str) -> str:
            """Resolve a path to its canonical full form via suffix matching."""
            p = p.replace("\\", "/")
            # Try deeper suffixes first to avoid ambiguous short matches
            for depth in (3, 2, 1):
                suffix = self._path_suffix(p, n=depth)
                if suffix in suffix_to_full:
                    return suffix_to_full[suffix]
            return p

        # --- Score every symbol and group by file ---
        file_symbols: dict[str, list[dict]] = {}  # path -> [{name, kind, score, language}]

        for func_dict in all_functions:
            fpath = func_dict.get("file", "")
            if not fpath:
                continue
            name = func_dict.get("name", "")
            cls_name = func_dict.get("class_name", "")
            src_tokens = self._get_source_tokens(name, fpath)
            score = ml_score_match(
                query_tokens, query_lower, name,
                func_dict.get("docstring", ""), fpath,
                token_weights=token_weights,
                source_tokens=src_tokens,
            )
            if score > 0.15:
                display_name = f"{cls_name}.{name}" if cls_name else name
                fpath_norm = _resolve_path(fpath)
                if fpath_norm not in file_symbols:
                    file_symbols[fpath_norm] = []
                file_symbols[fpath_norm].append({
                    "name": display_name,
                    "kind": "function",
                    "score": round(score, 3),
                    "language": func_dict.get("language", ""),
                })

        for cls_dict in all_classes:
            cpath = cls_dict.get("file", "")
            if not cpath:
                continue
            name = cls_dict.get("name", "")
            score = ml_score_match(
                query_tokens, query_lower, name,
                cls_dict.get("docstring", ""), cpath,
                token_weights=token_weights,
            )
            if score > 0.15:
                cpath_norm = _resolve_path(cpath)
                if cpath_norm not in file_symbols:
                    file_symbols[cpath_norm] = []
                file_symbols[cpath_norm].append({
                    "name": name,
                    "kind": "class",
                    "score": round(score, 3),
                    "language": cls_dict.get("language", ""),
                })

        # --- Score file names/paths directly ---
        # Catches files that match by directory or filename even if their
        # individual symbols don't.  E.g., "Zustand store" ->
        # frontend/src/stores/agentStore.ts
        file_name_scores: dict[str, float] = {}
        file_languages: dict[str, str] = {}
        for file_dict in all_files:
            path = file_dict.get("path", "")
            if not path:
                continue
            filename = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            score = ml_score_match(
                query_tokens, query_lower, filename,
                "", path,
                token_weights=token_weights,
            )
            if score > 0.12:
                path_norm = path.replace("\\", "/")
                file_name_scores[path_norm] = score
                file_languages[path_norm] = file_dict.get("language", "")

        # --- Merge symbol scores and filename scores per file ---
        all_file_paths = set(file_symbols.keys()) | set(file_name_scores.keys())
        file_ranks: list[tuple[str, float, list[dict]]] = []
        for fpath in all_file_paths:
            symbols = file_symbols.get(fpath, [])
            symbols.sort(key=lambda s: s["score"], reverse=True)
            symbol_score = symbols[0]["score"] if symbols else 0
            # File-name-only matches are weaker signals than symbol matches.
            # 0.6 multiplier prevents filenames (e.g., test_*.py matching "test")
            # from dominating over actual symbol-level relevance.
            name_score = file_name_scores.get(fpath, 0) * 0.6
            file_score = max(symbol_score, name_score)
            file_ranks.append((fpath, file_score, symbols))

        # --- Directory name bonus ---
        # Files in directories whose name matches a query token get a 1.3x boost.
        # "stores/agentStore.ts" matching "store" in the directory name is a
        # strong structural signal that the file is relevant.
        for i, (fpath, score, symbols) in enumerate(file_ranks):
            fpath_norm = fpath.replace("\\", "/")
            if "/" in fpath_norm:
                parent_dir = fpath_norm.rsplit("/", 1)[0].rsplit("/", 1)[-1]
                parent_tokens = set(split_tokens(parent_dir))
                if any(
                    qt in parent_tokens
                    or any(self._stem_match(qt, pt) for pt in parent_tokens)
                    for qt in query_tokens
                ):
                    file_ranks[i] = (fpath, score * 1.3, symbols)

        # --- Framework language bias ---
        # When a query contains a known frontend framework name (zustand, redux,
        # react...), boost TypeScript/JS results.  When it contains a backend
        # framework, boost Python results.  1.5x multiplier for matching language.
        frontend_bias = any(qt in _FRONTEND_FRAMEWORKS for qt in query_tokens)
        backend_bias = any(qt in _BACKEND_FRAMEWORKS for qt in query_tokens)
        if frontend_bias or backend_bias:
            for i, (fpath, score, symbols) in enumerate(file_ranks):
                lang = ""
                if symbols:
                    lang = symbols[0]["language"]
                elif fpath in file_languages:
                    lang = file_languages[fpath]
                if frontend_bias and lang in ("typescript", "javascript"):
                    file_ranks[i] = (fpath, score * 1.5, symbols)
                elif backend_bias and lang == "python":
                    file_ranks[i] = (fpath, score * 1.5, symbols)

        # --- Dual-engine: source + embedding always-on for search_files ---
        # Always run source token and embedding search, grouping by file.
        # Per-file: max(keyword_score, embedding_score) wins automatically.
        existing_files = {fp for fp, _, _ in file_ranks}

        # --- Path + aggregate-source combined admission ---
        # Adds files not yet in file_ranks when they match >=2 DISTINCT query tokens
        # across their full path (all directory segments) plus ALL of their functions'
        # source tokens (pre-pruning, so common-but-meaningful tokens like "admin"
        # aren't excluded by the 30% IDF pruning in the source search index).
        #
        # This catches FastAPI files like admin/platform.py where:
        # - "admin" appears in the directory path
        # - "analytics" appears in function names / route decorator strings
        # But no single function scores high enough to clear the 0.35 symbol threshold.
        #
        # Threshold: 2 distinct tokens + 40% of IDF-weighted query mass -> prevents
        # single-token false positives like models/analytics.py (only "analytics" matches).
        _file_src_toks: dict[str, set[str]] = getattr(self, '_file_source_tokens', {})
        _total_q_weight = sum(token_weights.get(qt, 1.0) for qt in query_tokens) or 1.0
        for _file_dict in all_files:
            _path = _file_dict.get("path", "").replace("\\", "/")
            if not _path:
                continue
            _path_norm = _resolve_path(_path)
            if _path_norm in existing_files:
                continue
            _path_parts = set(split_tokens(_path_norm))
            # Split path tokens into directory-only vs filename so we
            # can detect "weak" matches that come exclusively from
            # directory names (e.g. query "login flow" matching the
            # ``flows/`` directory in ``services/flows/activation.py``).
            _filename = _path_norm.rsplit("/", 1)[-1]
            _file_stem = _filename.rsplit(".", 1)[0] if "." in _filename else _filename
            _filename_tokens = set(split_tokens(_file_stem))
            _dir_only_tokens = _path_parts - _filename_tokens
            # Resolve file source tokens -- try full path, then descending suffixes.
            # The _file_source_tokens dict is keyed by path suffixes at multiple
            # depths (built in _ensure_source_index).
            _src_parts: set[str] = set()
            for _d in (5, 4, 3, 2, 1):
                _sfx = "/".join(_path_norm.split("/")[-_d:]) if _path_norm.count("/") >= _d else _path_norm
                _src_parts = _file_src_toks.get(_sfx, set())
                if _src_parts:
                    break
            _combined = _path_parts | _src_parts
            _distinct = sum(1 for qt in query_tokens if qt in _combined)
            # If exactly one query token matches and it's from the path, read the
            # full file (first 8 KB) to capture module-level strings that
            # _file_source_tokens misses -- e.g., APIRouter(prefix="/api/admin/analytics")
            # where "analytics" only appears in the module-level assignment, not in
            # any function body.
            if _distinct == 1 and any(qt in _path_parts for qt in query_tokens):
                _raw_path = _file_dict.get("path", "")
                try:
                    with open(_raw_path, "r", encoding="utf-8", errors="ignore") as _fh:
                        _file_content = _fh.read(8192)
                    _combined = _combined | set(_tokenize_source(_file_content))
                    _distinct = sum(1 for qt in query_tokens if qt in _combined)
                except OSError:
                    pass
            _hit_weight = sum(token_weights.get(qt, 1.0) for qt in query_tokens if qt in _combined)
            _coverage = _hit_weight / _total_q_weight
            # Module-name collision guard: if every matching token only
            # touches a directory name (not the filename, not source),
            # the file is matching a generic container ("flows/",
            # "services/", "models/") and not the query intent. Drop it.
            _substantive = (_filename_tokens | _src_parts)
            _substantive_hits = sum(
                1 for qt in query_tokens if qt in _substantive
            )
            if _distinct >= 2 and _coverage >= 0.30:
                if _substantive_hits == 0:
                    continue
                _score = _coverage * 0.70
                # Soft-penalize when only the directory carries the
                # match weight (one substantive match isn't enough to
                # outrank pure directory-noun coincidences).
                if _substantive_hits < 2:
                    _score *= 0.6
                # Apply inline directory bonus (same logic as the existing bonus above)
                _parent_dir = _path_norm.rsplit("/", 1)[0].rsplit("/", 1)[-1] if "/" in _path_norm else ""
                _parent_toks = set(split_tokens(_parent_dir))
                if any(
                    qt in _parent_toks or any(self._stem_match(qt, pt) for pt in _parent_toks)
                    for qt in query_tokens
                ):
                    _score = min(1.0, _score * 1.3)
                existing_files.add(_path_norm)
                _short = _path_norm.rsplit("/", 1)[-1]
                file_ranks.append((
                    _path_norm, _score,
                    [{"name": _short, "kind": "file", "score": round(_score, 3),
                      "language": _file_dict.get("language", "")}],
                ))

        # --- Collect all dual-engine hits: source + function embedding ---
        dual_hits: list[tuple[str, str, str, float]] = []
        dual_hits.extend(self._source_search(query_tokens, token_weights, max_results=20))
        for e_name, e_file, e_cls, e_score in self._embedding_search(description, top_k=20):
            if e_score > 0.25:
                dual_hits.append((e_name, e_file, e_cls, e_score))

        # File-level embedding search -- direct semantic match at file level
        for f_file, f_score in self._embedding_search_files(description, top_k=10):
            f_file_norm = _resolve_path(f_file) if f_file else f_file
            if f_file_norm not in existing_files:
                existing_files.add(f_file_norm)
                short = f_file_norm.rsplit("/", 1)[-1] if "/" in f_file_norm else f_file_norm
                file_ranks.append((
                    f_file_norm,
                    f_score,
                    [{"name": short, "kind": "file",
                      "score": round(f_score, 3), "language": ""}],
                ))
            else:
                # Upgrade existing file score if embedding score is higher
                for i, (fp, sc, syms) in enumerate(file_ranks):
                    if fp == f_file_norm and f_score > sc:
                        file_ranks[i] = (fp, f_score, syms)
                        break

        # Group dual-engine hits by file and merge into file_ranks
        for s_name, s_file, s_cls, s_score in dual_hits:
            s_file_norm = _resolve_path(s_file) if s_file else s_file
            display = f"{s_cls}.{s_name}" if s_cls else s_name
            if s_file_norm not in existing_files:
                existing_files.add(s_file_norm)
                file_ranks.append((
                    s_file_norm,
                    s_score * 0.95,
                    [{"name": display, "kind": "function",
                      "score": round(s_score, 3), "language": ""}],
                ))
            else:
                # File already in results -- upgrade score if dual-engine hit is better
                for i, (fp, sc, syms) in enumerate(file_ranks):
                    if fp == s_file_norm:
                        new_score = max(sc, s_score * 0.95)
                        syms_copy = list(syms)
                        if not any(sym["name"] == display for sym in syms_copy):
                            syms_copy.append({
                                "name": display, "kind": "function",
                                "score": round(s_score, 3), "language": "",
                            })
                        file_ranks[i] = (fp, new_score, syms_copy)
                        break

        # --- Graph expansion for search_files ---
        # For top-scoring files, add files that import them (consumers).
        # These are likely consumers of the feature being searched for.
        file_ranks.sort(key=lambda x: x[1], reverse=True)
        if len(file_ranks) < max_results:
            existing_files_exp = {fp for fp, _, _ in file_ranks}
            for fpath, score, _syms in file_ranks[:3]:
                if score < 0.3:
                    continue
                importers = (
                    self._graph.get_importers(fpath)
                    if hasattr(self._graph, "get_importers")
                    else []
                )
                for imp_file in importers:
                    imp_norm = _resolve_path(imp_file)
                    if imp_norm not in existing_files_exp:
                        existing_files_exp.add(imp_norm)
                        short_name = fpath.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                        file_ranks.append((
                            imp_norm,
                            score * 0.45,
                            [{"name": f"imports {short_name}", "kind": "import",
                              "score": round(score * 0.45, 3), "language": ""}],
                        ))

        # --- Path-relevance penalty ---
        # Files admitted only by semantic embedding with no path token overlap
        # get a 0.72x score penalty.  This suppresses false positives like
        # ShareTab.tsx (score 0.704) for a query like "admin analytics endpoints"
        # where the path contains none of the query tokens.
        for _i, (_fp, _sc, _sy) in enumerate(file_ranks):
            _fp_toks = set(split_tokens(_fp.replace("\\", "/")))
            if not any(
                qt in _fp_toks or any(self._stem_match(qt, pt) for pt in _fp_toks)
                for qt in query_tokens
            ):
                file_ranks[_i] = (_fp, _sc * 0.72, _sy)

        file_ranks.sort(key=lambda x: x[1], reverse=True)

        # --- Per-directory diversity cap ---
        # Prevent a single directory from dominating all results.
        # Max 2 files per directory in the final output; remaining
        # slots go to files from other directories for better coverage.
        dir_counts: dict[str, int] = {}
        diverse_ranks: list[tuple[str, float, list[dict]]] = []
        overflow: list[tuple[str, float, list[dict]]] = []
        for entry in file_ranks:
            fpath_norm = entry[0].replace("\\", "/")
            parent = fpath_norm.rsplit("/", 1)[0] if "/" in fpath_norm else ""
            count = dir_counts.get(parent, 0)
            if count < 2:
                diverse_ranks.append(entry)
                dir_counts[parent] = count + 1
            else:
                overflow.append(entry)
        file_ranks = diverse_ranks + overflow

        # --- Format output ---
        results = []
        for fpath, score, symbols in file_ranks[:max_results]:
            lang = ""
            if symbols:
                lang = symbols[0].get("language", "")
            if not lang and fpath in file_languages:
                lang = file_languages[fpath]
            # Cap score at 1.0 -- directory/framework bonuses can push above
            results.append({
                "file": fpath,
                "score": round(min(1.0, score), 3),
                "language": lang,
                "relevant_symbols": symbols[:5],
            })

        # --- Confidence flagging ---
        best_score = results[0]["score"] if results else 0.0
        if results and best_score < 0.55:
            for r in results:
                r["low_confidence"] = True
        return results
