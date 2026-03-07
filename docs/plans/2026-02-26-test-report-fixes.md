# Test Report Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 5 issues identified in the comprehensive test report (overall 8.8/10 → target 9.5/10).

**Architecture:** Targeted fixes across 5 files. No new modules needed. Issues 1/2/4 are independent; issues 3/5 both touch `_indexes.py` and are sequenced (5 before 3). All changes are backward-compatible — no MCP tool API changes.

**Tech Stack:** Python 3.12, graph-sitter, FastMCP, pytest

---

### Task 1: Fix cross_file_usages inflation in get_class_context

**Files:**
- Modify: `grafyx/server/_tools_introspection.py:517-569`
- Modify: `grafyx/graph/_indexes.py:732-747`
- Test: `tests/test_class_context_usages.py` (create)

**Step 1: Write the failing tests**

Create `tests/test_class_context_usages.py`:

```python
"""Tests for cross_file_usages filtering in get_class_context."""

import pytest
from unittest.mock import MagicMock, patch

from tests._tool_compat import call_tool


def _make_class_obj(name, filepath, methods=None, usages=None,
                    source="", docstring="", properties=None,
                    dependencies=None):
    """Build a mock class object with the given attributes."""
    cls = MagicMock()
    cls.name = name
    cls.filepath = filepath
    cls.source = source
    cls.docstring = docstring
    cls.methods = methods or []
    cls.usages = usages or []
    cls.properties = properties or []
    cls.dependencies = dependencies or []
    return cls


def _make_mock_graph(cls_obj, cls_file, importers=None,
                     class_instances=None, class_method_names=None):
    """Build a mock graph for class context testing."""
    graph = MagicMock()
    graph.get_class.return_value = ("python", cls_obj)
    graph.resolve_path = lambda p: p if p else ""
    graph.translate_path = lambda p: p if p else ""
    graph.get_line_number.return_value = 10
    graph.get_filepath_from_obj.return_value = cls_file
    graph.get_importers.return_value = importers or []
    graph._class_instances = class_instances or {}
    graph._class_method_names = class_method_names or {}
    graph._is_ignored_file_path = lambda p: False
    return graph


class TestCrossFileUsagesFiltering:
    """Verify that files with no actual reference lines are excluded."""

    def test_strategy2_excludes_empty_lines(self):
        """Strategy 2 (import index) should NOT include files where
        _find_reference_lines returns empty for both class name and instances."""
        cls = _make_class_obj("CrawlCache", "/proj/cache.py")
        graph = _make_mock_graph(
            cls, "/proj/cache.py",
            importers=["/proj/utils.py", "/proj/unrelated.py"],
            class_instances={"CrawlCache": [("cache", "/proj/state.py")]},
        )

        # _find_reference_lines returns lines for utils.py but empty for unrelated.py
        def mock_find_ref(filepath, name):
            if filepath == "/proj/utils.py" and name == "CrawlCache":
                return [15, 22]
            if filepath == "/proj/utils.py" and name == "cache":
                return [30]
            return []

        with patch("grafyx.server._state._graph", graph), \
             patch("grafyx.server._state._init_ready", True), \
             patch("grafyx.server._tools_introspection._find_reference_lines", mock_find_ref):
            result = call_tool("get_class_context", {"class_name": "CrawlCache"})

        usages = result.get("cross_file_usages", [])
        usage_files = [u["file"] for u in usages]
        # utils.py should be included (has lines), unrelated.py should NOT
        assert "/proj/utils.py" in usage_files
        assert "/proj/unrelated.py" not in usage_files

    def test_strategy3_excludes_empty_lines(self):
        """Strategy 3 (unique method callers) should NOT include files where
        _find_reference_lines returns empty."""
        method = MagicMock()
        method.name = "unique_method"
        method.decorators = []
        cls = _make_class_obj("MyClass", "/proj/myclass.py", methods=[method])
        graph = _make_mock_graph(
            cls, "/proj/myclass.py",
            importers=[],
            class_method_names={"MyClass": {"unique_method"}},
        )
        # unique_method has a caller in /proj/consumer.py
        graph.get_callers.return_value = [
            {"name": "do_stuff", "file": "/proj/consumer.py"},
        ]

        def mock_find_ref(filepath, name):
            # consumer.py doesn't contain the class name string
            return []

        with patch("grafyx.server._state._graph", graph), \
             patch("grafyx.server._state._init_ready", True), \
             patch("grafyx.server._tools_introspection._find_reference_lines", mock_find_ref):
            result = call_tool("get_class_context", {"class_name": "MyClass"})

        usages = result.get("cross_file_usages", [])
        usage_files = [u["file"] for u in usages]
        assert "/proj/consumer.py" not in usage_files
```

**Step 2: Run tests to verify they fail**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/test_class_context_usages.py -v`
Expected: FAIL (empty-lines files are currently included)

**Step 3: Implement the fix in _tools_introspection.py**

In `grafyx/server/_tools_introspection.py`, add `if lines:` guard to Strategy 2 (around line 518):

Change lines 517-522:
```python
                lines.sort()
                context["cross_file_usages"].append({
                    "file": imp_file_norm,
                    "lines": lines[:5],
                })
                found_files.add(imp_file_norm)
```

To:
```python
                lines.sort()
                if lines:
                    context["cross_file_usages"].append({
                        "file": imp_file_norm,
                        "lines": lines[:5],
                    })
                found_files.add(imp_file_norm)
```

And add `if lines:` guard to Strategy 3 (around line 566):

Change lines 565-569:
```python
            lines.sort()
            context["cross_file_usages"].append({
                "file": uf,
                "lines": lines[:5],
            })
```

To:
```python
            lines.sort()
            if lines:
                context["cross_file_usages"].append({
                    "file": uf,
                    "lines": lines[:5],
                })
```

**Step 4: Fix cross-language import resolution in _indexes.py**

In `grafyx/graph/_indexes.py`, change lines 732-747 to restrict cross-language fallback to same language family:

Change:
```python
                            # Try same-language candidates first, then cross-language
                            same_lang = [c for c, cl in all_candidates if not src_lang or cl == src_lang]
                            cross_lang = [c for c, cl in all_candidates if src_lang and cl != src_lang]

                            target = None
                            for candidate in same_lang:
                                matches = suffix_to_path.get(candidate)
                                if matches:
                                    target = _pick_closest(matches, fpath)
                                    break
                            if target is None:
                                for candidate in cross_lang:
                                    matches = suffix_to_path.get(candidate)
                                    if matches:
                                        target = _pick_closest(matches, fpath)
                                        break
```

To:
```python
                            # Try same-language candidates first, then cross-language
                            # (but only within the same language family to avoid
                            # Python <-> TypeScript false positives).
                            same_lang = [c for c, cl in all_candidates if not src_lang or cl == src_lang]

                            target = None
                            for candidate in same_lang:
                                matches = suffix_to_path.get(candidate)
                                if matches:
                                    target = _pick_closest(matches, fpath)
                                    break
                            if target is None:
                                # Cross-language fallback: only within the same
                                # language family (JS <-> TS is fine, Python <-> TS
                                # is almost always a false positive from coincidental
                                # module name overlap like "config" or "utils").
                                _WEB_LANGS = {"typescript", "javascript"}
                                src_family = "web" if src_lang in _WEB_LANGS else src_lang
                                for cand, cand_lang in all_candidates:
                                    if not src_lang or cand_lang == src_lang:
                                        continue  # Already tried in same_lang pass
                                    cand_family = "web" if cand_lang in _WEB_LANGS else cand_lang
                                    if src_family != cand_family:
                                        continue  # Skip cross-family resolution
                                    matches = suffix_to_path.get(cand)
                                    if matches:
                                        target = _pick_closest(matches, fpath)
                                        break
```

**Step 5: Run tests to verify they pass**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/test_class_context_usages.py -v`
Expected: PASS

**Step 6: Run full test suite for regression**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/ -v --tb=short`
Expected: All existing tests still pass

**Step 7: Commit**

```bash
git add grafyx/server/_tools_introspection.py grafyx/graph/_indexes.py tests/test_class_context_usages.py
git commit -m "fix: filter empty-lines entries from cross_file_usages and block cross-language import resolution"
```

---

### Task 2: Improve find_related_code recall

**Files:**
- Modify: `grafyx/search/searcher.py:289-311,467-580`
- Test: `tests/test_search.py` (add tests)

**Step 1: Write the failing tests**

Add to `tests/test_search.py`:

```python
class TestSearchRecallImprovements:
    """Tests for improved recall in find_related_code."""

    def test_source_blending_fires_for_strong_matches(self):
        """Source token blending should provide a mild boost even for
        functions with keyword score >= 0.65."""
        # Setup: a function named "authenticate" (strong keyword match for
        # query "authenticate user") that also has "jwt" in its source.
        # The source blending should give a mild boost.
        searcher = _make_searcher_with_functions([
            {"name": "authenticate", "file": "/proj/auth.py",
             "source": "def authenticate(user):\n    token = jwt.decode(...)"},
            {"name": "validate_jwt", "file": "/proj/auth.py",
             "source": "def validate_jwt(token):\n    return jwt.verify(token)"},
        ])
        results = searcher.search("authenticate user jwt", max_results=10)
        # authenticate should score higher than without source blending
        auth_result = next((r for r in results if r["name"] == "authenticate"), None)
        assert auth_result is not None
        # Score should be > 0.65 (the old threshold where blending kicked in)
        assert auth_result["score"] > 0.65

    def test_directory_affinity_expansion(self):
        """Files in the same directory as a top match should appear in results
        via directory affinity expansion."""
        searcher = _make_searcher_with_functions([
            {"name": "create_session", "file": "/proj/voice/session.py",
             "source": "def create_session(): pass"},
            {"name": "end_session", "file": "/proj/voice/session.py",
             "source": "def end_session(): pass"},
            {"name": "connect_webrtc", "file": "/proj/voice/webrtc.py",
             "source": "def connect_webrtc(): pass"},
            {"name": "unrelated", "file": "/proj/utils/helpers.py",
             "source": "def unrelated(): pass"},
        ])
        results = searcher.search("create session", max_results=10)
        names = [r["name"] for r in results]
        # connect_webrtc is in the same directory as the top match
        # and should appear via directory affinity
        assert "connect_webrtc" in names
```

Note: `_make_searcher_with_functions` is a test helper that creates a CodeSearcher with a mock graph. If this doesn't exist, create it. Adapt to match existing test patterns in `tests/test_search.py`.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_search.py::TestSearchRecallImprovements -v`
Expected: FAIL

**Step 3: Implement source blending improvement**

In `grafyx/search/searcher.py`, change lines 289-311 (source token blending):

Change:
```python
                # --- Source token blending ---
                # If function body contains query tokens (e.g., jwt.decode,
                # StreamingResponse), boost score.  This catches functions whose
                # names don't match but whose implementations are directly
                # relevant.  Only applied to weak matches (< 0.65) to avoid
                # over-boosting already strong name matches.
                if score < 0.65:
                    src_score = self._source_score_for(
                        name, func_file, query_tokens, token_weights,
                    )
                    if src_score > 0.15:
                        # Blend: max of original or source-boosted score.
                        # Two blend strategies compete:
                        # - Pure source signal (src_score * 0.55): for functions
                        #   where name is completely unrelated
                        # - Additive blend (score + src_score * 0.25): for functions
                        #   where name partially matches
                        source_boosted = max(
                            score,
                            src_score * 0.55,          # Pure source signal
                            score + src_score * 0.25,  # Additive blend
                        )
                        score = min(0.75, source_boosted)
```

To:
```python
                # --- Source token blending ---
                # If function body contains query tokens (e.g., jwt.decode,
                # StreamingResponse), boost score.  This catches functions whose
                # names don't match but whose implementations are directly
                # relevant.
                src_score = self._source_score_for(
                    name, func_file, query_tokens, token_weights,
                )
                if src_score > 0.15:
                    if score < 0.65:
                        # Weak keyword match -- source evidence can dominate.
                        # Two blend strategies compete:
                        # - Pure source signal: for functions where name
                        #   is completely unrelated
                        # - Additive blend: for functions where name
                        #   partially matches
                        source_boosted = max(
                            score,
                            src_score * 0.55,          # Pure source signal
                            score + src_score * 0.25,  # Additive blend
                        )
                        score = min(0.80, source_boosted)
                    else:
                        # Strong keyword match -- source provides a mild
                        # additive nudge (soft cap at 0.85 handles overflow).
                        score = score + src_score * 0.10
```

**Step 4: Increase graph expansion seed counts**

In `grafyx/search/searcher.py`, change line 473:

Change:
```python
        expansion_slots = min(5, max(0, max_results - 3))
```
To:
```python
        expansion_slots = min(max_results // 2, max(0, max_results - 3))
```

Change line 486 (caller expansion from top results):
```python
            for r in merged[:5]:
```
To:
```python
            for r in merged[:8]:
```

Change line 518 (import expansion from top files):
```python
            for r in merged[:3]:
```
To:
```python
            for r in merged[:5]:
```

Change line 550 (co-location from top files):
```python
            for r in merged[:3]:
```
To:
```python
            for r in merged[:5]:
```

**Step 5: Add directory affinity expansion**

After the co-location section (after line 576, before `expanded.sort(...)`), add:

```python
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
```

**Step 6: Add 2-hop import expansion**

After the existing import expansion section (after line 543, before co-location), add:

```python
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
```

**Step 7: Run tests**

Run: `python -m pytest tests/test_search.py -v --tb=short`
Expected: All tests pass

**Step 8: Commit**

```bash
git add grafyx/search/searcher.py tests/test_search.py
git commit -m "feat: improve find_related_code recall with source blending, directory affinity, and 2-hop imports"
```

---

### Task 3: Fix get_conventions counts

**Files:**
- Modify: `grafyx/conventions.py:64-66,103-111,261-264,324-327,352-357,443-466`
- Test: `tests/test_conventions.py` (add tests)

**Step 1: Write the failing test**

Add to `tests/test_conventions.py`:

```python
class TestConventionCounts:
    """Verify conventions report accurate total counts."""

    def test_function_count_exceeds_sample(self):
        """When project has >500 functions, the convention should report
        the true total, not the sample size."""
        # Create a mock graph with 800 functions
        graph = MagicMock()
        functions = [{"name": f"func_{i}", "signature": f"def func_{i}()"} for i in range(800)]
        graph.get_all_functions.return_value = functions
        graph._lock = threading.RLock()
        graph._codebases = {}

        detector = ConventionDetector(graph)
        conventions = detector.detect_naming_conventions()
        # The pattern string should mention "800" not "500"
        naming = [c for c in conventions if "snake_case" in c.pattern]
        assert naming
        assert "800" in naming[0].pattern

    def test_pascal_case_with_underscore_prefix(self):
        """_PrivateClass and __DunderClass should count as PascalCase."""
        graph = MagicMock()
        graph.get_all_classes.return_value = [
            {"name": "MyClass"},
            {"name": "_PrivateHelper"},
            {"name": "__InternalThing"},
            {"name": "PublicThing"},
        ]
        graph._lock = threading.RLock()
        graph._codebases = {}

        detector = ConventionDetector(graph)
        conventions = detector.detect_naming_conventions()
        pascal = [c for c in conventions if "PascalCase" in c.pattern]
        assert pascal
        assert "100%" in pascal[0].pattern
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_conventions.py::TestConventionCounts -v`
Expected: FAIL

**Step 3: Implement the fixes**

In `grafyx/conventions.py`, fix `_detect_function_naming` (line 65):

Change:
```python
    def _detect_function_naming(self, out: list[Convention]) -> None:
        functions = self._graph.get_all_functions(max_results=500)
        if not functions:
            return

        names = [f.get("name", "") for f in functions]
        names = [n for n in names if n]
```

To:
```python
    def _detect_function_naming(self, out: list[Convention]) -> None:
        functions = self._graph.get_all_functions(max_results=5000)
        if not functions:
            return
        true_total = len(functions)

        # Sample first 500 for pattern detection (sufficient for statistics)
        names = [f.get("name", "") for f in functions[:500]]
        names = [n for n in names if n]
```

And update the reporting lines (around line 88-101) to use `true_total` instead of `total`:

Change `total = len(names)` to keep it for ratio computation, but use `true_total` in pattern strings:
```python
        total = len(names)
        if total == 0:
            return

        if len(snake) >= len(camel):
            pct = len(snake) / total
            out.append(Convention(
                category="naming",
                pattern=f"Functions use snake_case ({int(pct * 100)}% of {true_total} functions)",
                confidence=pct,
                examples=snake[:3],
            ))
        else:
            pct = len(camel) / total
            out.append(Convention(
                category="naming",
                pattern=f"Functions use camelCase ({int(pct * 100)}% of {true_total} functions)",
                confidence=pct,
                examples=camel[:3],
            ))
```

Fix `_detect_class_naming` (line 103-122) similarly:

Change line 104:
```python
        classes = self._graph.get_all_classes(max_results=500)
```
To:
```python
        classes = self._graph.get_all_classes(max_results=5000)
```

And fix PascalCase detection (line 111):

Change:
```python
        pascal = [n for n in names if re.match(r"^[A-Z][a-zA-Z0-9]*$", n)]
```
To:
```python
        pascal = [
            n for n in names
            if (s := n.lstrip("_")) and re.match(r"^[A-Z][a-zA-Z0-9]*$", s)
        ]
```

And use `true_total` in the pattern string:
```python
        true_total = len(names)
        ...
        pattern=f"Classes use PascalCase ({int(pct * 100)}% of {true_total} classes)",
```

Fix `detect_typing_conventions` (line 264):
Change `max_results=500` to `max_results=5000` and add `true_total`.

Fix `detect_async_patterns` (line 327):
Change `max_results=500` to `max_results=5000` and add `true_total`.

Fix `detect_docstring_conventions` (line 357):
Change `max_results=500` to `max_results=5000` and add `true_total`.

Fix `detect_import_conventions` (lines 443-466):
Add deduplication by (file, module) to count import statements, not imported names.

Change the import collection loop at lines 456-465:
```python
                        for imp in imports:
                            source = safe_str(safe_get_attr(imp, "source", ""))
                            if not source:
                                source = safe_str(imp)
                            all_imports.append({
                                "source": source,
                                "file": fpath,
                                "language": lang,
                            })
```

To:
```python
                        seen_stmts: set[tuple[str, str]] = set()
                        for imp in imports:
                            source = safe_str(safe_get_attr(imp, "source", ""))
                            if not source:
                                source = safe_str(imp)
                            # Deduplicate: graph-sitter may expose each imported
                            # name as a separate import object. We want to count
                            # import STATEMENTS, not individual names.
                            # Key on (file, module) to collapse
                            # "from X import a, b, c" into one entry.
                            module = source.replace("from ", "").split(" import ")[0].strip()
                            if not module:
                                module = source
                            stmt_key = (fpath, module)
                            if stmt_key in seen_stmts:
                                continue
                            seen_stmts.add(stmt_key)
                            all_imports.append({
                                "source": source,
                                "file": fpath,
                                "language": lang,
                            })
```

Note: the `seen_stmts` set should be PER-FILE (inside the `for f in codebase.files:` loop), so declare it right after the `if not imports: continue` check.

**Step 4: Run tests**

Run: `python -m pytest tests/test_conventions.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add grafyx/conventions.py tests/test_conventions.py
git commit -m "fix: use accurate counts in conventions, fix PascalCase for _ prefix, deduplicate imports"
```

---

### Task 4: Add Celery task dispatch detection (dynamic dispatch part 1)

**Files:**
- Modify: `grafyx/graph/_indexes.py:80-81,155-162`
- Test: `tests/test_celery_dispatch.py` (create)

**Step 1: Write the failing test**

Create `tests/test_celery_dispatch.py`:

```python
"""Tests for Celery task dispatch detection in caller index."""

import re
import pytest
from unittest.mock import MagicMock

from grafyx.utils import safe_get_attr, safe_str


def _make_func(name, filepath, source="", decorators=None, function_calls=None):
    func = MagicMock()
    func.name = name
    func.filepath = filepath
    func.source = source
    func.decorators = decorators or []
    func.function_calls = function_calls or []
    func.parameters = []
    return func


def _make_class(name, filepath, methods=None):
    cls = MagicMock()
    cls.name = name
    cls.filepath = filepath
    cls.methods = methods or []
    return cls


class TestCeleryDispatchDetection:
    """Verify that .delay() and .apply_async() calls create caller entries."""

    def test_delay_creates_caller_entry(self):
        """process_asset_task.delay() should register the calling function
        as a caller of process_asset_task."""
        from grafyx.graph._indexes import IndexBuilderMixin

        mixin = MagicMock(spec=IndexBuilderMixin)
        mixin._caller_index = {}
        mixin._class_method_names = {}
        mixin._codebases = {
            "python": MagicMock(
                functions=[
                    _make_func(
                        "process_asset_task", "/proj/tasks.py",
                        source="@shared_task\ndef process_asset_task(asset_id):\n    pass",
                        decorators=[MagicMock(__str__=lambda s: "@shared_task")],
                    ),
                    _make_func(
                        "upload_handler", "/proj/api.py",
                        source="def upload_handler(file):\n    process_asset_task.delay(file.id)\n",
                    ),
                ],
                classes=[],
            ),
        }
        mixin.translate_path = lambda p: p
        mixin._is_ignored_file_path = lambda p: False

        IndexBuilderMixin._augment_index_with_celery_tasks(mixin)

        assert "process_asset_task" in mixin._caller_index
        callers = mixin._caller_index["process_asset_task"]
        assert any(c["name"] == "upload_handler" for c in callers)

    def test_apply_async_creates_caller_entry(self):
        """process_asset_task.apply_async() should also register a caller."""
        from grafyx.graph._indexes import IndexBuilderMixin

        mixin = MagicMock(spec=IndexBuilderMixin)
        mixin._caller_index = {}
        mixin._class_method_names = {}
        mixin._codebases = {
            "python": MagicMock(
                functions=[
                    _make_func(
                        "my_task", "/proj/tasks.py",
                        source="@task\ndef my_task(x):\n    pass",
                        decorators=[MagicMock(__str__=lambda s: "@task")],
                    ),
                    _make_func(
                        "trigger", "/proj/views.py",
                        source="def trigger():\n    my_task.apply_async(args=[1])\n",
                    ),
                ],
                classes=[],
            ),
        }
        mixin.translate_path = lambda p: p
        mixin._is_ignored_file_path = lambda p: False

        IndexBuilderMixin._augment_index_with_celery_tasks(mixin)

        assert "my_task" in mixin._caller_index
        callers = mixin._caller_index["my_task"]
        assert any(c["name"] == "trigger" for c in callers)

    def test_non_task_delay_is_ignored(self):
        """Functions not decorated with @task/@shared_task should NOT have
        their .delay() calls tracked."""
        from grafyx.graph._indexes import IndexBuilderMixin

        mixin = MagicMock(spec=IndexBuilderMixin)
        mixin._caller_index = {}
        mixin._class_method_names = {}
        mixin._codebases = {
            "python": MagicMock(
                functions=[
                    _make_func(
                        "not_a_task", "/proj/utils.py",
                        source="def not_a_task(): pass",
                    ),
                    _make_func(
                        "caller_fn", "/proj/views.py",
                        source="def caller_fn():\n    not_a_task.delay()\n",
                    ),
                ],
                classes=[],
            ),
        }
        mixin.translate_path = lambda p: p
        mixin._is_ignored_file_path = lambda p: False

        IndexBuilderMixin._augment_index_with_celery_tasks(mixin)

        assert "not_a_task" not in mixin._caller_index
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_celery_dispatch.py -v`
Expected: FAIL (method doesn't exist yet)

**Step 3: Implement _augment_index_with_celery_tasks**

In `grafyx/graph/_indexes.py`, after `_augment_index_with_class_attr_types` (after the method ends), add:

```python
    # --- Pass 4: Celery Task Dispatch Detection ---

    def _augment_index_with_celery_tasks(self) -> None:
        """Detect Celery task invocations via .delay() and .apply_async() (Pass 4).

        Celery tasks are called through a dynamic dispatch registry:
            my_task.delay(args)        -> equivalent to calling my_task()
            my_task.apply_async(args)  -> equivalent to calling my_task()
            my_task.s(args)            -> creates a signature (lazy invocation)
            my_task.si(args)           -> creates an immutable signature

        Algorithm:
            1. Collect all @task/@shared_task/@periodic_task decorated functions.
            2. Build a regex matching task_name.(delay|apply_async|s|si)(
            3. Scan all function/method sources for matches.
            4. Add synthetic caller entries linking the caller to the task.
        """
        task_decorators = {"task", "shared_task", "periodic_task"}
        known_tasks: dict[str, str] = {}  # task_name -> file_path

        for _lang, codebase in self._codebases.items():
            try:
                for func in codebase.functions:
                    f_name = safe_get_attr(func, "name", "")
                    if not f_name:
                        continue
                    f_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))
                    if self._is_ignored_file_path(f_file):
                        continue
                    decorators = safe_get_attr(func, "decorators", [])
                    for d in decorators:
                        d_str = safe_str(d).strip("@").split("(")[0].split(".")[-1]
                        if d_str in task_decorators:
                            known_tasks[f_name] = f_file
                            break
            except Exception:
                continue

        if not known_tasks:
            return

        celery_call_re = re.compile(
            r'\b(' + '|'.join(re.escape(t) for t in known_tasks)
            + r')\.(delay|apply_async|s|si)\s*\('
        )

        additions = 0
        for _lang, codebase in self._codebases.items():
            try:
                for func in codebase.functions:
                    additions += self._scan_celery_calls(
                        func, None, celery_call_re, known_tasks,
                    )
                for cls in codebase.classes:
                    cls_name_str = safe_get_attr(cls, "name", "")
                    cls_file = self.translate_path(str(safe_get_attr(cls, "filepath", "")))
                    if self._is_ignored_file_path(cls_file):
                        continue
                    for method in safe_get_attr(cls, "methods", []):
                        additions += self._scan_celery_calls(
                            method, cls_name_str, celery_call_re, known_tasks,
                        )
            except Exception as e:
                logger.debug("Error scanning Celery tasks for %s: %s", _lang, e)
        if additions:
            logger.debug("Celery task scan added %d caller entries", additions)

    def _scan_celery_calls(
        self,
        func: Any,
        caller_class: str | None,
        celery_call_re: re.Pattern,
        known_tasks: dict[str, str],
    ) -> int:
        """Scan a single function's source for Celery dispatch patterns."""
        caller_name = safe_get_attr(func, "name", "")
        if not caller_name:
            return 0
        caller_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))
        if self._is_ignored_file_path(caller_file):
            return 0
        source = safe_str(safe_get_attr(func, "source", ""))
        if not source:
            return 0

        additions = 0
        for match in celery_call_re.finditer(source):
            task_name = match.group(1)
            if task_name == caller_name:
                continue
            if task_name not in self._caller_index:
                self._caller_index[task_name] = []
            entry: dict[str, Any] = {
                "name": caller_name,
                "file": caller_file,
                "_trusted": True,
            }
            if caller_class:
                entry["class"] = caller_class
            if not any(
                e["name"] == caller_name and e["file"] == caller_file
                for e in self._caller_index[task_name]
            ):
                self._caller_index[task_name].append(entry)
                additions += 1
        return additions
```

And wire it into `_build_caller_index` at line 162, after `_augment_index_with_class_attr_types()`:

```python
        # Pass 4: detect Celery task dispatch (.delay(), .apply_async())
        self._augment_index_with_celery_tasks()
```

Also update the docstring at line 80-81 to say "Pass 1 of 5" and the module docstring at line 26 to mention 5 passes.

**Step 4: Run tests**

Run: `python -m pytest tests/test_celery_dispatch.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add grafyx/graph/_indexes.py tests/test_celery_dispatch.py
git commit -m "feat: detect Celery .delay()/.apply_async() as caller entries (Pass 4)"
```

---

### Task 5: Add unique-method heuristic for untyped instance calls (dynamic dispatch part 2)

**Files:**
- Modify: `grafyx/graph/_indexes.py` (add Pass 5 after Pass 4)
- Test: `tests/test_unique_method_dispatch.py` (create)

**Step 1: Write the failing test**

Create `tests/test_unique_method_dispatch.py`:

```python
"""Tests for unique-method heuristic in caller index."""

import pytest
from unittest.mock import MagicMock

from grafyx.utils import safe_get_attr, safe_str


def _make_func(name, filepath, source="", function_calls=None):
    func = MagicMock()
    func.name = name
    func.filepath = filepath
    func.source = source
    func.function_calls = function_calls or []
    func.decorators = []
    func.parameters = []
    return func


def _make_class(name, filepath, methods=None):
    cls = MagicMock()
    cls.name = name
    cls.filepath = filepath
    cls.methods = methods or []
    return cls


class TestUniqueMethodDispatch:
    """Verify unique-method heuristic creates caller entries for untyped vars."""

    def test_unique_method_creates_caller(self):
        """If send_message exists in exactly 1 class, var.send_message()
        should register as a caller of that class's method."""
        from grafyx.graph._indexes import IndexBuilderMixin

        mixin = MagicMock(spec=IndexBuilderMixin)
        mixin._caller_index = {}
        mixin._class_method_names = {
            "EddyService": {"send_message", "confirm_action", "__init__"},
        }
        mixin._codebases = {
            "python": MagicMock(
                functions=[
                    _make_func(
                        "handle_request", "/proj/api.py",
                        source="def handle_request(service):\n    service.send_message('hello')\n",
                    ),
                ],
                classes=[],
            ),
        }
        mixin.translate_path = lambda p: p
        mixin._is_ignored_file_path = lambda p: False

        IndexBuilderMixin._augment_index_with_unique_method_calls(mixin)

        assert "send_message" in mixin._caller_index
        callers = mixin._caller_index["send_message"]
        assert any(c["name"] == "handle_request" for c in callers)

    def test_non_unique_method_is_skipped(self):
        """If 'execute' exists in 2+ classes, var.execute() should NOT
        create a caller entry (ambiguous target)."""
        from grafyx.graph._indexes import IndexBuilderMixin

        mixin = MagicMock(spec=IndexBuilderMixin)
        mixin._caller_index = {}
        mixin._class_method_names = {
            "ToolExecutor": {"execute", "validate"},
            "QueryExecutor": {"execute", "prepare"},
        }
        mixin._codebases = {
            "python": MagicMock(
                functions=[
                    _make_func(
                        "run_tool", "/proj/runner.py",
                        source="def run_tool(executor):\n    executor.execute(args)\n",
                    ),
                ],
                classes=[],
            ),
        }
        mixin.translate_path = lambda p: p
        mixin._is_ignored_file_path = lambda p: False

        IndexBuilderMixin._augment_index_with_unique_method_calls(mixin)

        # execute should NOT be in caller_index (ambiguous)
        assert "execute" not in mixin._caller_index

    def test_self_calls_are_skipped(self):
        """self.send_message() inside EddyService methods should NOT create
        a duplicate caller entry (it's already handled by Pass 1)."""
        from grafyx.graph._indexes import IndexBuilderMixin

        method = _make_func(
            "process", "/proj/service.py",
            source="def process(self):\n    self.send_message('done')\n",
        )
        mixin = MagicMock(spec=IndexBuilderMixin)
        mixin._caller_index = {}
        mixin._class_method_names = {
            "EddyService": {"send_message", "process"},
        }
        mixin._codebases = {
            "python": MagicMock(
                functions=[],
                classes=[
                    _make_class("EddyService", "/proj/service.py", methods=[method]),
                ],
            ),
        }
        mixin.translate_path = lambda p: p
        mixin._is_ignored_file_path = lambda p: False

        IndexBuilderMixin._augment_index_with_unique_method_calls(mixin)

        # send_message should NOT get EddyService.process as caller
        # (self.method() within the same class is a self-call)
        assert "send_message" not in mixin._caller_index
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_unique_method_dispatch.py -v`
Expected: FAIL

**Step 3: Implement _augment_index_with_unique_method_calls**

In `grafyx/graph/_indexes.py`, after `_augment_index_with_celery_tasks` and its helper, add:

```python
    # --- Pass 5: Unique-Method Heuristic for Untyped Instance Calls ---

    def _augment_index_with_unique_method_calls(self) -> None:
        """Infer method call targets from unique method names (Pass 5).

        When a function calls ``var.method()`` and ``method`` is defined in
        exactly ONE class across the entire project, we can safely assume the
        call targets that class's method. This is the same logic as
        ``get_callers()`` Level 4 but applied proactively during index building.

        Safety guards:
            - Only methods that exist in exactly 1 class are eligible.
            - Dunder methods (__init__, etc.) are always skipped.
            - Common variable names (self, cls, os, etc.) are skipped.
            - Self-calls (caller's class == target class) are skipped.
        """
        # Build unique method -> class mapping
        method_to_classes: dict[str, list[str]] = {}
        for cls_name, methods in self._class_method_names.items():
            for m in methods:
                if m.startswith("__"):
                    continue
                if m not in method_to_classes:
                    method_to_classes[m] = []
                method_to_classes[m].append(cls_name)

        # Keep only truly unique methods (defined in exactly 1 class)
        unique_methods: dict[str, str] = {
            m: classes[0]
            for m, classes in method_to_classes.items()
            if len(classes) == 1
        }
        if not unique_methods:
            return

        method_call_re = re.compile(r'\b(\w+)\.(\w+)\s*\(')
        skip_vars = {
            "self", "cls", "super", "os", "sys", "re", "json", "math",
            "logging", "logger", "log", "print", "str", "int", "float",
            "list", "dict", "set", "tuple", "type", "object", "path",
            "Path", "datetime", "date", "time", "uuid",
        }

        additions = 0
        for _lang, codebase in self._codebases.items():
            try:
                for func in codebase.functions:
                    additions += self._scan_unique_method_calls(
                        func, None, unique_methods, method_call_re, skip_vars,
                    )
                for cls in codebase.classes:
                    cls_name = safe_get_attr(cls, "name", "")
                    cls_file = self.translate_path(
                        str(safe_get_attr(cls, "filepath", ""))
                    )
                    if self._is_ignored_file_path(cls_file):
                        continue
                    for method in safe_get_attr(cls, "methods", []):
                        additions += self._scan_unique_method_calls(
                            method, cls_name, unique_methods,
                            method_call_re, skip_vars,
                        )
            except Exception as e:
                logger.debug("Error scanning unique method calls for %s: %s", _lang, e)
        if additions:
            logger.debug("Unique method call scan added %d caller entries", additions)

    def _scan_unique_method_calls(
        self,
        func: Any,
        caller_class: str | None,
        unique_methods: dict[str, str],
        method_call_re: re.Pattern,
        skip_vars: set[str],
    ) -> int:
        """Scan a single function for calls to unique methods on untyped vars."""
        caller_name = safe_get_attr(func, "name", "")
        if not caller_name:
            return 0
        caller_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))
        if self._is_ignored_file_path(caller_file):
            return 0
        source = safe_str(safe_get_attr(func, "source", ""))
        if not source:
            return 0

        additions = 0
        for m in method_call_re.finditer(source):
            var_name, method_name = m.group(1), m.group(2)
            if var_name in skip_vars:
                continue
            target_cls = unique_methods.get(method_name)
            if not target_cls:
                continue
            # Skip self-calls: caller's own class IS the target class
            if caller_class == target_cls:
                continue
            if method_name not in self._caller_index:
                self._caller_index[method_name] = []
            entry: dict[str, Any] = {
                "name": caller_name,
                "file": caller_file,
                "_trusted": True,
            }
            if caller_class:
                entry["class"] = caller_class
            if not any(
                e["name"] == caller_name and e["file"] == caller_file
                for e in self._caller_index[method_name]
            ):
                self._caller_index[method_name].append(entry)
                additions += 1
        return additions
```

Wire it into `_build_caller_index` at line 163 (after Celery pass):

```python
        # Pass 5: infer method targets from unique method names
        self._augment_index_with_unique_method_calls()
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_unique_method_dispatch.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add grafyx/graph/_indexes.py tests/test_unique_method_dispatch.py
git commit -m "feat: infer untyped instance method call targets via unique-method heuristic (Pass 5)"
```

---

### Task 6: Fix module dependency misattribution

**Files:**
- Modify: `grafyx/graph/_indexes.py:596-780` (build top-level forward index)
- Modify: `grafyx/graph/_analysis.py:733-745` (use top-level index)
- Modify: `grafyx/graph/core.py:93` (init new attribute)
- Test: `tests/test_module_deps.py` (create)

**Step 1: Write the failing test**

Create `tests/test_module_deps.py`:

```python
"""Tests for module dependency accuracy (lazy import filtering)."""

import threading
import pytest
from unittest.mock import MagicMock, PropertyMock

from grafyx.utils import safe_get_attr, safe_str


class TestModuleDependencyLazyImports:
    """Verify lazy imports inside function bodies don't create module edges."""

    def test_lazy_import_excluded_from_module_deps(self):
        """A lazy import inside a function body should NOT create a
        module-level dependency edge."""
        from grafyx.graph._analysis import AnalysisMixin

        mixin = MagicMock(spec=AnalysisMixin)
        mixin._lock = threading.RLock()
        mixin.original_path = "/proj"

        # File structure: services/worker.py imports api/daily_rooms.py
        # BUT the import is inside a function body (lazy import).
        mixin.get_all_files.return_value = [
            {"path": "/proj/services/worker.py"},
            {"path": "/proj/api/daily_rooms.py"},
        ]
        # Full forward index includes the lazy import
        mixin._forward_import_index = {
            "/proj/services/worker.py": ["/proj/api/daily_rooms.py"],
        }
        # Top-level-only index does NOT include the lazy import
        mixin._top_level_forward_import_index = {}

        result = AnalysisMixin.get_module_dependencies(mixin, depth=1)

        # No edge from services -> api (the import is lazy/function-body only)
        edges = result.get("edges", [])
        services_to_api = [e for e in edges if e["from"] == "services" and e["to"] == "api"]
        assert len(services_to_api) == 0

    def test_top_level_import_creates_module_edge(self):
        """A top-level import SHOULD create a module-level dependency edge."""
        from grafyx.graph._analysis import AnalysisMixin

        mixin = MagicMock(spec=AnalysisMixin)
        mixin._lock = threading.RLock()
        mixin.original_path = "/proj"
        mixin.get_all_files.return_value = [
            {"path": "/proj/services/auth.py"},
            {"path": "/proj/models/user.py"},
        ]
        mixin._forward_import_index = {
            "/proj/services/auth.py": ["/proj/models/user.py"],
        }
        mixin._top_level_forward_import_index = {
            "/proj/services/auth.py": ["/proj/models/user.py"],
        }

        result = AnalysisMixin.get_module_dependencies(mixin, depth=1)

        edges = result.get("edges", [])
        services_to_models = [e for e in edges if e["from"] == "services" and e["to"] == "models"]
        assert len(services_to_models) == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_module_deps.py -v`
Expected: FAIL

**Step 3: Add _top_level_forward_import_index attribute in core.py**

In `grafyx/graph/core.py`, after line 93 (`_forward_import_index`), add:

```python
        self._top_level_forward_import_index: dict[str, list[str]] = {}  # source -> [imported] (top-level only)
```

**Step 4: Build the top-level-only forward index in _indexes.py**

In `grafyx/graph/_indexes.py`, inside `_build_import_index()`, right before Phase 2 (around line 674, before `for lang, codebase in self._codebases.items():`), build per-file function line ranges:

```python
            # Build function line ranges per file for scope detection.
            # Used to distinguish top-level imports from function-body
            # (lazy) imports. Lazy imports should not create module-level
            # dependency edges.
            file_func_ranges: dict[str, list[tuple[int, int]]] = {}
            for _lang2, _cb2 in self._codebases.items():
                try:
                    for _func in _cb2.functions:
                        _fp = self.translate_path(
                            str(safe_get_attr(_func, "filepath", ""))
                        ).replace("\\", "/")
                        if not _fp:
                            continue
                        _start = self._extract_line(_func)
                        _src = safe_str(safe_get_attr(_func, "source", ""))
                        if _start and _src:
                            _end = _start + _src.count("\n")
                            if _fp not in file_func_ranges:
                                file_func_ranges[_fp] = []
                            file_func_ranges[_fp].append((_start, _end))
                    for _cls in _cb2.classes:
                        for _meth in safe_get_attr(_cls, "methods", []):
                            _fp = self.translate_path(
                                str(safe_get_attr(_meth, "filepath", ""))
                            ).replace("\\", "/")
                            if not _fp:
                                continue
                            _start = self._extract_line(_meth)
                            _src = safe_str(safe_get_attr(_meth, "source", ""))
                            if _start and _src:
                                _end = _start + _src.count("\n")
                                if _fp not in file_func_ranges:
                                    file_func_ranges[_fp] = []
                                file_func_ranges[_fp].append((_start, _end))
                except Exception:
                    continue
```

Then, also initialize `top_level_forward` alongside `forward` at line 617:

```python
        top_level_forward: dict[str, list[str]] = {}
```

Inside the import processing loop, after the line `forward[fpath].append(target)` (around line 765), add the scope check:

```python
                                # Track whether this import is at top-level
                                # or inside a function body (lazy import).
                                imp_line = self._extract_line(imp)
                                is_top_level = True
                                if imp_line and fpath in file_func_ranges:
                                    for _fstart, _fend in file_func_ranges[fpath]:
                                        if _fstart <= imp_line <= _fend:
                                            is_top_level = False
                                            break
                                if is_top_level:
                                    if fpath not in top_level_forward:
                                        top_level_forward[fpath] = []
                                    if target not in top_level_forward[fpath]:
                                        top_level_forward[fpath].append(target)
```

At the end of `_build_import_index`, alongside the existing assignments (line 777), add:

```python
            self._top_level_forward_import_index = top_level_forward
```

And update the logger.debug line to include the new index size.

**Step 5: Use top-level index in get_module_dependencies**

In `grafyx/graph/_analysis.py`, change line 735:

Change:
```python
            for source_file, targets in self._forward_import_index.items():
```

To:
```python
            # Use top-level-only forward index to avoid ghost edges from
            # lazy imports inside function bodies (e.g., a function in
            # services/worker.py that does `from api.voice import X` would
            # create a false services->api module edge without this).
            _fwd_idx = getattr(self, '_top_level_forward_import_index', None)
            if not _fwd_idx:
                _fwd_idx = self._forward_import_index
            for source_file, targets in _fwd_idx.items():
```

Also update the debug section (line 801) to use the same index:

Change:
```python
                for source_file, targets in self._forward_import_index.items():
```

To:
```python
                for source_file, targets in _fwd_idx.items():
```

**Step 6: Run tests**

Run: `python -m pytest tests/test_module_deps.py -v`
Expected: PASS

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 7: Commit**

```bash
git add grafyx/graph/core.py grafyx/graph/_indexes.py grafyx/graph/_analysis.py tests/test_module_deps.py
git commit -m "fix: exclude function-body lazy imports from module dependency edges"
```

---

## Task Dependency Summary

```
Task 1 (cross_file_usages) ──── independent
Task 2 (search recall)     ──── independent
Task 3 (conventions)       ──── independent
Task 4 (Celery dispatch)   ──── depends on nothing, but Task 5 depends on it
Task 5 (unique methods)    ──── after Task 4 (same file, sequential edits)
Task 6 (module deps)       ──── after Task 5 (same file, sequential edits)
```

Tasks 1, 2, 3 can be parallelized. Tasks 4→5→6 must be sequential.
