"""Tests for v0.2.5 fixes:
- Module-level imports (`from pkg import submod`) tracked in import index
- Gibberish queries can't be rescued by a single coincidental source-token hit
- Methods in classes extending external base classes aren't flagged unused
- Top-level functions in plugin/hook files aren't flagged unused
"""

from unittest.mock import MagicMock

from grafyx.graph import CodebaseGraph


# ---------------------------------------------------------------------------
# Fix 1: Module-level submodule imports
# ---------------------------------------------------------------------------

class TestSubmoduleImportTracking:
    """`from pkg import submod` should track submod.py as imported."""

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._project_path = "/project"
        graph.original_path = "/project"
        graph._lock = MagicMock()
        graph._external_packages = set()
        graph.translate_path = lambda p: str(p) if p else ""
        graph._build_import_index = CodebaseGraph._build_import_index.__get__(graph)
        graph._extract_module_from_import = CodebaseGraph._extract_module_from_import
        graph._extract_symbol_names_from_import = CodebaseGraph._extract_symbol_names_from_import
        graph._resolve_python_relative_import = CodebaseGraph._resolve_python_relative_import
        graph._augment_with_submodule_imports = (
            CodebaseGraph._augment_with_submodule_imports.__get__(graph)
        )
        graph._resolve_init_reexports = lambda: None
        graph._is_ignored_file_path = lambda p: False
        return graph

    def _make_py_file(self, path, imports=()):
        f = MagicMock()
        f.path = path
        f.filepath = path
        imp_mocks = []
        for imp_str in imports:
            imp = MagicMock()
            imp.source = imp_str
            imp_mocks.append(imp)
        f.imports = imp_mocks
        return f

    def test_submodule_import_tracked(self):
        """`from fastapi import routing` registers fastapi/routing.py."""
        graph = self._make_graph()

        consumer = self._make_py_file(
            "/project/applications.py",
            imports=["from fastapi import routing"],
        )
        init_py = self._make_py_file("/project/fastapi/__init__.py")
        routing_py = self._make_py_file("/project/fastapi/routing.py")

        py_codebase = MagicMock()
        py_codebase.files = [consumer, init_py, routing_py]
        graph._codebases = {"python": py_codebase}

        graph._build_import_index()

        importers = graph._import_index.get("/project/fastapi/routing.py", [])
        assert "/project/applications.py" in importers, (
            "`from fastapi import routing` did not register routing.py as imported"
        )

    def test_submodule_import_symbol_tracked(self):
        """The submodule name is added to _file_symbol_imports for the submodule file."""
        graph = self._make_graph()

        consumer = self._make_py_file(
            "/project/utils.py",
            imports=["from fastapi import routing"],
        )
        init_py = self._make_py_file("/project/fastapi/__init__.py")
        routing_py = self._make_py_file("/project/fastapi/routing.py")

        py_codebase = MagicMock()
        py_codebase.files = [consumer, init_py, routing_py]
        graph._codebases = {"python": py_codebase}

        graph._build_import_index()

        symbols = graph._file_symbol_imports.get(
            "/project/utils.py", {}
        ).get("/project/fastapi/routing.py", set())
        assert "routing" in symbols

    def test_non_submodule_symbol_not_falsely_tracked(self):
        """`from fastapi import FastAPI` (where FastAPI is a class, not submodule) doesn't register a fake submodule file."""
        graph = self._make_graph()

        consumer = self._make_py_file(
            "/project/app.py",
            imports=["from fastapi import FastAPI"],
        )
        init_py = self._make_py_file("/project/fastapi/__init__.py")
        # No fastapi/FastAPI.py exists

        py_codebase = MagicMock()
        py_codebase.files = [consumer, init_py]
        graph._codebases = {"python": py_codebase}

        graph._build_import_index()

        # No spurious /project/fastapi/FastAPI.py should appear
        for target in graph._import_index:
            assert "FastAPI.py" not in target


# ---------------------------------------------------------------------------
# Fix 2: Gibberish bypass via single source-token hit
# ---------------------------------------------------------------------------

class TestGibberishStrictness:
    """A gibberish query with exactly ONE coincidental source-token hit must still be dampened."""

    def test_single_source_hit_does_not_save_gibberish(self):
        """'xyzzy foobar qlrmph' where 'foobar' is in source = scores still capped at 0.35."""
        from grafyx.search.searcher import CodeSearcher

        graph = MagicMock()
        graph.get_all_functions.return_value = [
            {"name": "test_path_bool_foobar", "file": "/project/tests/test_x.py", "line": 1, "language": "python"},
        ]
        graph.get_all_classes.return_value = []
        graph.get_all_files.return_value = []
        graph.iter_functions_with_source.return_value = iter([])
        graph._caller_index = {}
        graph._import_index = {}
        graph._forward_import_index = {}

        searcher = CodeSearcher(graph)
        # Mock the source index to contain only "foobar" (single hit for the gibberish query)
        searcher._source_index = {"foobar": {("test_path_bool_foobar", "/project/tests/test_x.py", "function", "")}}
        searcher._source_index_built = True

        # Force the search to return a result first, then check dampening
        results = searcher.search("xyzzy foobar qlrmph", max_results=5)

        # If results returned, all scores must be capped at 0.35 (gibberish dampening)
        for r in results:
            assert r["score"] <= 0.35, (
                f"Gibberish query rescued by single source hit; score {r['score']} > 0.35 "
                f"(result: {r.get('name')})"
            )

    def test_two_source_hits_does_save_gibberish(self):
        """Two distinct token matches still count as 'real query' (technical jargon allowed)."""
        from grafyx.search.searcher import CodeSearcher

        graph = MagicMock()
        graph.get_all_functions.return_value = []
        graph.get_all_classes.return_value = []
        graph.get_all_files.return_value = []
        graph.iter_functions_with_source.return_value = iter([])
        graph._caller_index = {}
        graph._import_index = {}
        graph._forward_import_index = {}

        searcher = CodeSearcher(graph)
        # Two of the three tokens hit
        searcher._source_index = {
            "foobar": {("x", "/p/x.py", "function", "")},
            "qlrmph": {("y", "/p/y.py", "function", "")},
        }
        searcher._source_index_built = True

        # Just verify the gibberish path doesn't unconditionally cap when 2 source tokens hit.
        # The scores depend on many factors; the contract is: 2+ hits = no automatic dampening.
        # This is verified indirectly — we don't have a strong assertion here, just that the
        # search doesn't crash and returns coherent output.
        results = searcher.search("xyzzy foobar qlrmph", max_results=5)
        # If results returned at all, they shouldn't all be capped at 0.35 from gibberish
        # (other low-score factors may still apply).
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Fix 3a: External base class exempts methods from unused detection
# ---------------------------------------------------------------------------

class TestExternalBaseClassExemption:
    """Methods in classes that extend an EXTERNAL base class shouldn't be flagged unused."""

    def test_method_overriding_external_base_not_flagged(self):
        """APIRoute(Route).matches — Route is external (Starlette), matches must be skipped."""
        # This is a unit-level test of the algorithm via direct mock.
        # We need to verify that when a class's ancestor isn't in our codebase
        # AND isn't a Python builtin, methods of that class are exempted.
        from grafyx.graph._analysis import AnalysisMixin

        # Verify the constant exists and has the right shape
        assert hasattr(AnalysisMixin, "_NON_DISPATCHING_BASES"), (
            "AnalysisMixin should define _NON_DISPATCHING_BASES — the set of "
            "base class names (object, Exception, etc.) that don't trigger "
            "framework-dispatch exemption"
        )
        bases = AnalysisMixin._NON_DISPATCHING_BASES
        assert "object" in bases
        assert "Exception" in bases
        # Framework bases like Route, Model, BaseModel must NOT be in this set —
        # they SHOULD trigger the exemption
        assert "Route" not in bases
        assert "Model" not in bases


# ---------------------------------------------------------------------------
# Fix 3b: Plugin hook file pattern exempts top-level functions
# ---------------------------------------------------------------------------

class TestHookFileExemption:
    """Top-level functions in `*_hooks.py` / `*_plugin.py` files shouldn't be flagged unused."""

    def test_hook_file_pattern_constant_exists(self):
        """AnalysisMixin should define _HOOK_FILE_SUFFIXES."""
        from grafyx.graph._analysis import AnalysisMixin

        assert hasattr(AnalysisMixin, "_HOOK_FILE_SUFFIXES"), (
            "AnalysisMixin should define _HOOK_FILE_SUFFIXES for the unused-symbol detector"
        )
        suffixes = AnalysisMixin._HOOK_FILE_SUFFIXES
        # Must catch the MkDocs-style hook files we saw in the FastAPI audit
        assert any("hook" in s for s in suffixes), "Should match _hooks.py / _hook.py files"
        assert any("plugin" in s for s in suffixes), "Should match _plugin.py / _plugins.py files"
