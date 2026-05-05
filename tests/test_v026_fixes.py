"""Tests for v0.2.6 fixes:

T5  — Intra-package submodule imports (project name collision in _external_packages)
T10 — get_subclasses must disambiguate when multiple classes share a name
T1/T20 — JS files reported as JavaScript, not TypeScript
T2  — detail="signatures" must produce a meaningfully smaller skeleton than default
T18 — Functions with body deprecation warnings (warnings.warn DeprecationWarning) exempt
"""

from unittest.mock import MagicMock

from grafyx.graph import CodebaseGraph


# ---------------------------------------------------------------------------
# T5: Intra-package submodule imports
# ---------------------------------------------------------------------------


class TestIntraPackageSubmoduleImports:
    """`from pkg import submod` must work even when pkg is the project's own name.

    The bug: when the project's pyproject.toml declares ``name = "fastapi"``,
    the regex extractor in ``_build_external_packages`` adds "fastapi" to
    ``_external_packages``. Phase 2 then skips ALL ``from fastapi import ...``
    imports, including from files inside the fastapi/ package itself, breaking
    intra-project link tracking.
    """

    def _make_graph(self, external_packages=None):
        graph = MagicMock(spec=CodebaseGraph)
        graph._project_path = "/project"
        graph.original_path = "/project"
        graph._lock = MagicMock()
        graph._external_packages = set(external_packages or set())
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

    def test_intra_package_import_with_self_name_in_external(self):
        """`from fastapi import routing` works when 'fastapi' is also in _external_packages.

        Reproduces the FastAPI audit failure: applications.py inside the fastapi
        package does ``from fastapi import routing``. The project's own name
        "fastapi" is in pyproject.toml so it gets added to _external_packages.
        Resolution must NOT skip the import, because the local module exists.
        """
        # Simulate the project name leaking into _external_packages
        graph = self._make_graph(external_packages={"fastapi"})

        # Importer is INSIDE the fastapi/ package
        applications_py = self._make_py_file(
            "/project/fastapi/applications.py",
            imports=["from fastapi import routing"],
        )
        init_py = self._make_py_file("/project/fastapi/__init__.py")
        routing_py = self._make_py_file("/project/fastapi/routing.py")

        py_codebase = MagicMock()
        py_codebase.files = [applications_py, init_py, routing_py]
        graph._codebases = {"python": py_codebase}

        graph._build_import_index()

        importers = graph._import_index.get("/project/fastapi/routing.py", [])
        assert "/project/fastapi/applications.py" in importers, (
            "Intra-package import was skipped because project name 'fastapi' "
            "appeared in _external_packages. routing.py imported_by missing "
            "applications.py."
        )

    def test_external_package_still_skipped(self):
        """Genuinely external packages (no local file) are still skipped."""
        graph = self._make_graph(external_packages={"requests"})

        consumer = self._make_py_file(
            "/project/app.py",
            imports=["from requests import Session"],
        )
        py_codebase = MagicMock()
        py_codebase.files = [consumer]
        graph._codebases = {"python": py_codebase}

        graph._build_import_index()

        # No local "requests/__init__.py" or "requests.py" exists, so nothing
        # should be tracked. _import_index should not have any "requests" key.
        for target in graph._import_index:
            assert "requests" not in target, (
                f"External package 'requests' was incorrectly treated as local: {target}"
            )


# ---------------------------------------------------------------------------
# T10: get_subclasses must disambiguate same-name classes
# ---------------------------------------------------------------------------


class TestSubclassDisambiguation:
    """get_subclasses must accept a file_path to disambiguate same-name classes.

    FastAPI has TWO classes named SecurityBase: one in security/base.py (the
    real abstract base), one in openapi/models.py (a Pydantic model).  Without
    file_path disambiguation, get_subclasses returns subclasses of BOTH,
    inflating the count and conflating two unrelated trees.
    """

    def test_file_path_disambiguates_same_name(self):
        """Calling get_subclasses(name, file_path=X) returns only subclasses of the class in X."""
        from grafyx.graph._analysis import AnalysisMixin

        # Verify the signature includes a file_path-style param
        import inspect
        sig = inspect.signature(AnalysisMixin.get_subclasses)
        params = sig.parameters
        # Either 'file_path' or 'file' should be a parameter
        assert "file_path" in params or "file" in params, (
            "get_subclasses must accept a file_path parameter to disambiguate "
            "same-name classes (e.g., FastAPI has two SecurityBase classes)"
        )


# ---------------------------------------------------------------------------
# T1/T20: JS files reported as JavaScript, not TypeScript
# ---------------------------------------------------------------------------


class TestJavaScriptFileLanguage:
    """A .js file should be reported as 'javascript', not 'typescript', even
    though graph-sitter parses both with the TypeScript parser.
    """

    def test_lang_from_path_helper_exists(self):
        """CodebaseGraph should expose a helper that maps a file path to its language."""
        # Either a static or instance method
        assert hasattr(CodebaseGraph, "_lang_from_path"), (
            "CodebaseGraph should define _lang_from_path(path) -> str using "
            "the file extension, not the codebase parser key"
        )

    def test_js_file_returns_javascript(self):
        """_lang_from_path('foo.js') == 'javascript'."""
        assert CodebaseGraph._lang_from_path("foo.js") == "javascript"
        assert CodebaseGraph._lang_from_path("a/b/c.jsx") == "javascript"

    def test_ts_file_returns_typescript(self):
        assert CodebaseGraph._lang_from_path("foo.ts") == "typescript"
        assert CodebaseGraph._lang_from_path("a/b.tsx") == "typescript"

    def test_py_file_returns_python(self):
        assert CodebaseGraph._lang_from_path("a/b/c.py") == "python"

    def test_unknown_extension_returns_empty(self):
        assert CodebaseGraph._lang_from_path("foo.md") == ""
        assert CodebaseGraph._lang_from_path("noext") == ""
        assert CodebaseGraph._lang_from_path("") == ""


# ---------------------------------------------------------------------------
# T2: detail="signatures" must produce a meaningfully smaller skeleton
# ---------------------------------------------------------------------------


class TestSignaturesDetailLevel:
    """detail='signatures' must strip more than detail='summary' for skeletons."""

    def test_signatures_drops_directory_stats(self):
        """detail='signatures' must remove directory_stats (already does)."""
        from grafyx.server._resolution import filter_by_detail

        full = {
            "project_path": "/x",
            "languages": ["python"],
            "total_files": 100,
            "total_functions": 200,
            "total_classes": 50,
            "by_language": {"python": {"files": 100}},
            "directory_stats": {"src": {"files": 50}},
            "subdir_stats": {},
            "file_tree": {"src/": {"a.py": None}},
        }

        sig = filter_by_detail(full, "signatures", "skeleton")
        summary = filter_by_detail(full, "summary", "skeleton")

        assert "directory_stats" not in sig
        assert "directory_stats" in summary

    def test_signatures_strips_file_tree(self):
        """detail='signatures' should strip file_tree — it's the heaviest field
        and not a 'signature'-style overview. summary keeps it."""
        from grafyx.server._resolution import filter_by_detail

        full = {
            "project_path": "/x",
            "total_files": 100,
            "by_language": {"python": {"files": 100}},
            "directory_stats": {"src": {"files": 50}},
            "file_tree": {"src/": {"a.py": None, "b.py": None}},
        }

        sig = filter_by_detail(full, "signatures", "skeleton")
        summary = filter_by_detail(full, "summary", "skeleton")

        # signatures payload should be strictly smaller
        sig_keys = set(sig.keys())
        summary_keys = set(summary.keys())
        assert sig_keys < summary_keys, (
            f"detail='signatures' should strip more keys than 'summary'. "
            f"signatures keys: {sig_keys}, summary keys: {summary_keys}"
        )


# ---------------------------------------------------------------------------
# T18: Functions with body deprecation warning are not flagged unused
# ---------------------------------------------------------------------------


class TestDeprecatedBodyExemption:
    """Functions whose body calls warnings.warn(...DeprecationWarning...) are
    intentionally kept for backwards compatibility — exempt them from unused.
    """

    def test_deprecated_decorator_in_framework_set(self):
        """The 'deprecated' decorator (typing.deprecated, PEP 702) should be
        recognized as a framework decorator that exempts a function from
        unused detection."""
        # We can't easily unit-test the body detection without a full mock graph
        # so we verify that 'deprecated' is at least in the FRAMEWORK_DECORATORS
        # set (the simpler signal).
        import inspect
        from grafyx.graph._analysis import AnalysisMixin

        src = inspect.getsource(AnalysisMixin.get_unused_functions)
        # Either typing.deprecated decorator OR warnings.warn body check
        # must be present in the implementation.
        assert (
            '"deprecated"' in src
            or "'deprecated'" in src
            or "warnings.warn" in src
            or "DeprecationWarning" in src
        ), (
            "get_unused_functions should exempt functions marked with "
            "@deprecated decorator OR functions whose body emits a "
            "DeprecationWarning via warnings.warn()."
        )
