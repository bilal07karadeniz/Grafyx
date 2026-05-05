"""Tests for v0.2.3 audit-driven fixes.

Each test pins behavior reported as a bug in the v0.2.2 audit so the
regressions don't return.

Covered fixes:
    - P0.1: get_dependency_graph.depended_on_by uses caller index for
      functions (was always empty for non-class symbols)
    - P0.2: __getattr__ lazy loader patterns beyond the dict-with-dot
      form (inline ``from .x import y``, no-leading-dot dicts,
      same-name submodule fallback when __getattr__ is defined)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from grafyx.graph import CodebaseGraph
from tests._tool_compat import call_tool

try:
    import watchdog  # noqa: F401
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

needs_watchdog = pytest.mark.skipif(
    not HAS_WATCHDOG, reason="watchdog not installed",
)


# ----------------------------------------------------------------------
# P0.1 — get_dependency_graph.depended_on_by for functions
# ----------------------------------------------------------------------


@needs_watchdog
class TestDependencyGraphFunctionCallers:
    """Functions called via singleton instances must surface in
    depended_on_by, not just for class symbols."""

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_function_callers_via_caller_index(self, mock_graph):
        """authenticate_user is called from api/auth.py via
        ``auth_service.authenticate_user(...)``. The import index can't
        see it (api/auth.py imports the module ``auth_service``, not the
        function name), but the caller index does. Source 4 should
        bridge that gap."""
        mock_graph.initialized = True
        mock_graph.original_path = "/project"

        # The target function symbol.
        target = MagicMock()
        target.filepath = "/project/services/auth/auth_service.py"
        target.dependencies = []
        target.usages = []

        # graph.get_class returns None (not a class)
        mock_graph.get_class.return_value = None
        # graph.get_function returns single match -- not in any class
        mock_graph.get_function.return_value = ("python", target, None)
        # graph.get_symbol used afterwards
        mock_graph.get_symbol.return_value = ("python", target, "function")

        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_filepath_from_obj.return_value = ""
        mock_graph.get_importers.return_value = []
        mock_graph._file_symbol_imports = {}
        mock_graph._class_method_names = {}
        mock_graph._class_instances = {}

        # Caller index has authenticate_user → login in api/auth.py
        mock_graph.get_callers.return_value = [
            {
                "name": "login",
                "file": "/project/api/auth.py",
                "has_dot_syntax": True,
                "receiver_token": "auth_service",
            },
        ]

        from grafyx.server import get_dependency_graph
        result = call_tool(
            get_dependency_graph,
            symbol_name="authenticate_user",
            depth=1,
        )

        # Caller index data should now appear in depended_on_by.
        files = [d["file"] for d in result["depended_on_by"]]
        assert "/project/api/auth.py" in files, (
            f"Expected api/auth.py in depended_on_by, got: {files}"
        )

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_method_callers_filtered_by_class(self, mock_graph):
        """For a method symbol, the caller index lookup must pass
        class_name= so callers of same-named methods on OTHER classes
        don't pollute depended_on_by."""
        mock_graph.initialized = True
        mock_graph.original_path = "/project"

        target = MagicMock()
        target.filepath = "/project/services/cache.py"
        target.dependencies = []
        target.usages = []

        mock_graph.get_class.return_value = None
        # Method-as-target: cls_name returned by get_function is "Cache"
        mock_graph.get_function.return_value = ("python", target, "Cache")
        mock_graph.get_symbol.return_value = ("python", target, "function")
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_filepath_from_obj.return_value = ""
        mock_graph.get_importers.return_value = []
        mock_graph._file_symbol_imports = {}
        mock_graph._class_method_names = {}
        mock_graph._class_instances = {}

        # Track whether class_name was forwarded.
        observed_class: list[str | None] = []

        def fake_callers(name, class_name=None):
            observed_class.append(class_name)
            return [
                {"name": "do_thing", "file": "/project/api/use.py",
                 "has_dot_syntax": True},
            ]

        mock_graph.get_callers.side_effect = fake_callers

        from grafyx.server import get_dependency_graph
        call_tool(get_dependency_graph, symbol_name="refresh", depth=1)

        # Source 4 must call get_callers with class_name="Cache".
        assert "Cache" in observed_class, (
            f"Expected get_callers(class_name='Cache'), saw {observed_class}"
        )

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_self_referencing_callers_excluded(self, mock_graph):
        """A caller in the same file as the symbol is a self-reference
        and shouldn't be reported as cross-file dependence."""
        mock_graph.initialized = True
        mock_graph.original_path = "/project"

        target = MagicMock()
        target.filepath = "/project/utils.py"
        target.dependencies = []
        target.usages = []

        mock_graph.get_class.return_value = None
        mock_graph.get_function.return_value = ("python", target, None)
        mock_graph.get_symbol.return_value = ("python", target, "function")
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_filepath_from_obj.return_value = ""
        mock_graph.get_importers.return_value = []
        mock_graph._file_symbol_imports = {}
        mock_graph._class_method_names = {}
        mock_graph._class_instances = {}

        # Single caller is in the SAME file.
        mock_graph.get_callers.return_value = [
            {"name": "wrapper", "file": "/project/utils.py",
             "has_dot_syntax": False},
        ]

        from grafyx.server import get_dependency_graph
        result = call_tool(
            get_dependency_graph, symbol_name="helper", depth=1,
        )

        files = [d["file"] for d in result["depended_on_by"]]
        assert "/project/utils.py" not in files, (
            f"Self-referencing caller should be excluded, got: {files}"
        )


# ----------------------------------------------------------------------
# P0.2 — __getattr__ lazy loader detection (non-dict forms)
# ----------------------------------------------------------------------


def _make_graph_for_imports():
    """Construct a minimal mock graph wired for _build_import_index."""
    graph = MagicMock(spec=CodebaseGraph)
    graph._project_path = "/project"
    graph.original_path = "/project"
    graph._lock = MagicMock()
    graph._external_packages = set()
    graph.translate_path = lambda p: str(p) if p else ""
    graph._build_import_index = CodebaseGraph._build_import_index.__get__(graph)
    graph._extract_module_from_import = CodebaseGraph._extract_module_from_import
    graph._extract_symbol_names_from_import = (
        CodebaseGraph._extract_symbol_names_from_import
    )
    graph._resolve_init_reexports = (
        CodebaseGraph._resolve_init_reexports.__get__(graph)
    )
    graph._is_ignored_file_path = lambda p: False
    return graph


class TestGetattrLazyLoaderForms:
    """v0.2.2 only handled dict-form lazy loaders. v0.2.3 extends to
    inline ``from`` imports, no-dot dicts, and a submodule-name
    fallback heuristic."""

    def test_inline_from_import_in_getattr(self):
        """``def __getattr__(name): from .auth_service import auth_service``
        should register auth_service as resolving to the submodule."""
        graph = _make_graph_for_imports()

        # __init__.py with a __getattr__ that uses inline relative imports.
        getattr_func = MagicMock()
        getattr_func.name = "__getattr__"
        getattr_func.source = (
            "def __getattr__(name):\n"
            "    if name == 'auth_service':\n"
            "        from .auth_service import auth_service\n"
            "        return auth_service\n"
            "    raise AttributeError(name)\n"
        )

        init_file = MagicMock()
        init_file.filepath = "/project/services/auth/__init__.py"
        init_file.path = "/project/services/auth/__init__.py"
        init_file.imports = []
        init_file.functions = [getattr_func]

        impl_file = MagicMock()
        impl_file.filepath = "/project/services/auth/auth_service.py"
        impl_file.path = "/project/services/auth/auth_service.py"
        impl_file.imports = []
        impl_file.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/api/auth.py"
        consumer.path = "/project/api/auth.py"
        consumer_imp = MagicMock()
        consumer_imp.source = "from services.auth import auth_service"
        consumer.imports = [consumer_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, impl_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # api/auth.py should now appear as importer of auth_service.py,
        # resolved through the __getattr__ inline-import body.
        impl_importers = graph._import_index.get(
            "/project/services/auth/auth_service.py", []
        )
        assert "/project/api/auth.py" in impl_importers, (
            f"Inline-from-import lazy loader not resolved; got {impl_importers}"
        )

    def test_submodule_fallback_when_getattr_present(self):
        """Even when __getattr__ has no parseable mapping, the
        fallback heuristic should resolve same-named submodules."""
        graph = _make_graph_for_imports()

        # __getattr__ body is opaque (does runtime lookup via a registry).
        getattr_func = MagicMock()
        getattr_func.name = "__getattr__"
        getattr_func.source = (
            "def __getattr__(name):\n"
            "    return _LOOKUP_REGISTRY.get(name) or _raise(name)\n"
        )

        init_file = MagicMock()
        init_file.filepath = "/project/services/auth/__init__.py"
        init_file.path = "/project/services/auth/__init__.py"
        init_file.imports = []
        init_file.functions = [getattr_func]

        impl_file = MagicMock()
        impl_file.filepath = "/project/services/auth/auth_service.py"
        impl_file.path = "/project/services/auth/auth_service.py"
        impl_file.imports = []
        impl_file.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/api/auth.py"
        consumer.path = "/project/api/auth.py"
        consumer_imp = MagicMock()
        consumer_imp.source = "from services.auth import auth_service"
        consumer.imports = [consumer_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, impl_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Heuristic: __getattr__ is defined and a same-named submodule
        # exists -> register the mapping so the consumer is linked.
        impl_importers = graph._import_index.get(
            "/project/services/auth/auth_service.py", []
        )
        assert "/project/api/auth.py" in impl_importers, (
            f"Same-name submodule fallback failed; got {impl_importers}"
        )

    def test_dict_without_leading_dot_resolves(self):
        """``"X": "submod"`` (no dot) should map to package/submod.py."""
        graph = _make_graph_for_imports()

        getattr_func = MagicMock()
        getattr_func.name = "__getattr__"
        getattr_func.source = (
            "_LAZY = {'auth_service': 'auth_service'}\n"
            "def __getattr__(name):\n"
            "    if name in _LAZY:\n"
            "        import importlib\n"
            "        return importlib.import_module('.' + _LAZY[name], __package__)\n"
            "    raise AttributeError(name)\n"
        )

        init_file = MagicMock()
        init_file.filepath = "/project/svc/__init__.py"
        init_file.path = "/project/svc/__init__.py"
        init_file.imports = []
        init_file.functions = [getattr_func]

        impl_file = MagicMock()
        impl_file.filepath = "/project/svc/auth_service.py"
        impl_file.path = "/project/svc/auth_service.py"
        impl_file.imports = []
        impl_file.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/api/use.py"
        consumer.path = "/project/api/use.py"
        consumer_imp = MagicMock()
        consumer_imp.source = "from svc import auth_service"
        consumer.imports = [consumer_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, impl_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        impl_importers = graph._import_index.get(
            "/project/svc/auth_service.py", []
        )
        assert "/project/api/use.py" in impl_importers, (
            f"No-dot dict form not resolved; got {impl_importers}"
        )

    def test_no_getattr_means_no_fallback(self):
        """If __init__.py does NOT define __getattr__, the same-name
        submodule fallback must NOT fire (would produce false positives
        for plain re-export-free packages)."""
        graph = _make_graph_for_imports()

        init_file = MagicMock()
        init_file.filepath = "/project/svc/__init__.py"
        init_file.path = "/project/svc/__init__.py"
        init_file.imports = []
        init_file.functions = []  # No __getattr__.

        impl_file = MagicMock()
        impl_file.filepath = "/project/svc/auth_service.py"
        impl_file.path = "/project/svc/auth_service.py"
        impl_file.imports = []
        impl_file.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/api/use.py"
        consumer.path = "/project/api/use.py"
        consumer_imp = MagicMock()
        consumer_imp.source = "from svc import auth_service"
        consumer.imports = [consumer_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, impl_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Without __getattr__, only the explicit __init__.py linkage
        # should exist — auth_service.py must NOT have api/use.py as an
        # importer simply because the names line up.
        impl_importers = graph._import_index.get(
            "/project/svc/auth_service.py", []
        )
        assert "/project/api/use.py" not in impl_importers, (
            f"Fallback fired without __getattr__; got {impl_importers}"
        )


# ----------------------------------------------------------------------
# P0.3 — Class cross_file_usages follows factory return types
# ----------------------------------------------------------------------


@needs_watchdog
class TestClassUsagesViaFactory:
    """A class accessed exclusively through a factory function (e.g.
    ``coord = await get_coordinator()``) is invisible to graph-sitter's
    .usages and to import-index lookups (consumers don't import the
    class). Strategy 4 in get_class_context uses the
    _factory_return_types map to surface those callers."""

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_factory_callers_appear_in_cross_file_usages(self, mock_graph):
        mock_graph.initialized = True
        mock_graph.original_path = "/project"

        cls = MagicMock()
        cls.name = "WorkerCoordinator"
        cls.filepath = "/project/voice/worker_coordinator.py"
        cls.methods = []
        cls.usages = []
        cls.dependencies = []
        cls.properties = []
        cls.docstring = ""
        cls.source = ""

        mock_graph.get_class.return_value = ("python", cls)
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_filepath_from_obj.return_value = ""
        mock_graph.get_importers.return_value = []
        mock_graph._class_method_names = {"WorkerCoordinator": set()}
        mock_graph._class_instances = {}
        mock_graph._factory_return_types = {
            "get_coordinator": "WorkerCoordinator",
        }
        mock_graph.get_line_number = lambda x: 1
        # Only get_coordinator's callers should be queried.
        seen_calls: list[str] = []

        def fake_callers(name, **_):
            seen_calls.append(name)
            if name == "get_coordinator":
                return [
                    {"name": "stop_session", "file": "/project/voice/session.py",
                     "has_dot_syntax": False},
                    {"name": "broadcast", "file": "/project/voice/bus.py",
                     "has_dot_syntax": False},
                ]
            return []

        mock_graph.get_callers.side_effect = fake_callers

        # _find_reference_lines is called by Strategy 4 for line numbers;
        # patch it to return [] so the test doesn't hit disk.
        with patch(
            "grafyx.server._tools_introspection._find_reference_lines",
            return_value=[],
        ):
            from grafyx.server import get_class_context
            result = call_tool(
                get_class_context,
                class_name="WorkerCoordinator",
                detail="summary",
                include_hints=False,
            )

        files = [u["file"] for u in result["cross_file_usages"]]
        assert "/project/voice/session.py" in files, (
            f"Factory-pattern caller missing; got {files}"
        )
        assert "/project/voice/bus.py" in files, (
            f"Factory-pattern caller missing; got {files}"
        )
        assert "get_coordinator" in seen_calls

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_no_factory_no_strategy4_calls(self, mock_graph):
        """Classes with no registered factory should NOT trigger
        get_callers from Strategy 4 — keeps the existing fast path."""
        mock_graph.initialized = True
        mock_graph.original_path = "/project"

        cls = MagicMock()
        cls.name = "PlainClass"
        cls.filepath = "/project/p.py"
        cls.methods = []
        cls.usages = []
        cls.dependencies = []
        cls.properties = []
        cls.docstring = ""
        cls.source = ""

        mock_graph.get_class.return_value = ("python", cls)
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_filepath_from_obj.return_value = ""
        mock_graph.get_importers.return_value = []
        mock_graph._class_method_names = {"PlainClass": set()}
        mock_graph._class_instances = {}
        mock_graph._factory_return_types = {}
        mock_graph.get_line_number = lambda x: 1

        seen: list[str] = []

        def fake_callers(name, **_):
            seen.append(name)
            return []

        mock_graph.get_callers.side_effect = fake_callers

        with patch(
            "grafyx.server._tools_introspection._find_reference_lines",
            return_value=[],
        ):
            from grafyx.server import get_class_context
            call_tool(
                get_class_context,
                class_name="PlainClass",
                detail="summary",
                include_hints=False,
            )

        # Strategy 4 must not have been entered (no factory map entry).
        # Strategy 3 also won't fire because there are no methods.
        # So get_callers should never have been called.
        assert seen == [], f"Strategy 4 fired without factory; saw {seen}"


# ----------------------------------------------------------------------
# P1.1 — file_path selector for get_function_context
# ----------------------------------------------------------------------


@needs_watchdog
class TestFunctionContextFilePath:
    """Two top-level functions with the same name in different files
    can't be disambiguated via ``ClassName.method``. ``file_path``
    selects between them."""

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_file_path_picks_correct_function(self, mock_graph):
        mock_graph.initialized = True

        # Two top-level create_agent functions, distinct files.
        f_route = MagicMock()
        f_route.name = "create_agent"
        f_route.filepath = "/project/api/agents.py"
        f_route.parameters = []
        f_route.decorators = []
        f_route.dependencies = []
        f_route.function_calls = []
        f_route.is_async = False
        f_route.docstring = "API route handler"
        f_route.return_type = ""
        f_route.source = ""

        f_service = MagicMock()
        f_service.name = "create_agent"
        f_service.filepath = "/project/services/agents/agent_service.py"
        f_service.parameters = []
        f_service.decorators = []
        f_service.dependencies = []
        f_service.function_calls = []
        f_service.is_async = False
        f_service.docstring = "Service-layer factory"
        f_service.return_type = ""
        f_service.source = ""

        mock_graph.get_function.return_value = [
            ("python", f_route, None),
            ("python", f_service, None),
        ]
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_line_number = lambda _: 1
        mock_graph.get_filepath_from_obj = lambda _: ""
        mock_graph.get_callers.return_value = []

        from grafyx.server import get_function_context
        result = call_tool(
            get_function_context,
            function_name="create_agent",
            file_path="api/agents.py",
            include_hints=False,
        )

        # Should have unambiguously picked the route handler.
        assert not result.get("ambiguous"), (
            f"Expected disambiguation by file, got: {result}"
        )
        assert result["file"].endswith("api/agents.py"), result["file"]
        assert result["docstring"] == "API route handler"

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_file_path_no_match_falls_through(self, mock_graph):
        """If file_path matches none, behave as if it wasn't supplied —
        return the disambiguation list rather than 'not found'."""
        mock_graph.initialized = True

        f1 = MagicMock()
        f1.name = "create_agent"
        f1.filepath = "/project/api/agents.py"
        f2 = MagicMock()
        f2.name = "create_agent"
        f2.filepath = "/project/services/agents/agent_service.py"

        mock_graph.get_function.return_value = [
            ("python", f1, None), ("python", f2, None),
        ]
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_line_number = lambda _: 1

        for f in (f1, f2):
            f.parameters = []
            f.decorators = []
            f.return_type = ""

        from grafyx.server import get_function_context
        result = call_tool(
            get_function_context,
            function_name="create_agent",
            file_path="nonexistent/path.py",
            include_hints=False,
        )

        assert result.get("ambiguous"), (
            f"Expected disambiguation list when filter matches nothing, got: {result}"
        )
        assert len(result["matches"]) == 2


# ----------------------------------------------------------------------
# P1.2 — get_module_context surfaces pagination hint on large modules
# ----------------------------------------------------------------------


@needs_watchdog
class TestModuleContextPaginationHint:
    """When a module has enough files to risk truncation, the
    response should include a concrete next-call suggestion using
    offset/limit so the agent knows it can page rather than treating
    the truncated body as the full picture."""

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_pagination_hint_on_large_module(self, mock_graph):
        mock_graph.initialized = True
        mock_graph.original_path = "/project"
        mock_graph.project_path = "/project"

        # Pretend the module has 120 files.
        files = [
            {"path": f"/project/big/file_{i:03}.py"}
            for i in range(120)
        ]
        mock_graph.get_all_files.return_value = files
        mock_graph.get_all_functions.return_value = []
        mock_graph.get_all_classes.return_value = []
        mock_graph.get_forward_imports.return_value = []
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""

        from grafyx.server import get_module_context
        result = call_tool(
            get_module_context,
            module_path="big",
            include_hints=False,
        )

        assert "pagination_hint" in result, (
            f"Expected pagination_hint for 120-file module, keys: {list(result)}"
        )
        hint = result["pagination_hint"]
        assert hint["total_files"] == 120
        assert "limit=50" in hint["suggested_call"]
        assert "offset=0" in hint["suggested_call"]
        assert "big" in hint["suggested_call"]

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_no_pagination_hint_when_paginated(self, mock_graph):
        """When offset/limit are already passed, the explicit ``page``
        block carries the metadata; pagination_hint is NOT added."""
        mock_graph.initialized = True
        mock_graph.original_path = "/project"
        mock_graph.project_path = "/project"

        files = [
            {"path": f"/project/big/file_{i:03}.py"}
            for i in range(120)
        ]
        mock_graph.get_all_files.return_value = files
        mock_graph.get_all_functions.return_value = []
        mock_graph.get_all_classes.return_value = []
        mock_graph.get_forward_imports.return_value = []
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""

        from grafyx.server import get_module_context
        result = call_tool(
            get_module_context,
            module_path="big",
            limit=50,
            offset=0,
            include_hints=False,
        )

        assert "page" in result
        assert "pagination_hint" not in result, (
            "Should not duplicate hint when explicit pagination is in use"
        )

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_no_pagination_hint_for_small_module(self, mock_graph):
        """Small modules don't risk truncation — no hint needed."""
        mock_graph.initialized = True
        mock_graph.original_path = "/project"
        mock_graph.project_path = "/project"

        files = [
            {"path": f"/project/small/file_{i}.py"}
            for i in range(5)
        ]
        mock_graph.get_all_files.return_value = files
        mock_graph.get_all_functions.return_value = []
        mock_graph.get_all_classes.return_value = []
        mock_graph.get_forward_imports.return_value = []
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""

        from grafyx.server import get_module_context
        result = call_tool(
            get_module_context,
            module_path="small",
            include_hints=False,
        )

        assert "pagination_hint" not in result
