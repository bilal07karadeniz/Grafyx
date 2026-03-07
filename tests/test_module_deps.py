"""Tests for get_module_dependencies in grafyx.graph module."""

import pytest
from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph


def _make_mock_graph_with_modules():
    """Build a mock CodebaseGraph with module-level import data."""
    graph = MagicMock(spec=CodebaseGraph)
    graph.original_path = "/project"
    graph._project_path = "/project"
    graph._lock = MagicMock()

    # Simulate _forward_import_index: file -> [files it imports]
    graph._forward_import_index = {
        "/project/api/routes.py": ["/project/services/user_service.py", "/project/models/user.py"],
        "/project/api/middleware.py": ["/project/services/auth_service.py"],
        "/project/services/user_service.py": ["/project/models/user.py", "/project/utils/helpers.py"],
        "/project/services/auth_service.py": ["/project/models/user.py"],
        "/project/models/user.py": ["/project/utils/helpers.py"],
        "/project/utils/helpers.py": [],
    }

    # Simulate get_all_files to list all known files
    graph.get_all_files.return_value = [
        {"path": "/project/api/routes.py", "language": "python"},
        {"path": "/project/api/middleware.py", "language": "python"},
        {"path": "/project/services/user_service.py", "language": "python"},
        {"path": "/project/services/auth_service.py", "language": "python"},
        {"path": "/project/models/user.py", "language": "python"},
        {"path": "/project/utils/helpers.py", "language": "python"},
    ]

    # translate_path is identity in this mock
    graph.translate_path = lambda p: p

    # Use the real method
    graph.get_module_dependencies = CodebaseGraph.get_module_dependencies.__get__(graph)
    return graph


class TestGetModuleDependencies:
    def test_basic_module_graph(self):
        graph = _make_mock_graph_with_modules()
        result = graph.get_module_dependencies()
        assert "modules" in result
        assert "edges" in result
        # Should have 4 modules: api, services, models, utils
        assert len(result["modules"]) == 4
        assert "api" in result["modules"]
        assert "services" in result["modules"]

    def test_edge_counts(self):
        graph = _make_mock_graph_with_modules()
        result = graph.get_module_dependencies()
        edges = result["edges"]
        # api -> services should exist (routes imports user_service, middleware imports auth_service)
        api_to_services = next((e for e in edges if e["from"] == "api" and e["to"] == "services"), None)
        assert api_to_services is not None
        assert api_to_services["import_count"] == 2

    def test_depends_on_and_depended_on_by(self):
        graph = _make_mock_graph_with_modules()
        result = graph.get_module_dependencies()
        modules = result["modules"]
        # utils should be depended on by services and models
        assert "services" in modules["utils"]["depended_on_by"]
        assert "models" in modules["utils"]["depended_on_by"]
        # utils depends on nothing
        assert modules["utils"]["depends_on"] == []

    def test_module_filter(self):
        graph = _make_mock_graph_with_modules()
        result = graph.get_module_dependencies(module_path="api")
        # Should only include modules that api touches
        assert "api" in result["modules"]
        # All edges should involve "api"
        for edge in result["edges"]:
            assert edge["from"] == "api" or edge["to"] == "api"

    def test_file_counts(self):
        graph = _make_mock_graph_with_modules()
        result = graph.get_module_dependencies()
        assert result["modules"]["api"]["file_count"] == 2
        assert result["modules"]["services"]["file_count"] == 2
        assert result["modules"]["models"]["file_count"] == 1
        assert result["modules"]["utils"]["file_count"] == 1

    def test_self_imports_excluded(self):
        """Imports within the same module should not appear as edges."""
        graph = _make_mock_graph_with_modules()
        # Add an intra-module import
        graph._forward_import_index["/project/api/routes.py"] = [
            "/project/services/user_service.py",
            "/project/models/user.py",
            "/project/api/middleware.py",  # intra-module
        ]
        result = graph.get_module_dependencies()
        # No edge from api -> api
        self_edges = [e for e in result["edges"] if e["from"] == e["to"]]
        assert len(self_edges) == 0

    def test_root_files_grouped(self):
        """Files not in any subdirectory should be grouped under '.'."""
        graph = _make_mock_graph_with_modules()
        graph.get_all_files.return_value.append(
            {"path": "/project/main.py", "language": "python"}
        )
        graph._forward_import_index["/project/main.py"] = ["/project/api/routes.py"]
        result = graph.get_module_dependencies()
        assert "." in result["modules"]

    def test_deep_filter_auto_adjusts_depth(self):
        """module_path deeper than depth should auto-bump depth."""
        graph = _make_mock_graph_with_deep_modules()
        # depth=2 would group backend/app/services into "backend/app",
        # making "backend/app/services" invisible. The fix auto-adjusts.
        result = graph.get_module_dependencies(
            module_path="backend/app/services", depth=2
        )
        assert result["module_count"] > 0
        assert any(
            "backend/app/services" in mod for mod in result["modules"]
        )
        # Should have edges involving services
        assert len(result["edges"]) > 0

    def test_prefix_filter_shows_submodules(self):
        """module_path='backend' at depth=2 shows backend/* sub-modules."""
        graph = _make_mock_graph_with_deep_modules()
        result = graph.get_module_dependencies(
            module_path="backend", depth=2
        )
        # Should see backend/app as a module (not just "backend" which doesn't exist)
        backend_modules = [
            m for m in result["modules"] if m.startswith("backend")
        ]
        assert len(backend_modules) > 0

    def test_deep_filter_finds_edges(self):
        """Drilling into backend/app/services should show its real dependencies."""
        graph = _make_mock_graph_with_deep_modules()
        result = graph.get_module_dependencies(
            module_path="backend/app/services", depth=3
        )
        # services imports models, so there should be an edge
        service_edges = [
            e for e in result["edges"]
            if "backend/app/services" in e["from"] or "backend/app/services" in e["to"]
        ]
        assert len(service_edges) > 0


def _make_mock_graph_with_deep_modules():
    """Build a mock graph with nested module structure like a real project.

    Structure:
        backend/app/services/user_service.py  -> backend/app/models/user.py
        backend/app/services/auth_service.py  -> backend/app/models/user.py
        backend/app/api/routes.py             -> backend/app/services/user_service.py
        backend/app/api/routes.py             -> backend/app/services/auth_service.py
        backend/app/workers/tasks.py          -> backend/app/services/user_service.py
        frontend/src/api.py                   -> backend/app/api/routes.py
    """
    graph = MagicMock(spec=CodebaseGraph)
    graph.original_path = "/project"
    graph._project_path = "/project"
    graph._lock = MagicMock()

    graph._forward_import_index = {
        "/project/backend/app/services/user_service.py": [
            "/project/backend/app/models/user.py",
        ],
        "/project/backend/app/services/auth_service.py": [
            "/project/backend/app/models/user.py",
        ],
        "/project/backend/app/api/routes.py": [
            "/project/backend/app/services/user_service.py",
            "/project/backend/app/services/auth_service.py",
        ],
        "/project/backend/app/workers/tasks.py": [
            "/project/backend/app/services/user_service.py",
        ],
        "/project/frontend/src/api.py": [
            "/project/backend/app/api/routes.py",
        ],
    }

    graph.get_all_files.return_value = [
        {"path": "/project/backend/app/services/user_service.py", "language": "python"},
        {"path": "/project/backend/app/services/auth_service.py", "language": "python"},
        {"path": "/project/backend/app/models/user.py", "language": "python"},
        {"path": "/project/backend/app/api/routes.py", "language": "python"},
        {"path": "/project/backend/app/workers/tasks.py", "language": "python"},
        {"path": "/project/frontend/src/api.py", "language": "python"},
    ]

    graph.translate_path = lambda p: p
    graph.get_module_dependencies = CodebaseGraph.get_module_dependencies.__get__(graph)
    return graph


import sys
from unittest.mock import patch
from tests._tool_compat import call_tool

try:
    import watchdog  # noqa: F401
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

needs_watchdog = pytest.mark.skipif(not HAS_WATCHDOG, reason="watchdog not installed")


@needs_watchdog
class TestGetModuleDepsTool:
    """Test the MCP tool wrapper in server.py."""

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_tool_returns_modules(self, mock_graph):
        mock_graph.initialized = True
        mock_graph.get_module_dependencies.return_value = {
            "project_path": "/project",
            "module_count": 2,
            "modules": {
                "api": {"file_count": 2, "depends_on": ["services"], "depended_on_by": []},
                "services": {"file_count": 2, "depends_on": [], "depended_on_by": ["api"]},
            },
            "edges": [{"from": "api", "to": "services", "import_count": 3}],
        }

        from grafyx.server import get_module_dependencies
        result = call_tool(get_module_dependencies)
        assert result["module_count"] == 2
        assert "api" in result["modules"]
        mock_graph.get_module_dependencies.assert_called_once_with("", 1, False)

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_tool_with_filter(self, mock_graph):
        mock_graph.initialized = True
        mock_graph.get_module_dependencies.return_value = {
            "project_path": "/project",
            "module_count": 1,
            "modules": {"api": {"file_count": 2, "depends_on": [], "depended_on_by": []}},
            "edges": [],
        }

        from grafyx.server import get_module_dependencies
        result = call_tool(get_module_dependencies,module_path="api", depth=2)
        mock_graph.get_module_dependencies.assert_called_once_with("api", 2, False)
