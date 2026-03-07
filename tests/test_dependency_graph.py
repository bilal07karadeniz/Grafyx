"""Tests for get_dependency_graph stdlib/third-party filtering."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from tests._tool_compat import call_tool

try:
    import watchdog  # noqa: F401
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

needs_watchdog = pytest.mark.skipif(not HAS_WATCHDOG, reason="watchdog not installed")


@needs_watchdog
class TestDependencyGraphFiltering:
    """Test that stdlib/third-party deps are filtered from depends_on."""

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_external_deps_filtered(self, mock_graph):
        """Dependencies outside the project root should not appear in depends_on."""
        mock_graph.initialized = True
        mock_graph.original_path = "/project"

        mock_symbol = MagicMock()
        mock_symbol.filepath = "/project/services/user.py"
        mock_graph.get_symbol.return_value = ("python", mock_symbol, "function")
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""

        dep_project = MagicMock()
        dep_project.name = "UserModel"
        dep_project.__class__.__name__ = "class_"
        dep_stdlib = MagicMock()
        dep_stdlib.name = "json"
        dep_stdlib.__class__.__name__ = "module"
        dep_typing = MagicMock()
        dep_typing.name = "Optional"
        dep_typing.__class__.__name__ = "type"

        mock_symbol.dependencies = [dep_project, dep_stdlib, dep_typing]
        mock_symbol.usages = []

        def fake_get_filepath(obj):
            name = getattr(obj, "name", "")
            if name == "UserModel":
                return "/project/models/user.py"
            if name == "json":
                return "/usr/lib/python3.12/json/__init__.py"
            return ""  # typing has no file
        mock_graph.get_filepath_from_obj.side_effect = fake_get_filepath
        mock_graph.get_importers.return_value = []

        from grafyx.server import get_dependency_graph
        result = call_tool(get_dependency_graph,symbol_name="process_user", depth=1)

        dep_names = [d["name"] for d in result["depends_on"]]
        assert "UserModel" in dep_names
        assert "json" not in dep_names
        assert "Optional" not in dep_names
        assert result.get("external_dependency_count") == 2

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_no_external_deps_no_count_field(self, mock_graph):
        """When all deps are project-internal, external_dependency_count is absent."""
        mock_graph.initialized = True
        mock_graph.original_path = "/project"

        mock_symbol = MagicMock()
        mock_symbol.filepath = "/project/main.py"
        mock_graph.get_symbol.return_value = ("python", mock_symbol, "function")
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""

        dep = MagicMock()
        dep.name = "helper"
        dep.__class__.__name__ = "function"
        mock_symbol.dependencies = [dep]
        mock_symbol.usages = []

        mock_graph.get_filepath_from_obj.return_value = "/project/utils.py"
        mock_graph.get_importers.return_value = []

        from grafyx.server import get_dependency_graph
        result = call_tool(get_dependency_graph,symbol_name="main_func", depth=1)

        assert len(result["depends_on"]) == 1
        assert "external_dependency_count" not in result

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_all_external_deps_empty_list(self, mock_graph):
        """When ALL deps are external, depends_on should be empty with a count."""
        mock_graph.initialized = True
        mock_graph.original_path = "/project"

        mock_symbol = MagicMock()
        mock_symbol.filepath = "/project/main.py"
        mock_graph.get_symbol.return_value = ("python", mock_symbol, "function")
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""

        dep1 = MagicMock()
        dep1.name = "asyncio"
        dep1.__class__.__name__ = "module"
        dep2 = MagicMock()
        dep2.name = "Any"
        dep2.__class__.__name__ = "type"
        mock_symbol.dependencies = [dep1, dep2]
        mock_symbol.usages = []

        def fake_get_filepath(obj):
            name = getattr(obj, "name", "")
            if name == "asyncio":
                return "/usr/lib/python3.12/asyncio/__init__.py"
            return ""
        mock_graph.get_filepath_from_obj.side_effect = fake_get_filepath
        mock_graph.get_importers.return_value = []

        from grafyx.server import get_dependency_graph
        result = call_tool(get_dependency_graph,symbol_name="main_func", depth=1)

        assert result["depends_on"] == []
        assert result["external_dependency_count"] == 2

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_stdlib_with_real_path_still_filtered(self, mock_graph):
        """Stdlib deps that resolve to real system paths should still be filtered."""
        mock_graph.initialized = True
        mock_graph.original_path = "/home/user/myproject"

        mock_symbol = MagicMock()
        mock_symbol.filepath = "/home/user/myproject/app.py"
        mock_graph.get_symbol.return_value = ("python", mock_symbol, "function")
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""

        dep_os = MagicMock()
        dep_os.name = "os"
        dep_os.__class__.__name__ = "module"
        dep_httpx = MagicMock()
        dep_httpx.name = "httpx"
        dep_httpx.__class__.__name__ = "module"
        dep_local = MagicMock()
        dep_local.name = "config"
        dep_local.__class__.__name__ = "module"

        mock_symbol.dependencies = [dep_os, dep_httpx, dep_local]
        mock_symbol.usages = []

        def fake_get_filepath(obj):
            name = getattr(obj, "name", "")
            if name == "os":
                return "/usr/lib/python3.12/os.py"
            if name == "httpx":
                return "/home/user/.venv/lib/python3.12/site-packages/httpx/__init__.py"
            if name == "config":
                return "/home/user/myproject/config.py"
            return ""
        mock_graph.get_filepath_from_obj.side_effect = fake_get_filepath
        mock_graph.get_importers.return_value = []

        from grafyx.server import get_dependency_graph
        result = call_tool(get_dependency_graph,symbol_name="main", depth=1)

        dep_names = [d["name"] for d in result["depends_on"]]
        assert "config" in dep_names
        assert "os" not in dep_names
        assert "httpx" not in dep_names
        assert result["external_dependency_count"] == 2
