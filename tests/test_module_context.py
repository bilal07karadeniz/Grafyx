"""Tests for get_module_context tool."""
from unittest.mock import MagicMock, patch
import pytest
from tests._tool_compat import call_tool

try:
    import watchdog
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

needs_watchdog = pytest.mark.skipif(not HAS_WATCHDOG, reason="watchdog not installed")


@needs_watchdog
class TestModuleContext:
    """Test the new get_module_context tool."""

    def _make_mock_graph(self):
        mock_graph = MagicMock()
        mock_graph.original_path = "/app"
        mock_graph.project_path = "/app"
        mock_graph.get_all_files.return_value = [
            {"path": "/app/services/orders.py", "function_count": 1, "class_count": 1},
            {"path": "/app/services/validation.py", "function_count": 1, "class_count": 0},
        ]
        mock_graph.get_all_functions.return_value = [
            {"name": "process_order", "file": "/app/services/orders.py", "signature": "def process_order()", "docstring": "Process an order.", "line": 10},
            {"name": "validate", "file": "/app/services/validation.py", "signature": "def validate()", "docstring": "Validate input.", "line": 5},
        ]
        mock_graph.get_all_classes.return_value = [
            {"name": "OrderService", "file": "/app/services/orders.py", "base_classes": [], "methods": [{"name": "process_order"}], "docstring": "Order service."},
        ]
        mock_graph.get_forward_imports.return_value = []
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph._caller_index = {}
        mock_graph._class_method_names = {}
        mock_graph._import_index = {}
        return mock_graph

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_basic_module_context(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_exploration import get_module_context
            result = call_tool(get_module_context, "services")
            assert result["module"] == "services"
            assert "symbols" in result
            assert "total_functions" in result
            assert "total_classes" in result
            assert result["total_functions"] == 2
            assert result["total_classes"] == 1

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_signatures_detail(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_exploration import get_module_context
            result = call_tool(get_module_context, "services", detail="signatures")
            assert "internal_imports" not in result
            for sym in result.get("symbols", []):
                for func in sym.get("functions", []):
                    assert "docstring" not in func

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_empty_module_path_shows_all(self, mock_graph_ref):
        """Empty module path should show the whole project."""
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_exploration import get_module_context
            result = call_tool(get_module_context, "")
            assert "symbols" in result
            assert result["module"] == "(root)"

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_no_hints_when_disabled(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_exploration import get_module_context
            result = call_tool(get_module_context, "services", include_hints=False)
            assert "suggested_next" not in result

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_nonexistent_module_returns_empty(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_exploration import get_module_context
            result = call_tool(get_module_context, "nonexistent")
            assert result["files"] == 0
            assert result["total_functions"] == 0

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_class_method_names_included(self, mock_graph_ref):
        """Class entries should include method names from the graph."""
        mock_graph = self._make_mock_graph()
        # Use string method names (as returned by get_all_classes with include_method_names=True)
        mock_graph.get_all_classes.return_value = [
            {"name": "OrderService", "file": "/app/services/orders.py", "base_classes": [],
             "methods": ["process_order", "cancel_order"], "docstring": "Order service."},
        ]
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_exploration import get_module_context
            result = call_tool(get_module_context, "services")
            cls = result["symbols"][0]["classes"][0]
            assert cls["name"] == "OrderService"
            assert "process_order" in cls["methods"]
            assert "cancel_order" in cls["methods"]
