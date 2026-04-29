"""Tests for structured error responses (P0 fix).

All 'not found' cases should return {"found": false, "message": "..."}
instead of raising ToolError, which would cancel parallel MCP calls.
"""

from unittest.mock import MagicMock, patch

import pytest

try:
    import watchdog  # noqa: F401
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

needs_watchdog = pytest.mark.skipif(not HAS_WATCHDOG, reason="watchdog not installed")

from tests._tool_compat import call_tool


def _patch_state():
    """Create a mock graph that returns None for all lookups."""
    mock_graph = MagicMock()
    mock_graph.get_function.return_value = None
    mock_graph.get_file.return_value = None
    mock_graph.get_class.return_value = None
    mock_graph.get_symbol.return_value = None
    mock_graph.get_subclasses.return_value = None
    mock_graph.get_all_functions.return_value = []
    mock_graph.get_all_classes.return_value = []
    return mock_graph


@needs_watchdog
class TestStructuredErrorResponses:
    def test_function_not_found_returns_dict(self):
        mock_graph = _patch_state()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_introspection import get_function_context
            result = call_tool(get_function_context, "nonexistent_function")
            assert result["found"] is False
            assert "not found" in result["message"].lower()

    def test_file_not_found_returns_dict(self):
        mock_graph = _patch_state()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_introspection import get_file_context
            result = call_tool(get_file_context, "nonexistent_file.py")
            assert result["found"] is False
            assert "not found" in result["message"].lower()

    def test_class_not_found_returns_dict(self):
        mock_graph = _patch_state()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_introspection import get_class_context
            result = call_tool(get_class_context, "NonexistentClass")
            assert result["found"] is False
            assert "not found" in result["message"].lower()

    def test_dependency_graph_not_found_returns_dict(self):
        mock_graph = _patch_state()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_graph import get_dependency_graph
            result = call_tool(get_dependency_graph, "nonexistent_symbol")
            assert result["found"] is False
            assert "not found" in result["message"].lower()
            assert "suggestions" in result

    def test_call_graph_not_found_returns_dict(self):
        mock_graph = _patch_state()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_graph import get_call_graph
            result = call_tool(get_call_graph, "nonexistent_function")
            assert result["found"] is False
            assert "not found" in result["message"].lower()
            assert "suggestions" in result

    def test_subclasses_not_found_returns_dict(self):
        mock_graph = _patch_state()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_quality import get_subclasses
            result = call_tool(get_subclasses, "NonexistentClass")
            assert result["found"] is False
            assert "not found" in result["message"].lower()
