"""Tests for get_subclasses in grafyx.graph module."""

import pytest
from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph


def _make_mock_graph_with_classes():
    """Build a mock CodebaseGraph with a class hierarchy for testing."""
    graph = MagicMock(spec=CodebaseGraph)
    graph.get_all_classes.return_value = [
        {"name": "BaseHandler", "base_classes": [], "file": "models.py", "language": "python", "line": 50, "method_count": 2},
        {"name": "HTTPHandler", "base_classes": ["BaseHandler"], "file": "models.py", "language": "python", "line": 60, "method_count": 1},
        {"name": "WebSocketHandler", "base_classes": ["BaseHandler"], "file": "models.py", "language": "python", "line": 70, "method_count": 1},
        {"name": "APIHandler", "base_classes": ["HTTPHandler"], "file": "models.py", "language": "python", "line": 80, "method_count": 1},
        {"name": "User", "base_classes": [], "file": "models.py", "language": "python", "line": 8, "method_count": 2},
        {"name": "Product", "base_classes": [], "file": "models.py", "language": "python", "line": 25, "method_count": 2},
    ]
    # Use the real method, bound to the mock
    graph.get_subclasses = CodebaseGraph.get_subclasses.__get__(graph)
    return graph


class TestGetSubclasses:
    def test_direct_subclasses(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("BaseHandler")
        assert result["class_name"] == "BaseHandler"
        assert result["direct_subclass_count"] == 2
        names = [s["name"] for s in result["subclasses"]]
        assert "HTTPHandler" in names
        assert "WebSocketHandler" in names

    def test_recursive_subclasses(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("BaseHandler", depth=3)
        assert result["total_subclass_count"] == 3  # HTTP, WebSocket, API
        # APIHandler should be nested under HTTPHandler
        http = next(s for s in result["subclasses"] if s["name"] == "HTTPHandler")
        assert len(http["subclasses"]) == 1
        assert http["subclasses"][0]["name"] == "APIHandler"

    def test_depth_1_limits_to_direct(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("BaseHandler", depth=1)
        assert result["direct_subclass_count"] == 2
        assert result["total_subclass_count"] == 2  # Only direct
        # HTTPHandler should have empty subclasses at depth=1
        http = next(s for s in result["subclasses"] if s["name"] == "HTTPHandler")
        assert http["subclasses"] == []

    def test_no_subclasses(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("User")
        assert result["direct_subclass_count"] == 0
        assert result["total_subclass_count"] == 0
        assert result["subclasses"] == []

    def test_class_not_found(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("NonExistentClass")
        assert result is None

    def test_leaf_class(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("APIHandler")
        assert result["direct_subclass_count"] == 0
        assert result["subclasses"] == []


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
class TestGetSubclassesTool:
    """Test the MCP tool wrapper in server.py."""

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_tool_returns_subclasses(self, mock_graph):
        mock_graph.initialized = True
        mock_graph.get_subclasses.return_value = {
            "class_name": "BaseHandler",
            "file": "models.py",
            "line": 50,
            "language": "python",
            "direct_subclass_count": 2,
            "total_subclass_count": 3,
            "subclasses": [
                {"name": "HTTPHandler", "file": "models.py", "line": 60, "language": "python", "subclasses": []},
            ],
        }

        from grafyx.server import get_subclasses
        # MCP tool functions are FunctionTool objects — call .fn directly
        result = call_tool(get_subclasses,class_name="BaseHandler")
        assert result["class_name"] == "BaseHandler"
        assert result["direct_subclass_count"] == 2
        mock_graph.get_subclasses.assert_called_once_with("BaseHandler", 3)

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_tool_class_not_found(self, mock_graph):
        mock_graph.initialized = True
        mock_graph.get_subclasses.return_value = None

        from grafyx.server import get_subclasses
        from fastmcp.exceptions import ToolError
        with pytest.raises(ToolError, match="not found"):
            call_tool(get_subclasses,class_name="Nonexistent")
