"""Tests for grafyx.server module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# server.py imports watcher.py which imports watchdog — skip if unavailable
try:
    import watchdog  # noqa: F401
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

needs_watchdog = pytest.mark.skipif(not HAS_WATCHDOG, reason="watchdog not installed")


@needs_watchdog
class TestCreateServer:
    @patch("grafyx.server._state.CodebaseWatcher")
    @patch("grafyx.server._state.ConventionDetector")
    @patch("grafyx.server._state.CodeSearcher")
    @patch("grafyx.server._state.CodebaseGraph")
    def test_create_server(self, mock_graph_cls, mock_search_cls, mock_detector_cls, mock_watcher_cls):
        mock_graph = MagicMock()
        mock_graph.initialize.return_value = {"languages": ["python"], "total_files": 10}
        mock_graph_cls.return_value = mock_graph

        from grafyx.server import create_server
        server = create_server("/tmp/test", watch=True)

        assert server is not None
        mock_graph_cls.assert_called_once()
        mock_graph.initialize.assert_called_once()
        mock_search_cls.assert_called_once_with(mock_graph)
        mock_detector_cls.assert_called_once_with(mock_graph)
        mock_watcher_cls.assert_called_once()

    @patch("grafyx.server._state.CodebaseWatcher")
    @patch("grafyx.server._state.ConventionDetector")
    @patch("grafyx.server._state.CodeSearcher")
    @patch("grafyx.server._state.CodebaseGraph")
    def test_create_server_no_watch(self, mock_graph_cls, mock_search_cls, mock_detector_cls, mock_watcher_cls):
        mock_graph = MagicMock()
        mock_graph.initialize.return_value = {"languages": ["python"]}
        mock_graph_cls.return_value = mock_graph

        from grafyx.server import create_server
        create_server("/tmp/test", watch=False)

        mock_watcher_cls.assert_not_called()
