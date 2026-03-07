"""Tests for grafyx.watcher module."""

import time
from unittest.mock import MagicMock, patch
from grafyx.watcher import CodebaseWatcher


class TestCodebaseWatcher:
    def test_init(self):
        graph = MagicMock()
        graph.project_path = "/tmp/test"
        watcher = CodebaseWatcher(graph)
        assert watcher.running is False

    def test_start_stop(self):
        graph = MagicMock()
        graph.project_path = "/tmp/test"
        watcher = CodebaseWatcher(graph)

        with patch("grafyx.watcher.Observer") as mock_observer_class:
            mock_observer = MagicMock()
            mock_observer_class.return_value = mock_observer

            watcher.start()
            assert watcher.running is True
            mock_observer.start.assert_called_once()

            watcher.stop()
            assert watcher.running is False
            mock_observer.stop.assert_called_once()

    def test_double_start_warning(self):
        graph = MagicMock()
        graph.project_path = "/tmp/test"
        watcher = CodebaseWatcher(graph)

        with patch("grafyx.watcher.Observer"):
            watcher.start()
            watcher.start()  # Should warn, not error
            watcher.stop()

    def test_stop_when_not_running(self):
        graph = MagicMock()
        graph.project_path = "/tmp/test"
        watcher = CodebaseWatcher(graph)
        watcher.stop()  # Should not raise
