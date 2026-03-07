"""Tests for P2 singleton/instance tracking in class context.

Covers:
- _build_class_instances detects module-level singleton patterns
- get_class_context fallback searches for instance names
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from grafyx.graph import CodebaseGraph
from grafyx.utils import safe_get_attr
from tests._tool_compat import call_tool

try:
    import watchdog  # noqa: F401
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

needs_watchdog = pytest.mark.skipif(not HAS_WATCHDOG, reason="watchdog not installed")


class TestBuildClassInstances:
    """Test that _build_class_instances detects singleton patterns."""

    def _make_graph_with_source_files(self, file_contents):
        """Create a graph mock and temp files with given contents.

        Args:
            file_contents: dict of {filename: source_code}

        Returns:
            (graph, temp_dir, file_paths)
        """
        temp_dir = tempfile.mkdtemp()
        file_paths = {}

        for filename, content in file_contents.items():
            path = os.path.join(temp_dir, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            file_paths[filename] = path

        graph = MagicMock(spec=CodebaseGraph)
        graph._project_path = temp_dir
        graph._lock = MagicMock()
        graph._class_method_names = {}
        graph._class_instances = {}

        # Build file mocks
        file_mocks = []
        for filename, path in file_paths.items():
            fm = MagicMock()
            fm.filepath = path
            fm.path = path
            file_mocks.append(fm)

        py_codebase = MagicMock()
        py_codebase.files = file_mocks
        py_codebase.classes = []
        graph._codebases = {"python": py_codebase}
        graph.translate_path = lambda p: str(p) if p else ""

        # Bind the real method
        graph._build_class_instances = CodebaseGraph._build_class_instances.__get__(graph)

        return graph, temp_dir, file_paths

    def _cleanup(self, temp_dir):
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_detects_simple_singleton(self):
        """Detects 'cache = TranscriptCache()' pattern."""
        graph, temp_dir, _ = self._make_graph_with_source_files({
            "cache.py": (
                "class TranscriptCache:\n"
                "    pass\n"
                "\n"
                "transcript_cache = TranscriptCache()\n"
            ),
        })
        # Register the class name
        graph._class_method_names = {"TranscriptCache": {"get", "set"}}

        try:
            graph._build_class_instances()
            instances = graph._class_instances
            assert "TranscriptCache" in instances
            assert any(name == "transcript_cache" for name, _ in instances["TranscriptCache"])
        finally:
            self._cleanup(temp_dir)

    def test_detects_typed_singleton(self):
        """Detects 'settings: AppSettings = AppSettings()' pattern."""
        graph, temp_dir, _ = self._make_graph_with_source_files({
            "config.py": (
                "class AppSettings:\n"
                "    pass\n"
                "\n"
                "settings: AppSettings = AppSettings()\n"
            ),
        })
        graph._class_method_names = {"AppSettings": {"get"}}

        try:
            graph._build_class_instances()
            instances = graph._class_instances
            assert "AppSettings" in instances
            assert any(name == "settings" for name, _ in instances["AppSettings"])
        finally:
            self._cleanup(temp_dir)

    def test_ignores_indented_assignments(self):
        """Indented assignments (inside functions/methods) are NOT module-level."""
        graph, temp_dir, _ = self._make_graph_with_source_files({
            "service.py": (
                "class MyService:\n"
                "    pass\n"
                "\n"
                "def setup():\n"
                "    svc = MyService()\n"
                "    return svc\n"
            ),
        })
        graph._class_method_names = {"MyService": {"run"}}

        try:
            graph._build_class_instances()
            instances = graph._class_instances
            # Should NOT detect indented assignment
            assert "MyService" not in instances
        finally:
            self._cleanup(temp_dir)

    def test_ignores_unknown_classes(self):
        """Assignments with class names not in the project are ignored."""
        graph, temp_dir, _ = self._make_graph_with_source_files({
            "app.py": (
                "client = SomeExternalClient()\n"
            ),
        })
        graph._class_method_names = {"MyLocalClass": {"method"}}

        try:
            graph._build_class_instances()
            instances = graph._class_instances
            assert "SomeExternalClient" not in instances
        finally:
            self._cleanup(temp_dir)

    def test_multiple_instances_same_class(self):
        """Multiple files can have instances of the same class."""
        graph, temp_dir, _ = self._make_graph_with_source_files({
            "a.py": "cache_a = RedisCache()\n",
            "b.py": "cache_b = RedisCache()\n",
        })
        graph._class_method_names = {"RedisCache": {"get", "set"}}

        try:
            graph._build_class_instances()
            instances = graph._class_instances
            assert "RedisCache" in instances
            names = [name for name, _ in instances["RedisCache"]]
            assert "cache_a" in names
            assert "cache_b" in names
        finally:
            self._cleanup(temp_dir)


@needs_watchdog
class TestClassContextSingletonConsumers:
    """Test that get_class_context finds files using singleton instances."""

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._tools_introspection._find_reference_lines")
    @patch("grafyx.server._state._graph")
    def test_instance_name_searched_in_importers(self, mock_graph, mock_find_lines):
        """When class has a singleton, importers should be searched for instance name too."""
        mock_graph.initialized = True

        mock_cls = MagicMock()
        mock_cls.name = "TranscriptCache"
        mock_cls.filepath = "/project/services/cache.py"
        mock_cls.docstring = "Cache for transcripts"
        mock_cls.usages = []
        mock_cls.methods = []
        mock_cls.properties = []
        mock_cls.dependencies = []

        mock_graph.get_class.return_value = ("python", mock_cls)
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_line_number.return_value = 1
        mock_graph.get_filepath_from_obj.return_value = ""
        mock_graph._class_method_names = {}
        mock_graph._class_instances = {
            "TranscriptCache": [("transcript_cache", "/project/services/cache.py")],
        }

        mock_graph.get_importers.return_value = [
            "/project/api/routes.py",
        ]

        # Simulate: routes.py has "transcript_cache" on line 3 but not "TranscriptCache"
        def find_lines(filepath, name, max_lines=3):
            if name == "TranscriptCache":
                return []
            if name == "transcript_cache" and filepath == "/project/api/routes.py":
                return [3]
            return []

        mock_find_lines.side_effect = find_lines

        from grafyx.server import get_class_context
        result = call_tool(get_class_context,class_name="TranscriptCache")

        assert len(result["cross_file_usages"]) == 1
        usage = result["cross_file_usages"][0]
        assert usage["file"] == "/project/api/routes.py"
        assert 3 in usage["lines"]

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._tools_introspection._find_reference_lines")
    @patch("grafyx.server._state._graph")
    def test_both_class_and_instance_lines_merged(self, mock_graph, mock_find_lines):
        """Lines from both class name and instance name should be merged and sorted."""
        mock_graph.initialized = True

        mock_cls = MagicMock()
        mock_cls.name = "AppConfig"
        mock_cls.filepath = "/project/config.py"
        mock_cls.docstring = ""
        mock_cls.usages = []
        mock_cls.methods = []
        mock_cls.properties = []
        mock_cls.dependencies = []

        mock_graph.get_class.return_value = ("python", mock_cls)
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_line_number.return_value = 1
        mock_graph.get_filepath_from_obj.return_value = ""
        mock_graph._class_method_names = {}
        mock_graph._class_instances = {
            "AppConfig": [("app_config", "/project/config.py")],
        }

        mock_graph.get_importers.return_value = ["/project/main.py"]

        def find_lines(filepath, name, max_lines=3):
            if name == "AppConfig":
                return [1]  # import line
            if name == "app_config":
                return [5, 10]  # usage lines
            return []

        mock_find_lines.side_effect = find_lines

        from grafyx.server import get_class_context
        result = call_tool(get_class_context,class_name="AppConfig")

        assert len(result["cross_file_usages"]) == 1
        usage = result["cross_file_usages"][0]
        # Lines should be merged and sorted
        assert usage["lines"] == [1, 5, 10]

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._tools_introspection._find_reference_lines", return_value=[4])
    @patch("grafyx.server._state._graph")
    def test_no_instances_still_works(self, mock_graph, mock_find_lines):
        """Classes without singleton instances should still work via normal fallback."""
        mock_graph.initialized = True

        mock_cls = MagicMock()
        mock_cls.name = "SimpleClass"
        mock_cls.filepath = "/project/simple.py"
        mock_cls.docstring = ""
        mock_cls.usages = []
        mock_cls.methods = []
        mock_cls.properties = []
        mock_cls.dependencies = []

        mock_graph.get_class.return_value = ("python", mock_cls)
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_line_number.return_value = 1
        mock_graph.get_filepath_from_obj.return_value = ""
        mock_graph._class_method_names = {}
        mock_graph._class_instances = {}  # No instances

        mock_graph.get_importers.return_value = ["/project/main.py"]

        from grafyx.server import get_class_context
        result = call_tool(get_class_context,class_name="SimpleClass")

        # Should still show importers even without instance names
        assert len(result["cross_file_usages"]) == 1
