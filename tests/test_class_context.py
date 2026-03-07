"""Tests for get_class_context cross-file usage fallback."""

import os
import tempfile
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
class TestClassContextUsageFallback:
    """Test that get_class_context falls back to import index when usages is empty."""

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._tools_introspection._find_reference_lines", return_value=[3])
    @patch("grafyx.server._state._graph")
    def test_import_index_fallback(self, mock_graph, mock_find_lines):
        """When graph-sitter usages is empty, import index should provide cross_file_usages."""
        mock_graph.initialized = True

        mock_cls = MagicMock()
        mock_cls.name = "UserService"
        mock_cls.filepath = "/project/services/user.py"
        mock_cls.docstring = "User service class"
        mock_cls.usages = []
        mock_cls.methods = []
        mock_cls.properties = []
        mock_cls.dependencies = []

        mock_graph.get_class.return_value = ("python", mock_cls)
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_line_number.return_value = 10
        mock_graph.get_filepath_from_obj.return_value = ""

        mock_graph.get_importers.return_value = [
            "/project/api/routes.py",
            "/project/tests/test_user.py",
        ]
        mock_graph._class_method_names = {}
        mock_graph._class_instances = {}

        from grafyx.server import get_class_context
        result = call_tool(get_class_context,class_name="UserService")

        assert len(result["cross_file_usages"]) == 2
        usage_files = [u["file"] for u in result["cross_file_usages"]]
        assert "/project/api/routes.py" in usage_files
        assert "/project/tests/test_user.py" in usage_files

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._tools_introspection._find_reference_lines", return_value=[1])
    @patch("grafyx.server._state._graph")
    def test_supplement_deduplicates_with_usages(self, mock_graph, mock_find_lines):
        """Import index supplement should add new files but not duplicate existing ones."""
        mock_graph.initialized = True

        mock_cls = MagicMock()
        mock_cls.name = "Config"
        mock_cls.filepath = "/project/config.py"
        mock_cls.docstring = ""
        mock_cls.methods = []
        mock_cls.properties = []
        mock_cls.dependencies = []

        usage = MagicMock()
        mock_graph.get_filepath_from_obj.return_value = "/project/main.py"
        mock_graph.get_line_number.side_effect = lambda x: 5 if x is usage else 1
        mock_cls.usages = [usage]

        mock_graph.get_class.return_value = ("python", mock_cls)
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""

        mock_graph.get_importers.return_value = [
            "/project/main.py", "/project/api.py", "/project/worker.py"
        ]
        mock_graph._class_method_names = {}
        mock_graph._class_instances = {}

        from grafyx.server import get_class_context
        result = call_tool(get_class_context,class_name="Config")

        # Import index supplement runs always — adds api.py and worker.py,
        # but main.py (already found via usages) is NOT duplicated.
        usage_files = [u["file"] for u in result["cross_file_usages"]]
        assert usage_files.count("/project/main.py") == 1
        assert "/project/api.py" in usage_files
        assert "/project/worker.py" in usage_files
        assert len(result["cross_file_usages"]) == 3

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._tools_introspection._find_reference_lines", return_value=[7])
    @patch("grafyx.server._state._graph")
    def test_caller_index_fallback(self, mock_graph, mock_find_lines):
        """When both usages and import index are empty, caller index should be tried."""
        mock_graph.initialized = True

        mock_method = MagicMock()
        mock_method.name = "unique_process"

        mock_cls = MagicMock()
        mock_cls.name = "Processor"
        mock_cls.filepath = "/project/processor.py"
        mock_cls.docstring = ""
        mock_cls.usages = []
        mock_cls.methods = [mock_method]
        mock_cls.properties = []
        mock_cls.dependencies = []

        mock_graph.get_class.return_value = ("python", mock_cls)
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_line_number.return_value = 5
        mock_graph.get_filepath_from_obj.return_value = ""

        mock_graph.get_importers.return_value = []
        mock_graph._class_method_names = {"Processor": {"unique_process"}}
        mock_graph._class_instances = {}
        mock_graph.get_callers.return_value = [
            {"name": "run_pipeline", "file": "/project/pipeline.py"},
        ]

        from grafyx.server import get_class_context
        result = call_tool(get_class_context,class_name="Processor")

        assert len(result["cross_file_usages"]) == 1
        assert result["cross_file_usages"][0]["file"] == "/project/pipeline.py"

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._tools_introspection._find_reference_lines", return_value=[2])
    @patch("grafyx.server._state._graph")
    def test_self_file_excluded_from_importers(self, mock_graph, mock_find_lines):
        """Import index fallback should not include the class's own file."""
        mock_graph.initialized = True

        mock_cls = MagicMock()
        mock_cls.name = "Helper"
        mock_cls.filepath = "/project/helpers.py"
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

        mock_graph.get_importers.return_value = [
            "/project/helpers.py",
            "/project/main.py",
        ]
        mock_graph._class_method_names = {}
        mock_graph._class_instances = {}

        from grafyx.server import get_class_context
        result = call_tool(get_class_context,class_name="Helper")

        assert len(result["cross_file_usages"]) == 1
        assert result["cross_file_usages"][0]["file"] == "/project/main.py"

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_fallback_populates_line_numbers(self, mock_graph):
        """Import index fallback should find real line numbers from file content."""
        mock_graph.initialized = True

        mock_cls = MagicMock()
        mock_cls.name = "UsageLog"
        mock_cls.filepath = "/project/models/usage.py"
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
        mock_graph._class_instances = {}

        # Create a real temp file that references the class
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write("from models.usage import UsageLog\n")
            tmp.write("\n")
            tmp.write("def get_logs() -> list[UsageLog]:\n")
            tmp.write("    return db.query(UsageLog).all()\n")
            tmp_path = tmp.name

        try:
            mock_graph.get_importers.return_value = [tmp_path]

            from grafyx.server import get_class_context
            result = call_tool(get_class_context,class_name="UsageLog")

            assert len(result["cross_file_usages"]) == 1
            usage = result["cross_file_usages"][0]
            assert usage["file"] == tmp_path
            # Should find line 1 (import), line 3 (type hint), line 4 (query)
            assert len(usage["lines"]) == 3
            assert 1 in usage["lines"]
            assert 3 in usage["lines"]
            assert 4 in usage["lines"]
        finally:
            os.unlink(tmp_path)


class TestFindReferenceLines:
    """Test the _find_reference_lines helper directly."""

    def test_finds_lines_with_name(self):
        from grafyx.server import _find_reference_lines

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write("import MyClass\n")
            tmp.write("x = 1\n")
            tmp.write("obj = MyClass()\n")
            tmp_path = tmp.name

        try:
            lines = _find_reference_lines(tmp_path, "MyClass")
            assert lines == [1, 3]
        finally:
            os.unlink(tmp_path)

    def test_returns_empty_for_missing_file(self):
        from grafyx.server import _find_reference_lines

        lines = _find_reference_lines("/nonexistent/file.py", "Foo")
        assert lines == []

    def test_respects_max_lines(self):
        from grafyx.server import _find_reference_lines

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            for i in range(10):
                tmp.write(f"use MyClass  # line {i+1}\n")
            tmp_path = tmp.name

        try:
            lines = _find_reference_lines(tmp_path, "MyClass", max_lines=3)
            assert len(lines) == 3
            assert lines == [1, 2, 3]
        finally:
            os.unlink(tmp_path)
