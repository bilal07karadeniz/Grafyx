"""Tests for import name resolution in dependency graph (P1 fix).

When a symbol name exists as both a third-party import and a local class,
the dependency graph should resolve to the actual import source, not guess
by name.
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


@needs_watchdog
class TestImportResolution:
    def test_external_import_not_resolved_to_local_file(self):
        """Session from sqlalchemy.orm should NOT resolve to app/models/session.py."""
        mock_graph = MagicMock()

        # Symbol: get_current_user in auth.py
        mock_symbol = MagicMock()
        mock_symbol.name = "get_current_user"
        mock_symbol.filepath = "auth.py"
        mock_symbol.usages = []
        mock_symbol.methods = []

        # Dependency: Session -- graph-sitter resolves to local session.py
        mock_dep = MagicMock()
        mock_dep.name = "Session"
        mock_dep.__class__.__name__ = "class"
        mock_dep.filepath = "app/models/session.py"  # Wrong resolution

        mock_symbol.dependencies = [mock_dep]

        mock_graph.get_function.return_value = None  # No disambiguation needed
        mock_graph.get_symbol.return_value = ("python", mock_symbol, "function")
        mock_graph.resolve_path.side_effect = lambda p: p
        mock_graph.get_filepath_from_obj.return_value = "app/models/session.py"
        mock_graph.original_path = "/project"
        mock_graph.get_importers.return_value = []
        mock_graph.get_all_functions.return_value = []
        mock_graph.get_all_classes.return_value = []
        mock_graph._class_method_names = {}

        # The actual import: from sqlalchemy.orm import Session
        # _file_symbol_imports says auth.py does NOT import Session from session.py
        mock_graph._file_symbol_imports = {
            "auth.py": {
                # auth.py imports from utils.py, but NOT from session.py
                "utils.py": {"get_db"},
            }
        }
        mock_graph._external_packages = {"sqlalchemy"}

        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_graph import get_dependency_graph
            result = call_tool(get_dependency_graph, "get_current_user")

            # Session should be counted as external, NOT listed in depends_on
            local_dep_names = [d["name"] for d in result.get("depends_on", [])]
            assert "Session" not in local_dep_names, (
                "Session should be external (sqlalchemy.orm), not local"
            )
            assert result.get("external_dependency_count", 0) >= 1

    def test_confirmed_local_import_not_filtered(self):
        """When a file DOES import a name from a local file, keep it as local dep."""
        mock_graph = MagicMock()

        mock_symbol = MagicMock()
        mock_symbol.name = "process_data"
        mock_symbol.filepath = "handler.py"
        mock_symbol.usages = []
        mock_symbol.methods = []

        # Dependency: DataModel -- correctly resolved to models.py
        mock_dep = MagicMock()
        mock_dep.name = "DataModel"
        mock_dep.__class__.__name__ = "class"
        mock_dep.filepath = "/project/models.py"

        mock_symbol.dependencies = [mock_dep]

        mock_graph.get_function.return_value = None
        mock_graph.get_symbol.return_value = ("python", mock_symbol, "function")
        mock_graph.resolve_path.side_effect = lambda p: p
        mock_graph.get_filepath_from_obj.return_value = "/project/models.py"
        mock_graph.original_path = "/project"
        mock_graph.get_importers.return_value = []
        mock_graph.get_all_functions.return_value = []
        mock_graph.get_all_classes.return_value = []
        mock_graph._class_method_names = {}

        # handler.py explicitly imports DataModel from models.py
        mock_graph._file_symbol_imports = {
            "handler.py": {
                "/project/models.py": {"DataModel"},
            }
        }
        mock_graph._external_packages = set()

        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_graph import get_dependency_graph
            result = call_tool(get_dependency_graph, "process_data")

            local_dep_names = [d["name"] for d in result.get("depends_on", [])]
            assert "DataModel" in local_dep_names, (
                "DataModel should be kept as local (explicitly imported from models.py)"
            )

    def test_transitive_dependency_through_init_py(self):
        """authenticate_user in auth_service.py should show router.py as dependent.

        Chain: router.py -> auth/__init__.py -> auth/auth_service.py
        """
        mock_graph = MagicMock()

        mock_symbol = MagicMock()
        mock_symbol.name = "authenticate_user"
        mock_symbol.filepath = "auth/auth_service.py"
        mock_symbol.usages = []
        mock_symbol.methods = []
        mock_symbol.dependencies = []

        mock_graph.get_function.return_value = None
        mock_graph.get_symbol.return_value = ("python", mock_symbol, "function")
        mock_graph.resolve_path.side_effect = lambda p: p
        mock_graph.get_filepath_from_obj.return_value = "auth/auth_service.py"
        mock_graph.original_path = "/project"
        mock_graph._class_method_names = {}
        mock_graph._external_packages = set()
        mock_graph._file_symbol_imports = {
            "auth/__init__.py": {
                "auth/auth_service.py": {"authenticate_user"},
            },
            "router.py": {
                "auth/__init__.py": {"authenticate_user"},
            },
        }
        mock_graph.get_all_functions.return_value = []
        mock_graph.get_all_classes.return_value = []

        # auth_service.py is imported by auth/__init__.py
        # auth/__init__.py is imported by router.py
        def mock_get_importers(path):
            importers_map = {
                "auth/auth_service.py": ["auth/__init__.py"],
                "auth/__init__.py": ["router.py"],
            }
            return importers_map.get(path, [])
        mock_graph.get_importers.side_effect = mock_get_importers

        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_graph import get_dependency_graph
            result = call_tool(get_dependency_graph, "authenticate_user")

            dep_files = [d["file"] for d in result.get("depended_on_by", [])]
            assert "router.py" in dep_files, (
                f"router.py should be in depended_on_by (through __init__.py). "
                f"Got: {dep_files}"
            )
