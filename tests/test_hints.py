"""Tests for navigation hint computation."""
from unittest.mock import MagicMock
from grafyx.server._hints import compute_hints


class TestComputeHints:
    """Test the shared navigation hint computation helper."""

    def _make_mock_graph(self):
        """Create a mock graph with enough data for hint computation."""
        graph = MagicMock()
        graph._caller_index = {
            "process_order": [
                {"caller": "handle_request", "file": "routes.py"},
                {"caller": "batch_run", "file": "jobs.py"},
            ],
            "validate": [
                {"caller": "process_order", "file": "orders.py"},
            ],
        }
        graph._class_method_names = {
            "OrderService": {"process_order", "cancel_order", "get_order"},
            "UserService": {"create_user", "get_user"},
        }
        graph._import_index = {
            "services/orders.py": ["routes.py", "jobs.py", "tests/test_orders.py"],
            "services/users.py": ["routes.py", "admin.py"],
        }
        graph.translate_path = lambda p: p
        graph.resolve_path = lambda p: p
        return graph

    def test_function_context_hints(self):
        """After viewing a function, hints should suggest top callers/callees."""
        graph = self._make_mock_graph()
        symbol_data = {
            "name": "process_order",
            "class": "OrderService",
            "file": "services/orders.py",
            "calls": [
                {"name": "validate", "file": "services/validation.py"},
                {"name": "save_to_db", "file": "services/db.py"},
            ],
            "called_by": [
                {"name": "handle_request", "file": "routes.py"},
                {"name": "batch_run", "file": "jobs.py"},
            ],
        }
        hints = compute_hints(graph, "function", symbol_data)
        assert isinstance(hints, list)
        assert len(hints) <= 3
        for hint in hints:
            assert "tool" in hint
            assert "args" in hint
            assert "reason" in hint

    def test_file_context_hints(self):
        """After viewing a file, hints should suggest important symbols in it."""
        graph = self._make_mock_graph()
        symbol_data = {
            "path": "services/orders.py",
            "functions": [
                {"name": "process_order", "signature": "def process_order()"},
                {"name": "validate", "signature": "def validate()"},
            ],
            "classes": [
                {"name": "OrderService", "method_count": 5},
            ],
        }
        hints = compute_hints(graph, "file", symbol_data)
        assert isinstance(hints, list)
        assert len(hints) <= 3

    def test_class_context_hints(self):
        """After viewing a class, hints should suggest related classes/methods."""
        graph = self._make_mock_graph()
        symbol_data = {
            "name": "OrderService",
            "file": "services/orders.py",
            "methods": [
                {"name": "process_order", "signature": "def process_order()"},
                {"name": "cancel_order", "signature": "def cancel_order()"},
            ],
            "base_classes": ["BaseService"],
            "cross_file_usages": [
                {"file": "routes.py", "lines": [10, 20]},
                {"file": "jobs.py", "lines": [5]},
            ],
        }
        hints = compute_hints(graph, "class", symbol_data)
        assert isinstance(hints, list)
        assert len(hints) <= 3

    def test_skeleton_context_hints(self):
        """After viewing skeleton, hints should suggest most interesting modules."""
        graph = self._make_mock_graph()
        symbol_data = {
            "directory_stats": {
                "services": {"files": 12, "functions": 45, "classes": 8},
                "tests": {"files": 10, "functions": 30, "classes": 2},
                "utils": {"files": 3, "functions": 10, "classes": 1},
            },
        }
        hints = compute_hints(graph, "skeleton", symbol_data)
        assert isinstance(hints, list)
        assert len(hints) <= 3
        tool_names = [h["tool"] for h in hints]
        assert all(t == "get_module_context" for t in tool_names)

    def test_skeleton_excludes_test_dirs(self):
        """Skeleton hints should not suggest test directories."""
        graph = self._make_mock_graph()
        symbol_data = {
            "directory_stats": {
                "tests": {"files": 50, "functions": 200, "classes": 10},
                "src": {"files": 5, "functions": 10, "classes": 2},
            },
        }
        hints = compute_hints(graph, "skeleton", symbol_data)
        suggested_modules = [h["args"]["module_path"] for h in hints]
        assert "tests" not in suggested_modules

    def test_module_context_hints(self):
        """After viewing module, hints should suggest important files/classes."""
        graph = self._make_mock_graph()
        symbol_data = {
            "module": "services",
            "symbols": [
                {
                    "file": "orders.py",
                    "functions": [{"name": "process_order"}, {"name": "validate"}],
                    "classes": [{"name": "OrderService", "methods": ["process_order"]}],
                },
                {
                    "file": "users.py",
                    "functions": [{"name": "create_user"}],
                    "classes": [{"name": "UserService", "methods": ["create_user"]}],
                },
            ],
        }
        hints = compute_hints(graph, "module", symbol_data)
        assert isinstance(hints, list)
        assert len(hints) <= 3

    def test_empty_data_returns_empty_hints(self):
        """If there's nothing interesting, return empty list."""
        graph = MagicMock()
        graph._caller_index = {}
        graph._class_method_names = {}
        graph._import_index = {}
        hints = compute_hints(graph, "function", {"name": "foo"})
        assert hints == []

    def test_max_three_hints(self):
        """Never return more than 3 hints regardless of available data."""
        graph = self._make_mock_graph()
        symbol_data = {
            "name": "process_order",
            "class": "OrderService",
            "file": "services/orders.py",
            "calls": [{"name": f"func_{i}", "file": f"file_{i}.py"} for i in range(20)],
            "called_by": [{"name": f"caller_{i}", "file": f"call_{i}.py"} for i in range(20)],
        }
        hints = compute_hints(graph, "function", symbol_data)
        assert len(hints) <= 3

    def test_unknown_context_type_returns_empty(self):
        """Unknown context types should return empty list, not crash."""
        graph = self._make_mock_graph()
        hints = compute_hints(graph, "unknown_type", {"name": "foo"})
        assert hints == []
