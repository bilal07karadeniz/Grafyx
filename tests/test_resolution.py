"""Tests for resolution filtering."""
import pytest
from grafyx.server._resolution import filter_by_detail, DETAIL_LEVELS


class TestFilterByDetail:
    """Test the shared detail-level filtering helper."""

    def test_valid_levels(self):
        """All three levels should be accepted."""
        assert set(DETAIL_LEVELS) == {"signatures", "summary", "full"}

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Invalid detail level"):
            filter_by_detail({}, "invalid", "function")

    def test_function_signatures_strips_docstring(self):
        """Signatures level should remove docstrings, callers, callees, source."""
        data = {
            "name": "process",
            "signature": "def process(x: int) -> str",
            "parameters": [{"name": "x", "type": "int"}],
            "return_type": "str",
            "docstring": "Process something.",
            "decorators": ["@cache"],
            "calls": [{"name": "helper"}],
            "called_by": [{"name": "main", "file": "app.py"}],
            "source": "def process(x): return str(x)",
            "dependencies": [{"name": "os"}],
            "is_async": False,
        }
        result = filter_by_detail(data, "signatures", "function")
        assert result["name"] == "process"
        assert result["signature"] == "def process(x: int) -> str"
        assert result["parameters"] == [{"name": "x", "type": "int"}]
        assert result["return_type"] == "str"
        assert "docstring" not in result
        assert "decorators" not in result
        assert "calls" not in result
        assert "called_by" not in result
        assert "source" not in result
        assert "dependencies" not in result
        assert "is_async" not in result

    def test_function_summary_keeps_docstring_strips_source(self):
        """Summary level keeps docstring and caller/callee names, strips source."""
        data = {
            "name": "process",
            "signature": "def process(x: int) -> str",
            "docstring": "Process something.\n\nDetailed explanation here.",
            "decorators": ["@cache"],
            "calls": [{"name": "helper", "file": "util.py"}],
            "called_by": [{"name": "main", "file": "app.py"}],
            "source": "def process(x): return str(x)",
        }
        result = filter_by_detail(data, "summary", "function")
        assert result["docstring"] is not None
        assert result["decorators"] == ["@cache"]
        assert "calls" in result
        assert "called_by" in result
        assert "source" not in result

    def test_function_full_keeps_everything(self):
        """Full level should return data unchanged."""
        data = {
            "name": "process",
            "signature": "def process(x)",
            "source": "def process(x): pass",
            "docstring": "Doc",
            "calls": [{"name": "a"}],
        }
        result = filter_by_detail(data, "full", "function")
        assert result["source"] == "def process(x): pass"
        assert result["docstring"] == "Doc"

    def test_file_signatures_strips_method_details(self):
        """File context at signatures level strips function docstrings."""
        data = {
            "path": "app.py",
            "functions": [
                {"name": "run", "signature": "def run()", "docstring": "Run the app.", "line": 1},
            ],
            "classes": [
                {"name": "App", "methods": [{"name": "start", "signature": "def start()", "docstring": "Start."}], "docstring": "Main app."},
            ],
            "imports": ["import os"],
            "imported_by": ["test_app.py"],
            "source": "import os\ndef run(): pass",
        }
        result = filter_by_detail(data, "signatures", "file")
        assert "docstring" not in result["functions"][0]
        assert "docstring" not in result["classes"][0]
        assert "source" not in result
        assert "imported_by" not in result

    def test_class_signatures_strips_details(self):
        """Class context at signatures level: methods have name+sig only."""
        data = {
            "name": "UserService",
            "methods": [
                {"name": "create", "signature": "def create(data)", "docstring": "Create user.", "is_async": False, "line": 10},
            ],
            "properties": [{"name": "db"}],
            "base_classes": ["BaseService"],
            "cross_file_usages": [{"file": "routes.py", "lines": [5]}],
            "source": "class UserService: pass",
        }
        result = filter_by_detail(data, "signatures", "class")
        assert result["methods"][0]["name"] == "create"
        assert result["methods"][0]["signature"] == "def create(data)"
        assert "docstring" not in result["methods"][0]
        assert "is_async" not in result["methods"][0]
        assert "line" not in result["methods"][0]
        assert "cross_file_usages" not in result
        assert "source" not in result

    def test_class_summary_keeps_usages(self):
        """Class context at summary level keeps cross-file usages."""
        data = {
            "name": "UserService",
            "methods": [{"name": "create", "signature": "def create()", "docstring": "Create."}],
            "cross_file_usages": [{"file": "routes.py", "lines": [5]}],
            "source": "class UserService: pass",
        }
        result = filter_by_detail(data, "summary", "class")
        assert "cross_file_usages" in result
        assert "source" not in result

    def test_skeleton_signatures_strips_dir_stats(self):
        """Skeleton at signatures level strips heavy keys.

        Strips directory_stats, by_language, subdir_stats, AND file_tree
        (file_tree was added in v0.2.6 — it's the heaviest field and must
        go away at signatures so the level is meaningfully smaller than
        summary).
        """
        data = {
            "project_path": "/app",
            "languages": ["python"],
            "total_files": 10,
            "total_functions": 20,
            "total_classes": 5,
            "directory_stats": {"src": {"files": 8}},
            "file_tree": "src/\n  app.py",
            "by_language": {"python": {"files": 10}},
        }
        result = filter_by_detail(data, "signatures", "skeleton")
        assert "directory_stats" not in result
        assert "by_language" not in result
        assert "file_tree" not in result
        # Stat totals + project_path remain at signatures
        assert result["total_files"] == 10
        assert result["project_path"] == "/app"

    def test_missing_keys_handled_gracefully(self):
        """Should not crash if data doesn't have all expected keys."""
        data = {"name": "foo"}
        result = filter_by_detail(data, "signatures", "function")
        assert result["name"] == "foo"

    def test_module_signatures(self):
        """Module context at signatures level strips docstrings from symbols."""
        data = {
            "module": "services",
            "files": 3,
            "symbols": [
                {
                    "file": "auth.py",
                    "functions": [{"name": "login", "signature": "def login()", "docstring": "Log in."}],
                    "classes": [{"name": "AuthService", "methods": ["login"], "docstring": "Auth service."}],
                }
            ],
            "internal_imports": [{"from": "auth.py", "imports": ["models.py"]}],
        }
        result = filter_by_detail(data, "signatures", "module")
        assert "docstring" not in result["symbols"][0]["functions"][0]
        assert "docstring" not in result["symbols"][0]["classes"][0]
        assert "internal_imports" not in result

    def test_does_not_mutate_original(self):
        """filter_by_detail should not mutate the original data dict."""
        data = {
            "name": "process",
            "source": "def process(): pass",
            "docstring": "Doc",
        }
        original_keys = set(data.keys())
        filter_by_detail(data, "signatures", "function")
        assert set(data.keys()) == original_keys
