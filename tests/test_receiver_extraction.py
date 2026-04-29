"""Tests for receiver token extraction and has_dot_syntax flag in caller entries.

Verifies that get_callers() exposes `receiver_token` and `has_dot_syntax`
fields extracted from the internal `_receivers` and `_has_dot_syntax` data
built by `_index_calls_from()`.
"""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph
from grafyx.graph._callers import _extract_immediate_receiver


def _make_graph():
    """Create a minimal CodebaseGraph mock with real index + caller methods."""
    graph = MagicMock(spec=CodebaseGraph)
    graph._project_path = "/project"
    graph.original_path = "/project"
    graph._lock = MagicMock()
    graph._external_packages = set()
    graph._caller_index = {}
    graph._class_method_names = {}
    graph._file_class_methods = {}
    graph._class_defined_in = {}
    graph._import_index = {}
    graph._forward_import_index = {}
    graph._file_symbol_imports = {}
    graph.translate_path = lambda p: str(p) if p else ""
    graph._is_ignored_file_path = lambda p: False
    # Bind the real methods
    graph._index_calls_from = CodebaseGraph._index_calls_from.__get__(graph)
    graph.get_callers = CodebaseGraph.get_callers.__get__(graph)
    return graph


def _make_func(name, filepath, source, callees, parent_class=None):
    """Create a mock function with function_calls and source."""
    func = MagicMock()
    func.name = name
    func.filepath = filepath
    func.source = source
    func.parent_class = parent_class

    calls = []
    for callee_name in callees:
        c = MagicMock()
        c.name = callee_name
        calls.append(c)
    func.function_calls = calls
    return func


class TestExtractImmediateReceiver:
    """Unit tests for _extract_immediate_receiver helper."""

    def test_none_input(self):
        assert _extract_immediate_receiver(None) is None

    def test_empty_set(self):
        assert _extract_immediate_receiver(set()) is None

    def test_bare_self(self):
        """Bare 'self' alone should return None."""
        assert _extract_immediate_receiver({"self"}) is None

    def test_self_dot_attr(self):
        """self.db -> 'db'."""
        assert _extract_immediate_receiver({"self.db"}) == "db"

    def test_self_dot_chain(self):
        """self.db.session -> 'db' (first part after self)."""
        assert _extract_immediate_receiver({"self.db.session"}) == "db"

    def test_plain_variable(self):
        """response -> 'response'."""
        assert _extract_immediate_receiver({"response"}) == "response"

    def test_dotted_variable(self):
        """app.config -> 'app'."""
        assert _extract_immediate_receiver({"app.config"}) == "app"

    def test_multiple_receivers_picks_deepest(self):
        """When multiple receivers, pick the most specific (longest chain)."""
        result = _extract_immediate_receiver({"self", "self.db.session"})
        assert result == "db"

    def test_multiple_non_self_picks_deepest(self):
        """Among non-self receivers, pick deepest."""
        result = _extract_immediate_receiver({"response", "app.config.db"})
        assert result == "app"


class TestHasDotSyntaxFlag:
    """Verify _has_dot_syntax is set correctly on every entry."""

    def test_dot_syntax_true_for_method_call(self):
        """db.refresh(obj) should set _has_dot_syntax=True."""
        graph = _make_graph()
        func = _make_func(
            "my_handler", "/project/handlers.py",
            "def my_handler(db):\n    db.refresh(obj)\n",
            ["refresh"],
        )
        graph._index_calls_from(func, graph._caller_index)

        entries = graph._caller_index.get("refresh", [])
        assert len(entries) == 1
        assert entries[0]["_has_dot_syntax"] is True

    def test_dot_syntax_false_for_standalone_call(self):
        """process_data() should set _has_dot_syntax=False."""
        graph = _make_graph()
        func = _make_func(
            "my_handler", "/project/handlers.py",
            "def my_handler():\n    process_data()\n",
            ["process_data"],
        )
        graph._index_calls_from(func, graph._caller_index)

        entries = graph._caller_index.get("process_data", [])
        assert len(entries) == 1
        assert entries[0]["_has_dot_syntax"] is False

    def test_self_validate_has_dot_syntax(self):
        """self.validate() is dot syntax even though receiver is bare self."""
        graph = _make_graph()
        parent = MagicMock()
        parent.name = "MyModel"
        func = _make_func(
            "save", "/project/models.py",
            "def save(self):\n    self.validate()\n",
            ["validate"],
            parent_class=parent,
        )
        graph._index_calls_from(func, graph._caller_index)

        entries = graph._caller_index.get("validate", [])
        assert len(entries) == 1
        assert entries[0]["_has_dot_syntax"] is True


class TestReceiverTokenInGetCallers:
    """Verify get_callers() output includes receiver_token and has_dot_syntax."""

    def test_db_refresh_receiver(self):
        """db.refresh(obj) -> receiver_token='db', has_dot_syntax=True."""
        graph = _make_graph()
        func = _make_func(
            "my_handler", "/project/handlers.py",
            "def my_handler(db):\n    db.refresh(obj)\n",
            ["refresh"],
        )
        graph._index_calls_from(func, graph._caller_index)
        result = graph.get_callers("refresh")

        assert len(result) == 1
        assert result[0]["receiver_token"] == "db"
        assert result[0]["has_dot_syntax"] is True

    def test_standalone_call(self):
        """process_data() -> receiver_token=None, has_dot_syntax=False."""
        graph = _make_graph()
        func = _make_func(
            "my_handler", "/project/handlers.py",
            "def my_handler():\n    process_data()\n",
            ["process_data"],
        )
        graph._index_calls_from(func, graph._caller_index)
        result = graph.get_callers("process_data")

        assert len(result) == 1
        assert result[0]["receiver_token"] is None
        assert result[0]["has_dot_syntax"] is False

    def test_bare_self_validate(self):
        """self.validate() -> receiver_token=None (bare self doesn't count), has_dot_syntax=True."""
        graph = _make_graph()
        parent = MagicMock()
        parent.name = "MyModel"
        func = _make_func(
            "save", "/project/models.py",
            "def save(self):\n    self.validate()\n",
            ["validate"],
            parent_class=parent,
        )
        graph._index_calls_from(func, graph._caller_index)
        result = graph.get_callers("validate")

        assert len(result) == 1
        assert result[0]["receiver_token"] is None
        assert result[0]["has_dot_syntax"] is True

    def test_self_db_query(self):
        """self.db.query() -> receiver_token='db', has_dot_syntax=True."""
        graph = _make_graph()
        parent = MagicMock()
        parent.name = "Repository"
        func = _make_func(
            "find_all", "/project/repo.py",
            "def find_all(self):\n    self.db.query()\n",
            ["query"],
            parent_class=parent,
        )
        graph._index_calls_from(func, graph._caller_index)
        result = graph.get_callers("query")

        assert len(result) == 1
        assert result[0]["receiver_token"] == "db"
        assert result[0]["has_dot_syntax"] is True

    def test_response_json(self):
        """response.json() -> receiver_token='response', has_dot_syntax=True."""
        graph = _make_graph()
        func = _make_func(
            "fetch_data", "/project/client.py",
            "def fetch_data():\n    response.json()\n",
            ["json"],
        )
        graph._index_calls_from(func, graph._caller_index)
        result = graph.get_callers("json")

        assert len(result) == 1
        assert result[0]["receiver_token"] == "response"
        assert result[0]["has_dot_syntax"] is True

    def test_internal_fields_stripped(self):
        """Output should not contain _receivers or _has_dot_syntax keys."""
        graph = _make_graph()
        func = _make_func(
            "handler", "/project/app.py",
            "def handler(db):\n    db.refresh(x)\n",
            ["refresh"],
        )
        graph._index_calls_from(func, graph._caller_index)
        result = graph.get_callers("refresh")

        assert len(result) == 1
        for key in result[0]:
            assert not key.startswith("_"), f"Internal key {key!r} leaked to output"

    def test_standard_fields_preserved(self):
        """Output should still include name and file."""
        graph = _make_graph()
        func = _make_func(
            "handler", "/project/app.py",
            "def handler():\n    do_stuff()\n",
            ["do_stuff"],
        )
        graph._index_calls_from(func, graph._caller_index)
        result = graph.get_callers("do_stuff")

        assert len(result) == 1
        assert result[0]["name"] == "handler"
        assert result[0]["file"] == "/project/app.py"

    def test_mixed_calls_in_same_function(self):
        """A function with both dot-syntax and standalone calls."""
        graph = _make_graph()
        func = _make_func(
            "process", "/project/service.py",
            "def process(db):\n    db.refresh(obj)\n    notify()\n",
            ["refresh", "notify"],
        )
        graph._index_calls_from(func, graph._caller_index)

        refresh_callers = graph.get_callers("refresh")
        assert len(refresh_callers) == 1
        assert refresh_callers[0]["receiver_token"] == "db"
        assert refresh_callers[0]["has_dot_syntax"] is True

        notify_callers = graph.get_callers("notify")
        assert len(notify_callers) == 1
        assert notify_callers[0]["receiver_token"] is None
        assert notify_callers[0]["has_dot_syntax"] is False


class TestMethodNameCollisionFilter:
    """When querying a standalone function whose name collides with a class
    method, dot-syntax callers (obj.method()) should be filtered out."""

    def test_standalone_with_dot_collision_filtered(self):
        """get_callers('refresh') with no class_name should filter db.refresh() callers."""
        graph = _make_graph()
        # Register 'refresh' as a class method of Session
        graph._class_method_names = {"Session": {"refresh", "commit", "close"}}

        # Two callers: one dot-syntax (db.refresh), one standalone (refresh())
        graph._caller_index = {
            "refresh": [
                {"name": "handler_a", "file": "/project/a.py", "_has_dot_syntax": True, "_receivers": {"db"}},
                {"name": "handler_b", "file": "/project/b.py", "_has_dot_syntax": False, "_receivers": None},
            ],
        }
        result = graph.get_callers("refresh")
        names = [r["name"] for r in result]
        # Dot-syntax caller filtered, standalone kept
        assert "handler_b" in names
        assert "handler_a" not in names

    def test_standalone_without_collision_all_returned(self):
        """get_callers('unique_func') with no class collision should return all callers."""
        graph = _make_graph()
        graph._class_method_names = {"Session": {"refresh", "commit"}}

        graph._caller_index = {
            "unique_func": [
                {"name": "caller_x", "file": "/project/x.py", "_has_dot_syntax": True, "_receivers": {"obj"}},
                {"name": "caller_y", "file": "/project/y.py", "_has_dot_syntax": False, "_receivers": None},
            ],
        }
        result = graph.get_callers("unique_func")
        names = [r["name"] for r in result]
        # No collision with class methods -> all returned
        assert "caller_x" in names
        assert "caller_y" in names

    def test_trusted_entries_bypass_filter(self):
        """Trusted callers should be kept even with dot-syntax and collision."""
        graph = _make_graph()
        graph._class_method_names = {"Session": {"refresh"}}

        graph._caller_index = {
            "refresh": [
                {"name": "handler_trusted", "file": "/project/t.py", "_has_dot_syntax": True,
                 "_receivers": {"session"}, "_trusted": True},
                {"name": "handler_dot", "file": "/project/d.py", "_has_dot_syntax": True,
                 "_receivers": {"db"}, "_trusted": False},
                {"name": "handler_plain", "file": "/project/p.py", "_has_dot_syntax": False,
                 "_receivers": None},
            ],
        }
        result = graph.get_callers("refresh")
        names = [r["name"] for r in result]
        # Trusted kept, non-trusted dot-syntax filtered, standalone kept
        assert "handler_trusted" in names
        assert "handler_plain" in names
        assert "handler_dot" not in names

    def test_external_library_method_detected(self):
        """External library methods (e.g. SQLAlchemy Session.refresh) should
        be detected via caller dot-syntax ratio, even without _class_method_names."""
        graph = _make_graph()
        # No codebase class defines 'refresh' — it's from an external library
        graph._class_method_names = {}

        # 4+ callers, all using dot syntax (db.refresh(), session.refresh(), etc.)
        graph._caller_index = {
            "refresh": [
                {"name": f"handler_{i}", "file": f"/project/{i}.py",
                 "_has_dot_syntax": True, "_receivers": {"db"}}
                for i in range(5)
            ],
        }
        result = graph.get_callers("refresh")
        # >80% dot syntax → detected as external method → all filtered
        assert len(result) == 0

    def test_mixed_callers_below_threshold_not_filtered(self):
        """When dot-syntax ratio is below 80%, no filtering applied."""
        graph = _make_graph()
        graph._class_method_names = {}

        graph._caller_index = {
            "process": [
                {"name": "a", "file": "/project/a.py", "_has_dot_syntax": True, "_receivers": {"mod"}},
                {"name": "b", "file": "/project/b.py", "_has_dot_syntax": False, "_receivers": None},
                {"name": "c", "file": "/project/c.py", "_has_dot_syntax": False, "_receivers": None},
                {"name": "d", "file": "/project/d.py", "_has_dot_syntax": False, "_receivers": None},
            ],
        }
        result = graph.get_callers("process")
        # 25% dot syntax → below threshold → all returned
        assert len(result) == 4
