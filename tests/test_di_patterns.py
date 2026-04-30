"""Tests for DI/callback pattern detection in caller index."""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph


class TestDIPatternDetection:
    """Test that function references passed as arguments are detected."""

    def _make_graph(self):
        """Create a graph mock with real caller index methods."""
        graph = MagicMock(spec=CodebaseGraph)
        graph._project_path = "/project"
        graph.original_path = "/project"
        graph._lock = MagicMock()
        graph.translate_path = lambda p: str(p) if p else ""

        # Bind real methods
        graph._build_caller_index = CodebaseGraph._build_caller_index.__get__(graph)
        graph._index_calls_from = CodebaseGraph._index_calls_from.__get__(graph)
        graph._augment_index_with_di_patterns = CodebaseGraph._augment_index_with_di_patterns.__get__(graph)
        graph._scan_di_refs = CodebaseGraph._scan_di_refs.__get__(graph)
        graph._augment_index_with_local_var_types = CodebaseGraph._augment_index_with_local_var_types.__get__(graph)
        graph._scan_local_var_types = CodebaseGraph._scan_local_var_types.__get__(graph)
        graph._is_ignored_file_path = lambda p: False
        return graph

    def test_depends_pattern_detected(self):
        """Depends(get_current_admin) should create a caller entry."""
        graph = self._make_graph()

        # The DI-target function
        get_admin = MagicMock()
        get_admin.name = "get_current_admin"
        get_admin.filepath = "/project/auth.py"
        get_admin.function_calls = []
        get_admin.source = "def get_current_admin(token):\n    return verify(token)"

        # The route function that uses Depends()
        route_func = MagicMock()
        route_func.name = "list_users"
        route_func.filepath = "/project/routes.py"
        route_func.function_calls = []
        route_func.source = (
            "async def list_users(admin = Depends(get_current_admin)):\n"
            "    return await db.get_users()"
        )

        py_codebase = MagicMock()
        py_codebase.functions = [get_admin, route_func]
        py_codebase.classes = []

        graph._codebases = {"python": py_codebase}

        graph._build_caller_index()

        callers = graph._caller_index.get("get_current_admin", [])
        caller_names = [c["name"] for c in callers]
        assert "list_users" in caller_names

    def test_callback_keyword_pattern(self):
        """callback=handler should create a caller entry."""
        graph = self._make_graph()

        handler = MagicMock()
        handler.name = "on_message"
        handler.filepath = "/project/handlers.py"
        handler.function_calls = []
        handler.source = "def on_message(msg):\n    process(msg)"

        setup = MagicMock()
        setup.name = "setup_ws"
        setup.filepath = "/project/ws.py"
        setup.function_calls = []
        setup.source = "def setup_ws():\n    ws.connect(callback=on_message)\n"

        py_codebase = MagicMock()
        py_codebase.functions = [handler, setup]
        py_codebase.classes = []

        graph._codebases = {"python": py_codebase}

        graph._build_caller_index()

        callers = graph._caller_index.get("on_message", [])
        caller_names = [c["name"] for c in callers]
        assert "setup_ws" in caller_names

    def test_unknown_identifiers_not_indexed(self):
        """Variable names that aren't known functions should not be indexed."""
        graph = self._make_graph()

        func = MagicMock()
        func.name = "process"
        func.filepath = "/project/main.py"
        func.function_calls = []
        func.source = "def process():\n    result = compute(data_variable)\n    return result"

        py_codebase = MagicMock()
        py_codebase.functions = [func]
        py_codebase.classes = []

        graph._codebases = {"python": py_codebase}

        graph._build_caller_index()

        # data_variable is not a known function, should not appear
        assert "data_variable" not in graph._caller_index

    def test_builtin_names_skipped(self):
        """Python builtins like self, None, True should not be indexed."""
        graph = self._make_graph()

        func = MagicMock()
        func.name = "do_stuff"
        func.filepath = "/project/main.py"
        func.function_calls = []
        func.source = "def do_stuff():\n    x = isinstance(None, type)\n    return True"

        py_codebase = MagicMock()
        py_codebase.functions = [func]
        py_codebase.classes = []

        graph._codebases = {"python": py_codebase}

        graph._build_caller_index()

        assert "None" not in graph._caller_index
        assert "True" not in graph._caller_index
        assert "self" not in graph._caller_index

    def test_no_duplicate_entries(self):
        """Same caller should not appear twice for the same reference."""
        graph = self._make_graph()

        target = MagicMock()
        target.name = "authenticate"
        target.filepath = "/project/auth.py"
        target.function_calls = []
        target.source = "def authenticate(token):\n    pass"

        # This function references authenticate twice
        caller = MagicMock()
        caller.name = "handler"
        caller.filepath = "/project/api.py"
        caller.function_calls = []
        caller.source = (
            "def handler():\n"
            "    a = Depends(authenticate)\n"
            "    b = Security(authenticate)\n"
        )

        py_codebase = MagicMock()
        py_codebase.functions = [target, caller]
        py_codebase.classes = []

        graph._codebases = {"python": py_codebase}

        graph._build_caller_index()

        callers = graph._caller_index.get("authenticate", [])
        handler_entries = [c for c in callers if c["name"] == "handler"]
        assert len(handler_entries) == 1

    def test_async_factory_pattern_resolves_method_calls(self):
        """`coord = await get_coordinator()` should link `coord.method()` to the class."""
        graph = self._make_graph()

        # Factory function returns WorkerCoordinator
        factory = MagicMock()
        factory.name = "get_coordinator"
        factory.filepath = "/project/factories.py"
        factory.function_calls = []
        factory.return_type = "WorkerCoordinator"
        factory.source = "async def get_coordinator() -> WorkerCoordinator:\n    return _instance"

        # Caller awaits the factory then calls a method on the result
        caller = MagicMock()
        caller.name = "send_transcript"
        caller.filepath = "/project/api/voice/helpers.py"
        caller.function_calls = []
        caller.return_type = ""
        caller.source = (
            "async def send_transcript(text):\n"
            "    coord = await get_coordinator()\n"
            "    await coord.broadcast_transcript(text)\n"
        )

        # The class with the method
        cls = MagicMock()
        cls.name = "WorkerCoordinator"
        cls.filepath = "/project/voice/coordinator.py"
        broadcast_method = MagicMock()
        broadcast_method.name = "broadcast_transcript"
        broadcast_method.filepath = "/project/voice/coordinator.py"
        broadcast_method.function_calls = []
        broadcast_method.source = "async def broadcast_transcript(self, text): pass"
        cls.methods = [broadcast_method]

        py_codebase = MagicMock()
        py_codebase.functions = [factory, caller]
        py_codebase.classes = [cls]

        graph._codebases = {"python": py_codebase}
        graph._class_method_names = {"WorkerCoordinator": {"broadcast_transcript"}}
        graph._caller_index = {}

        graph._augment_index_with_local_var_types()

        callers = graph._caller_index.get("broadcast_transcript", [])
        caller_names = [c["name"] for c in callers]
        assert "send_transcript" in caller_names, (
            f"async-factory-resolved caller missing; got {caller_names}"
        )

    def test_method_di_includes_class(self):
        """DI refs from class methods should include the class name."""
        graph = self._make_graph()

        target = MagicMock()
        target.name = "get_db"
        target.filepath = "/project/deps.py"
        target.function_calls = []
        target.source = "def get_db():\n    pass"

        method = MagicMock()
        method.name = "create_user"
        method.filepath = "/project/api.py"
        method.function_calls = []
        method.source = "def create_user(db=Depends(get_db)):\n    pass"

        cls = MagicMock()
        cls.name = "UserRouter"
        cls.filepath = "/project/api.py"
        cls.methods = [method]

        py_codebase = MagicMock()
        py_codebase.functions = [target]
        py_codebase.classes = [cls]

        graph._codebases = {"python": py_codebase}

        graph._build_caller_index()

        callers = graph._caller_index.get("get_db", [])
        method_caller = next((c for c in callers if c["name"] == "create_user"), None)
        assert method_caller is not None
        assert method_caller.get("class") == "UserRouter"
