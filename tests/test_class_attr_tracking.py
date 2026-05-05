"""Tests for class attribute type tracking in caller index.

Covers:
- self.field.method() calls where field type is declared in __init__ or class body
- Should add trusted caller entries so methods aren't falsely flagged as unused
"""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph
from grafyx.utils import safe_get_attr


class TestClassAttributeTypeTracking:
    """Methods called via self.typed_field.method() should be tracked."""

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._lock = MagicMock()
        graph._caller_index = {}
        graph._class_method_names = {
            "ToolExecutor": {"execute", "validate"},
            "OrchestratorService": {"run_tools"},
        }
        graph._file_class_methods = {}
        graph._class_defined_in = {}
        graph._external_packages = set()
        graph.translate_path = lambda p: str(p) if p else ""
        graph._is_ignored_file_path = lambda p: False
        graph._codebases = {}
        return graph

    def test_self_field_method_call_tracked(self):
        """self.executor.execute() should create a caller for ToolExecutor.execute
        when __init__ has self.executor = ToolExecutor(...)."""
        graph = self._make_graph()

        init_method = MagicMock()
        init_method.name = "__init__"
        init_method.filepath = "/project/service.py"
        init_method.source = (
            "def __init__(self):\n"
            "    self.executor = ToolExecutor(config)\n"
        )

        run_method = MagicMock()
        run_method.name = "run_tools"
        run_method.filepath = "/project/service.py"
        run_method.source = (
            "def run_tools(self, tools):\n"
            "    result = self.executor.execute(tools)\n"
            "    return result\n"
        )
        run_method.function_calls = []

        cls = MagicMock()
        cls.name = "OrchestratorService"
        cls.filepath = "/project/service.py"
        cls.methods = [init_method, run_method]

        codebase = MagicMock()
        codebase.functions = []
        codebase.classes = [cls]
        graph._codebases = {"python": codebase}

        graph._augment_index_with_class_attr_types = (
            CodebaseGraph._augment_index_with_class_attr_types.__get__(graph)
        )
        graph._augment_index_with_class_attr_types()

        callers = graph._caller_index.get("execute", [])
        assert any(
            c["name"] == "run_tools" and c.get("_trusted")
            for c in callers
        ), f"Expected trusted caller from run_tools, got: {callers}"

    def test_typed_annotation_field(self):
        """self.executor: ToolExecutor in __init__ should also work."""
        graph = self._make_graph()

        init_method = MagicMock()
        init_method.name = "__init__"
        init_method.filepath = "/project/service.py"
        init_method.source = (
            "def __init__(self, executor: ToolExecutor):\n"
            "    self.executor: ToolExecutor = executor\n"
        )

        use_method = MagicMock()
        use_method.name = "process"
        use_method.filepath = "/project/service.py"
        use_method.source = (
            "def process(self):\n"
            "    self.executor.validate(data)\n"
        )
        use_method.function_calls = []

        cls = MagicMock()
        cls.name = "MyService"
        cls.filepath = "/project/service.py"
        cls.methods = [init_method, use_method]

        codebase = MagicMock()
        codebase.functions = []
        codebase.classes = [cls]
        graph._codebases = {"python": codebase}

        graph._class_method_names["MyService"] = {"process", "__init__"}
        graph._augment_index_with_class_attr_types = (
            CodebaseGraph._augment_index_with_class_attr_types.__get__(graph)
        )
        graph._augment_index_with_class_attr_types()

        callers = graph._caller_index.get("validate", [])
        assert any(
            c["name"] == "process" and c.get("_trusted")
            for c in callers
        ), f"Expected trusted caller from process, got: {callers}"

    def test_no_false_positives_untyped_field(self):
        """self.something.method() should NOT be tracked if 'something' has no known type."""
        graph = self._make_graph()

        init_method = MagicMock()
        init_method.name = "__init__"
        init_method.filepath = "/project/service.py"
        init_method.source = (
            "def __init__(self):\n"
            "    self.something = get_unknown()\n"
        )

        use_method = MagicMock()
        use_method.name = "do_work"
        use_method.filepath = "/project/service.py"
        use_method.source = (
            "def do_work(self):\n"
            "    self.something.execute()\n"
        )

        cls = MagicMock()
        cls.name = "SomeService"
        cls.filepath = "/project/service.py"
        cls.methods = [init_method, use_method]

        codebase = MagicMock()
        codebase.functions = []
        codebase.classes = [cls]
        graph._codebases = {"python": codebase}

        graph._augment_index_with_class_attr_types = (
            CodebaseGraph._augment_index_with_class_attr_types.__get__(graph)
        )
        graph._augment_index_with_class_attr_types()

        callers = graph._caller_index.get("execute", [])
        assert len(callers) == 0, f"Expected no callers for untyped field, got: {callers}"

    def test_async_factory_class_attr_resolves(self):
        """self.coord = await get_coordinator() should resolve to
        WorkerCoordinator when get_coordinator() is annotated as
        returning WorkerCoordinator (audit Test 9 — v0.2.2 fix)."""
        graph = self._make_graph()
        graph._class_method_names["WorkerCoordinator"] = {"shutdown", "start"}

        # Factory function with return-type annotation -> seeds
        # _factory_return_types so Pass 3b can resolve it.
        factory_fn = MagicMock()
        factory_fn.name = "get_coordinator"
        factory_fn.return_type = "WorkerCoordinator"

        init_method = MagicMock()
        init_method.name = "__init__"
        init_method.filepath = "/project/service.py"
        init_method.source = (
            "async def __init__(self):\n"
            "    self.coord = await get_coordinator()\n"
        )

        use_method = MagicMock()
        use_method.name = "stop"
        use_method.filepath = "/project/service.py"
        use_method.source = (
            "async def stop(self):\n"
            "    await self.coord.shutdown()\n"
        )
        use_method.function_calls = []

        cls = MagicMock()
        cls.name = "WorkerService"
        cls.filepath = "/project/service.py"
        cls.methods = [init_method, use_method]
        cls.source = "class WorkerService:\n    pass\n"

        codebase = MagicMock()
        codebase.functions = [factory_fn]
        codebase.classes = [cls]
        graph._codebases = {"python": codebase}
        graph._class_method_names["WorkerService"] = {"__init__", "stop"}

        graph._augment_index_with_class_attr_types = (
            CodebaseGraph._augment_index_with_class_attr_types.__get__(graph)
        )
        graph._augment_index_with_class_attr_types()

        callers = graph._caller_index.get("shutdown", [])
        assert any(
            c["name"] == "stop" and c.get("_trusted")
            for c in callers
        ), f"Expected stop() to be tracked as caller of WorkerCoordinator.shutdown via async factory, got: {callers}"

    def test_class_body_typed_attribute_resolves(self):
        """``coord: WorkerCoordinator = None`` declared at class
        scope should seed field_types so self.coord.shutdown() resolves."""
        graph = self._make_graph()
        graph._class_method_names["WorkerCoordinator"] = {"shutdown"}

        # Field declared at class body, no method assignment.
        use_method = MagicMock()
        use_method.name = "stop"
        use_method.filepath = "/project/service.py"
        use_method.source = (
            "def stop(self):\n"
            "    self.coord.shutdown()\n"
        )
        use_method.function_calls = []

        cls = MagicMock()
        cls.name = "Worker"
        cls.filepath = "/project/service.py"
        cls.methods = [use_method]
        cls.source = (
            "class Worker:\n"
            "    coord: WorkerCoordinator = None\n"
            "\n"
            "    def stop(self):\n"
            "        self.coord.shutdown()\n"
        )

        codebase = MagicMock()
        codebase.functions = []
        codebase.classes = [cls]
        graph._codebases = {"python": codebase}
        graph._class_method_names["Worker"] = {"stop"}

        graph._augment_index_with_class_attr_types = (
            CodebaseGraph._augment_index_with_class_attr_types.__get__(graph)
        )
        graph._augment_index_with_class_attr_types()

        callers = graph._caller_index.get("shutdown", [])
        assert any(
            c["name"] == "stop" and c.get("_trusted")
            for c in callers
        ), f"Expected class-body-typed attr to resolve, got: {callers}"
