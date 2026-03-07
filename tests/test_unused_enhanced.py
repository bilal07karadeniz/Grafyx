"""Tests for P1 unused symbols false-positive fixes.

Covers:
- Expanded FRAMEWORK_DECORATORS (websocket, exception_handler, Celery)
- Compound decorator detection (signal.connect patterns)
- Protocol/ABC method skip
- Protocol/ABC class skip in get_unused_classes
"""

from unittest.mock import MagicMock, patch

from grafyx.graph import CodebaseGraph
from grafyx.utils import safe_get_attr


class TestExpandedFrameworkDecorators:
    """Verify that newly added decorator patterns skip unused detection."""

    def _make_graph_with_decorated_func(self, decorator_str):
        """Create a graph with a single function decorated with the given string."""
        graph = MagicMock(spec=CodebaseGraph)
        graph._lock = MagicMock()
        graph._codebases = {}
        graph._caller_index = {}
        graph._class_method_names = {}
        graph._file_class_methods = {}
        graph._class_defined_in = {}
        graph._import_index = {}
        graph._forward_import_index = {}

        decorator = MagicMock()
        decorator.__str__ = lambda self: decorator_str

        func = MagicMock()
        func.name = "my_handler"
        func.filepath = "/project/handlers.py"
        func.decorators = [decorator]
        func.usages = []

        graph.get_all_functions = MagicMock(return_value=[{
            "name": "my_handler",
            "file": "/project/handlers.py",
            "line": 10,
            "language": "python",
        }])
        graph.get_function = MagicMock(return_value=("python", func, None))
        graph._is_test_path = CodebaseGraph._is_test_path
        graph._is_migration_path = CodebaseGraph._is_migration_path
        graph._build_imported_names = MagicMock(return_value=set())
        graph.get_unused_functions = CodebaseGraph.get_unused_functions.__get__(graph)
        graph._get_class_bases = MagicMock(return_value=set())
        graph._ABSTRACT_BASE_NAMES = CodebaseGraph._ABSTRACT_BASE_NAMES
        graph.translate_path = lambda p: str(p) if p else ""
        return graph

    def test_websocket_decorator_skipped(self):
        """@router.websocket('/ws') should not be flagged as unused."""
        graph = self._make_graph_with_decorated_func("@router.websocket('/ws')")
        unused = graph.get_unused_functions()
        assert len(unused) == 0

    def test_exception_handler_decorator_skipped(self):
        """@app.exception_handler(404) should not be flagged as unused."""
        graph = self._make_graph_with_decorated_func("@app.exception_handler(404)")
        unused = graph.get_unused_functions()
        assert len(unused) == 0

    def test_celery_task_decorator_skipped(self):
        """@shared_task should not be flagged as unused."""
        graph = self._make_graph_with_decorated_func("@shared_task")
        unused = graph.get_unused_functions()
        assert len(unused) == 0

    def test_celery_periodic_task_skipped(self):
        """@periodic_task(run_every=...) should not be flagged as unused."""
        graph = self._make_graph_with_decorated_func("@periodic_task(run_every=60)")
        unused = graph.get_unused_functions()
        assert len(unused) == 0

    def test_lifespan_decorator_skipped(self):
        """@app.lifespan should not be flagged as unused."""
        graph = self._make_graph_with_decorated_func("@app.lifespan")
        unused = graph.get_unused_functions()
        assert len(unused) == 0

    def test_middleware_decorator_skipped(self):
        """@app.middleware('http') should not be flagged as unused."""
        graph = self._make_graph_with_decorated_func("@app.middleware('http')")
        unused = graph.get_unused_functions()
        assert len(unused) == 0


class TestCompoundDecoratorDetection:
    """Verify signal.connect compound decorator pattern detection."""

    def _make_graph_with_decorated_func(self, decorator_str):
        graph = MagicMock(spec=CodebaseGraph)
        graph._lock = MagicMock()
        graph._codebases = {}
        graph._caller_index = {}
        graph._class_method_names = {}

        decorator = MagicMock()
        decorator.__str__ = lambda self: decorator_str

        func = MagicMock()
        func.name = "on_worker_init"
        func.filepath = "/project/celery_app.py"
        func.decorators = [decorator]
        func.usages = []

        graph.get_all_functions = MagicMock(return_value=[{
            "name": "on_worker_init",
            "file": "/project/celery_app.py",
            "line": 5,
            "language": "python",
        }])
        graph.get_function = MagicMock(return_value=("python", func, None))
        graph._is_test_path = CodebaseGraph._is_test_path
        graph._is_migration_path = CodebaseGraph._is_migration_path
        graph._build_imported_names = MagicMock(return_value=set())
        graph.get_unused_functions = CodebaseGraph.get_unused_functions.__get__(graph)
        graph._get_class_bases = MagicMock(return_value=set())
        graph._ABSTRACT_BASE_NAMES = CodebaseGraph._ABSTRACT_BASE_NAMES
        graph.translate_path = lambda p: str(p) if p else ""
        return graph

    def test_signal_connect_decorator_skipped(self):
        """@worker_process_init.connect should not be flagged as unused."""
        graph = self._make_graph_with_decorated_func("@worker_process_init.connect")
        unused = graph.get_unused_functions()
        assert len(unused) == 0

    def test_signal_connect_with_args_skipped(self):
        """@post_save.connect(sender=MyModel) should not be flagged as unused."""
        graph = self._make_graph_with_decorated_func("@post_save.connect(sender=MyModel)")
        unused = graph.get_unused_functions()
        assert len(unused) == 0


class TestProtocolMethodSkip:
    """Verify that methods on Protocol/ABC classes are not flagged as unused."""

    def _make_graph_with_protocol_method(self, base_class_name):
        graph = MagicMock(spec=CodebaseGraph)
        graph._lock = MagicMock()
        graph._caller_index = {}
        graph._class_method_names = {"MyInterface": {"do_something"}}

        func = MagicMock()
        func.name = "do_something"
        func.filepath = "/project/interfaces.py"
        func.decorators = []
        func.usages = []

        # Provide a mock codebase with a class whose base_classes can be resolved
        # by extract_base_classes() -- used by the _inh dict in get_unused_functions
        mock_cls = MagicMock()
        mock_cls.name = "MyInterface"
        mock_cls.base_classes = [base_class_name]
        mock_codebase = MagicMock()
        mock_codebase.classes = [mock_cls]
        mock_codebase.functions = []
        graph._codebases = {"python": mock_codebase}

        graph.get_all_functions = MagicMock(return_value=[{
            "name": "do_something",
            "file": "/project/interfaces.py",
            "line": 8,
            "language": "python",
            "class_name": "MyInterface",
        }])
        graph.get_function = MagicMock(return_value=("python", func, "MyInterface"))
        graph._is_test_path = CodebaseGraph._is_test_path
        graph._is_migration_path = CodebaseGraph._is_migration_path
        graph._build_imported_names = MagicMock(return_value=set())
        graph.get_unused_functions = CodebaseGraph.get_unused_functions.__get__(graph)
        graph._ABSTRACT_BASE_NAMES = CodebaseGraph._ABSTRACT_BASE_NAMES
        graph.translate_path = lambda p: str(p) if p else ""
        return graph

    def test_protocol_method_not_flagged(self):
        """Methods on Protocol classes should not be flagged as unused."""
        graph = self._make_graph_with_protocol_method("Protocol")
        unused = graph.get_unused_functions()
        assert len(unused) == 0

    def test_abc_method_not_flagged(self):
        """Methods on ABC classes should not be flagged as unused."""
        graph = self._make_graph_with_protocol_method("ABC")
        unused = graph.get_unused_functions()
        assert len(unused) == 0

    def test_abcmeta_method_not_flagged(self):
        """Methods on ABCMeta classes should not be flagged as unused."""
        graph = self._make_graph_with_protocol_method("ABCMeta")
        unused = graph.get_unused_functions()
        assert len(unused) == 0

    def test_regular_class_method_still_flagged(self):
        """Methods on non-Protocol/ABC classes should still be flagged if unused."""
        graph = self._make_graph_with_protocol_method("SomeOtherBase")
        unused = graph.get_unused_functions()
        assert len(unused) == 1
        assert unused[0]["name"] == "do_something"
