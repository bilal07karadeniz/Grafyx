"""Tests for __init__.py re-export tracking.

When package/__init__.py re-exports a symbol from a submodule,
files importing from the package should be linked to the original
defining file in the import index.
"""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph


class TestReExportTracking:

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._project_path = "/project"
        graph.original_path = "/project"
        graph._lock = MagicMock()
        graph._external_packages = set()
        graph.translate_path = lambda p: str(p) if p else ""
        graph._build_import_index = CodebaseGraph._build_import_index.__get__(graph)
        graph._extract_module_from_import = CodebaseGraph._extract_module_from_import
        graph._extract_symbol_names_from_import = CodebaseGraph._extract_symbol_names_from_import
        graph._is_ignored_file_path = lambda p: False
        return graph

    def test_reexport_links_consumer_to_init(self):
        """Files importing from package/__init__.py should appear in
        _import_index for the __init__.py file itself."""
        graph = self._make_graph()

        # package/__init__.py re-exports from .impl (relative import)
        init_file = MagicMock()
        init_file.filepath = "/project/package/__init__.py"
        init_file.path = "/project/package/__init__.py"
        init_imp = MagicMock()
        init_imp.source = "from .impl import ServiceClass"
        init_file.imports = [init_imp]

        impl_file = MagicMock()
        impl_file.filepath = "/project/package/impl.py"
        impl_file.path = "/project/package/impl.py"
        impl_file.imports = []

        # Consumer imports from the package
        consumer = MagicMock()
        consumer.filepath = "/project/app/main.py"
        consumer.path = "/project/app/main.py"
        consumer_imp = MagicMock()
        consumer_imp.source = "from package import ServiceClass"
        consumer.imports = [consumer_imp]

        codebase = MagicMock()
        codebase.files = [init_file, impl_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Consumer should import the __init__.py (how package imports resolve)
        init_importers = graph._import_index.get("/project/package/__init__.py", [])
        assert "/project/app/main.py" in init_importers, (
            f"Consumer should be in init importers, got: {init_importers}"
        )

    def test_init_self_import_recorded_for_reexport(self):
        """__init__.py importing its own submodule should be recorded in the
        import index so that transitive dependencies through re-exports work.

        e.g., router.py -> auth/__init__.py -> auth/auth_service.py
        """
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/package/__init__.py"
        init_file.path = "/project/package/__init__.py"
        init_imp = MagicMock()
        init_imp.source = "from package.impl import ServiceClass"
        init_file.imports = [init_imp]

        impl_file = MagicMock()
        impl_file.filepath = "/project/package/impl.py"
        impl_file.path = "/project/package/impl.py"
        impl_file.imports = []

        codebase = MagicMock()
        codebase.files = [init_file, impl_file]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # __init__.py SHOULD appear as an importer of impl.py (for re-export tracking)
        impl_importers = graph._import_index.get("/project/package/impl.py", [])
        assert "/project/package/__init__.py" in impl_importers
