"""Tests for local module preference in import resolution.

When multiple files match a module name suffix (e.g., config.py exists in
both load_tests/ and backend/app/api/voice/), the resolver should prefer
the file closest to the importing file's directory.
"""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph


class TestLocalModuleResolution:
    """Import resolution should prefer same-directory modules."""

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

    def test_same_dir_config_preferred_over_distant(self):
        """load_tests/worker.py importing 'config' should resolve to
        load_tests/config.py, not backend/app/api/voice/config.py."""
        graph = self._make_graph()

        worker = MagicMock()
        worker.filepath = "/project/load_tests/worker.py"
        worker.path = "/project/load_tests/worker.py"
        imp = MagicMock()
        imp.source = "from config import Settings"
        worker.imports = [imp]

        local_config = MagicMock()
        local_config.filepath = "/project/load_tests/config.py"
        local_config.path = "/project/load_tests/config.py"
        local_config.imports = []

        distant_config = MagicMock()
        distant_config.filepath = "/project/backend/app/api/voice/config.py"
        distant_config.path = "/project/backend/app/api/voice/config.py"
        distant_config.imports = []

        codebase = MagicMock()
        # distant_config listed FIRST so suffix_to_path picks it first
        # without disambiguation — this is the bug scenario
        codebase.files = [worker, distant_config, local_config]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        local_importers = graph._import_index.get("/project/load_tests/config.py", [])
        distant_importers = graph._import_index.get("/project/backend/app/api/voice/config.py", [])
        assert "/project/load_tests/worker.py" in local_importers
        assert "/project/load_tests/worker.py" not in distant_importers

    def test_unambiguous_module_unaffected(self):
        """When only one file matches (e.g., 'database'), resolve as before."""
        graph = self._make_graph()

        main = MagicMock()
        main.filepath = "/project/app/main.py"
        main.path = "/project/app/main.py"
        imp = MagicMock()
        imp.source = "from database import get_db"
        main.imports = [imp]

        db_file = MagicMock()
        db_file.filepath = "/project/app/database.py"
        db_file.path = "/project/app/database.py"
        db_file.imports = []

        codebase = MagicMock()
        codebase.files = [main, db_file]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        assert "/project/app/main.py" in graph._import_index.get("/project/app/database.py", [])
