"""Tests for _build_import_index cross-language filtering."""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph


class TestImportIndexCrossLanguage:
    """Verify that import resolution prefers same-language matches."""

    def _make_graph_with_mixed_files(self):
        """Create a graph mock with both .py and .tsx files sharing a module name."""
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

    def test_python_import_prefers_py_file(self):
        """A Python file importing 'helpers' should resolve to helpers.py, not helpers.tsx."""
        graph = self._make_graph_with_mixed_files()

        # Python file that imports "helpers"
        py_main = MagicMock()
        py_main.filepath = "/project/main.py"
        py_main.path = "/project/main.py"
        imp = MagicMock()
        imp.source = "from helpers import func"
        py_main.imports = [imp]

        py_helpers = MagicMock()
        py_helpers.filepath = "/project/helpers.py"
        py_helpers.path = "/project/helpers.py"
        py_helpers.imports = []

        tsx_helpers = MagicMock()
        tsx_helpers.filepath = "/project/helpers.tsx"
        tsx_helpers.path = "/project/helpers.tsx"
        tsx_helpers.imports = []

        py_codebase = MagicMock()
        py_codebase.files = [py_main, py_helpers]
        ts_codebase = MagicMock()
        ts_codebase.files = [tsx_helpers]

        graph._codebases = {"python": py_codebase, "typescript": ts_codebase}
        graph._build_import_index()

        # helpers.py should be imported by main.py
        assert "/project/main.py" in graph._import_index.get("/project/helpers.py", [])
        # helpers.tsx should NOT be imported by main.py
        assert "/project/main.py" not in graph._import_index.get("/project/helpers.tsx", [])

    def test_ts_import_prefers_tsx_file(self):
        """A TypeScript file importing './helpers' should resolve to helpers.tsx, not helpers.py."""
        graph = self._make_graph_with_mixed_files()

        ts_app = MagicMock()
        ts_app.filepath = "/project/app.tsx"
        ts_app.path = "/project/app.tsx"
        imp = MagicMock()
        imp.source = "import { func } from './helpers'"
        ts_app.imports = [imp]

        tsx_helpers = MagicMock()
        tsx_helpers.filepath = "/project/helpers.tsx"
        tsx_helpers.path = "/project/helpers.tsx"
        tsx_helpers.imports = []

        py_helpers = MagicMock()
        py_helpers.filepath = "/project/helpers.py"
        py_helpers.path = "/project/helpers.py"
        py_helpers.imports = []

        ts_codebase = MagicMock()
        ts_codebase.files = [ts_app, tsx_helpers]
        py_codebase = MagicMock()
        py_codebase.files = [py_helpers]

        graph._codebases = {"python": py_codebase, "typescript": ts_codebase}
        graph._build_import_index()

        # helpers.tsx should be imported by app.tsx
        assert "/project/app.tsx" in graph._import_index.get("/project/helpers.tsx", [])
        # helpers.py should NOT be imported by app.tsx
        assert "/project/app.tsx" not in graph._import_index.get("/project/helpers.py", [])

    def test_cross_language_fallback_blocked_across_families(self):
        """JS -> Python cross-language resolution should be blocked (different families)."""
        graph = self._make_graph_with_mixed_files()

        # JS file imports "utils" but only utils.py exists (no .js/.ts version)
        js_app = MagicMock()
        js_app.filepath = "/project/app.js"
        js_app.path = "/project/app.js"
        imp = MagicMock()
        imp.source = "import { util } from './utils'"
        js_app.imports = [imp]

        py_utils = MagicMock()
        py_utils.filepath = "/project/utils.py"
        py_utils.path = "/project/utils.py"
        py_utils.imports = []

        js_codebase = MagicMock()
        js_codebase.files = [js_app]
        py_codebase = MagicMock()
        py_codebase.files = [py_utils]

        graph._codebases = {"python": py_codebase, "javascript": js_codebase}
        graph._build_import_index()

        # Cross-family (JS -> Python) should NOT resolve — this avoids false
        # positives from coincidental module name overlap like "config" or "utils".
        assert "/project/app.js" not in graph._import_index.get("/project/utils.py", [])

    def test_same_language_import_unchanged(self):
        """Normal same-language imports should work as before."""
        graph = self._make_graph_with_mixed_files()

        py_main = MagicMock()
        py_main.filepath = "/project/main.py"
        py_main.path = "/project/main.py"
        imp = MagicMock()
        imp.source = "from models import User"
        py_main.imports = [imp]

        py_models = MagicMock()
        py_models.filepath = "/project/models.py"
        py_models.path = "/project/models.py"
        py_models.imports = []

        py_codebase = MagicMock()
        py_codebase.files = [py_main, py_models]
        ts_codebase = MagicMock()
        ts_codebase.files = []

        graph._codebases = {"python": py_codebase, "typescript": ts_codebase}
        graph._build_import_index()

        assert "/project/main.py" in graph._import_index.get("/project/models.py", [])
        # Forward index should also be correct
        assert "/project/models.py" in graph._forward_import_index.get("/project/main.py", [])
