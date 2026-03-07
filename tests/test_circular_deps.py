"""Tests for P0 circular dependency false-positive fixes.

Covers:
- External package filtering (pip packages, stdlib)
- Bare npm specifier detection (no ./ or ../ prefix)
- __init__.py self-import skip (package importing own sub-modules)
"""

from unittest.mock import MagicMock

from grafyx.graph import CodebaseGraph


class TestExternalPackageFiltering:
    """Verify that known external packages are filtered from import resolution."""

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._project_path = "/project"
        graph.original_path = "/project"
        graph._lock = MagicMock()
        graph._external_packages = {"requests", "flask", "os", "sys", "json"}
        graph.translate_path = lambda p: str(p) if p else ""
        graph._is_ignored_file_path = lambda p: False
        graph._build_import_index = CodebaseGraph._build_import_index.__get__(graph)
        graph._extract_module_from_import = CodebaseGraph._extract_module_from_import
        graph._extract_symbol_names_from_import = CodebaseGraph._extract_symbol_names_from_import
        return graph

    def test_pip_package_not_resolved_to_local_file(self):
        """'from requests import get' should NOT resolve to a local requests.py."""
        graph = self._make_graph()

        py_main = MagicMock()
        py_main.filepath = "/project/main.py"
        py_main.path = "/project/main.py"
        imp = MagicMock()
        imp.source = "from requests import get"
        py_main.imports = [imp]

        py_requests = MagicMock()
        py_requests.filepath = "/project/requests.py"
        py_requests.path = "/project/requests.py"
        py_requests.imports = []

        py_codebase = MagicMock()
        py_codebase.files = [py_main, py_requests]

        graph._codebases = {"python": py_codebase}
        graph._build_import_index()

        # main.py should NOT be listed as an importer of requests.py
        assert "/project/main.py" not in graph._import_index.get("/project/requests.py", [])

    def test_stdlib_module_not_resolved_to_local_file(self):
        """'import json' should NOT resolve to a local json.py."""
        graph = self._make_graph()

        py_app = MagicMock()
        py_app.filepath = "/project/app.py"
        py_app.path = "/project/app.py"
        imp = MagicMock()
        imp.source = "import json"
        py_app.imports = [imp]

        py_json = MagicMock()
        py_json.filepath = "/project/json.py"
        py_json.path = "/project/json.py"
        py_json.imports = []

        py_codebase = MagicMock()
        py_codebase.files = [py_app, py_json]

        graph._codebases = {"python": py_codebase}
        graph._build_import_index()

        assert "/project/app.py" not in graph._import_index.get("/project/json.py", [])

    def test_local_module_still_resolves_when_not_external(self):
        """Local modules NOT in external packages should still resolve."""
        graph = self._make_graph()

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

        graph._codebases = {"python": py_codebase}
        graph._build_import_index()

        assert "/project/main.py" in graph._import_index.get("/project/models.py", [])


class TestBareNpmSpecifiers:
    """Verify that bare npm imports (no ./ prefix) are skipped."""

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._project_path = "/project"
        graph.original_path = "/project"
        graph._lock = MagicMock()
        graph._external_packages = set()
        graph.translate_path = lambda p: str(p) if p else ""
        graph._is_ignored_file_path = lambda p: False
        graph._build_import_index = CodebaseGraph._build_import_index.__get__(graph)
        graph._extract_module_from_import = CodebaseGraph._extract_module_from_import
        graph._extract_symbol_names_from_import = CodebaseGraph._extract_symbol_names_from_import
        return graph

    def test_bare_npm_import_not_resolved(self):
        """'import { Toaster } from 'sonner'' should NOT match local sonner.tsx."""
        graph = self._make_graph()

        tsx_app = MagicMock()
        tsx_app.filepath = "/project/app.tsx"
        tsx_app.path = "/project/app.tsx"
        imp = MagicMock()
        imp.source = "import { Toaster } from 'sonner'"
        tsx_app.imports = [imp]

        tsx_sonner = MagicMock()
        tsx_sonner.filepath = "/project/sonner.tsx"
        tsx_sonner.path = "/project/sonner.tsx"
        tsx_sonner.imports = []

        ts_codebase = MagicMock()
        ts_codebase.files = [tsx_app, tsx_sonner]

        graph._codebases = {"typescript": ts_codebase}
        graph._build_import_index()

        assert "/project/app.tsx" not in graph._import_index.get("/project/sonner.tsx", [])

    def test_relative_import_still_resolves(self):
        """'import { func } from './helpers'' should resolve to helpers.tsx."""
        graph = self._make_graph()

        tsx_app = MagicMock()
        tsx_app.filepath = "/project/app.tsx"
        tsx_app.path = "/project/app.tsx"
        imp = MagicMock()
        imp.source = "import { func } from './helpers'"
        tsx_app.imports = [imp]

        tsx_helpers = MagicMock()
        tsx_helpers.filepath = "/project/helpers.tsx"
        tsx_helpers.path = "/project/helpers.tsx"
        tsx_helpers.imports = []

        ts_codebase = MagicMock()
        ts_codebase.files = [tsx_app, tsx_helpers]

        graph._codebases = {"typescript": ts_codebase}
        graph._build_import_index()

        assert "/project/app.tsx" in graph._import_index.get("/project/helpers.tsx", [])

    def test_bare_react_import_not_resolved(self):
        """'import React from 'react'' should not match any local file."""
        graph = self._make_graph()

        tsx_comp = MagicMock()
        tsx_comp.filepath = "/project/Button.tsx"
        tsx_comp.path = "/project/Button.tsx"
        imp = MagicMock()
        imp.source = "import React from 'react'"
        tsx_comp.imports = [imp]

        ts_codebase = MagicMock()
        ts_codebase.files = [tsx_comp]

        graph._codebases = {"typescript": ts_codebase}
        graph._build_import_index()

        # No local file should appear as a match
        for key, importers in graph._import_index.items():
            assert "/project/Button.tsx" not in importers


class TestInitPySelfImport:
    """Verify that __init__.py importing own sub-modules is not a circular edge."""

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._project_path = "/project"
        graph.original_path = "/project"
        graph._lock = MagicMock()
        graph._external_packages = set()
        graph.translate_path = lambda p: str(p) if p else ""
        graph._is_ignored_file_path = lambda p: False
        graph._build_import_index = CodebaseGraph._build_import_index.__get__(graph)
        graph._extract_module_from_import = CodebaseGraph._extract_module_from_import
        graph._extract_symbol_names_from_import = CodebaseGraph._extract_symbol_names_from_import
        return graph

    def test_init_py_importing_submodule_skipped(self):
        """mypackage/__init__.py importing from mypackage.sub should not create a cycle edge."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/mypackage/__init__.py"
        init_file.path = "/project/mypackage/__init__.py"
        imp = MagicMock()
        imp.source = "from mypackage.sub import helper"
        init_file.imports = [imp]

        sub_file = MagicMock()
        sub_file.filepath = "/project/mypackage/sub.py"
        sub_file.path = "/project/mypackage/sub.py"
        sub_file.imports = []

        py_codebase = MagicMock()
        py_codebase.files = [init_file, sub_file]

        graph._codebases = {"python": py_codebase}
        graph._build_import_index()

        # __init__.py should NOT be listed as importer of sub.py (self-import)
        init_path = "/project/mypackage/__init__.py"
        sub_path = "/project/mypackage/sub.py"
        assert init_path not in graph._import_index.get(sub_path, [])

    def test_external_import_of_submodule_still_works(self):
        """A file outside the package importing the sub-module should still resolve."""
        graph = self._make_graph()

        main_file = MagicMock()
        main_file.filepath = "/project/main.py"
        main_file.path = "/project/main.py"
        imp = MagicMock()
        imp.source = "from mypackage.sub import helper"
        main_file.imports = [imp]

        sub_file = MagicMock()
        sub_file.filepath = "/project/mypackage/sub.py"
        sub_file.path = "/project/mypackage/sub.py"
        sub_file.imports = []

        py_codebase = MagicMock()
        py_codebase.files = [main_file, sub_file]

        graph._codebases = {"python": py_codebase}
        graph._build_import_index()

        # main.py should be listed as importer of sub.py
        assert "/project/main.py" in graph._import_index.get("/project/mypackage/sub.py", [])
