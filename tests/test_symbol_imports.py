"""Tests for symbol-level import index and dependency graph filtering.

Covers:
- _extract_symbol_names_from_import() for Python and TS/JS patterns
- _file_symbol_imports population in _build_import_index()
- Symbol-level filter in get_dependency_graph (server.py)
"""

from unittest.mock import MagicMock, patch

from grafyx.graph import CodebaseGraph
from tests._tool_compat import call_tool


class TestExtractSymbolNames:
    """Test _extract_symbol_names_from_import() for various import patterns."""

    def test_python_from_import(self):
        """from module import A, B, C → {A, B, C}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "from auth.jwt import create_token, verify_token"
        )
        assert names == {"create_token", "verify_token"}

    def test_python_from_import_with_alias(self):
        """from module import Foo as Bar → {Foo, Bar}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "from models import User as AppUser"
        )
        assert names == {"User", "AppUser"}

    def test_python_wildcard_import(self):
        """from module import * → empty set (wildcard)"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "from models import *"
        )
        assert names == set()

    def test_python_bare_import(self):
        """import os → empty set (no specific symbols)"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "import os"
        )
        assert names == set()

    def test_ts_named_imports(self):
        """import { A, B } from './module' → {A, B}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "import { useState, useEffect } from 'react'"
        )
        assert names == {"useState", "useEffect"}

    def test_ts_named_imports_with_alias(self):
        """import { Foo as Bar } from './module' → {Foo, Bar}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "import { Config as AppConfig } from './config'"
        )
        assert names == {"Config", "AppConfig"}

    def test_ts_default_import(self):
        """import React from 'react' → {React}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "import React from 'react'"
        )
        assert names == {"React"}

    def test_ts_wildcard_import(self):
        """import * as utils from './utils' → empty set"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "import * as utils from './utils'"
        )
        assert names == set()

    def test_ts_type_import(self):
        """import type { Foo } from './types' → {Foo}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "import type { UserProfile } from './types'"
        )
        assert names == {"UserProfile"}

    def test_empty_string(self):
        """Empty string → empty set."""
        names = CodebaseGraph._extract_symbol_names_from_import("")
        assert names == set()


class TestFileSymbolImports:
    """Test that _file_symbol_imports is correctly populated."""

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

    def test_symbol_names_tracked(self):
        """Symbol names from imports should be stored in _file_symbol_imports."""
        graph = self._make_graph()

        py_main = MagicMock()
        py_main.filepath = "/project/main.py"
        py_main.path = "/project/main.py"
        imp = MagicMock()
        imp.source = "from utils import helper_a, helper_b"
        py_main.imports = [imp]

        py_utils = MagicMock()
        py_utils.filepath = "/project/utils.py"
        py_utils.path = "/project/utils.py"
        py_utils.imports = []

        py_codebase = MagicMock()
        py_codebase.files = [py_main, py_utils]

        graph._codebases = {"python": py_codebase}
        graph._build_import_index()

        sym_imports = graph._file_symbol_imports
        assert "/project/main.py" in sym_imports
        assert "/project/utils.py" in sym_imports["/project/main.py"]
        symbols = sym_imports["/project/main.py"]["/project/utils.py"]
        assert "helper_a" in symbols
        assert "helper_b" in symbols

    def test_wildcard_import_not_tracked(self):
        """Wildcard imports should NOT have a symbol entry (treated conservatively)."""
        graph = self._make_graph()

        py_main = MagicMock()
        py_main.filepath = "/project/main.py"
        py_main.path = "/project/main.py"
        imp = MagicMock()
        imp.source = "from utils import *"
        py_main.imports = [imp]

        py_utils = MagicMock()
        py_utils.filepath = "/project/utils.py"
        py_utils.path = "/project/utils.py"
        py_utils.imports = []

        py_codebase = MagicMock()
        py_codebase.files = [py_main, py_utils]

        graph._codebases = {"python": py_codebase}
        graph._build_import_index()

        sym_imports = graph._file_symbol_imports
        # Wildcard has no specific symbols → not tracked in symbol_imports
        # This means the dependency graph filter includes it conservatively
        if "/project/main.py" in sym_imports:
            assert sym_imports["/project/main.py"].get("/project/utils.py", set()) == set()
        # Also valid: main.py not in sym_imports at all


class TestDependencyGraphSymbolFilter:
    """Test that get_dependency_graph filters importers by symbol name."""

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_importer_filtered_when_importing_different_symbol(self, mock_graph):
        """A file importing other_func from same module should NOT appear
        as a dependent of target_func."""
        from grafyx.server import get_dependency_graph
        from grafyx.utils import safe_get_attr

        mock_graph.initialized = True

        # Symbol under test
        symbol = MagicMock()
        symbol.name = "target_func"
        symbol.filepath = "/project/utils.py"
        safe_get_attr_orig = safe_get_attr

        mock_graph.get_symbol.return_value = ("python", symbol, "function")
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.original_path = "/project"
        mock_graph.get_filepath_from_obj.side_effect = lambda o: getattr(o, "filepath", "")

        # Symbol dependencies (what it depends ON)
        symbol.dependencies = []
        symbol.usages = []
        symbol.methods = []

        # Import index: two files import utils.py
        mock_graph.get_importers.return_value = [
            "/project/a.py",  # imports target_func
            "/project/b.py",  # imports other_func (NOT target_func)
        ]

        # Symbol-level import info
        mock_graph._file_symbol_imports = {
            "/project/a.py": {"/project/utils.py": {"target_func"}},
            "/project/b.py": {"/project/utils.py": {"other_func"}},
        }
        mock_graph._class_method_names = {}

        result = call_tool(get_dependency_graph,symbol_name="target_func")

        dep_files = [d["file"] for d in result.get("depended_on_by", [])]
        # a.py imports target_func → should be included
        assert "/project/a.py" in dep_files
        # b.py imports other_func → should be EXCLUDED
        assert "/project/b.py" not in dep_files

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_wildcard_importer_still_included(self, mock_graph):
        """A file with wildcard import (empty symbol set) should be included
        conservatively."""
        from grafyx.server import get_dependency_graph

        mock_graph.initialized = True

        symbol = MagicMock()
        symbol.name = "my_func"
        symbol.filepath = "/project/lib.py"

        mock_graph.get_symbol.return_value = ("python", symbol, "function")
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.original_path = "/project"
        mock_graph.get_filepath_from_obj.side_effect = lambda o: getattr(o, "filepath", "")

        symbol.dependencies = []
        symbol.usages = []
        symbol.methods = []

        mock_graph.get_importers.return_value = ["/project/consumer.py"]

        # Empty symbol set = wildcard → include conservatively
        mock_graph._file_symbol_imports = {
            "/project/consumer.py": {"/project/lib.py": set()},
        }
        mock_graph._class_method_names = {}

        result = call_tool(get_dependency_graph,symbol_name="my_func")

        dep_files = [d["file"] for d in result.get("depended_on_by", [])]
        assert "/project/consumer.py" in dep_files
