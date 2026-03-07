"""Tests for import alias tracking in the symbol index.

Covers:
- _extract_symbol_names_from_import returns both original AND alias names
- _file_symbol_imports stores alias mappings
- Aliased symbols are findable via _build_imported_names
"""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph


class TestExtractSymbolNamesWithAliases:
    """_extract_symbol_names_from_import should return both original and alias."""

    def test_python_from_import_with_alias_returns_both(self):
        """from models import User as AppUser -> {User, AppUser}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "from models import User as AppUser"
        )
        assert "User" in names
        assert "AppUser" in names

    def test_python_from_import_multiple_aliases(self):
        """from utils import A as X, B as Y, C -> {A, X, B, Y, C}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "from utils import A as X, B as Y, C"
        )
        assert names == {"A", "X", "B", "Y", "C"}

    def test_python_from_import_no_alias_unchanged(self):
        """from auth import create_token, verify -> {create_token, verify}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "from auth import create_token, verify"
        )
        assert names == {"create_token", "verify"}

    def test_ts_named_import_with_alias_returns_both(self):
        """import { Config as AppConfig } from './config' -> {Config, AppConfig}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "import { Config as AppConfig } from './config'"
        )
        assert "Config" in names
        assert "AppConfig" in names

    def test_ts_named_import_multiple_aliases(self):
        """import { A as X, B } from './mod' -> {A, X, B}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "import { A as X, B } from './mod'"
        )
        assert names == {"A", "X", "B"}


class TestImportedNamesIncludesAliases:
    """_build_imported_names should include alias names so dead code
    detection doesn't flag aliased re-exports as unused."""

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._lock = MagicMock()
        graph._external_packages = set()
        graph.translate_path = lambda p: str(p) if p else ""
        graph._is_ignored_file_path = lambda p: False
        graph._build_imported_names = CodebaseGraph._build_imported_names.__get__(graph)
        return graph

    def test_alias_appears_in_imported_names(self):
        """If file imports 'from X import store_chunks as store_chunks_in_qdrant',
        both 'store_chunks' and 'store_chunks_in_qdrant' should be in imported names."""
        graph = self._make_graph()

        imp = MagicMock()
        imp.source = "from rag.chunking import store_chunks as store_chunks_in_qdrant"
        f = MagicMock()
        f.imports = [imp]
        codebase = MagicMock()
        codebase.files = [f]
        graph._codebases = {"python": codebase}

        names = graph._build_imported_names()
        assert "store_chunks" in names
        assert "store_chunks_in_qdrant" in names

    def test_ts_alias_appears_in_imported_names(self):
        """TS named import with alias should include both names."""
        graph = self._make_graph()

        imp = MagicMock()
        imp.source = "import { Config as AppConfig } from './config'"
        f = MagicMock()
        f.imports = [imp]
        codebase = MagicMock()
        codebase.files = [f]
        graph._codebases = {"typescript": codebase}

        names = graph._build_imported_names()
        assert "Config" in names
        assert "AppConfig" in names
