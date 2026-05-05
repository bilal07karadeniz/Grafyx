"""Tests for v0.2.4 fixes:
- Python relative imports tracked in _build_import_index
- EmbeddingSearcher._build_error surfaced on build failure
"""

import logging
from unittest.mock import MagicMock, patch

from grafyx.graph import CodebaseGraph


# ---------------------------------------------------------------------------
# Fix 1: Python relative import tracking
# ---------------------------------------------------------------------------

class TestPythonRelativeImportTracking:
    """'from .module import X' must appear in the import index."""

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
        graph._resolve_python_relative_import = CodebaseGraph._resolve_python_relative_import
        graph._resolve_init_reexports = lambda: None
        graph._is_ignored_file_path = lambda p: False
        return graph

    def _make_py_file(self, path, imports=()):
        f = MagicMock()
        f.path = path
        f.filepath = path
        imp_mocks = []
        for imp_str in imports:
            imp = MagicMock()
            imp.source = imp_str
            imp_mocks.append(imp)
        f.imports = imp_mocks
        return f

    def test_same_package_relative_import_tracked(self):
        """'from .database import get_db' registers database.py in the import index."""
        graph = self._make_graph()

        config_loader = self._make_py_file(
            "/project/agents/base/config_loader.py",
            imports=["from .database import get_db"],
        )
        database = self._make_py_file("/project/agents/base/database.py")

        py_codebase = MagicMock()
        py_codebase.files = [config_loader, database]
        graph._codebases = {"python": py_codebase}

        graph._build_import_index()

        importers = graph._import_index.get("/project/agents/base/database.py", [])
        assert "/project/agents/base/config_loader.py" in importers, (
            "Relative import 'from .database import get_db' was not tracked"
        )

    def test_same_package_relative_import_symbol_tracked(self):
        """Symbol names from relative imports appear in _file_symbol_imports."""
        graph = self._make_graph()

        config_loader = self._make_py_file(
            "/project/agents/base/config_loader.py",
            imports=["from .database import get_db"],
        )
        database = self._make_py_file("/project/agents/base/database.py")

        py_codebase = MagicMock()
        py_codebase.files = [config_loader, database]
        graph._codebases = {"python": py_codebase}

        graph._build_import_index()

        target = "/project/agents/base/database.py"
        importer = "/project/agents/base/config_loader.py"
        symbols = graph._file_symbol_imports.get(importer, {}).get(target, set())
        assert "get_db" in symbols, (
            "Symbol 'get_db' not tracked in _file_symbol_imports for relative import"
        )

    def test_parent_package_relative_import_tracked(self):
        """'from ..models import User' (two dots) registers models.py."""
        graph = self._make_graph()

        service = self._make_py_file(
            "/project/app/services/user_service.py",
            imports=["from ..models import User"],
        )
        models = self._make_py_file("/project/app/models.py")

        py_codebase = MagicMock()
        py_codebase.files = [service, models]
        graph._codebases = {"python": py_codebase}

        graph._build_import_index()

        importers = graph._import_index.get("/project/app/models.py", [])
        assert "/project/app/services/user_service.py" in importers, (
            "Two-dot relative import 'from ..models import User' was not tracked"
        )

    def test_forward_index_populated_for_relative_import(self):
        """_forward_import_index is updated alongside the reverse index."""
        graph = self._make_graph()

        config_loader = self._make_py_file(
            "/project/app/loader.py",
            imports=["from .utils import helper"],
        )
        utils = self._make_py_file("/project/app/utils.py")

        py_codebase = MagicMock()
        py_codebase.files = [config_loader, utils]
        graph._codebases = {"python": py_codebase}

        graph._build_import_index()

        imported = graph._forward_import_index.get("/project/app/loader.py", [])
        assert "/project/app/utils.py" in imported, (
            "_forward_import_index not updated for relative import"
        )

    def test_absolute_import_still_works(self):
        """Normal absolute imports are unaffected by the relative-import fix."""
        graph = self._make_graph()

        main = self._make_py_file(
            "/project/main.py",
            imports=["from models import User"],
        )
        models = self._make_py_file("/project/models.py")

        py_codebase = MagicMock()
        py_codebase.files = [main, models]
        graph._codebases = {"python": py_codebase}

        graph._build_import_index()

        importers = graph._import_index.get("/project/models.py", [])
        assert "/project/main.py" in importers, "Absolute import broken by relative-import fix"

    def test_ambiguous_relative_import_picks_closest_file(self):
        """When two files share a basename, the one in the same directory wins."""
        graph = self._make_graph()

        loader = self._make_py_file(
            "/project/agents/base/loader.py",
            imports=["from .database import get_db"],
        )
        db_base = self._make_py_file("/project/agents/base/database.py")
        db_app = self._make_py_file("/project/app/database.py")

        py_codebase = MagicMock()
        py_codebase.files = [loader, db_base, db_app]
        graph._codebases = {"python": py_codebase}

        graph._build_import_index()

        # The sibling database.py must be picked, not the distant one
        assert "/project/agents/base/loader.py" in graph._import_index.get(
            "/project/agents/base/database.py", []
        )
        assert "/project/agents/base/loader.py" not in graph._import_index.get(
            "/project/app/database.py", []
        )


# ---------------------------------------------------------------------------
# Fix 2: EmbeddingSearcher build error tracking
# ---------------------------------------------------------------------------

class TestEmbeddingBuildErrorTracking:
    """EmbeddingSearcher must set _build_error when build() fails."""

    def _make_searcher(self, tmp_path):
        from grafyx.search._embeddings import EmbeddingSearcher
        graph = MagicMock()
        graph.iter_functions_with_source.return_value = iter([])
        graph.get_all_classes.return_value = []
        return EmbeddingSearcher(graph, cache_dir=str(tmp_path))

    def test_build_error_initially_none(self, tmp_path):
        """_build_error is None on a fresh EmbeddingSearcher."""
        searcher = self._make_searcher(tmp_path)
        assert searcher._build_error is None

    def test_build_failure_sets_build_error(self, tmp_path):
        """When the TextEmbedding model fails to load, _build_error is populated."""
        from grafyx.search._embeddings import EmbeddingSearcher, _HAS_EMBEDDINGS

        if not _HAS_EMBEDDINGS:
            return  # fastembed not installed; skip

        searcher = self._make_searcher(tmp_path)

        # Inject a crash: pretend the model raises on instantiation
        with patch("grafyx.search._embeddings.TextEmbedding") as mock_te:
            mock_te.side_effect = RuntimeError("model download failed")
            searcher.build()

        assert searcher._build_error is not None
        assert "model download failed" in searcher._build_error
        assert not searcher._ready

    def test_build_error_surfaced_in_degraded_reason(self):
        """When _build_error is set, find_related_code returns degraded_reason='build_failed'."""
        from grafyx.server import _state, _tools_search
        from tests._tool_compat import call_tool

        mock_searcher = MagicMock()
        mock_searcher.degraded = True
        mock_searcher._embedding_searcher = MagicMock()
        mock_searcher._embedding_searcher._build_error = "model download failed"
        mock_searcher._embedding_searcher._building = False
        mock_searcher.search.return_value = []
        mock_searcher.encoder_meta = {"model": "tokens", "version": "", "configured": "jina-v2"}

        mock_graph = MagicMock()
        mock_graph.get_all_functions.return_value = []
        mock_graph.get_all_classes.return_value = []
        mock_graph.get_all_files.return_value = []

        with (
            patch.object(_state, "_graph", mock_graph),
            patch.object(_state, "_searcher", mock_searcher),
            patch.object(_state, "_init_ready", True),
        ):
            result = call_tool(_tools_search.find_related_code, "test query")

        assert result.get("degraded") is True
        assert result.get("degraded_reason") == "build_failed", (
            f"Expected 'build_failed', got {result.get('degraded_reason')!r}"
        )


# ---------------------------------------------------------------------------
# Fix 2b: EmbeddingSearcher build() emits info-level log messages
# ---------------------------------------------------------------------------

class TestEmbeddingBuildLogging:
    """Key milestones in build() must be logged at INFO, not DEBUG."""

    def test_cache_hit_logged_at_info(self, tmp_path, caplog):
        """A cache hit during build() emits at least one INFO record."""
        import numpy as np
        import json
        from grafyx.search._embeddings import EmbeddingSearcher, _HAS_EMBEDDINGS

        if not _HAS_EMBEDDINGS:
            return

        graph = MagicMock()
        graph.iter_functions_with_source.return_value = iter([
            ("my_func", "/project/main.py", "def my_func(): pass", ""),
        ])
        graph.get_all_classes.return_value = []
        searcher = EmbeddingSearcher(graph, cache_dir=str(tmp_path))

        # Pre-populate cache so the fast path triggers
        fp = searcher._compute_fingerprint()
        cache_dir = searcher._cache_dir
        vectors = np.zeros((1, 768), dtype=np.float32)
        np.save(str(cache_dir / f"{fp}_vectors.npy"), vectors)
        np.save(str(cache_dir / f"{fp}_file_vectors.npy"), vectors)
        with open(str(cache_dir / f"{fp}_meta.json"), "w") as f:
            json.dump([{"name": "my_func", "file": "/project/main.py", "class_name": ""}], f)
        with open(str(cache_dir / f"{fp}_file_meta.json"), "w") as f:
            json.dump([{"file": "/project/main.py"}], f)

        with caplog.at_level(logging.INFO, logger="grafyx.search._embeddings"):
            searcher.build()

        info_messages = [r for r in caplog.records if r.levelno >= logging.INFO]
        assert info_messages, "build() emitted no INFO-level log messages on cache hit"
