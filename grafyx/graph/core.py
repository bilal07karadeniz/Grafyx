"""CodebaseGraph facade -- the main entry point for codebase analysis.

This is the composition root that brings all mixin modules together into a
single ``CodebaseGraph`` class. It is the only file that:
    - Creates ``graph_sitter.Codebase`` instances (one per detected language)
    - Orchestrates the index-building pipeline (caller, import, instance, etc.)
    - Owns the ``__init__``, ``initialize()``, and ``refresh()`` lifecycle

The MRO (Method Resolution Order) composes five mixins:
    1. ``PathMixin``          -- path resolution, mirror syncing, line extraction
    2. ``IndexBuilderMixin``  -- builds all reverse indexes from parsed codebases
    3. ``CallerQueryMixin``   -- multi-level caller disambiguation queries
    4. ``SymbolQueryMixin``   -- function/class/file/symbol lookups, stats
    5. ``AnalysisMixin``      -- dead code detection, subclass trees

All mixin state lives on ``self`` (the CodebaseGraph instance). The
``__init__`` method initializes all ``self._*`` attributes that the mixins
read and write. See each mixin's module docstring for which attributes it
expects.

Thread safety: All public methods acquire ``self._lock`` (a reentrant lock)
before accessing ``self._codebases`` or any index. The RLock allows methods
to call other methods that also acquire the lock without deadlocking.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any

from grafyx.utils import (
    DEFAULT_IGNORE_PATTERNS,
    detect_languages,
)

from grafyx.graph._patches import _patch_graph_sitter_assert
from grafyx.graph._paths import PathMixin
from grafyx.graph._indexes import IndexBuilderMixin
from grafyx.graph._callers import CallerQueryMixin
from grafyx.graph._query import SymbolQueryMixin
from grafyx.graph._analysis import AnalysisMixin

logger = logging.getLogger(__name__)


class CodebaseGraph(
    PathMixin,
    IndexBuilderMixin,
    CallerQueryMixin,
    SymbolQueryMixin,
    AnalysisMixin,
):
    """Manages multiple graph-sitter Codebase instances (one per language)
    and provides a unified query interface across all of them.

    Lifecycle:
        1. ``__init__(project_path)`` -- initializes state, syncs mirror if needed.
        2. ``initialize()`` -- detects languages, parses codebases, builds indexes.
        3. ``refresh()`` -- re-parses and rebuilds indexes (called by file watcher).

    After ``initialize()`` returns, all query and analysis methods are ready.
    """

    def __init__(
        self,
        project_path: str,
        languages: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        sync_enabled: bool = True,
    ):
        resolved = str(Path(project_path).resolve())
        self._original_path = resolved
        # If on a Windows mount (/mnt/c/...), creates a native Linux mirror
        self._project_path = self._ensure_linux_path(resolved)
        self._languages = languages
        self._ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)
        if ignore_patterns:
            self._ignore_patterns.extend(ignore_patterns)
        self._sync_enabled = sync_enabled

        # --- graph-sitter Codebase instances (one per language) ---
        self._codebases: dict[str, Any] = {}

        # --- Caller index: who calls what (built by IndexBuilderMixin) ---
        self._caller_index: dict[str, list[dict]] = {}          # callee_name -> [{name, file, class?, ...}]
        self._class_method_names: dict[str, set[str]] = {}      # class_name -> {method_names}
        self._file_class_methods: dict[str, dict[str, set[str]]] = {}  # file -> {class -> {methods}}
        self._class_defined_in: dict[str, set[str]] = {}        # class_name -> {file_paths}

        # --- Import indexes: file-level dependency graph (bidirectional) ---
        self._import_index: dict[str, list[str]] = {}           # target -> [importers] (reverse)
        self._forward_import_index: dict[str, list[str]] = {}   # source -> [imported] (forward)
        self._file_symbol_imports: dict[str, dict[str, set[str]]] = {}  # importer -> {target -> {names}}

        # --- Supporting indexes ---
        self._external_packages: set[str] = set()               # pip/npm/stdlib names to skip
        self._class_instances: dict[str, list[tuple[str, str]]] = {}  # class -> [(var_name, file)]

        # --- Convention caches: collected during index building for ConventionDetector ---
        self._convention_import_sources: list[dict[str, str]] = []
        self._convention_decorator_info: dict[str, dict[str, tuple[int, list[str]]]] = {}
        self._convention_method_counts: dict[str, int] = {}

        # --- Object literal methods (TS/JS arrow functions in exports) ---
        self._object_literal_methods: list[dict] = []

        # --- Lifecycle state ---
        self._init_errors: list[str] = []
        self._initialized = False
        self._lock = threading.RLock()  # RLock allows reentrant acquisition
        self._last_refresh_time: float | None = None

    # ===================================================================
    # Lifecycle: Initialize and Refresh
    # ===================================================================

    def initialize(self) -> dict:
        """Detect languages, create Codebase instances, build all indexes.

        This is the main setup method. It:
            1. Auto-detects languages (or uses explicitly provided ones).
            2. Creates one ``graph_sitter.Codebase`` per language.
            3. Builds all indexes in order: external packages -> caller ->
               class instances -> import.
            4. Returns aggregate stats.

        The index build order matters:
            - ``_build_external_packages()`` must run first so import index
              can skip external packages.
            - ``_build_caller_index()`` must run before ``_build_import_index()``
              because the caller index augmentation passes don't depend on
              the import index, but later queries (get_callers Level 3) do.

        Returns:
            Dict with languages, file/function/class counts, duration_seconds,
            and optionally errors/warnings.
        """
        with self._lock:
            start = time.time()

            # Detect or use explicit languages
            if self._languages:
                langs = self._languages
                logger.info("Using explicit languages: %s", langs)
            else:
                logger.debug("Auto-detecting languages in %s", self._project_path)
                langs = detect_languages(self._project_path, self._ignore_patterns)
                logger.info("Detected languages: %s", langs)

            if not langs:
                self._initialized = True
                return {
                    "languages": [],
                    "total_files": 0,
                    "total_functions": 0,
                    "total_classes": 0,
                    "by_language": {},
                    "duration_seconds": 0,
                    "warning": "No supported languages detected in this project.",
                }

            # Create one graph-sitter Codebase per language.
            # Patch the circular reference assertion before any parsing.
            from graph_sitter import Codebase
            _patch_graph_sitter_assert()
            self._codebases = {}
            self._init_errors = []
            for lang in langs:
                try:
                    logger.info("Parsing %s codebase at %s...", lang, self._project_path)
                    codebase = Codebase(self._project_path, language=lang)
                    self._codebases[lang] = codebase
                    logger.info(
                        "Parsed %s: %d functions, %d classes, %d files",
                        lang,
                        len(list(codebase.functions)),
                        len(list(codebase.classes)),
                        len(list(codebase.files)),
                    )
                except Exception as e:
                    error_msg = f"{lang}: {type(e).__name__}: {e}"
                    self._init_errors.append(error_msg)
                    logger.error("Failed to parse %s codebase: %s", lang, e, exc_info=True)

            self._initialized = True
            self._build_external_packages()
            self._build_caller_index()
            self._build_class_instances()
            self._build_import_index()
            self._augment_index_with_import_disambiguated_calls()  # Pass 7
            self._extract_object_literal_methods()  # Pass 8
            self._last_refresh_time = time.time()
            duration = time.time() - start

            stats = self.get_stats()
            stats["duration_seconds"] = round(duration, 2)
            if self._init_errors:
                stats["errors"] = self._init_errors
            if not self._codebases and langs:
                stats["warning"] = (
                    f"All languages failed to parse. "
                    f"Detected {langs} but graph-sitter raised errors. "
                    f"See 'errors' field for details."
                )
            return stats

    def refresh(self, languages: list[str] | None = None) -> dict:
        """Destroy and re-create Codebase instances. Full re-parse.

        Called by the file watcher when source files change. Re-creates
        graph-sitter Codebase instances and rebuilds all indexes from scratch.
        Does NOT re-run ``_build_external_packages()`` since package manifests
        rarely change during development.

        Args:
            languages: If provided, only refresh these specific languages.
                      If None, refresh all currently loaded languages.

        Returns:
            Same format as ``initialize()``.
        """
        with self._lock:
            start = time.time()

            langs_to_refresh = languages or list(self._codebases.keys())

            from graph_sitter import Codebase
            for lang in langs_to_refresh:
                try:
                    logger.info(f"Refreshing {lang} codebase...")
                    self._codebases[lang] = Codebase(self._project_path, language=lang)
                except Exception as e:
                    logger.error(f"Failed to refresh {lang}: {e}")

            self._build_caller_index()
            self._build_class_instances()
            self._build_import_index()
            self._augment_index_with_import_disambiguated_calls()  # Pass 7
            self._extract_object_literal_methods()  # Pass 8
            self._last_refresh_time = time.time()
            duration = time.time() - start

            stats = self.get_stats()
            stats["duration_seconds"] = round(duration, 2)
            return stats

    # ===================================================================
    # Read-Only Properties
    # ===================================================================

    @property
    def project_path(self) -> str:
        """The usable project path (original or mirror if on Windows mount)."""
        return self._project_path

    @property
    def original_path(self) -> str:
        """The original project path as provided by the user."""
        return self._original_path

    @property
    def initialized(self) -> bool:
        """True after initialize() has completed (even if some languages failed)."""
        return self._initialized

    @property
    def languages(self) -> list[str]:
        """List of successfully parsed language names (e.g., ['python', 'typescript'])."""
        return list(self._codebases.keys())
