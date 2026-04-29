"""Path resolution and translation utilities for CodebaseGraph.

This module provides PathMixin, which handles all filesystem path operations
needed by the graph engine:

    - **Mirror syncing**: When the project lives on a Windows-mounted filesystem
      (``/mnt/c/...`` in WSL), graph-sitter cannot use git/chmod properly.
      ``_ensure_linux_path()`` creates a native Linux mirror copy under
      ``~/.grafyx/mirrors/`` with rsync (or shutil fallback).

    - **Path translation**: All paths stored in indexes and returned to users
      go through ``translate_path()`` and ``resolve_path()`` to convert mirror
      paths back to the original project path that the user recognizes.

    - **Path extraction**: Graph-sitter objects store file/line info under
      varying attribute names across versions. The ``get_filepath_from_obj()``
      and ``get_line_number()`` methods try multiple attribute names with
      unwrapping of wrapper objects (usage/call_site nodes).

    - **Path filtering**: ``_is_ignored_file_path()``, ``_is_test_path()``,
      and ``_is_migration_path()`` classify paths for index building and
      dead code analysis.

Mixin: PathMixin
Reads: self._original_path, self._project_path, self._ignore_patterns
Writes: nothing (pure utility methods)
"""

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from grafyx.utils import safe_get_attr

logger = logging.getLogger(__name__)


class PathMixin:
    """Path resolution, translation, and filtering methods.

    This mixin is the foundation for all other mixins -- every index builder
    and query method calls ``translate_path()`` or ``resolve_path()`` to
    normalize graph-sitter paths before storing or returning them.

    Reads:
        self._original_path: The original (possibly Windows-mounted) project path.
        self._project_path: The usable path (original or synced mirror).
                           Equals _original_path unless on a Windows mount.
        self._ignore_patterns: List of directory names to skip (node_modules, etc.).
    """

    # --- Mirror Syncing (Windows/WSL Compatibility) ---

    @staticmethod
    def _ensure_linux_path(project_path: str) -> str:
        """If the path is on a Windows mount (/mnt/c/...), sync it to the
        native Linux filesystem where graph-sitter can use git/chmod.
        Returns the usable path (original or synced copy).

        The mirror is stored at ``~/.grafyx/mirrors/<project>-<hash>/``
        where the hash ensures uniqueness for projects with the same name
        in different directories.
        """
        if not project_path.startswith("/mnt/"):
            return project_path

        # Create a stable mirror directory based on the original path.
        # MD5 hash ensures different /mnt/c paths get different mirrors.
        path_hash = hashlib.md5(project_path.encode()).hexdigest()[:10]
        project_name = Path(project_path).name
        mirror_dir = Path.home() / ".grafyx" / "mirrors" / f"{project_name}-{path_hash}"

        logger.info(
            "Windows mount detected (%s). Syncing to %s...",
            project_path,
            mirror_dir,
        )

        mirror_dir.parent.mkdir(parents=True, exist_ok=True)

        # Prefer rsync for incremental sync (fast on subsequent calls);
        # fall back to shutil.copytree if rsync isn't installed.
        try:
            subprocess.run(
                [
                    "rsync", "-a", "--delete",
                    "--exclude", ".git",
                    "--exclude", "node_modules",
                    "--exclude", "__pycache__",
                    "--exclude", ".venv",
                    "--exclude", "venv",
                    f"{project_path}/",
                    str(mirror_dir) + "/",
                ],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            # rsync not available -- full copy (slower but always works)
            if mirror_dir.exists():
                shutil.rmtree(mirror_dir)
            shutil.copytree(
                project_path,
                mirror_dir,
                ignore=shutil.ignore_patterns(
                    ".git", "node_modules", "__pycache__", ".venv", "venv",
                ),
            )

        # Graph-sitter requires a git repo to function. Initialize a
        # throwaway repo in the mirror so graph-sitter's git operations
        # don't fail. Uses fake author info to avoid git config prompts.
        git_dir = mirror_dir / ".git"
        if not git_dir.exists():
            subprocess.run(
                ["git", "init"], cwd=str(mirror_dir),
                check=True, capture_output=True,
            )
            subprocess.run(
                ["git", "add", "-A"], cwd=str(mirror_dir),
                check=True, capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "grafyx mirror", "--allow-empty"],
                cwd=str(mirror_dir), check=True, capture_output=True,
                env={**os.environ, "GIT_AUTHOR_NAME": "grafyx", "GIT_AUTHOR_EMAIL": "grafyx@local",
                     "GIT_COMMITTER_NAME": "grafyx", "GIT_COMMITTER_EMAIL": "grafyx@local"},
            )

        logger.info("Synced to %s", mirror_dir)
        return str(mirror_dir)

    # --- Path Translation (Mirror <-> Original) ---

    def translate_path(self, path: str) -> str:
        """Translate a mirror path back to the original project path.

        When running on a Windows mount, ``_project_path`` points to the
        Linux mirror but users expect paths relative to their real project.
        This method does a simple string replacement of the mirror prefix
        with the original prefix. No-op when there is no mirror.
        """
        if self._original_path != self._project_path and path:
            return path.replace(self._project_path, self._original_path)
        return path

    def resolve_path(self, path: str) -> str:
        """Normalize a raw graph-sitter path to a canonical absolute path.

        Three-step pipeline:
        1. ``translate_path()``: convert mirror path to original project path.
        2. Normalize backslashes to forward slashes.
        3. If still relative (no leading ``/`` and no Windows drive letter),
           prepend project root to make it absolute.

        This unifies paths from different sources (graph-sitter usage objects,
        import index, caller index) that may arrive in inconsistent formats.
        Every path returned to the user should go through this method.
        """
        if not path:
            return ""
        result = self.translate_path(path).replace("\\", "/")
        # Detect absolute paths: Unix (/) or Windows (C:)
        is_absolute = result.startswith("/") or (
            len(result) >= 2 and result[1] == ":" and result[0].isalpha()
        )
        if not is_absolute:
            root = self._original_path.replace("\\", "/").rstrip("/")
            result = root + "/" + result
        return result

    # --- Graph-Sitter Object Introspection ---

    @staticmethod
    def get_filepath_from_obj(obj: Any) -> str:
        """Extract file path from a graph-sitter object, trying multiple attrs.

        Graph-sitter objects (Function, Class, Usage, CallSite) store file
        paths under different attribute names depending on the object type
        and graph-sitter version. This method tries them in priority order.

        For wrapper objects (Usage wraps a Match/Symbol/Node), we also
        unwrap one level to find the path on the inner object.
        """
        # Direct attributes -- most graph-sitter objects have one of these
        for attr in ("filepath", "file_path", "file", "source_file", "path"):
            val = safe_get_attr(obj, attr, None)
            if val is not None and str(val).strip():
                return str(val)
        # Usage objects may wrap a match/symbol node that has the path
        for wrapper_attr in ("match", "symbol", "node"):
            wrapper = safe_get_attr(obj, wrapper_attr, None)
            if wrapper:
                for attr in ("filepath", "file_path", "file", "path"):
                    val = safe_get_attr(wrapper, attr, None)
                    if val is not None and str(val).strip():
                        return str(val)
        return ""

    @staticmethod
    def _extract_line(obj: Any) -> int | None:
        """Try to extract a line number from a single graph-sitter object.

        Tries three strategies in order:
        1. Direct ``line`` attribute (most common).
        2. ``start_point`` tuple from tree-sitter (0-indexed row, so we add 1).
        3. Alternative names: ``line_number``, ``start_line``, ``lineno``.
        """
        line = safe_get_attr(obj, "line", None)
        if line is not None:
            try:
                return int(line)
            except (TypeError, ValueError):
                pass
        # tree-sitter exposes position as (row, col) tuple, 0-indexed
        start_point = safe_get_attr(obj, "start_point", None)
        if start_point is not None:
            try:
                if isinstance(start_point, (tuple, list)) and len(start_point) >= 1:
                    return int(start_point[0]) + 1
            except (TypeError, ValueError):
                pass
        # Fallback attribute names used by some graph-sitter versions
        for attr in ("line_number", "start_line", "lineno"):
            val = safe_get_attr(obj, attr, None)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def get_line_number(obj: Any) -> int | None:
        """Extract line number from a graph-sitter object, trying multiple attrs.

        Like ``get_filepath_from_obj()``, also unwraps usage/call_site wrapper
        objects (match, symbol, node) that may contain the actual source location.
        """
        result = PathMixin._extract_line(obj)
        if result is not None:
            return result
        # Usage objects may wrap a match/symbol/node that has the line info
        for wrapper_attr in ("match", "symbol", "node"):
            wrapper = safe_get_attr(obj, wrapper_attr, None)
            if wrapper:
                result = PathMixin._extract_line(wrapper)
                if result is not None:
                    return result
        return None

    # --- Path Classification / Filtering ---

    def _is_ignored_file_path(self, path: str) -> bool:
        """Return True if any component of 'path' is in self._ignore_patterns.

        Works on both forward-slash and backslash paths since we split into
        components. Catches nested ignored dirs such as
        ``project/packages/node_modules/lodash/index.ts``.
        """
        if not path:
            return False
        parts = path.replace("\\", "/").split("/")
        ignore_set = set(self._ignore_patterns)
        return any(part in ignore_set for part in parts)

    @staticmethod
    def _is_test_path(path: str) -> bool:
        """Check if a file path is in a test directory.

        Used in two contexts:
        1. **Symbol deduplication**: When multiple symbols share a name (e.g.,
           ``EventBus`` in src/ and tests/), prefer the source definition.
        2. **Dead code analysis**: Optionally exclude test files from unused
           symbol detection (``include_tests=False``).
        """
        parts = path.replace("\\", "/").lower().split("/")
        return any(
            p in ("test", "tests", "spec", "specs", "testing", "__tests__",
                  "load_tests", "__mocks__")
            for p in parts
        )

    @staticmethod
    def _is_migration_path(path: str) -> bool:
        """Check if a file is in a migration/alembic directory.

        Migration files are auto-generated and always excluded from dead code
        detection since their functions are called by the migration framework,
        not by user code.
        """
        norm = path.replace("\\", "/")
        return (
            "/alembic/versions/" in norm
            or "/migrations/" in norm
        )
