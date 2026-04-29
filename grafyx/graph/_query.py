"""Symbol query methods for CodebaseGraph.

This module provides SymbolQueryMixin, the read-only query layer that all
MCP tools use to look up functions, classes, files, and generic symbols
across all language codebases parsed by graph-sitter.

Key design decisions:

    - **Multi-language iteration**: Every query method iterates over
      ``self._codebases`` (keyed by language) so results span Python,
      TypeScript, JavaScript, etc. transparently.

    - **Test file deprioritization**: When multiple symbols share a name
      (e.g., ``EventBus`` in ``src/`` and ``tests/``), results are sorted
      so source definitions come first. Users almost always want the source.

    - **Fuzzy fallback**: ``get_function()`` falls back to token-based
      matching when exact match fails. This allows queries like
      ``"execute_tool"`` to match ``ToolExecutor.execute`` by checking
      that all query tokens (``execute``, ``tool``) appear in the
      combined class + method name tokens.

    - **Prioritized file matching**: ``get_file()`` uses a 3-tier priority
      (exact > suffix > filename-only) so users can provide partial paths.

    - **Import graph queries**: ``get_importers()`` and ``get_forward_imports()``
      expose the bidirectional import index for file-level dependency analysis.

Mixin: SymbolQueryMixin
Reads: self._codebases, self._lock, self._import_index,
       self._forward_import_index, self._ignore_patterns, self._init_errors
Writes: nothing (read-only queries)
"""

import logging
from typing import Any

from grafyx.utils import (
    format_class_summary,
    format_function_signature,
    safe_get_attr,
    safe_str,
    split_tokens,
)

logger = logging.getLogger(__name__)


class SymbolQueryMixin:
    """Lookup functions, classes, files, and symbols across all codebases.

    This mixin provides the query methods that the MCP server tools call
    to retrieve information about the codebase. All methods are read-only
    and acquire ``self._lock`` for thread safety.

    Reads: _codebases, _lock, _import_index, _forward_import_index,
    _ignore_patterns, _init_errors
    """

    # ===================================================================
    # Single-Symbol Lookups
    # ===================================================================

    def get_function(self, name: str) -> tuple[str, Any, str | None] | None:
        """Search all codebases for a function by name.
        Also searches class methods.

        Supports ``ClassName.method_name`` syntax for disambiguation.
        If exact match fails, tries fuzzy matching where query tokens
        appear in class name + method name combinations.

        Returns:
            - Single tuple ``(language, Function, class_name)`` if exactly one match.
            - List of such tuples if multiple matches (caller handles ambiguity).
            - None if not found.
            ``class_name`` is None for top-level functions.
        """
        with self._lock:
            # Support ClassName.method syntax
            class_filter = None
            method_name = name
            if "." in name:
                class_filter, method_name = name.rsplit(".", 1)

            matches: list[tuple[str, Any, str | None]] = []
            for lang, codebase in self._codebases.items():
                try:
                    # Search top-level functions (skip if class filter is set)
                    if class_filter is None:
                        for func in codebase.functions:
                            if safe_get_attr(func, "name") == method_name:
                                matches.append((lang, func, None))
                    # Search class methods
                    for cls in codebase.classes:
                        cls_name = safe_get_attr(cls, "name", "")
                        if class_filter and cls_name != class_filter:
                            continue
                        methods = safe_get_attr(cls, "methods", [])
                        if methods:
                            for method in methods:
                                if safe_get_attr(method, "name") == method_name:
                                    matches.append((lang, method, cls_name))
                except Exception:
                    continue

            # Fuzzy fallback: if no exact match, try token-based matching
            # e.g., "execute_tool" -> find "execute" in classes with "tool" in name
            if not matches and class_filter is None:
                matches = self._fuzzy_function_search(name)

            if not matches:
                return None
            # Prefer source files over test files
            if len(matches) > 1:
                matches.sort(key=lambda m: self._is_test_path(
                    str(safe_get_attr(m[1], "filepath", ""))
                ))
            if len(matches) == 1:
                return matches[0]
            return matches  # Multiple matches -- caller handles ambiguity

    def _fuzzy_function_search(self, query: str) -> list[tuple[str, Any, str | None]]:
        """Fuzzy fallback: find methods where ALL query tokens appear in the
        union of ClassName tokens + method_name tokens.

        Example: ``'execute_tool'`` -> tokens ``{'execute', 'tool'}``
        matches ``ToolExecutor.execute`` because ``'tool'`` is in the class
        name and ``'execute'`` is the method name.

        Requires at least 2 tokens to avoid overly broad matches.
        Only searches class methods (not top-level functions) since the
        cross-name matching only makes sense when class context is available.
        """
        tokens = set(split_tokens(query))
        if not tokens or len(tokens) < 2:
            return []

        matches: list[tuple[str, Any, str | None]] = []
        for lang, codebase in self._codebases.items():
            try:
                for cls in codebase.classes:
                    cls_name = safe_get_attr(cls, "name", "")
                    cls_tokens = set(split_tokens(cls_name))
                    methods = safe_get_attr(cls, "methods", [])
                    if not methods:
                        continue
                    for method in methods:
                        m_name = safe_get_attr(method, "name", "")
                        m_tokens = set(split_tokens(m_name))
                        # All query tokens must appear in class_tokens | method_tokens
                        combined = cls_tokens | m_tokens
                        if tokens and tokens.issubset(combined):
                            matches.append((lang, method, cls_name))
            except Exception:
                continue
        return matches

    def get_class(self, name: str) -> tuple[str, Any] | None:
        """Search all codebases for a class by exact name match.

        Returns:
            - Single tuple ``(language, Class)`` if exactly one match.
            - List of such tuples if multiple matches.
            - None if not found.
        """
        with self._lock:
            matches = []
            for lang, codebase in self._codebases.items():
                try:
                    for cls in codebase.classes:
                        if safe_get_attr(cls, "name") == name:
                            matches.append((lang, cls))
                except Exception:
                    continue

            if not matches:
                return None
            # Prefer source files over test files -- when EventBus exists in
            # both src/events/bus.py and tests/test_event_bus.py, the source
            # definition should come first.
            if len(matches) > 1:
                matches.sort(key=lambda m: self._is_test_path(
                    str(safe_get_attr(m[1], "filepath", ""))
                ))
            if len(matches) == 1:
                return matches[0]
            return matches

    def get_file(self, path: str) -> tuple[str, Any] | None:
        """Search all codebases for a file by path.

        Uses a 3-tier priority system to handle partial paths:
            1. **Exact match**: full path equality (fastest, most precise).
            2. **Suffix match**: one path ends with the other (handles
               relative vs. absolute path mismatches).
            3. **Filename-only**: same filename, scored by matching path
               components from the end (weakest, last resort).
        """
        with self._lock:
            # Normalize search: also try translating original path to mirror
            search_str = path.replace("\\", "/")
            # If user passes original path, translate to mirror for matching
            if self._original_path != self._project_path:
                search_str = search_str.replace(
                    self._original_path.replace("\\", "/"),
                    self._project_path.replace("\\", "/"),
                )

            best_match: tuple[str, Any] | None = None
            best_score = 0  # 3=exact, 2=suffix, 1=filename

            for lang, codebase in self._codebases.items():
                try:
                    for f in codebase.files:
                        fp = str(safe_get_attr(f, "path", safe_get_attr(f, "filepath", "")))
                        fp_norm = fp.replace("\\", "/")

                        # Priority 1: Exact match
                        if fp_norm == search_str or fp == path:
                            return (lang, f)

                        # Priority 2: Suffix match (longer suffix = better)
                        if fp_norm.endswith(search_str) or search_str.endswith(fp_norm):
                            overlap = min(len(fp_norm), len(search_str))
                            if overlap > best_score:
                                best_match = (lang, f)
                                best_score = overlap

                        # Priority 3: Filename-only (weakest, only if no suffix match)
                        elif best_score < 2:
                            fp_name = fp_norm.rsplit("/", 1)[-1]
                            search_name = search_str.rsplit("/", 1)[-1]
                            if fp_name == search_name:
                                # Score by how many path components match
                                fp_parts = fp_norm.split("/")
                                search_parts = search_str.split("/")
                                common = sum(
                                    1 for a, b in zip(reversed(fp_parts), reversed(search_parts))
                                    if a == b
                                )
                                if common > best_score:
                                    best_match = (lang, f)
                                    best_score = common
                except Exception:
                    continue

            return best_match

    def get_symbol(self, name: str) -> tuple[str, Any, str] | None:
        """Search for a symbol (function or class) by name.

        Tries class lookup first because it's exact-only. Function lookup
        includes fuzzy matching which can produce false positives when class
        name tokens overlap with method names (e.g., ``"ToolExecutor"``
        fuzzy-matching ``ExecutorAgent.execute_tool``). By checking classes
        first, we avoid this trap.

        Returns:
            Tuple of ``(language, symbol, kind)`` where kind is
            ``'function'`` or ``'class'``. None if not found.
        """
        # Try classes first -- exact-only, no fuzzy false positives
        result = self.get_class(name)
        if result:
            if isinstance(result, list):
                return (result[0][0], result[0][1], "class")
            return (result[0], result[1], "class")

        # Then try functions (includes fuzzy fallback)
        result = self.get_function(name)
        if result:
            if isinstance(result, list):
                return (result[0][0], result[0][1], "function")
            return (result[0], result[1], "function")

        return None

    # ===================================================================
    # Bulk Listing Methods (used by MCP tools and analysis)
    # ===================================================================

    def get_all_functions(
        self,
        language: str | None = None,
        max_results: int = 200,
        include_methods: bool = False,
    ) -> list[dict]:
        """Return all functions across all language codebases as dicts.

        Args:
            language: Filter to a specific language. None means all.
            max_results: Cap on number of results to prevent huge responses.
            include_methods: If True, also include class methods with a
                ``class_name`` field. Used by dead code detection which
                needs to check both functions and methods.

        Returns:
            List of FunctionDict-shaped dicts (see ``_types.py``).
        """
        with self._lock:
            results: list[dict] = []
            codebases = self._codebases.items()
            if language:
                cb = self._codebases.get(language)
                codebases = [(language, cb)] if cb else []
            for lang, codebase in codebases:
                try:
                    for func in codebase.functions:
                        if len(results) >= max_results:
                            break
                        results.append({
                            "name": safe_get_attr(func, "name", ""),
                            "signature": format_function_signature(func),
                            "file": self.translate_path(str(safe_get_attr(func, "filepath", ""))),
                            "language": lang,
                            "line": self.get_line_number(func),
                            "docstring": safe_str(safe_get_attr(func, "docstring", "")),
                        })
                    # Include class methods if requested
                    if include_methods:
                        for cls in codebase.classes:
                            if len(results) >= max_results:
                                break
                            cls_name = safe_get_attr(cls, "name", "")
                            methods = safe_get_attr(cls, "methods", [])
                            if methods:
                                for method in methods:
                                    if len(results) >= max_results:
                                        break
                                    results.append({
                                        "name": safe_get_attr(method, "name", ""),
                                        "signature": format_function_signature(method),
                                        "file": self.translate_path(str(safe_get_attr(method, "filepath", ""))),
                                        "language": lang,
                                        "line": self.get_line_number(method),
                                        "docstring": safe_str(safe_get_attr(method, "docstring", "")),
                                        "class_name": cls_name,
                                    })
                except Exception as e:
                    logger.error(f"Error getting functions from {lang}: {e}")
                if len(results) >= max_results:
                    break
            # Also include object literal methods (TS/JS)
            for olm in getattr(self, "_object_literal_methods", []):
                if len(results) >= max_results:
                    break
                results.append({
                    "name": olm["name"],
                    "file": olm["file"],
                    "line": olm["line"],
                    "class_name": olm["parent"],
                    "language": olm["language"],
                    "signature": f"{olm['name']}: (...) => ...",
                    "docstring": "",
                })
            return results

    def iter_functions_with_source(self):
        """Yield ``(name, file_path, source, class_name)`` for all functions and methods.

        This is a lightweight iterator that avoids the dict serialization
        overhead of ``get_all_functions()``. Used by ``CodeSearcher`` to
        build the source token index for full-text search over function
        bodies. The source is the raw graph-sitter source attribute (the
        function's full text including def line and body).
        """
        with self._lock:
            for _lang, codebase in self._codebases.items():
                try:
                    for func in safe_get_attr(codebase, "functions", []):
                        name = safe_get_attr(func, "name", "")
                        if not name:
                            continue
                        yield (
                            name,
                            self.translate_path(
                                str(safe_get_attr(func, "filepath", ""))
                            ),
                            safe_get_attr(func, "source", ""),
                            "",
                        )
                    for cls in safe_get_attr(codebase, "classes", []):
                        cls_name = safe_get_attr(cls, "name", "")
                        for method in safe_get_attr(cls, "methods", []):
                            mname = safe_get_attr(method, "name", "")
                            if not mname:
                                continue
                            yield (
                                mname,
                                self.translate_path(
                                    str(safe_get_attr(method, "filepath", ""))
                                ),
                                safe_get_attr(method, "source", ""),
                                cls_name,
                            )
                except Exception:
                    continue
            # Also yield object literal methods (TS/JS)
            for olm in getattr(self, "_object_literal_methods", []):
                yield (olm["name"], olm["file"], olm.get("source", ""), olm["parent"])

    def get_all_classes(
        self,
        language: str | None = None,
        max_results: int = 200,
        include_method_names: bool = False,
    ) -> list[dict]:
        """Return all classes across all language codebases as dicts.

        Args:
            language: Filter to a specific language. None means all.
            max_results: Cap on number of results.
            include_method_names: If True, include a ``methods`` list with
                method name strings. Used by ``get_module_context`` to show
                what methods each class has.

        Returns:
            List of ClassDict-shaped dicts (see ``_types.py``).
            Uses ``format_class_summary()`` from utils to extract base
            classes with the multi-attribute + regex fallback chain.
        """
        with self._lock:
            results: list[dict] = []
            codebases = self._codebases.items()
            if language:
                cb = self._codebases.get(language)
                codebases = [(language, cb)] if cb else []
            for lang, codebase in codebases:
                try:
                    for cls in codebase.classes:
                        if len(results) >= max_results:
                            break
                        summary = format_class_summary(cls)
                        summary["file"] = self.translate_path(str(safe_get_attr(cls, "filepath", "")))
                        summary["language"] = lang
                        summary["line"] = self.get_line_number(cls)
                        if include_method_names:
                            methods = safe_get_attr(cls, "methods", [])
                            summary["methods"] = [
                                safe_get_attr(m, "name", "?")
                                for m in (methods or [])
                            ]
                        results.append(summary)
                except Exception as e:
                    logger.error(f"Error getting classes from {lang}: {e}")
                if len(results) >= max_results:
                    break
            return results

    def get_all_files(self, language: str | None = None, max_results: int = 500) -> list[dict]:
        """Return all source files across all language codebases as dicts.

        Filters out ignored paths (node_modules, .git, etc.) automatically.

        Args:
            language: Filter to a specific language. None means all.
            max_results: Cap on number of results.

        Returns:
            List of FileDict-shaped dicts (see ``_types.py``).
        """
        with self._lock:
            results: list[dict] = []
            codebases = self._codebases.items()
            if language:
                cb = self._codebases.get(language)
                codebases = [(language, cb)] if cb else []
            for lang, codebase in codebases:
                try:
                    for f in codebase.files:
                        if len(results) >= max_results:
                            break
                        translated = self.translate_path(
                            str(safe_get_attr(f, "path", safe_get_attr(f, "filepath", "")))
                        )
                        if self._is_ignored_file_path(translated):
                            continue
                        functions = safe_get_attr(f, "functions", [])
                        classes = safe_get_attr(f, "classes", [])
                        imports = safe_get_attr(f, "imports", [])
                        results.append({
                            "path": translated,
                            "function_count": len(list(functions)) if functions else 0,
                            "class_count": len(list(classes)) if classes else 0,
                            "import_count": len(list(imports)) if imports else 0,
                            "language": lang,
                            "docstring": safe_str(safe_get_attr(f, "docstring", "")),
                        })
                except Exception as e:
                    logger.error(f"Error getting files from {lang}: {e}")
                if len(results) >= max_results:
                    break
            return results

    # ===================================================================
    # Import Graph Queries
    # ===================================================================

    def get_importers(self, file_path: str) -> list[str]:
        """Return list of file paths that import the given file (reverse direction).

        Tries exact path match first, then falls back to suffix matching
        for partial paths.  Suffix matching aggregates ALL matches instead
        of returning on the first hit.
        """
        with self._lock:
            fp_norm = file_path.replace("\\", "/")
            # Try exact match first
            result = self._import_index.get(fp_norm, [])
            if result:
                return list(result)
            # Aggregate ALL suffix matches (not just first hit)
            aggregated: list[str] = []
            seen: set[str] = set()
            for key, importers in self._import_index.items():
                if key.endswith(fp_norm) or fp_norm.endswith(key):
                    for imp in importers:
                        if imp not in seen:
                            seen.add(imp)
                            aggregated.append(imp)
            return aggregated

    def get_forward_imports(self, file_path: str) -> list[str]:
        """Return list of file paths that the given file imports (forward direction).

        Tries exact path match first, then falls back to suffix matching.
        """
        with self._lock:
            fp_norm = file_path.replace("\\", "/")
            result = self._forward_import_index.get(fp_norm, [])
            if result:
                return list(result)
            for key, imports in self._forward_import_index.items():
                if key.endswith(fp_norm) or fp_norm.endswith(key):
                    return list(imports)
            return []

    # ===================================================================
    # Aggregate Statistics
    # ===================================================================

    def get_stats(self) -> dict:
        """Return aggregate stats per language and totals.

        Excludes ignored paths from counts. Includes any initialization
        errors in the result if present. This is the data returned by
        ``initialize()`` and ``refresh()`` as well as the MCP
        ``get_project_skeleton`` tool.
        """
        with self._lock:
            by_language = {}
            total_files = 0
            total_functions = 0
            total_classes = 0

            for lang, codebase in self._codebases.items():
                try:
                    files = sum(
                        1 for f in codebase.files
                        if not self._is_ignored_file_path(
                            self.translate_path(str(safe_get_attr(f, "path", safe_get_attr(f, "filepath", ""))))
                        )
                    )
                    functions = sum(
                        1 for func in codebase.functions
                        if not self._is_ignored_file_path(
                            self.translate_path(str(safe_get_attr(func, "filepath", "")))
                        )
                    )
                    classes = sum(
                        1 for cls in codebase.classes
                        if not self._is_ignored_file_path(
                            self.translate_path(str(safe_get_attr(cls, "filepath", "")))
                        )
                    )
                    by_language[lang] = {
                        "files": files,
                        "functions": functions,
                        "classes": classes,
                    }
                    total_files += files
                    total_functions += functions
                    total_classes += classes
                except Exception as e:
                    logger.error(f"Error getting stats for {lang}: {e}")
                    by_language[lang] = {
                        "files": 0,
                        "functions": 0,
                        "classes": 0,
                        "error": str(e),
                    }

            result = {
                "languages": list(self._codebases.keys()),
                "total_files": total_files,
                "total_functions": total_functions,
                "total_classes": total_classes,
                "by_language": by_language,
            }
            if self._init_errors:
                result["errors"] = self._init_errors
            return result
