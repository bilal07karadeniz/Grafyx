"""MCP Tools for graph traversal: dependency graphs and call graphs.

This module provides two tools that let an AI assistant explore
**relationships** between symbols:

- **get_dependency_graph** (Tool 7) -- what a symbol depends ON and what
  depends on IT, with transitive expansion up to a configurable depth.
  Combines three data sources: graph-sitter usages, import index, and
  caller index.
- **get_call_graph** (Tool 9) -- recursive call tree (both outgoing calls
  and incoming callers) for a function.  Supports disambiguation and
  cross-language filtering.

These tools are the core of "blast radius" analysis -- understanding what
would break if a symbol or module were changed.

Registered on the shared ``mcp`` instance from ``_state.py``.
"""

from typing import Any

from fastmcp.exceptions import ToolError

from grafyx.constants import PYTHON_BUILTINS
from grafyx.server._state import _ensure_initialized, mcp
from grafyx.utils import (
    EXTENSION_TO_LANGUAGE,
    format_function_signature,
    safe_get_attr,
    safe_str,
    truncate_response,
)


def _suggest_alternatives(graph, query: str, max_suggestions: int = 5) -> list[str]:
    """Find symbols with names containing the query tokens."""
    query_tokens = set(query.lower().replace(".", " ").split())
    suggestions: list[tuple[str, int]] = []

    for func in graph.get_all_functions(max_results=2000):
        name = func.get("name", "")
        name_lower = name.lower()
        hits = sum(1 for t in query_tokens if t in name_lower)
        if hits > 0:
            file_path = func.get("file", "")
            label = f"{name} (function in {file_path.rsplit('/', 1)[-1]})"
            suggestions.append((label, hits))

    for cls in graph.get_all_classes(max_results=1000):
        name = cls.get("name", "")
        name_lower = name.lower()
        hits = sum(1 for t in query_tokens if t in name_lower)
        if hits > 0:
            file_path = cls.get("file", "")
            label = f"{name} (class in {file_path.rsplit('/', 1)[-1]})"
            suggestions.append((label, hits))

    suggestions.sort(key=lambda x: -x[1])
    return [s[0] for s in suggestions[:max_suggestions]]


def _is_external_dependency(graph, symbol_file: str, dep_name: str, dep_file: str) -> bool:
    """Check if a dependency is actually external (not imported from the resolved local file).

    When graph-sitter resolves a dependency name to a local file, but the actual
    import statement imports that name from an external package, the dependency
    should be treated as external.

    Strategy:
    1. Check if the symbol's file imports this name from the dep's resolved file.
       If yes -> it's local (correct resolution).
    2. If the file has imports but none import this name from dep_file,
       it's likely an external import that graph-sitter misresolved.
    """
    sym_imports = getattr(graph, "_file_symbol_imports", None)
    if not isinstance(sym_imports, dict):
        return False  # No import tracking data available

    file_imports = sym_imports.get(symbol_file, {})

    # If we have no symbol-level import info for this file, can't verify
    if not file_imports:
        return False

    # Check if this file explicitly imports this name from the resolved file
    if dep_file in file_imports:
        imported_names = file_imports[dep_file]
        if dep_name in imported_names:
            return False  # Confirmed: this file imports dep_name from dep_file

    # Check if this name is imported from a DIFFERENT local file
    for target_file, names in file_imports.items():
        if dep_name in names:
            return False  # Imported from a different local file, not external

    # Not found in any local import — likely external
    return True


# --- Tool 7: get_dependency_graph ---


@mcp.tool
def get_dependency_graph(symbol_name: str, depth: int = 2) -> dict:
    """Get the dependency graph for a symbol: what it depends ON and
    what depends on IT, up to the specified depth.

    Use this to understand the impact radius of changing a symbol.
    """
    graph = _ensure_initialized()
    try:
        # --- Symbol resolution ---
        # Check classes first (exact match, never ambiguous) so that
        # passing a class name like "CrawlCache" works directly instead
        # of falling into function disambiguation with its 20 methods.
        class_result = graph.get_class(symbol_name)
        if class_result is not None and isinstance(class_result, (tuple, list)):
            if isinstance(class_result, list):
                lang, symbol = class_result[0]
            else:
                lang, symbol = class_result
            kind = "class"
        else:
            # Not a class — try function lookup with disambiguation
            func_result = graph.get_function(symbol_name)
            if isinstance(func_result, list) and len(func_result) > 1:
                disambiguation = []
                for _lang, func, cls_name in func_result:
                    qualified = f"{cls_name}.{safe_get_attr(func, 'name', '?')}" if cls_name else safe_get_attr(func, "name", "?")
                    disambiguation.append({
                        "qualified_name": qualified,
                        "file": graph.translate_path(str(safe_get_attr(func, "filepath", ""))),
                        "signature": format_function_signature(func),
                    })
                return truncate_response({
                    "ambiguous": True,
                    "message": (
                        f"Multiple symbols named '{symbol_name}' found. "
                        f"Use 'ClassName.method_name' syntax to disambiguate."
                    ),
                    "matches": disambiguation,
                })

            symbol_result = graph.get_symbol(symbol_name)
            if symbol_result is None:
                suggestions = _suggest_alternatives(graph, symbol_name)
                msg = f"Symbol '{symbol_name}' not found."
                if suggestions:
                    msg += " Did you mean: " + ", ".join(suggestions)
                return {"found": False, "message": msg, "suggestions": suggestions}

            lang, symbol, kind = symbol_result

        context = {
            "symbol": symbol_name,
            "kind": kind,
            "file": graph.resolve_path(str(safe_get_attr(symbol, "filepath", ""))),
            "language": lang,
            "depends_on": [],
            "depended_on_by": [],
        }

        # ================================================================
        # Forward dependencies: what this symbol depends ON
        # ================================================================
        # Recursively expand graph-sitter's .dependencies attribute.
        # External (stdlib/third-party) dependencies are counted but not
        # listed, to keep the output focused on project-internal coupling.

        external_dep_count = [0]  # Mutable counter for closure access
        project_root = graph.original_path.replace("\\", "/").rstrip("/") + "/"

        def _is_project_file(fpath: str) -> bool:
            """Check if a file path is inside the project root."""
            if not fpath:
                return False
            fp = fpath.replace("\\", "/")
            return fp.startswith(project_root) or project_root.rstrip("/") in fp

        def _get_deps(sym: Any, current_depth: int, visited: set) -> list[dict]:
            """Recursively collect dependencies, filtering out externals.

            Each dependency entry contains name, file, and kind.  At
            depth > 1, entries also contain a nested ``depends_on`` list
            for transitive dependencies.  Cycles are prevented via the
            ``visited`` set.
            """
            deps = safe_get_attr(sym, "dependencies", [])
            result = []
            if not deps:
                return result
            for d in deps:
                d_name = safe_str(safe_get_attr(d, "name", str(d)))
                # Skip already-visited to prevent infinite recursion
                if d_name in visited:
                    continue
                visited.add(d_name)
                entry: dict = {
                    "name": d_name,
                    "file": graph.resolve_path(graph.get_filepath_from_obj(d)),
                    "kind": (
                        type(d).__name__.lower()
                        if hasattr(d, "__class__")
                        else "unknown"
                    ),
                }
                # Filter out stdlib/third-party deps (outside project root).
                # We count them so the AI knows they exist but don't list
                # them to avoid noise.
                if not _is_project_file(entry["file"]):
                    external_dep_count[0] += 1
                    continue
                # P1 fix: also filter deps that graph-sitter misresolved to local files
                # when the actual import is from an external package.
                if _is_external_dependency(graph, context["file"], d_name, entry["file"]):
                    external_dep_count[0] += 1
                    continue
                # At depth > 1, resolve the dependency symbol and recurse
                # to find its transitive dependencies.
                if current_depth > 1:
                    resolved = graph.get_symbol(d_name)
                    if resolved:
                        _, dep_sym, _ = resolved
                        transitive = _get_deps(dep_sym, current_depth - 1, visited)
                        if transitive:
                            entry["depends_on"] = transitive
                result.append(entry)
            return result

        # Start recursive dependency collection from the target symbol.
        # The symbol itself is pre-added to the visited set.
        context["depends_on"] = _get_deps(symbol, depth, {symbol_name})
        if external_dep_count[0] > 0:
            context["external_dependency_count"] = external_dep_count[0]

        # ================================================================
        # Reverse dependencies: what depends ON this symbol
        # ================================================================
        # Three complementary data sources are combined for comprehensive
        # coverage.  Self-references (same file) are excluded.
        #
        #   Source 1: graph-sitter .usages (direct AST references)
        #   Source 2: import index (files importing this symbol's module)
        #   Source 3: caller index (for classes: callers of unique methods)

        symbol_file = context["file"]
        by_file: dict[str, int] = {}  # file -> reference count

        # --- Source 1: graph-sitter usages ---
        # Direct AST-level references found by the parser.
        usages = safe_get_attr(symbol, "usages", [])
        if usages:
            for u in usages:
                u_file = graph.resolve_path(graph.get_filepath_from_obj(u))
                if u_file and u_file != symbol_file:
                    by_file[u_file] = by_file.get(u_file, 0) + 1

        # --- Source 2: import index ---
        # Files that import this symbol's module.  graph-sitter usages may
        # miss import-based references, especially for classes used as type
        # hints or constructor calls.
        # Symbol-level filter: only count files that ACTUALLY import THIS
        # symbol (not just any symbol from the same file).
        if symbol_file:
            importers = graph.get_importers(symbol_file)
            sym_imports = getattr(graph, "_file_symbol_imports", {})
            for imp_file in importers:
                if imp_file == symbol_file or imp_file in by_file:
                    continue
                # If symbol-level import info is available, verify this
                # file actually imports the target symbol specifically.
                # An empty set means wildcard/bare import -> include conservatively.
                if sym_imports and imp_file in sym_imports:
                    imported_names = sym_imports[imp_file].get(symbol_file, set())
                    # Empty set = wildcard or bare import -> include conservatively
                    if imported_names and symbol_name not in imported_names:
                        continue
                by_file[imp_file] = 1

        # --- Source 2b: Transitive through __init__.py ---
        # If auth_service.py is imported by auth/__init__.py, and router.py
        # imports auth/__init__.py with the same symbol name, then router.py
        # transitively depends on auth_service.py.
        if symbol_file:
            init_importers = graph.get_importers(symbol_file)
            for init_file in init_importers:
                init_norm = graph.resolve_path(init_file) if init_file else init_file
                if not init_norm or not init_norm.endswith("/__init__.py"):
                    continue
                # Verify __init__.py actually re-exports our symbol
                sym_imports = getattr(graph, "_file_symbol_imports", {})
                if not isinstance(sym_imports, dict):
                    continue
                init_imports = sym_imports.get(init_norm, {})
                imported_names = init_imports.get(symbol_file, set())
                if imported_names and symbol_name not in imported_names:
                    continue  # __init__.py doesn't re-export this symbol
                # Find files that import this __init__.py
                transitive_importers = graph.get_importers(init_norm)
                for trans_file in transitive_importers:
                    trans_norm = graph.resolve_path(trans_file) if trans_file else trans_file
                    if not trans_norm or trans_norm == symbol_file or trans_norm in by_file:
                        continue
                    # Optionally verify the transitive importer imports our symbol name
                    trans_imports = sym_imports.get(trans_norm, {})
                    trans_names = trans_imports.get(init_norm, set())
                    if trans_names and symbol_name not in trans_names:
                        continue
                    by_file[trans_norm] = 1

        # --- Source 3: caller index for unique class methods ---
        # For class symbols only.  Methods like "emit" that only exist on
        # EventBus are great signals -- any caller of "emit" is definitely
        # an EventBus dependent.  But "execute" exists on 5+ classes, so
        # callers of "execute" can't be reliably attributed -- skip those.
        if kind == "class":
            methods = safe_get_attr(symbol, "methods", [])
            if methods:
                for method in methods:
                    m_name = safe_get_attr(method, "name", "")
                    if not m_name or m_name.startswith("__"):
                        continue
                    # Skip methods whose name exists in OTHER classes --
                    # these would cause cross-class contamination.
                    other_classes = sum(
                        1 for cls_n, meths in graph._class_method_names.items()
                        if cls_n != symbol_name and m_name in meths
                    )
                    if other_classes > 0:
                        continue
                    callers = graph.get_callers(m_name)
                    for caller in callers:
                        c_file = graph.resolve_path(caller.get("file", ""))
                        if c_file and c_file != symbol_file and c_file not in by_file:
                            by_file[c_file] = 1

        # --- Normalize and deduplicate paths ---
        # Different sources may produce the same file under different path
        # formats (absolute vs relative, mirror vs original).  resolve_path()
        # canonicalizes all of them so we don't double-count.
        if by_file:
            normalized_by_file: dict[str, int] = {}
            for raw_path, count in by_file.items():
                key = graph.resolve_path(raw_path)
                if key:
                    normalized_by_file[key] = normalized_by_file.get(key, 0) + count
            by_file = normalized_by_file

        # Sort by reference count descending
        if by_file:
            context["depended_on_by"] = [
                {"file": f, "count": c}
                for f, c in sorted(by_file.items(), key=lambda x: -x[1])
            ]

        return truncate_response(context)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_dependency_graph: {e}")


# --- Tool 9: get_call_graph ---


@mcp.tool
def get_call_graph(
    function_name: str, depth: int = 3, include_builtins: bool = False,
) -> dict:
    """Get the call graph for a function: what it calls and what calls it,
    recursively up to the specified depth.

    Supports 'ClassName.method_name' syntax (e.g., 'SkillManager.initialize').
    If multiple functions share the same name, returns a disambiguation list.

    Use this to trace execution flow and understand how a function fits
    into the larger call chain. By default, Python builtins and common
    stdlib methods (len, append, strip, etc.) are filtered out for clarity.
    Set include_builtins=True to see everything.
    """
    graph = _ensure_initialized()
    try:
        result = graph.get_function(function_name)
        if result is None:
            suggestions = _suggest_alternatives(graph, function_name)
            msg = f"Function '{function_name}' not found."
            if suggestions:
                msg += " Did you mean: " + ", ".join(suggestions)
            return {"found": False, "message": msg, "suggestions": suggestions}

        # --- Disambiguation (same pattern as get_function_context) ---
        if isinstance(result, list):
            if len(result) > 1:
                disambiguation = []
                for lang, func, cls_name in result:
                    qualified = f"{cls_name}.{safe_get_attr(func, 'name', '?')}" if cls_name else safe_get_attr(func, "name", "?")
                    disambiguation.append({
                        "qualified_name": qualified,
                        "file": graph.translate_path(str(safe_get_attr(func, "filepath", ""))),
                        "signature": format_function_signature(func),
                    })
                return truncate_response({
                    "ambiguous": True,
                    "message": (
                        f"Multiple functions named '{function_name}' found. "
                        f"Use 'ClassName.method_name' syntax to disambiguate."
                    ),
                    "matches": disambiguation,
                })
            lang, func, cls_name = result[0]
        else:
            lang, func, cls_name = result

        # Determine which names to skip in the tree.  By default we filter
        # PYTHON_BUILTINS to keep the graph focused on application logic.
        skip = set() if include_builtins else PYTHON_BUILTINS

        # ================================================================
        # Build outgoing call tree (what this function calls, recursively)
        # ================================================================
        def _build_calls_tree(fn, current_depth: int, visited: set) -> dict:
            """Build a nested dict of {callee_name: {its_callees...}}.

            Each level represents one hop in the call chain.  Cycles are
            detected via the ``visited`` set.  The tree terminates when
            depth reaches 0 or a cycle is detected.
            """
            name = safe_get_attr(fn, "name", "?")
            if current_depth <= 0 or name in visited:
                return {}
            visited.add(name)
            tree = {}
            calls = safe_get_attr(fn, "function_calls", [])
            if calls:
                for called in calls:
                    called_name = safe_get_attr(called, "name", "?")
                    if called_name not in visited and called_name not in skip:
                        # Resolve FunctionCall reference to actual Function
                        # object for deeper expansion.  FunctionCall objects
                        # (from .function_calls) don't have their own
                        # .function_calls, so we look up the real Function
                        # by name in the graph.
                        resolved = called
                        if current_depth > 1:
                            result = graph.get_function(called_name)
                            if result is not None:
                                if isinstance(result, list):
                                    resolved = result[0][1]
                                else:
                                    resolved = result[1]
                        tree[called_name] = _build_calls_tree(
                            resolved, current_depth - 1, visited
                        )
            return tree

        # ================================================================
        # Build incoming caller tree (who calls this function, recursively)
        # ================================================================
        def _build_callers_tree(fn_name: str, current_depth: int, visited: set,
                                fn_class: str | None = None,
                                lang_filter: str = "") -> dict:
            """Build a nested dict of {caller_name: {its_callers...}}.

            Uses the reverse caller index (graph.get_callers).  The
            fn_class parameter ensures we don't mix up methods with the
            same name from different classes.

            Cross-language filtering: if we're analyzing a Python function,
            callers from TypeScript files are excluded.  This prevents
            false positives when frontend and backend share endpoint names.
            """
            if current_depth <= 0 or fn_name in visited:
                return {}
            visited.add(fn_name)
            tree = {}
            callers = graph.get_callers(fn_name, class_name=fn_class)
            for caller in callers:
                caller_name = caller.get("name", "?")
                caller_file = caller.get("file", "")

                # --- Cross-language filter ---
                # A TypeScript component calling a same-named wrapper must
                # not appear as a caller of the Python backend function.
                if lang_filter and caller_file and "." in caller_file:
                    caller_ext = "." + caller_file.rsplit(".", 1)[-1]
                    caller_lang = EXTENSION_TO_LANGUAGE.get(caller_ext, "")
                    if caller_lang and caller_lang != lang_filter:
                        continue

                caller_class = caller.get("class")
                # Resolve class from file when parent_class wasn't detected
                # by the caller index.  Without this, depth-2 expansion of
                # a common name like "execute" passes fn_class=None and
                # returns unfiltered callers from all classes.
                if not caller_class:
                    caller_class = graph.resolve_method_class(
                        caller_name, caller_file
                    )
                if caller_name not in visited and caller_name not in skip:
                    tree[caller_name] = _build_callers_tree(
                        caller_name, current_depth - 1, visited,
                        fn_class=caller_class, lang_filter=lang_filter,
                    )
            return tree

        # Build both trees starting from the target function
        calls_tree = _build_calls_tree(func, depth, set())
        callers_tree = _build_callers_tree(
            safe_get_attr(func, "name", function_name), depth, set(),
            fn_class=cls_name, lang_filter=lang,
        )

        # --- Flatten trees for convenience ---
        # The tree structure is great for visualization, but a flat list
        # is easier for the AI to scan quickly.
        def _flatten(tree: dict) -> list[str]:
            """Depth-first flattening of a nested name dict."""
            names = []
            for name, subtree in tree.items():
                names.append(name)
                names.extend(_flatten(subtree))
            return names

        context = {
            "function": function_name,
            "file": graph.translate_path(str(safe_get_attr(func, "filepath", ""))),
            "language": lang,
            "depth": depth,
            "builtins_filtered": not include_builtins,
            "calls_tree": {function_name: calls_tree},
            "callers_tree": {function_name: callers_tree},
            "flat_calls": _flatten(calls_tree),
            "flat_callers": _flatten(callers_tree),
        }

        return truncate_response(context)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_call_graph: {e}")
