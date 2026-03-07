"""MCP Tools for code quality analysis and maintenance.

This module groups five tools focused on codebase health and maintenance:

- **get_conventions** (Tool 8) -- detects naming conventions, style
  patterns, and coding norms so the AI writes code that matches the
  project's existing style.
- **refresh_graph** (Tool 10) -- forces a full re-parse of the codebase
  when the file watcher hasn't caught up or after bulk changes.
- **get_unused_symbols** (Tool 11) -- dead code detection.  Finds
  functions and classes with zero inbound references.
- **get_circular_dependencies** (Tool 12) -- finds import cycles that
  could cause runtime ImportError or indicate tight coupling.
- **get_subclasses** (Tool 13) -- recursively finds all subclasses of a
  given class, useful for understanding the blast radius of interface
  changes.

Registered on the shared ``mcp`` instance from ``_state.py``.
"""

import time
from typing import Any

from fastmcp.exceptions import ToolError

from grafyx.server import _state
from grafyx.server._state import _ensure_initialized, mcp
from grafyx.utils import truncate_response


# --- Tool 8: get_conventions ---


@mcp.tool
def get_conventions() -> dict:
    """Detect and report coding conventions used in this codebase.

    Use this to understand the project's style BEFORE writing new code,
    so your additions are consistent with existing patterns.
    """
    _ensure_initialized()
    try:
        if _state._detector is None:
            raise ToolError("Convention detector not initialized.")

        # ConventionDetector.detect_all() scans the graph for patterns like
        # naming style (snake_case vs camelCase), docstring format, import
        # ordering, etc.  Each convention has a confidence score 0.0-1.0.
        conventions = _state._detector.detect_all()

        # Build a quick summary from high-confidence conventions (>= 70%).
        # This gives the AI a one-line overview without reading every entry.
        high_conf = [c for c in conventions if c.get("confidence", 0) >= 0.7]
        summary_parts = [c["pattern"] for c in high_conf[:5]]

        return truncate_response({
            "conventions": conventions,
            "summary": (
                "; ".join(summary_parts)
                if summary_parts
                else "No strong conventions detected."
            ),
            "total_detected": len(conventions),
        })
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_conventions: {e}")


# --- Tool 10: refresh_graph ---


@mcp.tool
def refresh_graph() -> dict:
    """Force a full re-parse of the entire codebase graph.

    Use this after making significant changes to the codebase, or if
    the graph seems stale. Normally the file watcher handles this automatically.
    """
    graph = _ensure_initialized()
    try:
        start = time.time()
        # graph.refresh() re-initializes the entire CodebaseGraph: re-parses
        # all files, rebuilds the caller index, import index, and all other
        # derived data structures.
        result = graph.refresh()
        duration = round(time.time() - start, 2)

        return {
            "status": "refreshed",
            "duration_seconds": duration,
            "languages": result.get("languages", []),
            "stats": graph.get_stats(),
        }
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in refresh_graph: {e}")


# --- Tool 11: get_unused_symbols ---


@mcp.tool
def get_unused_symbols(
    symbol_type: str = "all",
    include_tests: bool = False,
    max_results: int = 50,
) -> dict:
    """Find functions and classes with zero inbound references (dead code detection).

    Analyzes the caller index and import graph to identify symbols that
    are never called, subclassed, or referenced from other code.

    Excludes common false positives: dunder methods, framework-decorated
    handlers, entry points, and exception classes.

    Args:
        symbol_type: What to search for: "all", "functions", or "classes".
        include_tests: If True, also scan test files for unused code.
        max_results: Maximum results per category.
    """
    graph = _ensure_initialized()
    try:
        result: dict[str, Any] = {
            "include_tests": include_tests,
        }

        # --- Unused functions ---
        # graph.get_unused_functions() uses a fast pre-check on the
        # _caller_index (O(1) lookup) and only falls back to full
        # get_callers() disambiguation for ambiguous names.  This keeps
        # the operation fast even on large codebases.
        if symbol_type in ("all", "functions"):
            unused_funcs = graph.get_unused_functions(
                include_tests=include_tests,
                max_results=max_results,
            )
            result["unused_functions"] = unused_funcs
            result["unused_function_count"] = len(unused_funcs)

        # --- Unused classes ---
        # Similar approach: checks usages, import index, and subclass
        # references.  Exception classes are excluded as false positives
        # (they're "used" by being raised, which graph-sitter may not track).
        if symbol_type in ("all", "classes"):
            unused_classes = graph.get_unused_classes(
                include_tests=include_tests,
                max_results=max_results,
            )
            result["unused_classes"] = unused_classes
            result["unused_class_count"] = len(unused_classes)

        # --- Summary ---
        total = result.get("unused_function_count", 0) + result.get("unused_class_count", 0)
        result["total_unused"] = total

        if total == 0:
            result["summary"] = "No unused symbols detected."
        else:
            parts = []
            if "unused_function_count" in result:
                parts.append(f"{result['unused_function_count']} functions/methods")
            if "unused_class_count" in result:
                parts.append(f"{result['unused_class_count']} classes")
            result["summary"] = f"Found {total} potentially unused symbols: {', '.join(parts)}."

        return truncate_response(result)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_unused_symbols: {e}")


# --- Tool 12: get_circular_dependencies ---


@mcp.tool
def get_circular_dependencies(
    include_tests: bool = False,
    max_chain_length: int = 10,
) -> dict:
    """Find circular import chains in the codebase.

    Detects files that form import cycles (A imports B, B imports A),
    including transitive cycles (A -> B -> C -> A).

    Circular imports can cause ImportError at runtime, make code harder
    to reason about, and indicate architectural coupling that should
    be refactored.

    Args:
        include_tests: If True, include test files in analysis.
        max_chain_length: Maximum cycle length to report.
    """
    graph = _ensure_initialized()
    try:
        # graph.find_import_cycles() uses DFS on the _forward_import_index
        # to detect all cycles up to max_chain_length.
        cycles = graph.find_import_cycles(max_chain_length=max_chain_length)

        # Filter out test-only cycles unless explicitly requested.
        # A cycle where ALL files are tests is usually harmless (test
        # helpers importing each other).
        if not include_tests:
            cycles = [
                c for c in cycles
                if not all(graph._is_test_path(f) for f in c[:-1])
            ]

        # Translate internal mirror paths back to user-facing paths.
        # Each cycle is a list like [A, B, C, A] where the last element
        # repeats the first to show the cycle completes.
        translated_cycles = []
        for cycle in cycles:
            translated = [graph.translate_path(f) for f in cycle]
            translated_cycles.append({
                "chain": translated,
                "length": len(translated) - 1,  # Subtract 1 because last == first
                # Deduplicated list of unique files in the cycle
                "files_involved": list(dict.fromkeys(translated[:-1])),
            })

        result: dict[str, Any] = {
            "cycles": translated_cycles,
            "cycle_count": len(translated_cycles),
            "include_tests": include_tests,
        }

        if not translated_cycles:
            result["summary"] = "No circular import dependencies detected."
        else:
            lengths = [c["length"] for c in translated_cycles]
            result["summary"] = (
                f"Found {len(translated_cycles)} circular import chain(s). "
                f"Shortest: {min(lengths)} files, longest: {max(lengths)} files."
            )

        return truncate_response(result)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_circular_dependencies: {e}")


# --- Tool 13: get_subclasses ---


@mcp.tool
def get_subclasses(class_name: str, depth: int = 3) -> dict:
    """Find all classes that extend a given class, recursively.

    Use this to understand the impact of changing a class's interface:
    which subclasses would need updating?

    Supports multi-level inheritance trees up to the specified depth.
    """
    graph = _ensure_initialized()
    try:
        # graph.get_subclasses() walks the _subclass_index (built during
        # graph initialization from extract_base_classes) and returns a
        # nested tree of {class_name, file, subclasses: [...]}.
        result = graph.get_subclasses(class_name, depth)
        if result is None:
            raise ToolError(f"Class '{class_name}' not found in the codebase.")
        return truncate_response(result)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_subclasses: {e}")
