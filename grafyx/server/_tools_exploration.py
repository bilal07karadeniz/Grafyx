"""MCP Tool: get_module_context -- intermediate zoom level.

This module provides the "Level 1.5" tool that bridges the gap between
`get_project_skeleton` (bird's eye view) and `get_file_context` (single
file deep-dive).

It answers: "What symbols exist in this directory/package?" with
configurable resolution:

    - **signatures**: Just names and signatures (minimal tokens)
    - **summary**: Names, signatures, first-line docstrings, internal
      import relationships (default)
    - **full**: Everything including source code

This is the key tool for the "Fractal Context" pattern -- an LLM sees
the project skeleton, then zooms into a module to see what's inside,
then zooms into a specific class or function.

    get_project_skeleton  ->  get_module_context  ->  get_file/class/function_context

Registered on the shared ``mcp`` instance from ``_state.py``.
"""

import os

from fastmcp.exceptions import ToolError

from grafyx.server._hints import compute_hints
from grafyx.server._resolution import filter_by_detail
from grafyx.server._state import _ensure_initialized, mcp
from grafyx.utils import safe_get_attr, safe_str, truncate_response


@mcp.tool
def get_module_context(
    module_path: str = "",
    detail: str = "summary",
    include_hints: bool = True,
    offset: int = 0,
    limit: int = 0,
) -> dict:
    """Get an overview of all symbols in a directory/package.

    This is the intermediate zoom level between get_project_skeleton
    (whole project) and get_file_context (single file). Use it to
    understand what a package contains before drilling into specifics.

    Args:
        module_path: Directory path relative to project root (e.g. "services",
            "grafyx/graph"). Empty string means the whole project.
        detail: Level of detail: "signatures", "summary" (default), or "full".
        include_hints: If True, append navigation suggestions.
        offset: Number of files to skip (for paginating large modules).
        limit: Maximum number of files to return (0 = no limit, default).
            Combine with offset to page through modules with hundreds
            of files where the response would otherwise be truncated.
            ``total_files`` in the response always reflects the full
            count, regardless of pagination.
    """
    graph = _ensure_initialized()
    try:
        # Normalize paths for comparison
        project_root = graph.original_path.replace("\\", "/").rstrip("/") + "/"
        mirror_root = graph.project_path.replace("\\", "/").rstrip("/") + "/"
        module_filter = module_path.replace("\\", "/").strip("/")

        # Collect all files, functions, classes from the graph
        # Use high max_results to avoid truncation on large projects
        all_files = graph.get_all_files(max_results=5000)
        all_functions = graph.get_all_functions(max_results=5000)
        all_classes = graph.get_all_classes(max_results=5000, include_method_names=True)

        def _get_rel_path(abs_path: str) -> str | None:
            """Convert absolute or relative path to project-relative path.

            Handles three cases:
            - Absolute with original root: /mnt/c/.../Grafyx/grafyx/graph/core.py
            - Absolute with mirror root: /home/bilal/.grafyx/mirrors/.../grafyx/graph/core.py
            - Already relative: grafyx/graph/core.py (from translate_path)
            """
            if not abs_path:
                return None
            p = abs_path.replace("\\", "/")
            for root in (project_root, mirror_root):
                if p.startswith(root):
                    return p[len(root):]
            # Already a relative path (translate_path already stripped root)
            return p

        def _matches_module(rel_path: str) -> bool:
            """Check if a relative path belongs to the target module."""
            if not module_filter:
                return True
            return (
                rel_path.startswith(module_filter + "/")
                or rel_path.startswith(module_filter + "\\")
            )

        # Group symbols by file, filtering to the requested module_path
        file_symbols: dict[str, dict] = {}

        # Index files first
        for f_info in all_files:
            fpath = f_info.get("path", "")
            rel = _get_rel_path(fpath)
            if rel and _matches_module(rel):
                if rel not in file_symbols:
                    file_symbols[rel] = {"functions": [], "classes": []}

        # Index functions into their files
        # Note: get_all_functions() returns "file" key, not "path"
        for func_info in all_functions:
            fpath = func_info.get("file", func_info.get("path", ""))
            rel = _get_rel_path(fpath)
            if rel and _matches_module(rel):
                if rel not in file_symbols:
                    file_symbols[rel] = {"functions": [], "classes": []}
                entry = {
                    "name": func_info.get("name", "?"),
                    "signature": func_info.get("signature", ""),
                }
                if func_info.get("docstring"):
                    entry["docstring"] = func_info["docstring"]
                file_symbols[rel]["functions"].append(entry)

        # Index classes into their files
        # Note: get_all_classes() returns "file" key, not "path"
        for cls_info in all_classes:
            fpath = cls_info.get("file", cls_info.get("path", ""))
            rel = _get_rel_path(fpath)
            if rel and _matches_module(rel):
                if rel not in file_symbols:
                    file_symbols[rel] = {"functions": [], "classes": []}
                # Extract method names from class info
                methods = cls_info.get("methods", [])
                method_names = []
                if isinstance(methods, list):
                    for m in methods:
                        if isinstance(m, dict):
                            method_names.append(m.get("name", "?"))
                        elif isinstance(m, str):
                            method_names.append(m)
                        else:
                            method_names.append(safe_get_attr(m, "name", "?"))
                entry = {
                    "name": cls_info.get("name", "?"),
                    "methods": method_names,
                }
                if cls_info.get("docstring"):
                    entry["docstring"] = cls_info["docstring"]
                if cls_info.get("base_classes"):
                    entry["base_classes"] = cls_info["base_classes"]
                file_symbols[rel]["classes"].append(entry)

        # Order all files first so totals reflect the full module,
        # then slice the symbols list by offset/limit so callers can
        # page through large modules (hundreds of files) without
        # blowing past the response size limit.
        sorted_paths = sorted(file_symbols.keys())
        total_functions = sum(
            len(file_symbols[p]["functions"]) for p in sorted_paths
        )
        total_classes = sum(
            len(file_symbols[p]["classes"]) for p in sorted_paths
        )

        if offset < 0:
            offset = 0
        if limit and limit > 0:
            paged_paths = sorted_paths[offset:offset + limit]
        elif offset > 0:
            paged_paths = sorted_paths[offset:]
        else:
            paged_paths = sorted_paths

        symbols = []
        for file_path in paged_paths:
            data = file_symbols[file_path]
            filename = os.path.basename(file_path)
            symbols.append({
                "file": filename,
                "file_path": file_path,
                "functions": data["functions"],
                "classes": data["classes"],
            })

        # Build internal import relationships within this module.
        # Only compute for files in the paged window so paginated
        # responses don't carry import graphs for files the caller
        # can't see anyway.
        paged_set = set(paged_paths)
        internal_imports = []
        for file_path in paged_paths:
            abs_path = project_root + file_path
            forward = graph.get_forward_imports(abs_path)
            if not forward:
                forward = graph.get_forward_imports(mirror_root + file_path)
            internal = []
            for imp in (forward or []):
                imp_rel = _get_rel_path(imp)
                if imp_rel and imp_rel in file_symbols:
                    internal.append(os.path.basename(imp_rel))
            if internal:
                internal_imports.append({
                    "from": os.path.basename(file_path),
                    "imports": internal,
                })

        result = {
            "module": module_filter or "(root)",
            "files": len(file_symbols),
            "total_functions": total_functions,
            "total_classes": total_classes,
            "symbols": symbols,
            "internal_imports": internal_imports,
        }
        # Surface pagination state when it's actually paged so the
        # caller knows there's more to fetch.
        if offset > 0 or (limit and limit > 0 and len(file_symbols) > limit):
            result["page"] = {
                "offset": offset,
                "limit": limit if limit > 0 else None,
                "returned": len(paged_paths),
                "total_files": len(file_symbols),
                "has_more": (offset + len(paged_paths)) < len(file_symbols),
            }
        elif limit == 0 and len(file_symbols) > 80:
            # Caller didn't paginate but the module is large enough
            # that ``truncate_response`` will trim symbol entries.
            # Surface a concrete next-call suggestion so the AI knows
            # the tool can paginate instead of just being "too big".
            example_module = module_filter or "."
            result["pagination_hint"] = {
                "reason": (
                    f"Module has {len(file_symbols)} files; the default "
                    f"response truncates large lists. Use offset/limit "
                    f"to page through the full module."
                ),
                "suggested_call": (
                    f"get_module_context(module_path={example_module!r}, "
                    f"limit=50, offset=0)  # then increment offset by 50"
                ),
                "total_files": len(file_symbols),
            }

        # Apply detail-level filtering
        result = filter_by_detail(result, detail, "module")

        # Compute navigation hints
        if include_hints:
            hints = compute_hints(graph, "module", result)
            if hints:
                result["suggested_next"] = hints

        return truncate_response(result)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_module_context: {e}")
