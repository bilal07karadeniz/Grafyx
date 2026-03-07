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

        # Build symbols list grouped by file
        symbols = []
        for file_path in sorted(file_symbols.keys()):
            data = file_symbols[file_path]
            filename = os.path.basename(file_path)
            symbols.append({
                "file": filename,
                "file_path": file_path,
                "functions": data["functions"],
                "classes": data["classes"],
            })

        # Build internal import relationships within this module
        internal_imports = []
        for file_path in sorted(file_symbols.keys()):
            # Try both original and mirror paths for import lookup
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

        # Count totals
        total_functions = sum(len(s["functions"]) for s in symbols)
        total_classes = sum(len(s["classes"]) for s in symbols)

        result = {
            "module": module_filter or "(root)",
            "files": len(file_symbols),
            "total_functions": total_functions,
            "total_classes": total_classes,
            "symbols": symbols,
            "internal_imports": internal_imports,
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
