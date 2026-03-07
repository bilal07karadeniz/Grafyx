"""MCP Tool: get_project_skeleton -- high-level project overview.

This module defines a single tool that gives an AI assistant its first
look at a codebase.  It returns:

- Global statistics (file/function/class counts, languages detected)
- Per-directory breakdown (so the AI knows which packages are code-heavy)
- A visual file tree (respecting ``max_depth`` to avoid flooding context)

This is typically the **first tool an AI calls** when it starts working on
a project, before drilling into specific files or symbols.

Registered on the shared ``mcp`` instance from ``_state.py``.
"""

from fastmcp.exceptions import ToolError

from grafyx.server._hints import compute_hints
from grafyx.server._resolution import filter_by_detail
from grafyx.server._state import _ensure_initialized, mcp
from grafyx.utils import build_directory_tree, truncate_response


# --- Tool 1: get_project_skeleton ---


@mcp.tool
def get_project_skeleton(max_depth: int = 3, detail: str = "summary", include_hints: bool = True) -> dict:
    """Get the full project structure with file tree and statistics.

    Call this first when starting work on a project to understand its
    overall shape, module layout, and code distribution.

    Args:
        max_depth: Maximum depth for the file tree display.
        detail: Level of detail: "signatures", "summary" (default), or "full".
        include_hints: If True, append navigation suggestions.
    """
    graph = _ensure_initialized()
    try:
        stats = graph.get_stats()

        # Collect every file the graph knows about.  Each entry is a dict
        # with keys like "path", "function_count", "class_count".
        all_files = graph.get_all_files()
        file_paths = [f.get("path", "") for f in all_files if f.get("path")]

        # build_directory_tree() produces a nested dict like:
        #   {"grafyx/": {"server/": {"__init__.py": None, ...}, ...}}
        # max_depth controls how many levels deep the tree goes.
        tree = build_directory_tree(file_paths, graph.original_path, max_depth=max_depth)

        # --- Per-directory statistics ---
        # Aggregate files, functions, and classes by top-level directory.
        # This tells the AI which packages are most code-dense.
        dir_stats: dict[str, dict[str, int]] = {}

        # Normalize both the user-facing path (original_path) and the
        # internal mirror path (project_path) so prefix stripping works
        # regardless of OS path separators or symlinks.
        roots = [
            graph.original_path.replace("\\", "/").rstrip("/") + "/",
            graph.project_path.replace("\\", "/").rstrip("/") + "/",
        ]
        for f in all_files:
            fpath = f.get("path", "").replace("\\", "/")
            if not fpath:
                continue

            # Strip the project root prefix to get a relative path
            # (e.g. "grafyx/server/__init__.py")
            rel = fpath
            for root in roots:
                if fpath.startswith(root):
                    rel = fpath[len(root):]
                    break

            # Use the first path component as the top-level directory key.
            # Files directly in the project root get the "." bucket.
            parts = rel.split("/")
            dir_key = parts[0] if len(parts) > 1 else "."
            if dir_key not in dir_stats:
                dir_stats[dir_key] = {"files": 0, "functions": 0, "classes": 0}
            dir_stats[dir_key]["files"] += 1
            dir_stats[dir_key]["functions"] += f.get("function_count", 0)
            dir_stats[dir_key]["classes"] += f.get("class_count", 0)

        # Sort directories by file count descending so the most important
        # packages appear first in the response.
        sorted_dir_stats = dict(
            sorted(dir_stats.items(), key=lambda x: x[1]["files"], reverse=True)
        )

        result = {
            "project_path": graph.original_path,
            "languages": stats.get("languages", []),
            "total_files": stats.get("total_files", 0),
            "total_functions": stats.get("total_functions", 0),
            "total_classes": stats.get("total_classes", 0),
            "by_language": stats.get("by_language", {}),
            "directory_stats": sorted_dir_stats,
            "file_tree": tree,
        }
        # Apply detail-level filtering
        result = filter_by_detail(result, detail, "skeleton")

        # Compute navigation hints (suggest most interesting modules)
        if include_hints:
            hints = compute_hints(graph, "skeleton", result)
            if hints:
                result["suggested_next"] = hints

        # truncate_response() caps the JSON size to avoid exceeding the
        # MCP client's context window.
        return truncate_response(result)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_project_skeleton: {e}")
