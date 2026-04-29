"""Grafyx MCP server package -- public API surface and tool registration.

This is the entry point for the ``grafyx.server`` package.  It has three
responsibilities:

1. **Create the shared FastMCP instance** (``mcp``) and server factory
   (``create_server``) -- both live in ``_state.py``.

2. **Register all MCP tools** by importing each ``_tools_*.py`` sub-module.
   Every sub-module decorates its functions with ``@mcp.tool``, which
   automatically registers them on the shared ``mcp`` instance at import
   time.  The order of imports below defines the tool registration order.

3. **Re-export tool functions** individually so that call-sites outside
   this package (e.g. tests) can do
   ``from grafyx.server import get_class_context`` without knowing which
   sub-module a tool lives in.

Architecture overview::

    __init__.py          <-- you are here (public API + registration)
    _state.py            <-- shared state: graph, searcher, detector, watcher
    _tools_structure.py  <-- Tool 1:  get_project_skeleton
    _tools_introspection.py  <-- Tools 2-4: function / file / class context
    _tools_search.py     <-- Tools 5-6: find_related_code / find_related_files
    _tools_graph.py      <-- Tools 7, 9: dependency graph / call graph
    _tools_quality.py    <-- Tools 8, 10-11, 13: conventions, refresh, unused, subclasses
"""

# --- Core server state and helpers ---
from grafyx.server._state import (
    _find_reference_lines,
    create_server,
    mcp,
)

# --- Tool registration ---
# Import all tool modules to trigger @mcp.tool registration.
# Each module decorates functions with @mcp.tool, which registers
# them on the shared `mcp` FastMCP instance defined in _state.py.
from grafyx.server import (  # noqa: F401
    _tools_structure,
    _tools_introspection,
    _tools_exploration,
    _tools_search,
    _tools_graph,
    _tools_quality,
)

# --- Re-exports for convenient external access ---
# Re-export individual tool functions so that existing imports like
# `from grafyx.server import get_class_context` continue to work.
# This keeps the public API stable regardless of internal refactors.
from grafyx.server._tools_structure import get_project_skeleton  # noqa: F401
from grafyx.server._tools_introspection import (  # noqa: F401
    get_class_context,
    get_file_context,
    get_function_context,
)
from grafyx.server._tools_exploration import get_module_context  # noqa: F401
from grafyx.server._tools_search import (  # noqa: F401
    find_related_code,
    find_related_files,
)
from grafyx.server._tools_graph import (  # noqa: F401
    get_call_graph,
    get_dependency_graph,
)
from grafyx.server._tools_quality import (  # noqa: F401
    get_conventions,
    get_subclasses,
    get_unused_symbols,
    refresh_graph,
)

__all__ = [
    # Server infrastructure
    "create_server",
    "mcp",
    "_find_reference_lines",
    # Tool functions (one per MCP tool endpoint)
    "get_project_skeleton",
    "get_module_context",
    "get_function_context",
    "get_file_context",
    "get_class_context",
    "find_related_code",
    "find_related_files",
    "get_dependency_graph",
    "get_call_graph",
    "get_conventions",
    "get_unused_symbols",
    "get_subclasses",
    "refresh_graph",
]
