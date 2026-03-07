"""Shared MCP server state, initialization, and helper functions.

This module holds the **singleton state** for the entire Grafyx MCP server:

- ``_graph``    -- the CodebaseGraph that parses and indexes the project
- ``_searcher`` -- fuzzy/semantic code search powered by the graph
- ``_detector`` -- coding convention detector
- ``_watcher``  -- file-system watcher that keeps the graph in sync

It also provides:

- ``mcp``              -- the shared FastMCP instance all tools register on
- ``create_server()``  -- factory that wires everything together and
                          optionally defers heavy init to a background thread
- ``_ensure_initialized()`` -- gate that every tool calls first; blocks
                               until the graph is ready
- ``_find_reference_lines()`` -- lightweight grep helper used by
                                 introspection tools when graph-sitter
                                 usages are unavailable

Lifecycle::

    create_server(path, ...)
        |
        +--> _background_init()   [in daemon thread if lazy=True]
        |       |
        |       +--> CodebaseGraph.initialize()
        |       +--> CodeSearcher(graph)
        |       +--> ConventionDetector(graph)
        |       +--> CodebaseWatcher(graph).start()
        |       +--> _init_ready = True
        |
        +--> return mcp            [immediately, even before init completes]

    (any tool call)
        |
        +--> _ensure_initialized() [blocks up to 120 s until _init_ready]
        +--> use _graph / _searcher / _detector
"""

import logging
import threading
import time

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from grafyx.conventions import ConventionDetector
from grafyx.graph import CodebaseGraph
from grafyx.search import CodeSearcher
from grafyx.watcher import CodebaseWatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton state.  These are set once during initialization
# and then read by every tool function.  Access is safe because:
#   - writes happen in _background_init() before _init_ready is set
#   - reads happen in tools only AFTER _ensure_initialized() confirms
#     _init_ready is True (memory ordering via time.sleep loop)
# ---------------------------------------------------------------------------
_graph: CodebaseGraph | None = None
_searcher: CodeSearcher | None = None
_detector: ConventionDetector | None = None
_watcher: CodebaseWatcher | None = None
_init_ready = False  # Set True after background initialization completes

# The single FastMCP instance that all tool modules register on.
# Created at module-load time so @mcp.tool decorators work immediately.
mcp = FastMCP("Grafyx")


# ---------------------------------------------------------------------------
# Helper: lightweight line-number search
# ---------------------------------------------------------------------------

def _find_reference_lines(file_path: str, name: str, max_lines: int = 3) -> list[int]:
    """Find line numbers where ``name`` appears in a source file.

    This is a simple text search (not AST-aware) used as a fallback when
    graph-sitter usages are unavailable.  It is called by the
    introspection tools (get_class_context, get_dependency_graph) to
    populate ``cross_file_usages`` line numbers from the import/caller
    index.

    Args:
        file_path: Absolute path to the source file to scan.
        name: The symbol name to search for (substring match per line).
        max_lines: Stop after finding this many matching lines.

    Returns:
        A list of 1-based line numbers where ``name`` appears, up to
        ``max_lines`` entries.
    """
    try:
        with open(file_path, "r", errors="replace") as f:
            result = []
            for line_num, line in enumerate(f, 1):
                if name in line:
                    result.append(line_num)
                    if len(result) >= max_lines:
                        break
            return result
    except (OSError, IOError):
        return []


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def _background_init(
    project_path: str,
    languages: list[str] | None,
    ignore_patterns: list[str] | None,
    watch: bool,
) -> None:
    """Heavy initialization in background thread -- called after MCP handshake.

    This function does the expensive work of parsing the entire codebase
    into a graph-sitter graph, building indexes (caller index, import
    index, class-method index), and starting the file watcher.  It is
    designed to run in a daemon thread so the MCP transport can complete
    its handshake immediately.

    Order of operations:
        1. Build CodebaseGraph (parses all source files, builds indexes)
        2. Create CodeSearcher (wraps the graph with fuzzy search)
        3. Create ConventionDetector (analyzes naming/style patterns)
        4. Start CodebaseWatcher (file-system watcher for incremental updates)
        5. Set _init_ready flag (unblocks all waiting tool calls)
    """
    global _graph, _searcher, _detector, _watcher, _init_ready

    logger.info("Initializing Grafyx for %s", project_path)
    _graph = CodebaseGraph(
        project_path,
        languages=languages,
        ignore_patterns=ignore_patterns,
    )
    init_result = _graph.initialize()
    logger.info("Graph initialized: %s", init_result)

    _searcher = CodeSearcher(_graph)
    _detector = ConventionDetector(_graph)

    if watch:
        _watcher = CodebaseWatcher(_graph)
        _watcher.start()
        logger.info("File watcher started")

    # Signal that all state is ready.  Tools blocked in _wait_for_init()
    # will see this on their next polling iteration.
    _init_ready = True
    logger.info("Grafyx ready")


def _wait_for_init(timeout: float = 120.0) -> None:
    """Block until background initialization is complete.

    Polls ``_init_ready`` every 200 ms.  If the timeout is exceeded,
    raises a ToolError so the MCP client gets a useful error message
    rather than hanging forever.
    """
    deadline = time.monotonic() + timeout
    while not _init_ready:
        if time.monotonic() > deadline:
            raise ToolError("Grafyx is still initializing. Please try again in a moment.")
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# Public API: server factory
# ---------------------------------------------------------------------------

def create_server(
    project_path: str,
    languages: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    watch: bool = True,
    lazy: bool = False,
) -> FastMCP:
    """Initialize all components and return the configured MCP server.

    This is the main entry point called by ``grafyx.__main__``.

    Args:
        project_path: Root directory of the codebase to analyze.
        languages: Optional list of languages to parse (e.g. ["python"]).
                   If None, all supported languages are auto-detected.
        ignore_patterns: Glob patterns for files/dirs to skip
                         (e.g. ["node_modules", ".git"]).
        watch: If True, start a file-system watcher for live updates.
        lazy: If True, defer heavy initialization (graph parsing) to a
            background thread.  The MCP server starts immediately so it
            can complete the protocol handshake within the client's
            timeout.  Tools block until init finishes via
            ``_ensure_initialized()``.
    """
    if lazy:
        # Start heavy work in a daemon thread so the MCP transport layer
        # can respond to the handshake while parsing is still running.
        thread = threading.Thread(
            target=_background_init,
            args=(project_path, languages, ignore_patterns, watch),
            daemon=True,
        )
        thread.start()
    else:
        # Synchronous init -- useful for tests and small projects.
        _background_init(project_path, languages, ignore_patterns, watch)

    return mcp


# ---------------------------------------------------------------------------
# Internal gate: every tool must call this before accessing _graph
# ---------------------------------------------------------------------------

def _ensure_initialized() -> CodebaseGraph:
    """Wait for lazy init (if active), then return the graph.

    Every ``@mcp.tool`` function starts by calling this.  It serves two
    purposes:
        1. Block until ``_init_ready`` is True (handles the lazy=True case).
        2. Validate that the graph is actually initialized (catches startup
           failures where _background_init raised an exception).

    Returns:
        The initialized ``CodebaseGraph`` instance.

    Raises:
        ToolError: If initialization timed out or the graph is not available.
    """
    if not _init_ready:
        _wait_for_init()
    if _graph is None or not _graph.initialized:
        raise ToolError("Grafyx not initialized. Server may have failed to start.")
    return _graph
