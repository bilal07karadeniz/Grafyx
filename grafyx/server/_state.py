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
import os
import sys
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
_init_error: str | None = None  # Set if background init fails
_init_thread: threading.Thread | None = None  # Reference to background init thread

# Saved server config for set_project / deferred init
_server_languages: list[str] | None = None
_server_ignore: list[str] | None = None
_server_watch: bool = True

# Usage instructions injected into every connected client's system prompt.
# Captures hard-won lessons about when this MCP helps vs. when it can mislead.
# See https://modelcontextprotocol.io for how clients surface server instructions.
GRAFYX_INSTRUCTIONS = """\
Grafyx provides 14 pre-indexed tools for codebase navigation: call graphs, dependency maps, symbol lookups. Use these tools instead of grep when working with existing code — they are faster and structurally aware. But Grafyx is not a 100% replacement for reading source. Verify when results look thin.

## When to use Grafyx

Always use BEFORE modifying existing code:
- get_function_context or get_class_context before editing any function/class
- get_call_graph before refactoring (to see what will break)
- get_file_context before editing any file (to understand imports and dependents)
- get_subclasses before changing a base class interface

Use for exploration:
- get_project_skeleton when starting work on an unfamiliar area
- get_module_context to understand a directory/package
- find_related_code to find where a concept lives by description
- get_unused_symbols when cleaning up dead code

Use for safety checks:
- get_dependency_graph to check blast radius before changes
- get_call_graph depth=2 to trace execution flow

## When NOT to use Grafyx

- Simple single-file edits where context is already clear
- Creating brand new files
- Config / env / build-tooling changes

## Known limitations (verify before trusting)

- TypeScript object literal methods (`const api = { fn: () => ... }`) sometimes show as 0 functions. Read the file source instead of trusting function count.
- find_related_files can miss important files. If results look incomplete, fall back to grep.
- Python analysis is highly accurate. TypeScript analysis is good but verify caller counts for React components.
- Celery `.delay()` and `.apply_async()` callers are NOT detected (dynamic dispatch through the task registry).
- get_all_functions(include_methods=False) is the default — only returns top-level functions, not class methods.
- ML relevance ranker (find_related_code) is a small MLP (42 features). Expect it to surface plausible-but-wrong matches on ambiguous queries; cross-check with get_function_context before acting.

## Recommended workflow

1. Before ANY code change: get_function_context on what you're about to modify
2. Check callers — if callers > 0, understand the impact before editing
3. Make the change
4. Verify callers still work (run tests, or get_function_context again on the callers)

If a tool returns suspicious or empty results, do NOT silently trust it — read the source file directly.
"""

# The single FastMCP instance that all tool modules register on.
# Created at module-load time so @mcp.tool decorators work immediately.
mcp = FastMCP("Grafyx", instructions=GRAFYX_INSTRUCTIONS)

# ---------------------------------------------------------------------------
# Project markers — files that indicate a directory is a project root
# ---------------------------------------------------------------------------
_PROJECT_MARKERS = (
    ".git", "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "Cargo.toml", "go.mod", "pom.xml",
    "Makefile", "CMakeLists.txt", ".hg", ".svn",
)


def _is_project_dir(path: str) -> bool:
    """Check if a directory looks like a project root."""
    return any(os.path.exists(os.path.join(path, m)) for m in _PROJECT_MARKERS)


def _find_project_root(start: str) -> str | None:
    """Walk up from start looking for a project root. Returns None if not found."""
    path = os.path.abspath(start)
    if _is_project_dir(path):
        return path
    parent = path
    for _ in range(10):
        parent = os.path.dirname(parent)
        if parent == os.path.dirname(parent):
            break
        if _is_project_dir(parent):
            return parent
    return None


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


def _progress(msg: str) -> None:
    """Write progress to stderr (always visible in MCP client logs)."""
    print(f"[grafyx] {msg}", file=sys.stderr, flush=True)


def _silence_graph_sitter() -> None:
    """Suppress all graph_sitter loggers to prevent stdout pollution."""
    _OFF = logging.CRITICAL + 1
    for name in list(logging.Logger.manager.loggerDict):
        if "graph_sitter" in name:
            lgr = logging.getLogger(name)
            lgr.handlers.clear()
            lgr.setLevel(_OFF)
            lgr.propagate = False
    gs = logging.getLogger("graph_sitter")
    gs.handlers.clear()
    gs.setLevel(_OFF)
    gs.propagate = False


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
    global _graph, _searcher, _detector, _watcher, _init_ready, _init_error

    try:
        # Lower thread priority so graph parsing doesn't starve the system.
        # nice(10) is a mild deprioritization — still finishes fast but
        # won't pin CPU at 95% on single/dual-core machines.
        try:
            os.nice(10)
        except (OSError, AttributeError):
            pass  # Windows or permission denied — no-op

        abs_path = os.path.abspath(project_path)
        _progress(f"Initializing for: {abs_path}")

        # Validate project path
        root = _find_project_root(abs_path)
        if root is None:
            raise RuntimeError(
                f"'{abs_path}' is not a project directory "
                f"(no .git, pyproject.toml, package.json, etc.). "
                f"Use --project /path/to/project or run grafyx "
                f"from within a project directory."
            )
        if root != abs_path:
            project_path = root
            _progress(f"Auto-detected project root: {root}")

        # Silence graph_sitter BEFORE it creates any loggers during init
        _silence_graph_sitter()

        _progress("Building graph...")
        _graph = CodebaseGraph(
            project_path,
            languages=languages,
            ignore_patterns=ignore_patterns,
        )
        init_result = _graph.initialize()
        _progress(f"Graph ready: {init_result}")

        # Silence again — build_graph() creates new child loggers lazily
        _silence_graph_sitter()

        _progress("Building search index...")
        _searcher = CodeSearcher(_graph)
        _progress("Detecting conventions...")
        _detector = ConventionDetector(_graph)

        # Only start watcher if the project is on native Linux.
        # When using a mirror (~/.grafyx/mirrors/...), the watcher would
        # watch the mirror — not the original project — so it can never
        # detect real file changes.  Skip it to save CPU.
        if watch and _graph.project_path == _graph.original_path:
            _watcher = CodebaseWatcher(_graph)
            _watcher.start()
            _progress("File watcher started")
        elif watch:
            _progress("Skipping file watcher (mirror mode — use refresh_graph to update)")

        _init_ready = True
        _progress("Grafyx ready")
    except BaseException as e:
        _init_error = f"{type(e).__name__}: {e}"
        _init_ready = True  # Unblock waiters so they get the error
        _progress(f"Initialization FAILED: {_init_error}")


def _wait_for_init(timeout: float = 120.0) -> None:
    """Block until background initialization is complete.

    Polls ``_init_ready`` every 200 ms.  If the timeout is exceeded or the
    init thread dies unexpectedly, raises a ToolError so the MCP client
    gets a useful error message rather than hanging forever.
    """
    deadline = time.monotonic() + timeout
    while not _init_ready:
        if time.monotonic() > deadline:
            raise ToolError("Grafyx is still initializing. Please try again in a moment.")
        # Detect if the init thread crashed without setting _init_ready
        if _init_thread is not None and not _init_thread.is_alive() and not _init_ready:
            raise ToolError(
                "Grafyx initialization thread died unexpectedly. "
                "Check stderr/MCP logs for details."
            )
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
    global _init_thread, _server_languages, _server_ignore, _server_watch

    # Save config for set_project
    _server_languages = languages
    _server_ignore = ignore_patterns
    _server_watch = watch

    # Check if the project path is a valid project
    abs_path = os.path.abspath(project_path)
    root = _find_project_root(abs_path)

    if root is not None:
        # Valid project — initialize immediately or in background
        if lazy:
            _init_thread = threading.Thread(
                target=_background_init,
                args=(root, languages, ignore_patterns, watch),
                daemon=True,
            )
            _init_thread.start()
        else:
            _background_init(root, languages, ignore_patterns, watch)
    else:
        # No project detected — server starts but tools will prompt
        # the AI to call set_project() with the workspace path.
        _progress(
            f"No project at '{abs_path}'. "
            f"Waiting for set_project() call."
        )

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
    if not _init_ready and _init_thread is None:
        # No init started — server is waiting for set_project
        raise ToolError(
            "No project loaded. Call set_project with the workspace "
            "root path first (e.g. set_project(project_path='/path/to/project'))."
        )
    if not _init_ready:
        _wait_for_init()
    if _init_error:
        raise ToolError(f"Grafyx initialization failed: {_init_error}")
    if _graph is None or not _graph.initialized:
        raise ToolError("Grafyx not initialized. Server may have failed to start.")
    return _graph
