"""File system watcher for automatic graph updates."""

import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from grafyx.graph import CodebaseGraph
from grafyx.utils import DEFAULT_IGNORE_PATTERNS, EXTENSION_TO_LANGUAGE

logger = logging.getLogger(__name__)


class CodebaseWatcher:
    """Watches for file changes and triggers graph re-parsing with debounce.

    The watcher runs the Observer in a background (daemon) thread.
    A debounce timer prevents rapid-fire re-parses during batch saves.
    """

    def __init__(
        self,
        graph: CodebaseGraph,
        debounce_seconds: float = 2.0,
        extra_ignore: list[str] | None = None,
    ):
        self._graph = graph
        self._debounce_seconds = debounce_seconds
        self._observer: Observer | None = None
        self._debounce_timer: threading.Timer | None = None
        self._pending_changes: list[dict] = []
        self._lock = threading.Lock()
        self._running = False

        self._ignore_dirs = set(DEFAULT_IGNORE_PATTERNS)
        if extra_ignore:
            self._ignore_dirs.update(extra_ignore)

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start watching the project directory in a background thread."""
        if self._running:
            logger.warning("Watcher is already running")
            return

        handler = _GrafyxEventHandler(
            callback=self._on_file_change,
            ignore_dirs=self._ignore_dirs,
        )

        self._observer = Observer()
        self._observer.daemon = True  # So it doesn't prevent process exit
        self._observer.schedule(handler, self._graph.project_path, recursive=True)
        self._observer.start()
        self._running = True
        logger.info("File watcher started for %s", self._graph.project_path)

    def stop(self) -> None:
        """Stop the file watcher."""
        if not self._running:
            return

        if self._debounce_timer:
            self._debounce_timer.cancel()
            self._debounce_timer = None

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        self._running = False
        self._pending_changes.clear()
        logger.info("File watcher stopped")

    def _on_file_change(self, event_type: str, file_path: str) -> None:
        """Called by the event handler when a relevant file changes."""
        with self._lock:
            self._pending_changes.append({
                "type": event_type,
                "path": file_path,
            })

            # Reset debounce timer
            if self._debounce_timer:
                self._debounce_timer.cancel()

            self._debounce_timer = threading.Timer(
                self._debounce_seconds,
                self._execute_refresh,
            )
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _execute_refresh(self) -> None:
        """Execute the graph refresh after debounce period."""
        with self._lock:
            if not self._pending_changes:
                return

            changes = list(self._pending_changes)
            self._pending_changes.clear()

        change_count = len(changes)
        changed_files = [c["path"] for c in changes]
        logger.info(
            "Refreshing graph after %d file change(s): %s",
            change_count,
            ", ".join(changed_files[:5]) + ("..." if change_count > 5 else ""),
        )

        try:
            result = self._graph.refresh()
            logger.info("Graph refresh complete: %s", result.get("status", "unknown"))
        except Exception as exc:
            logger.error("Graph refresh failed: %s", exc)


class _GrafyxEventHandler(FileSystemEventHandler):
    """Watchdog event handler that filters and delegates file changes."""

    def __init__(self, callback, ignore_dirs: set[str]):
        super().__init__()
        self._callback = callback
        self._ignore_dirs = ignore_dirs

    def _should_process(self, path: str) -> bool:
        """Check if this file change should trigger a refresh."""
        p = Path(path)

        # Skip directories
        if p.is_dir():
            return False

        # Skip files in ignored directories
        for part in p.parts:
            if part in self._ignore_dirs:
                return False

        # Only process supported code file extensions
        ext = p.suffix.lower()
        return ext in EXTENSION_TO_LANGUAGE

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self._callback("created", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self._callback("modified", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self._callback("deleted", event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if self._should_process(event.src_path) or self._should_process(event.dest_path):
            self._callback("moved", event.dest_path)
