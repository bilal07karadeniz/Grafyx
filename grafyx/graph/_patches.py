"""Compatibility patches for graph-sitter and its dependencies.

This module applies monkey-patches to fix known issues in graph-sitter
and its transitive dependency py_mini_racer. These patches are applied
once at import time (py_mini_racer) or once per initialize() call
(graph-sitter Chainable).

Patches applied:
    1. py_mini_racer: Fixes JSEvalException import path for newer versions
       where the exception class moved from ``_types`` to the top-level module.
    2. graph-sitter Chainable: Replaces a crashing ``assert child is not self``
       in ``with_resolution_frame`` with a logged warning + skip, preventing
       circular reference crashes during codebase parsing.

This module has no mixin class -- it exposes two standalone functions called
by ``__init__.py`` and ``core.py`` respectively.
"""

import functools
import logging

logger = logging.getLogger(__name__)


# --- py_mini_racer JSEvalException Patch ---

def apply_py_mini_racer_patch() -> None:
    """Patch py_mini_racer compatibility issue with graph-sitter.

    Graph-sitter imports ``JSEvalException`` from ``py_mini_racer._types``,
    but newer versions of mini-racer moved it to the top-level module.
    We bridge the gap by re-exporting it on the ``_types`` submodule.

    Called once from ``grafyx.graph.__init__`` at package import time.
    Safe to call multiple times (idempotent).
    """
    try:
        import py_mini_racer._types as _pmr_types
        if not hasattr(_pmr_types, "JSEvalException"):
            import py_mini_racer
            _pmr_types.JSEvalException = py_mini_racer.JSEvalException
    except ImportError:
        # py_mini_racer not installed -- graph-sitter will fail later
        # with a clearer error, so we silently skip here.
        pass


# --- graph-sitter Circular Reference Assertion Patch ---

def _patch_graph_sitter_assert() -> None:
    """Patch graph-sitter's Chainable.with_resolution_frame to skip the
    ``assert child is not self`` check that crashes on circular references.

    Some codebases (e.g., files importing ``redis``) trigger a circular
    reference in graph-sitter's dependency resolution.  The assertion
    kills the entire parse.  This patch turns the assert into a logged
    warning + skip, allowing the rest of the codebase to parse normally.

    Called once from ``CodebaseGraph.initialize()`` right before creating
    Codebase instances, so it takes effect before any parsing happens.
    """
    try:
        from graph_sitter.core.interfaces.chainable import Chainable
        original = Chainable.with_resolution_frame

        @functools.wraps(original)
        def _safe_with_resolution_frame(self, child, *args, **kwargs):
            # Replaces the original ``assert child is not self`` with a
            # graceful skip. The ``_resolving`` flag is set by graph-sitter
            # during dependency resolution to track the resolution stack.
            if hasattr(child, '_resolving') and child is self:
                logger.warning(
                    "Skipping circular resolution: %s references itself",
                    getattr(self, 'name', '?'),
                )
                return  # yield nothing -- skip this resolution
            yield from original(self, child, *args, **kwargs)

        Chainable.with_resolution_frame = _safe_with_resolution_frame
        logger.debug("Patched graph-sitter Chainable.with_resolution_frame")
    except Exception as e:
        # May fail if graph-sitter API changed -- not critical, just log.
        logger.debug("Could not patch graph-sitter assertion: %s", e)
