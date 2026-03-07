"""Graph package -- CodebaseGraph and its mixin components.

This package contains the core graph engine for Grafyx, split into focused
mixin modules for maintainability:

    core.py        - CodebaseGraph facade: composes all mixins, owns __init__/refresh
    _patches.py    - Monkey-patches for graph-sitter / py_mini_racer compatibility
    _types.py      - TypedDict definitions for all internal index data structures
    _paths.py      - PathMixin: path resolution, mirror syncing, line extraction
    _indexes.py    - IndexBuilderMixin: builds caller, import, instance indexes
    _callers.py    - CallerQueryMixin: multi-level caller disambiguation queries
    _query.py      - SymbolQueryMixin: function/class/file/symbol lookups, stats
    _analysis.py   - AnalysisMixin: dead code, cycles, subclass trees, module deps

All mixins read/write attributes on ``self`` (the CodebaseGraph instance).
See each module's docstring for which ``self.*`` attributes it expects.

Re-exports CodebaseGraph so that ``from grafyx.graph import CodebaseGraph``
continues to work after the single-file-to-package refactor.
"""

from grafyx.graph._patches import apply_py_mini_racer_patch
from grafyx.graph.core import CodebaseGraph

# Apply py_mini_racer compatibility patch at import time, matching
# the original graph.py behavior where this ran as module-level code.
apply_py_mini_racer_patch()

__all__ = ["CodebaseGraph"]
