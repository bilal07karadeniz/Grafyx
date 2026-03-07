# Fractal Context Design

## Summary

Three features that make Grafyx work like "Google Earth for code" -- LLMs see the codebase at the right zoom level without flooding their context window.

### Decisions Made
- **Resolution control**: 3 levels (`signatures` / `summary` / `full`) across all 5 context tools, replacing `include_source` boolean
- **New tool**: `get_module_context` -- intermediate zoom between skeleton and file context
- **Navigation hints**: `suggested_next` list on exploration tools (skeleton, module, file, class, function context), opt-out via `include_hints=False`
- **Architecture**: Shared helpers in `server/_resolution.py` and `server/_hints.py` (Approach B: Shared Middleware)
- **Deep-dive tools skip hints**: dependency_graph, call_graph, module_dependencies, quality tools, search tools
