# Design: Two New Grafyx MCP Tools

**Date**: 2026-02-13
**Status**: Approved

## Context

Grafyx has 12 MCP tools covering navigation, search, analysis, and maintenance. After a full codebase review, two gaps were identified:

1. **Subclass discovery** — `get_class_context` shows base classes but not who extends a class
2. **Module-level dependencies** — `get_dependency_graph` works at symbol level; no way to see how packages/directories depend on each other

## Tool 13: `get_subclasses`

### Purpose

Given a class name, find all classes that extend it — recursively if needed. Answers "who would break if I change this class's interface?"

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `class_name` | `str` | required | The base class to search for subclasses of |
| `depth` | `int` | `3` | Inheritance levels to traverse (1 = direct only) |

### Output

```json
{
  "class_name": "BaseHandler",
  "file": "src/handlers/base.py",
  "line": 10,
  "language": "python",
  "direct_subclass_count": 4,
  "total_subclass_count": 7,
  "subclasses": [
    {
      "name": "HTTPHandler",
      "file": "src/handlers/http.py",
      "line": 15,
      "language": "python",
      "subclasses": [
        {"name": "APIHandler", "file": "src/handlers/api.py", "line": 8, "language": "python", "subclasses": []}
      ]
    }
  ]
}
```

### Implementation

- **graph.py**: Add `get_subclasses(class_name, depth)` method
  - Uses `get_all_classes()` to get every class with its `base_classes`
  - Builds a reverse map: `base_name -> [child_classes]`
  - Recursively traverses up to `depth` levels
  - Returns tree structure
- **server.py**: Add `@mcp.tool` wrapper with standard error handling
- No new indexes needed — `get_all_classes()` already extracts `base_classes`

## Tool 14: `get_module_dependencies`

### Purpose

Show how directories/packages depend on each other at the module level. Answers "which packages would be affected if I refactor this one?"

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `module_path` | `str` | `""` | Optional filter to a specific directory |
| `depth` | `int` | `1` | Subdirectory grouping depth (1 = top-level dirs) |

### Output

```json
{
  "project_path": "/path/to/project",
  "module_count": 5,
  "modules": {
    "services": {"file_count": 8, "depends_on": ["models", "utils"], "depended_on_by": ["api"]},
    "models": {"file_count": 4, "depends_on": ["utils"], "depended_on_by": ["services", "api"]},
    "api": {"file_count": 6, "depends_on": ["services", "models"], "depended_on_by": []},
    "utils": {"file_count": 3, "depends_on": [], "depended_on_by": ["services", "models"]}
  },
  "edges": [
    {"from": "api", "to": "services", "import_count": 12},
    {"from": "api", "to": "models", "import_count": 8},
    {"from": "services", "to": "models", "import_count": 15},
    {"from": "services", "to": "utils", "import_count": 6}
  ]
}
```

### Implementation

- **graph.py**: Add `get_module_dependencies(module_path, depth)` method
  - Reads `_forward_import_index` (file -> [imported_files])
  - Groups files into directories at the specified depth
  - Aggregates import edges between directory groups
  - Optionally filters to a specific module's perspective
- **server.py**: Add `@mcp.tool` wrapper with standard error handling
- No new indexes needed — `_forward_import_index` is already built

## Design Principles

Both tools follow existing patterns:

- `_ensure_initialized()` guard
- `truncate_response()` on output
- `safe_get_attr()` for graph-sitter access
- `translate_path()` for mirror/original path mapping
- `ToolError` for user-facing errors, catch-all for unexpected exceptions
- Graph methods in `graph.py`, tool wrappers in `server.py`

## Test Plan

- Unit tests in `tests/test_server.py` (or new test files) using the existing fixture project
- Test `get_subclasses` with the fixture's inheritance chain (if any) or extend fixtures
- Test `get_module_dependencies` against the fixture project's import structure
- Edge cases: class with no subclasses, circular inheritance, empty module path
