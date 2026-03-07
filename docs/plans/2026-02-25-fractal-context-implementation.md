# Fractal Context Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three "Fractal Context" features to Grafyx: resolution control (3-level detail parameter), a new `get_module_context` tool (intermediate zoom), and smart navigation hints on exploration tools.

**Architecture:** Shared middleware approach -- two new helper modules (`server/_resolution.py` for detail filtering, `server/_hints.py` for navigation hint computation) called by each tool at the end of its response pipeline. Existing `include_source` booleans replaced with `detail: str` parameter.

**Tech Stack:** Python 3.11+, FastMCP, graph-sitter, pytest

---

### Task 1: Create `server/_resolution.py` -- detail level filtering

**Files:**
- Create: `grafyx/server/_resolution.py`
- Test: `tests/test_resolution.py`

**Step 1: Write the failing tests**

```python
"""Tests for resolution filtering."""
import pytest
from grafyx.server._resolution import filter_by_detail, DETAIL_LEVELS


class TestFilterByDetail:
    """Test the shared detail-level filtering helper."""

    def test_valid_levels(self):
        """All three levels should be accepted."""
        assert set(DETAIL_LEVELS) == {"signatures", "summary", "full"}

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Invalid detail level"):
            filter_by_detail({}, "invalid", "function")

    def test_function_signatures_strips_docstring(self):
        """Signatures level should remove docstrings, callers, callees, source."""
        data = {
            "name": "process",
            "signature": "def process(x: int) -> str",
            "parameters": [{"name": "x", "type": "int"}],
            "return_type": "str",
            "docstring": "Process something.",
            "decorators": ["@cache"],
            "calls": [{"name": "helper"}],
            "called_by": [{"name": "main", "file": "app.py"}],
            "source": "def process(x): return str(x)",
            "dependencies": [{"name": "os"}],
        }
        result = filter_by_detail(data, "signatures", "function")
        assert result["name"] == "process"
        assert result["signature"] == "def process(x: int) -> str"
        assert result["parameters"] == [{"name": "x", "type": "int"}]
        assert result["return_type"] == "str"
        assert "docstring" not in result
        assert "decorators" not in result
        assert "calls" not in result
        assert "called_by" not in result
        assert "source" not in result

    def test_function_summary_keeps_docstring_strips_source(self):
        """Summary level keeps docstring and caller/callee names, strips source."""
        data = {
            "name": "process",
            "signature": "def process(x: int) -> str",
            "docstring": "Process something.\n\nDetailed explanation here.",
            "decorators": ["@cache"],
            "calls": [{"name": "helper", "file": "util.py"}],
            "called_by": [{"name": "main", "file": "app.py"}],
            "source": "def process(x): return str(x)",
        }
        result = filter_by_detail(data, "summary", "function")
        assert result["docstring"] is not None
        assert result["decorators"] == ["@cache"]
        assert "calls" in result
        assert "called_by" in result
        assert "source" not in result

    def test_function_full_keeps_everything(self):
        """Full level should return data unchanged."""
        data = {
            "name": "process",
            "signature": "def process(x)",
            "source": "def process(x): pass",
            "docstring": "Doc",
            "calls": [{"name": "a"}],
        }
        result = filter_by_detail(data, "full", "function")
        assert result["source"] == "def process(x): pass"
        assert result["docstring"] == "Doc"

    def test_file_signatures_strips_method_details(self):
        """File context at signatures level strips function docstrings."""
        data = {
            "path": "app.py",
            "functions": [
                {"name": "run", "signature": "def run()", "docstring": "Run the app.", "line": 1},
            ],
            "classes": [
                {"name": "App", "methods": [{"name": "start", "signature": "def start()", "docstring": "Start."}], "docstring": "Main app."},
            ],
            "imports": ["import os"],
            "imported_by": ["test_app.py"],
            "source": "import os\ndef run(): pass",
        }
        result = filter_by_detail(data, "signatures", "file")
        # Functions should lose docstrings
        assert "docstring" not in result["functions"][0]
        # Classes should lose docstrings
        assert "docstring" not in result["classes"][0]
        # No source
        assert "source" not in result
        # No imported_by at signatures level
        assert "imported_by" not in result

    def test_class_signatures_strips_details(self):
        """Class context at signatures level: methods have name+sig only."""
        data = {
            "name": "UserService",
            "methods": [
                {"name": "create", "signature": "def create(data)", "docstring": "Create user.", "is_async": False, "line": 10},
            ],
            "properties": [{"name": "db"}],
            "base_classes": ["BaseService"],
            "cross_file_usages": [{"file": "routes.py", "lines": [5]}],
            "source": "class UserService: pass",
        }
        result = filter_by_detail(data, "signatures", "class")
        assert result["methods"][0]["name"] == "create"
        assert result["methods"][0]["signature"] == "def create(data)"
        assert "docstring" not in result["methods"][0]
        assert "cross_file_usages" not in result
        assert "source" not in result

    def test_class_summary_keeps_usages(self):
        """Class context at summary level keeps cross-file usages."""
        data = {
            "name": "UserService",
            "methods": [{"name": "create", "signature": "def create()", "docstring": "Create."}],
            "cross_file_usages": [{"file": "routes.py", "lines": [5]}],
            "source": "class UserService: pass",
        }
        result = filter_by_detail(data, "summary", "class")
        assert "cross_file_usages" in result
        assert "source" not in result

    def test_skeleton_signatures_strips_dir_stats(self):
        """Skeleton at signatures level strips directory_stats."""
        data = {
            "project_path": "/app",
            "languages": ["python"],
            "total_files": 10,
            "total_functions": 20,
            "total_classes": 5,
            "directory_stats": {"src": {"files": 8}},
            "file_tree": "src/\n  app.py",
            "by_language": {"python": {"files": 10}},
        }
        result = filter_by_detail(data, "signatures", "skeleton")
        assert "directory_stats" not in result
        assert "by_language" not in result
        assert result["file_tree"] is not None

    def test_missing_keys_handled_gracefully(self):
        """Should not crash if data doesn't have all expected keys."""
        data = {"name": "foo"}
        result = filter_by_detail(data, "signatures", "function")
        assert result["name"] == "foo"

    def test_module_signatures(self):
        """Module context at signatures level strips docstrings from symbols."""
        data = {
            "module": "services",
            "files": 3,
            "symbols": [
                {
                    "file": "auth.py",
                    "functions": [{"name": "login", "signature": "def login()", "docstring": "Log in."}],
                    "classes": [{"name": "AuthService", "methods": ["login"], "docstring": "Auth service."}],
                }
            ],
            "internal_imports": [{"from": "auth.py", "imports": ["models.py"]}],
        }
        result = filter_by_detail(data, "signatures", "module")
        assert "docstring" not in result["symbols"][0]["functions"][0]
        assert "docstring" not in result["symbols"][0]["classes"][0]
        assert "internal_imports" not in result
```

**Step 2: Run tests to verify they fail**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/test_resolution.py -v`
Expected: FAIL (module not found)

**Step 3: Implement `server/_resolution.py`**

```python
"""Detail-level resolution filtering for MCP tool responses.

This module provides the shared `filter_by_detail()` function that all
context tools call before returning their response. It strips fields
based on the requested detail level:

    - **signatures**: Bare minimum -- names, signatures, types only.
      Saves the most context-window tokens. Use when scanning many symbols.
    - **summary** (default): Names, signatures, docstrings, caller/callee
      names, usage counts. The workhorse for most exploration.
    - **full**: Everything including source code. Use when you need to
      read the actual implementation.

Architecture:
    Each tool builds its FULL response dict, then calls:
        return truncate_response(filter_by_detail(data, detail, context_type))
    The filter removes keys that don't belong at the requested level.

Why a shared helper instead of per-tool logic:
    - Consistent behavior across all tools (same `detail` param everywhere)
    - Single place to tune what each level includes
    - Easy to add new levels later without touching every tool
"""

from typing import Any

# Valid detail levels, ordered from least to most verbose
DETAIL_LEVELS = ("signatures", "summary", "full")

# -----------------------------------------------------------------------
# Per-context-type field definitions
# -----------------------------------------------------------------------
# For each context type, we define which fields to REMOVE at each level.
# "full" never removes anything. "summary" removes source. "signatures"
# removes everything except structural identity fields.

# Fields to strip at "signatures" level, by context type
_SIGNATURES_STRIP: dict[str, set[str]] = {
    "function": {
        "docstring", "decorators", "calls", "called_by",
        "dependencies", "source", "is_async",
    },
    "file": {
        "source", "imported_by",
    },
    "class": {
        "cross_file_usages", "internal_usage_count",
        "dependencies", "source",
    },
    "skeleton": {
        "directory_stats", "by_language",
    },
    "module": {
        "internal_imports",
    },
}

# Fields to strip at "summary" level, by context type
_SUMMARY_STRIP: dict[str, set[str]] = {
    "function": {"source"},
    "file": {"source"},
    "class": {"source"},
    "skeleton": set(),
    "module": set(),
}


def filter_by_detail(data: dict, detail: str, context_type: str) -> dict:
    """Filter a tool response dict based on the requested detail level.

    Args:
        data: The full response dict built by the tool.
        detail: One of "signatures", "summary", "full".
        context_type: What kind of response this is -- "function", "file",
            "class", "skeleton", or "module". Determines which fields
            get stripped at each level.

    Returns:
        A new dict with fields removed according to the detail level.
        The original dict is not modified.

    Raises:
        ValueError: If detail is not a valid level.
    """
    if detail not in DETAIL_LEVELS:
        raise ValueError(
            f"Invalid detail level '{detail}'. Must be one of: {', '.join(DETAIL_LEVELS)}"
        )

    # "full" returns everything unchanged
    if detail == "full":
        return data

    # Make a shallow copy so we don't mutate the caller's dict
    result = dict(data)

    # Determine which top-level keys to strip
    if detail == "signatures":
        strip_keys = _SIGNATURES_STRIP.get(context_type, set())
    else:  # summary
        strip_keys = _SUMMARY_STRIP.get(context_type, set())

    for key in strip_keys:
        result.pop(key, None)

    # --- Nested filtering for "signatures" level ---
    # Strip docstrings and detail fields from nested items
    if detail == "signatures":
        _strip_nested_details(result, context_type)

    return result


def _strip_nested_details(data: dict, context_type: str) -> None:
    """Remove docstrings and detail fields from nested structures.

    Modifies data in-place. Called only at "signatures" level to strip
    docstrings from function/class/method entries within file/module/class
    contexts.
    """
    # Strip docstrings from functions listed in file/module context
    if context_type in ("file", "module"):
        for sym_group in data.get("symbols", [data]):
            for func in sym_group.get("functions", []):
                func.pop("docstring", None)
                func.pop("decorators", None)
            for cls in sym_group.get("classes", []):
                cls.pop("docstring", None)

    # Strip docstrings from methods listed in class context
    if context_type == "class":
        for method in data.get("methods", []):
            method.pop("docstring", None)
            method.pop("is_async", None)
            method.pop("line", None)
```

**Step 4: Run tests to verify they pass**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/test_resolution.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add grafyx/server/_resolution.py tests/test_resolution.py
git commit -m "feat: add resolution filtering helper (signatures/summary/full)"
```

---

### Task 2: Create `server/_hints.py` -- navigation hint computation

**Files:**
- Create: `grafyx/server/_hints.py`
- Test: `tests/test_hints.py`

**Step 1: Write the failing tests**

```python
"""Tests for navigation hint computation."""
from unittest.mock import MagicMock, patch
import pytest
from grafyx.server._hints import compute_hints


class TestComputeHints:
    """Test the shared navigation hint computation helper."""

    def _make_mock_graph(self):
        """Create a mock graph with enough data for hint computation."""
        graph = MagicMock()
        graph._caller_index = {
            "process_order": [
                {"caller": "handle_request", "file": "routes.py"},
                {"caller": "batch_run", "file": "jobs.py"},
            ],
            "validate": [
                {"caller": "process_order", "file": "orders.py"},
            ],
        }
        graph._class_method_names = {
            "OrderService": {"process_order", "cancel_order", "get_order"},
            "UserService": {"create_user", "get_user"},
        }
        graph._import_index = {
            "services/orders.py": ["routes.py", "jobs.py", "tests/test_orders.py"],
            "services/users.py": ["routes.py", "admin.py"],
        }
        graph.translate_path = lambda p: p
        graph.resolve_path = lambda p: p
        return graph

    def test_function_context_hints(self):
        """After viewing a function, hints should suggest top callers/callees."""
        graph = self._make_mock_graph()
        symbol_data = {
            "name": "process_order",
            "class": "OrderService",
            "file": "services/orders.py",
            "calls": [
                {"name": "validate", "file": "services/validation.py"},
                {"name": "save_to_db", "file": "services/db.py"},
            ],
            "called_by": [
                {"name": "handle_request", "file": "routes.py"},
                {"name": "batch_run", "file": "jobs.py"},
            ],
        }
        hints = compute_hints(graph, "function", symbol_data)
        assert isinstance(hints, list)
        assert len(hints) <= 3
        # Each hint should have tool, args, reason
        for hint in hints:
            assert "tool" in hint
            assert "args" in hint
            assert "reason" in hint

    def test_file_context_hints(self):
        """After viewing a file, hints should suggest important symbols in it."""
        graph = self._make_mock_graph()
        symbol_data = {
            "path": "services/orders.py",
            "functions": [
                {"name": "process_order", "signature": "def process_order()"},
                {"name": "validate", "signature": "def validate()"},
            ],
            "classes": [
                {"name": "OrderService", "method_count": 5},
            ],
        }
        hints = compute_hints(graph, "file", symbol_data)
        assert isinstance(hints, list)
        assert len(hints) <= 3

    def test_class_context_hints(self):
        """After viewing a class, hints should suggest related classes/methods."""
        graph = self._make_mock_graph()
        symbol_data = {
            "name": "OrderService",
            "file": "services/orders.py",
            "methods": [
                {"name": "process_order", "signature": "def process_order()"},
                {"name": "cancel_order", "signature": "def cancel_order()"},
            ],
            "base_classes": ["BaseService"],
            "cross_file_usages": [
                {"file": "routes.py", "lines": [10, 20]},
                {"file": "jobs.py", "lines": [5]},
            ],
        }
        hints = compute_hints(graph, "class", symbol_data)
        assert isinstance(hints, list)
        assert len(hints) <= 3

    def test_skeleton_context_hints(self):
        """After viewing skeleton, hints should suggest most interesting modules."""
        graph = self._make_mock_graph()
        symbol_data = {
            "directory_stats": {
                "services": {"files": 12, "functions": 45, "classes": 8},
                "tests": {"files": 10, "functions": 30, "classes": 2},
                "utils": {"files": 3, "functions": 10, "classes": 1},
            },
        }
        hints = compute_hints(graph, "skeleton", symbol_data)
        assert isinstance(hints, list)
        assert len(hints) <= 3
        # Should suggest modules, not test dirs
        tool_names = [h["tool"] for h in hints]
        assert all(t == "get_module_context" for t in tool_names)

    def test_module_context_hints(self):
        """After viewing module, hints should suggest important files/classes."""
        graph = self._make_mock_graph()
        symbol_data = {
            "module": "services",
            "symbols": [
                {
                    "file": "orders.py",
                    "functions": [{"name": "process_order"}, {"name": "validate"}],
                    "classes": [{"name": "OrderService", "methods": ["process_order"]}],
                },
                {
                    "file": "users.py",
                    "functions": [{"name": "create_user"}],
                    "classes": [{"name": "UserService", "methods": ["create_user"]}],
                },
            ],
        }
        hints = compute_hints(graph, "module", symbol_data)
        assert isinstance(hints, list)
        assert len(hints) <= 3

    def test_empty_data_returns_empty_hints(self):
        """If there's nothing interesting, return empty list."""
        graph = MagicMock()
        graph._caller_index = {}
        graph._class_method_names = {}
        graph._import_index = {}
        hints = compute_hints(graph, "function", {"name": "foo"})
        assert hints == []

    def test_max_three_hints(self):
        """Never return more than 3 hints regardless of available data."""
        graph = self._make_mock_graph()
        # Give lots of data
        symbol_data = {
            "name": "process_order",
            "class": "OrderService",
            "file": "services/orders.py",
            "calls": [{"name": f"func_{i}", "file": f"file_{i}.py"} for i in range(20)],
            "called_by": [{"name": f"caller_{i}", "file": f"call_{i}.py"} for i in range(20)],
        }
        hints = compute_hints(graph, "function", symbol_data)
        assert len(hints) <= 3
```

**Step 2: Run tests to verify they fail**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/test_hints.py -v`
Expected: FAIL (module not found)

**Step 3: Implement `server/_hints.py`**

```python
"""Navigation hint computation for MCP tool responses.

This module provides the `compute_hints()` function that exploration tools
call to generate "suggested next steps" for the LLM. Hints are ranked
by interestingness (caller count, reference count, import density) and
capped at 3 per response.

Which tools use hints:
    - get_project_skeleton  (suggests most interesting modules)
    - get_module_context    (suggests most important files/classes)
    - get_file_context      (suggests most-called functions, used classes)
    - get_class_context     (suggests most-called methods, dependent classes)
    - get_function_context  (suggests top callers, important callees)

Which tools DO NOT use hints:
    - get_dependency_graph, get_call_graph, get_module_dependencies
      (already deep-dive -- user is navigating deliberately)
    - Quality/search tools (analysis results, not navigation)

Hint format:
    [
        {
            "tool": "get_function_context",
            "args": {"function_name": "process_order"},
            "reason": "most-called function (8 callers)"
        },
        ...
    ]

Why max 3 hints:
    More than 3 creates decision paralysis for the LLM and wastes tokens.
    The ranking ensures the most valuable explorations surface first.
"""

from typing import Any

# Maximum number of hints per response
MAX_HINTS = 3

# Directories to deprioritize when suggesting modules to explore
_DEPRIORITIZED_DIRS = {"tests", "test", "__pycache__", "migrations", "docs", "scripts"}


def compute_hints(graph: Any, context_type: str, symbol_data: dict) -> list[dict]:
    """Compute navigation hints based on the current context.

    Args:
        graph: The CodebaseGraph instance (used for caller/import indexes).
        context_type: What kind of context was just viewed -- "skeleton",
            "module", "file", "class", or "function".
        symbol_data: The response dict from the tool (before filtering).

    Returns:
        List of hint dicts, max 3, sorted by interestingness.
    """
    if context_type == "skeleton":
        return _hints_for_skeleton(symbol_data)
    elif context_type == "module":
        return _hints_for_module(graph, symbol_data)
    elif context_type == "file":
        return _hints_for_file(graph, symbol_data)
    elif context_type == "class":
        return _hints_for_class(graph, symbol_data)
    elif context_type == "function":
        return _hints_for_function(graph, symbol_data)
    return []


def _hints_for_skeleton(data: dict) -> list[dict]:
    """After viewing the project skeleton, suggest the most interesting modules.

    Ranking: by total symbols (functions + classes), excluding test dirs.
    """
    dir_stats = data.get("directory_stats", {})
    if not dir_stats:
        return []

    scored = []
    for dir_name, stats in dir_stats.items():
        # Skip test/build directories
        if dir_name.lower() in _DEPRIORITIZED_DIRS or dir_name.startswith("."):
            continue
        score = stats.get("functions", 0) + stats.get("classes", 0)
        if score > 0:
            scored.append((score, dir_name, stats))

    scored.sort(reverse=True)
    hints = []
    for score, dir_name, stats in scored[:MAX_HINTS]:
        hints.append({
            "tool": "get_module_context",
            "args": {"module_path": dir_name},
            "reason": f"{stats.get('files', 0)} files, {stats.get('functions', 0)} functions, {stats.get('classes', 0)} classes",
        })
    return hints


def _hints_for_module(graph: Any, data: dict) -> list[dict]:
    """After viewing a module, suggest the most important files/classes in it.

    Ranking: classes by method count, then files by function count.
    """
    hints = []
    caller_index = getattr(graph, "_caller_index", {})

    # Collect all classes and functions from module symbols
    all_classes = []
    all_functions = []
    for sym_group in data.get("symbols", []):
        file_name = sym_group.get("file", "")
        for cls in sym_group.get("classes", []):
            cls_name = cls.get("name", "")
            method_count = len(cls.get("methods", []))
            all_classes.append((method_count, cls_name, file_name))
        for func in sym_group.get("functions", []):
            func_name = func.get("name", "")
            caller_count = len(caller_index.get(func_name, []))
            all_functions.append((caller_count, func_name, file_name))

    # Suggest top classes first (by method count)
    all_classes.sort(reverse=True)
    for _score, cls_name, _file in all_classes[:2]:
        hints.append({
            "tool": "get_class_context",
            "args": {"class_name": cls_name},
            "reason": f"{_score} methods",
        })

    # Then top function (by caller count)
    all_functions.sort(reverse=True)
    for _score, func_name, _file in all_functions[:1]:
        reason = f"{_score} callers" if _score > 0 else "top function in module"
        hints.append({
            "tool": "get_function_context",
            "args": {"function_name": func_name},
            "reason": reason,
        })

    return hints[:MAX_HINTS]


def _hints_for_file(graph: Any, data: dict) -> list[dict]:
    """After viewing a file, suggest the most important symbols in it.

    Ranking: functions by caller count, classes by method count.
    """
    hints = []
    caller_index = getattr(graph, "_caller_index", {})

    # Score functions by caller count
    functions = data.get("functions", [])
    scored_funcs = []
    for func in functions:
        name = func.get("name", "")
        caller_count = len(caller_index.get(name, []))
        scored_funcs.append((caller_count, name))
    scored_funcs.sort(reverse=True)

    # Score classes by method count
    classes = data.get("classes", [])
    scored_classes = []
    for cls in classes:
        name = cls.get("name", "")
        method_count = cls.get("method_count", len(cls.get("methods", [])))
        scored_classes.append((method_count, name))
    scored_classes.sort(reverse=True)

    # Interleave: top class, then top 2 functions (or vice versa)
    if scored_classes:
        _score, cls_name = scored_classes[0]
        hints.append({
            "tool": "get_class_context",
            "args": {"class_name": cls_name},
            "reason": f"{_score} methods" if _score > 0 else "main class in file",
        })

    for _score, func_name in scored_funcs[:2]:
        reason = f"{_score} callers" if _score > 0 else "defined in this file"
        hints.append({
            "tool": "get_function_context",
            "args": {"function_name": func_name},
            "reason": reason,
        })

    return hints[:MAX_HINTS]


def _hints_for_class(graph: Any, data: dict) -> list[dict]:
    """After viewing a class, suggest related exploration.

    Suggests: base class (if has one), most-called method, heaviest usage file.
    """
    hints = []
    caller_index = getattr(graph, "_caller_index", {})

    # Suggest base class exploration
    base_classes = data.get("base_classes", [])
    for base in base_classes[:1]:
        if base and base not in ("object", "ABC", "Protocol"):
            hints.append({
                "tool": "get_class_context",
                "args": {"class_name": base},
                "reason": "parent class",
            })

    # Suggest most-called method
    methods = data.get("methods", [])
    scored_methods = []
    for method in methods:
        name = method.get("name", "")
        if name.startswith("__"):
            continue
        caller_count = len(caller_index.get(name, []))
        scored_methods.append((caller_count, name))
    scored_methods.sort(reverse=True)

    cls_name = data.get("name", "")
    for _score, method_name in scored_methods[:1]:
        qualified = f"{cls_name}.{method_name}" if cls_name else method_name
        reason = f"{_score} callers" if _score > 0 else "key method"
        hints.append({
            "tool": "get_function_context",
            "args": {"function_name": qualified},
            "reason": reason,
        })

    # Suggest heaviest usage file
    usages = data.get("cross_file_usages", [])
    if usages:
        top_usage = max(usages, key=lambda u: len(u.get("lines", [])))
        hints.append({
            "tool": "get_file_context",
            "args": {"file_path": top_usage["file"]},
            "reason": f"references this class ({len(top_usage.get('lines', []))} lines)",
        })

    return hints[:MAX_HINTS]


def _hints_for_function(graph: Any, data: dict) -> list[dict]:
    """After viewing a function, suggest callers and important callees.

    Suggests: top caller, most important callee, parent class (if method).
    """
    hints = []
    caller_index = getattr(graph, "_caller_index", {})

    # Suggest top caller
    called_by = data.get("called_by", [])
    if called_by:
        top_caller = called_by[0]
        caller_name = top_caller.get("name", top_caller.get("caller", ""))
        if caller_name:
            hints.append({
                "tool": "get_function_context",
                "args": {"function_name": caller_name},
                "reason": "top caller of this function",
            })

    # Suggest most important callee (by its own caller count)
    calls = data.get("calls", [])
    scored_calls = []
    for call in calls:
        name = call.get("name", "")
        callee_callers = len(caller_index.get(name, []))
        scored_calls.append((callee_callers, name, call.get("file", "")))
    scored_calls.sort(reverse=True)

    for _score, callee_name, callee_file in scored_calls[:1]:
        reason = f"called by this function, has {_score} callers" if _score > 0 else "called by this function"
        hints.append({
            "tool": "get_function_context",
            "args": {"function_name": callee_name},
            "reason": reason,
        })

    # Suggest parent class if this is a method
    cls_name = data.get("class")
    if cls_name:
        hints.append({
            "tool": "get_class_context",
            "args": {"class_name": cls_name},
            "reason": "parent class of this method",
        })

    return hints[:MAX_HINTS]
```

**Step 4: Run tests to verify they pass**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/test_hints.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add grafyx/server/_hints.py tests/test_hints.py
git commit -m "feat: add navigation hint computation helper"
```

---

### Task 3: Add `detail` parameter to `get_function_context`

**Files:**
- Modify: `grafyx/server/_tools_introspection.py` (lines 43-188)
- Test: `tests/test_resolution_integration.py` (new)

**Step 1: Write the failing test**

```python
"""Integration tests for detail parameter on MCP tools."""
from unittest.mock import MagicMock, patch
import pytest

try:
    import watchdog
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

needs_watchdog = pytest.mark.skipif(not HAS_WATCHDOG, reason="watchdog not installed")


@needs_watchdog
class TestFunctionContextDetail:
    """Test detail parameter on get_function_context."""

    def _make_mock_graph(self):
        mock_graph = MagicMock()
        mock_func = MagicMock()
        mock_func.name = "process"
        mock_func.filepath = "/app/services/orders.py"
        mock_func.is_async = False
        mock_func.docstring = "Process an order."
        mock_func.parameters = []
        mock_func.return_type = "dict"
        mock_func.decorators = ["@cache"]
        mock_func.function_calls = []
        mock_func.dependencies = []
        mock_func.source = "def process(): return {}"

        mock_graph.get_function.return_value = ("python", mock_func, None)
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_line_number.return_value = 10
        mock_graph.get_filepath_from_obj.return_value = ""
        mock_graph.get_callers.return_value = []
        mock_graph._caller_index = {}
        mock_graph._class_method_names = {}
        mock_graph._import_index = {}
        return mock_graph

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_default_detail_is_summary(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        mock_graph_ref.__class__ = type(mock_graph)
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_introspection import get_function_context
            result = get_function_context.fn("process")
            # Summary should have docstring but no source
            assert result.get("docstring") is not None
            assert "source" not in result

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_signatures_strips_docstring(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_introspection import get_function_context
            result = get_function_context.fn("process", detail="signatures")
            assert "docstring" not in result
            assert "source" not in result
            assert "calls" not in result

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_full_includes_source(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_introspection import get_function_context
            result = get_function_context.fn("process", detail="full")
            assert result.get("source") is not None
            assert result.get("docstring") is not None
```

**Step 2: Run tests to verify they fail**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/test_resolution_integration.py::TestFunctionContextDetail -v`
Expected: FAIL (parameter not accepted)

**Step 3: Modify `get_function_context`**

In `grafyx/server/_tools_introspection.py`:

1. Add import at top:
```python
from grafyx.server._resolution import filter_by_detail
from grafyx.server._hints import compute_hints
```

2. Change function signature from:
```python
def get_function_context(function_name: str, include_source: bool = False) -> dict:
```
to:
```python
def get_function_context(function_name: str, detail: str = "summary", include_hints: bool = True) -> dict:
```

3. Replace `if include_source:` block at line 181-182 with:
```python
    # Always include source in the raw dict -- filter_by_detail will strip
    # it for non-"full" levels
    source = safe_str(safe_get_attr(func, "source", "")) or None
    if source:
        context["source"] = source
```

4. Replace `return truncate_response(context)` at line 184 with:
```python
    # Apply detail-level filtering
    context = filter_by_detail(context, detail, "function")

    # Compute navigation hints
    if include_hints:
        hints = compute_hints(graph, "function", context)
        if hints:
            context["suggested_next"] = hints

    return truncate_response(context)
```

**Step 4: Run tests to verify they pass**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/test_resolution_integration.py::TestFunctionContextDetail -v`
Expected: ALL PASS

Also run existing tests: `python -m pytest tests/ -v --tb=short`
Expected: ALL 157+ PASS

**Step 5: Commit**

```bash
git add grafyx/server/_tools_introspection.py tests/test_resolution_integration.py
git commit -m "feat: add detail parameter to get_function_context"
```

---

### Task 4: Add `detail` parameter to `get_file_context` and `get_class_context`

**Files:**
- Modify: `grafyx/server/_tools_introspection.py` (lines 195-551)

**Step 1: Write the failing tests**

Add to `tests/test_resolution_integration.py`:

```python
@needs_watchdog
class TestFileContextDetail:
    """Test detail parameter on get_file_context."""

    def _make_mock_graph(self):
        mock_graph = MagicMock()
        mock_file = MagicMock()
        mock_file.filepath = "/app/services/orders.py"
        mock_file.path = "/app/services/orders.py"
        mock_file.functions = []
        mock_file.classes = []
        mock_file.imports = []
        mock_file.source = "# orders module"

        mock_graph.get_file.return_value = ("python", mock_file)
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_line_number.return_value = 1
        mock_graph.get_importers.return_value = []
        mock_graph._caller_index = {}
        mock_graph._class_method_names = {}
        mock_graph._import_index = {}
        return mock_graph

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_signatures_no_source(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_introspection import get_file_context
            result = get_file_context.fn("orders.py", detail="signatures")
            assert "source" not in result

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_full_has_source(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_introspection import get_file_context
            result = get_file_context.fn("orders.py", detail="full")
            assert "source" in result


@needs_watchdog
class TestClassContextDetail:
    """Test detail parameter on get_class_context."""

    def _make_mock_graph(self):
        mock_graph = MagicMock()
        mock_cls = MagicMock()
        mock_cls.name = "OrderService"
        mock_cls.filepath = "/app/services/orders.py"
        mock_cls.docstring = "Handles orders."
        mock_cls.methods = []
        mock_cls.properties = []
        mock_cls.base_classes = []
        mock_cls.usages = []
        mock_cls.dependencies = []
        mock_cls.source = "class OrderService: pass"

        mock_graph.get_class.return_value = ("python", mock_cls)
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.get_line_number.return_value = 1
        mock_graph.get_filepath_from_obj.return_value = ""
        mock_graph.get_importers.return_value = []
        mock_graph.get_callers.return_value = []
        mock_graph._class_instances = {}
        mock_graph._class_method_names = {}
        mock_graph._caller_index = {}
        mock_graph._import_index = {}
        return mock_graph

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_signatures_no_usages(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_introspection import get_class_context
            result = get_class_context.fn("OrderService", detail="signatures")
            assert "cross_file_usages" not in result
            assert "source" not in result

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_full_has_source(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_introspection import get_class_context
            result = get_class_context.fn("OrderService", detail="full")
            assert "source" in result
```

**Step 2: Run tests to verify they fail**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/test_resolution_integration.py -v`
Expected: New tests FAIL

**Step 3: Modify `get_file_context` and `get_class_context`**

In `get_file_context`:
1. Change signature: `def get_file_context(file_path: str, detail: str = "summary", include_hints: bool = True) -> dict:`
2. Always include source in raw dict, then filter
3. Add `filter_by_detail` + `compute_hints` before return

In `get_class_context`:
1. Change signature: `def get_class_context(class_name: str, detail: str = "summary", include_hints: bool = True) -> dict:`
2. Always include source in raw dict, then filter
3. Add `filter_by_detail` + `compute_hints` before return

**Step 4: Run full test suite**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add grafyx/server/_tools_introspection.py tests/test_resolution_integration.py
git commit -m "feat: add detail parameter to get_file_context and get_class_context"
```

---

### Task 5: Add `detail` + hints to `get_project_skeleton`

**Files:**
- Modify: `grafyx/server/_tools_structure.py`

**Step 1: Write the failing test**

Add to `tests/test_resolution_integration.py`:

```python
@needs_watchdog
class TestSkeletonDetail:
    """Test detail parameter on get_project_skeleton."""

    def _make_mock_graph(self):
        mock_graph = MagicMock()
        mock_graph.get_stats.return_value = {
            "languages": ["python"],
            "total_files": 5,
            "total_functions": 10,
            "total_classes": 3,
            "by_language": {"python": {"files": 5}},
        }
        mock_graph.get_all_files.return_value = []
        mock_graph.original_path = "/app"
        mock_graph.project_path = "/app"
        mock_graph._caller_index = {}
        mock_graph._class_method_names = {}
        mock_graph._import_index = {}
        return mock_graph

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_signatures_strips_dir_stats(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_structure import get_project_skeleton
            result = get_project_skeleton.fn(detail="signatures")
            assert "directory_stats" not in result
            assert "by_language" not in result

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_full_has_everything(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_structure import get_project_skeleton
            result = get_project_skeleton.fn(detail="full")
            assert "directory_stats" in result
```

**Step 2: Run test, verify fail**

**Step 3: Modify `get_project_skeleton`**

1. Add imports for `filter_by_detail` and `compute_hints`
2. Change signature: `def get_project_skeleton(max_depth: int = 3, detail: str = "summary", include_hints: bool = True) -> dict:`
3. Add filtering + hints before return

**Step 4: Run full test suite**

**Step 5: Commit**

```bash
git add grafyx/server/_tools_structure.py tests/test_resolution_integration.py
git commit -m "feat: add detail parameter and hints to get_project_skeleton"
```

---

### Task 6: Create `get_module_context` tool

**Files:**
- Create: `grafyx/server/_tools_exploration.py`
- Modify: `grafyx/server/__init__.py` (add import + re-export)
- Test: `tests/test_module_context.py`

**Step 1: Write the failing tests**

```python
"""Tests for get_module_context tool."""
from unittest.mock import MagicMock, patch
import pytest

try:
    import watchdog
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

needs_watchdog = pytest.mark.skipif(not HAS_WATCHDOG, reason="watchdog not installed")


@needs_watchdog
class TestModuleContext:
    """Test the new get_module_context tool."""

    def _make_mock_graph(self):
        mock_graph = MagicMock()

        # Mock functions
        mock_func1 = MagicMock()
        mock_func1.name = "process_order"
        mock_func1.filepath = "/app/services/orders.py"
        mock_func1.is_async = False
        mock_func1.parameters = []
        mock_func1.return_type = "dict"
        mock_func1.docstring = "Process an order."

        mock_func2 = MagicMock()
        mock_func2.name = "validate"
        mock_func2.filepath = "/app/services/validation.py"
        mock_func2.is_async = False
        mock_func2.parameters = []
        mock_func2.return_type = "bool"
        mock_func2.docstring = "Validate input."

        # Mock classes
        mock_cls = MagicMock()
        mock_cls.name = "OrderService"
        mock_cls.filepath = "/app/services/orders.py"
        mock_cls.docstring = "Order service."
        mock_cls.methods = [mock_func1]
        mock_cls.properties = []
        mock_cls.base_classes = []

        mock_graph.get_all_functions.return_value = [
            {"name": "process_order", "path": "/app/services/orders.py", "signature": "def process_order()", "docstring": "Process an order.", "line": 10},
            {"name": "validate", "path": "/app/services/validation.py", "signature": "def validate()", "docstring": "Validate input.", "line": 5},
        ]
        mock_graph.get_all_classes.return_value = [
            {"name": "OrderService", "path": "/app/services/orders.py", "base_classes": [], "method_count": 1, "docstring": "Order service."},
        ]
        mock_graph.get_all_files.return_value = [
            {"path": "/app/services/orders.py", "function_count": 1, "class_count": 1},
            {"path": "/app/services/validation.py", "function_count": 1, "class_count": 0},
        ]
        mock_graph.get_forward_imports.return_value = ["/app/services/validation.py"]
        mock_graph.translate_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.resolve_path.side_effect = lambda p: str(p) if p else ""
        mock_graph.original_path = "/app"
        mock_graph.project_path = "/app"
        mock_graph._caller_index = {}
        mock_graph._class_method_names = {}
        mock_graph._import_index = {}
        return mock_graph

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_basic_module_context(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_exploration import get_module_context
            result = get_module_context.fn("services")
            assert result["module"] == "services"
            assert "symbols" in result
            assert "total_functions" in result
            assert "total_classes" in result

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_signatures_detail(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_exploration import get_module_context
            result = get_module_context.fn("services", detail="signatures")
            # Signatures should strip docstrings and internal imports
            assert "internal_imports" not in result
            for sym in result.get("symbols", []):
                for func in sym.get("functions", []):
                    assert "docstring" not in func

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_empty_module_path(self, mock_graph_ref):
        """Empty module path should show the whole project."""
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_exploration import get_module_context
            result = get_module_context.fn("")
            assert "symbols" in result

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_hints_included_by_default(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_exploration import get_module_context
            result = get_module_context.fn("services")
            # Hints may or may not be present depending on data, but
            # the key should NOT raise an error
            # (hints are only added if non-empty)

    @patch("grafyx.server._state._init_ready", True)
    @patch("grafyx.server._state._graph")
    def test_no_hints_when_disabled(self, mock_graph_ref):
        mock_graph = self._make_mock_graph()
        with patch("grafyx.server._state._graph", mock_graph), \
             patch("grafyx.server._state._init_ready", True):
            from grafyx.server._tools_exploration import get_module_context
            result = get_module_context.fn("services", include_hints=False)
            assert "suggested_next" not in result
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement `server/_tools_exploration.py`**

```python
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

    get_project_skeleton  →  get_module_context  →  get_file/class/function_context

Registered on the shared ``mcp`` instance from ``_state.py``.
"""

import os

from fastmcp.exceptions import ToolError

from grafyx.server._hints import compute_hints
from grafyx.server._resolution import filter_by_detail
from grafyx.server._state import _ensure_initialized, mcp
from grafyx.utils import format_function_signature, safe_get_attr, safe_str, truncate_response


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

        # Collect all files matching the module path
        all_files = graph.get_all_files()
        all_functions = graph.get_all_functions()
        all_classes = graph.get_all_classes()

        # Group by file, filtering to module_path
        file_symbols: dict[str, dict] = {}  # rel_path -> {functions: [], classes: []}

        def _get_rel_path(abs_path: str) -> str | None:
            """Convert absolute path to project-relative path."""
            p = abs_path.replace("\\", "/")
            for root in (project_root, mirror_root):
                if p.startswith(root):
                    return p[len(root):]
            return p

        def _matches_module(rel_path: str) -> bool:
            """Check if a relative path belongs to the target module."""
            if not module_filter:
                return True
            return rel_path.startswith(module_filter + "/") or rel_path.startswith(module_filter + "\\")

        # Index files
        for f_info in all_files:
            fpath = f_info.get("path", "")
            rel = _get_rel_path(fpath)
            if rel and _matches_module(rel):
                file_key = rel
                if file_key not in file_symbols:
                    file_symbols[file_key] = {"functions": [], "classes": []}

        # Index functions into their files
        for func_info in all_functions:
            fpath = func_info.get("path", "")
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
        for cls_info in all_classes:
            fpath = cls_info.get("path", "")
            rel = _get_rel_path(fpath)
            if rel and _matches_module(rel):
                if rel not in file_symbols:
                    file_symbols[rel] = {"functions": [], "classes": []}
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
            # Extract just the filename from the rel path
            filename = os.path.basename(file_path)
            symbols.append({
                "file": filename,
                "file_path": file_path,
                "functions": data["functions"],
                "classes": data["classes"],
            })

        # Build internal import relationships
        internal_imports = []
        for file_path in sorted(file_symbols.keys()):
            abs_path = project_root + file_path
            forward = graph.get_forward_imports(abs_path)
            if not forward:
                # Try with mirror path
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
```

**Step 4: Update `server/__init__.py`**

Add to imports:
```python
from grafyx.server import _tools_exploration  # noqa: F401
```

Add to re-exports:
```python
from grafyx.server._tools_exploration import get_module_context  # noqa: F401
```

Add `"get_module_context"` to `__all__`.

**Step 5: Run full test suite**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add grafyx/server/_tools_exploration.py grafyx/server/__init__.py tests/test_module_context.py
git commit -m "feat: add get_module_context tool (intermediate zoom level)"
```

---

### Task 7: Update existing tool tests for new parameter names

**Files:**
- Modify: `tests/test_class_context.py` -- change `include_source=True` to `detail="full"`
- Modify: `tests/test_dependency_graph.py` -- update if needed
- Modify: `tests/test_singleton_tracking.py` -- change `include_source` references

**Step 1: Find all tests using `include_source`**

Run: `grep -rn "include_source" tests/`

**Step 2: Update each test to use `detail="full"` instead of `include_source=True`**

For each test calling a tool with `include_source=True`, change to `detail="full"`.
For each test calling with `include_source=False` (or default), change to `detail="summary"` or remove the param (summary is default).

**Step 3: Run full test suite**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/
git commit -m "refactor: update tests to use detail parameter instead of include_source"
```

---

### Task 8: Verify tool registration and end-to-end

**Step 1: Verify all 15 tools register**

Run:
```bash
cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate
python -c "from grafyx.server import mcp; print(f'Tools: {len(mcp._tool_manager._tools)}'); [print(f'  - {name}') for name in sorted(mcp._tool_manager._tools.keys())]"
```
Expected: 15 tools (14 original + get_module_context)

**Step 2: Run full test suite one final time**

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && source ~/grafyx-venv/bin/activate && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS (157 original + ~25 new tests)

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Fractal Context features (resolution control, module context, navigation hints)"
```
