# New Grafyx Tools Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two new MCP tools to Grafyx: `get_subclasses` (find all classes extending a given class) and `get_module_dependencies` (show package-level dependency graph).

**Architecture:** Both tools follow the existing pattern — graph method in `graph.py` for logic, thin `@mcp.tool` wrapper in `server.py` for error handling and response formatting. Tests use mock graphs matching the pattern in `tests/test_search.py`.

**Tech Stack:** Python 3.12+, graph-sitter, FastMCP, pytest

---

### Task 1: Extend test fixtures with inheritance hierarchy

**Files:**
- Modify: `tests/fixtures/python_project/models.py`

The current fixture has `User`, `Product`, `Inventory` with no inheritance. We need a class hierarchy to test `get_subclasses`.

**Step 1: Add base class and subclasses to the fixture**

Add these classes at the end of `tests/fixtures/python_project/models.py`:

```python
class BaseHandler:
    """Base class for all handlers."""

    def handle(self, data: dict) -> dict:
        """Process incoming data."""
        raise NotImplementedError

    def validate(self, data: dict) -> bool:
        """Validate incoming data."""
        return bool(data)


class HTTPHandler(BaseHandler):
    """Handles HTTP requests."""

    def handle(self, data: dict) -> dict:
        return {"status": 200, "body": data}


class WebSocketHandler(BaseHandler):
    """Handles WebSocket connections."""

    def handle(self, data: dict) -> dict:
        return {"type": "ws", "payload": data}


class APIHandler(HTTPHandler):
    """Handles API-specific HTTP requests. Extends HTTPHandler."""

    def handle(self, data: dict) -> dict:
        result = super().handle(data)
        result["api_version"] = "v1"
        return result
```

**Step 2: Verify fixture is valid Python**

Run: `python -c "import ast; ast.parse(open('tests/fixtures/python_project/models.py').read()); print('OK')"`
Expected: `OK`

---

### Task 2: Add `get_subclasses` method to `graph.py`

**Files:**
- Modify: `grafyx/graph.py` (add method before the `_is_test_path` static method, around line 1585)

**Step 1: Write the failing test**

Create `tests/test_subclasses.py`:

```python
"""Tests for get_subclasses in grafyx.graph module."""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph


def _make_mock_graph_with_classes():
    """Build a mock CodebaseGraph with a class hierarchy for testing."""
    graph = MagicMock(spec=CodebaseGraph)
    graph.get_all_classes.return_value = [
        {"name": "BaseHandler", "base_classes": [], "file": "models.py", "language": "python", "line": 50, "method_count": 2},
        {"name": "HTTPHandler", "base_classes": ["BaseHandler"], "file": "models.py", "language": "python", "line": 60, "method_count": 1},
        {"name": "WebSocketHandler", "base_classes": ["BaseHandler"], "file": "models.py", "language": "python", "line": 70, "method_count": 1},
        {"name": "APIHandler", "base_classes": ["HTTPHandler"], "file": "models.py", "language": "python", "line": 80, "method_count": 1},
        {"name": "User", "base_classes": [], "file": "models.py", "language": "python", "line": 8, "method_count": 2},
        {"name": "Product", "base_classes": [], "file": "models.py", "language": "python", "line": 25, "method_count": 2},
    ]
    # Use the real method, bound to the mock
    graph.get_subclasses = CodebaseGraph.get_subclasses.__get__(graph)
    return graph


class TestGetSubclasses:
    def test_direct_subclasses(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("BaseHandler")
        assert result["class_name"] == "BaseHandler"
        assert result["direct_subclass_count"] == 2
        names = [s["name"] for s in result["subclasses"]]
        assert "HTTPHandler" in names
        assert "WebSocketHandler" in names

    def test_recursive_subclasses(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("BaseHandler", depth=3)
        assert result["total_subclass_count"] == 3  # HTTP, WebSocket, API
        # APIHandler should be nested under HTTPHandler
        http = next(s for s in result["subclasses"] if s["name"] == "HTTPHandler")
        assert len(http["subclasses"]) == 1
        assert http["subclasses"][0]["name"] == "APIHandler"

    def test_depth_1_limits_to_direct(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("BaseHandler", depth=1)
        assert result["direct_subclass_count"] == 2
        assert result["total_subclass_count"] == 2  # Only direct
        # HTTPHandler should have empty subclasses at depth=1
        http = next(s for s in result["subclasses"] if s["name"] == "HTTPHandler")
        assert http["subclasses"] == []

    def test_no_subclasses(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("User")
        assert result["direct_subclass_count"] == 0
        assert result["total_subclass_count"] == 0
        assert result["subclasses"] == []

    def test_class_not_found(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("NonExistentClass")
        assert result is None

    def test_leaf_class(self):
        graph = _make_mock_graph_with_classes()
        result = graph.get_subclasses("APIHandler")
        assert result["direct_subclass_count"] == 0
        assert result["subclasses"] == []
```

**Step 2: Run test to verify it fails**

Run (in WSL): `source ~/grafyx-venv/bin/activate && cd "/mnt/c/Kişisel Projelerim/Grafyx" && pytest tests/test_subclasses.py -v`
Expected: FAIL — `CodebaseGraph` has no `get_subclasses` method

**Step 3: Implement `get_subclasses` in `graph.py`**

Add this method to `CodebaseGraph` class, right before `_is_test_path` (around line 1585 in `grafyx/graph.py`):

```python
    def get_subclasses(self, class_name: str, depth: int = 3) -> dict | None:
        """Find all classes that extend the given class, recursively.

        Args:
            class_name: The base class to find subclasses of.
            depth: How many inheritance levels to traverse (1 = direct only).

        Returns:
            Dict with class info and nested subclasses tree, or None if
            the class is not found in the codebase.
        """
        with self._lock:
            all_classes = self.get_all_classes(max_results=2000)

            # Check that the target class exists
            target = None
            for cls_dict in all_classes:
                if cls_dict.get("name") == class_name:
                    target = cls_dict
                    break
            if target is None:
                return None

            # Build reverse map: base_name -> [child_class_dicts]
            children_of: dict[str, list[dict]] = {}
            for cls_dict in all_classes:
                for base in cls_dict.get("base_classes", []):
                    base_name = str(base).split(".")[-1].strip()
                    if base_name:
                        if base_name not in children_of:
                            children_of[base_name] = []
                        children_of[base_name].append(cls_dict)

            # Recursively build subclass tree
            def _build_tree(name: str, current_depth: int, visited: set) -> list[dict]:
                if current_depth <= 0 or name in visited:
                    return []
                visited.add(name)
                direct = children_of.get(name, [])
                result = []
                for child in direct:
                    child_name = child.get("name", "")
                    entry = {
                        "name": child_name,
                        "file": child.get("file", ""),
                        "line": child.get("line"),
                        "language": child.get("language", ""),
                        "subclasses": _build_tree(child_name, current_depth - 1, visited.copy()),
                    }
                    result.append(entry)
                return result

            tree = _build_tree(class_name, depth, set())

            # Count totals
            def _count(nodes: list[dict]) -> int:
                total = len(nodes)
                for node in nodes:
                    total += _count(node.get("subclasses", []))
                return total

            direct_count = len(children_of.get(class_name, []))
            total_count = _count(tree)

            return {
                "class_name": class_name,
                "file": target.get("file", ""),
                "line": target.get("line"),
                "language": target.get("language", ""),
                "direct_subclass_count": direct_count,
                "total_subclass_count": total_count,
                "subclasses": tree,
            }
```

**Step 4: Run tests to verify they pass**

Run (in WSL): `source ~/grafyx-venv/bin/activate && cd "/mnt/c/Kişisel Projelerim/Grafyx" && pytest tests/test_subclasses.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add tests/fixtures/python_project/models.py grafyx/graph.py tests/test_subclasses.py
git commit -m "feat: add get_subclasses method to CodebaseGraph"
```

---

### Task 3: Add `get_subclasses` MCP tool to `server.py`

**Files:**
- Modify: `grafyx/server.py` (add Tool 13 after line 1047)

**Step 1: Write the failing test**

Add to `tests/test_subclasses.py`:

```python
import sys
from unittest.mock import patch

try:
    import watchdog  # noqa: F401
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

needs_watchdog = pytest.mark.skipif(not HAS_WATCHDOG, reason="watchdog not installed")


@needs_watchdog
class TestGetSubclassesTool:
    """Test the MCP tool wrapper in server.py."""

    @patch("grafyx.server._graph")
    def test_tool_returns_subclasses(self, mock_graph):
        mock_graph.initialized = True
        mock_graph.get_subclasses.return_value = {
            "class_name": "BaseHandler",
            "file": "models.py",
            "line": 50,
            "language": "python",
            "direct_subclass_count": 2,
            "total_subclass_count": 3,
            "subclasses": [
                {"name": "HTTPHandler", "file": "models.py", "line": 60, "language": "python", "subclasses": []},
            ],
        }

        from grafyx.server import get_subclasses
        # MCP tool functions are FunctionTool objects — call .fn directly
        result = get_subclasses.fn(class_name="BaseHandler")
        assert result["class_name"] == "BaseHandler"
        assert result["direct_subclass_count"] == 2
        mock_graph.get_subclasses.assert_called_once_with("BaseHandler", 3)

    @patch("grafyx.server._graph")
    def test_tool_class_not_found(self, mock_graph):
        mock_graph.initialized = True
        mock_graph.get_subclasses.return_value = None

        from grafyx.server import get_subclasses
        from fastmcp.exceptions import ToolError
        import pytest as pt
        with pt.raises(ToolError, match="not found"):
            get_subclasses.fn(class_name="Nonexistent")
```

Add `import pytest` to the top of `tests/test_subclasses.py` if not already present.

**Step 2: Run test to verify it fails**

Run (in WSL): `source ~/grafyx-venv/bin/activate && cd "/mnt/c/Kişisel Projelerim/Grafyx" && pytest tests/test_subclasses.py::TestGetSubclassesTool -v`
Expected: FAIL — `get_subclasses` tool not defined in server.py

**Step 3: Add the MCP tool wrapper to `server.py`**

Append after line 1047 (end of `get_circular_dependencies`) in `grafyx/server.py`:

```python


# ── Tool 13: get_subclasses ──


@mcp.tool
def get_subclasses(class_name: str, depth: int = 3) -> dict:
    """Find all classes that extend a given class, recursively.

    Use this to understand the impact of changing a class's interface:
    which subclasses would need updating?

    Supports multi-level inheritance trees up to the specified depth.
    """
    graph = _ensure_initialized()
    try:
        result = graph.get_subclasses(class_name, depth)
        if result is None:
            raise ToolError(f"Class '{class_name}' not found in the codebase.")
        return truncate_response(result)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_subclasses: {e}")
```

**Step 4: Run tests to verify they pass**

Run (in WSL): `source ~/grafyx-venv/bin/activate && cd "/mnt/c/Kişisel Projelerim/Grafyx" && pytest tests/test_subclasses.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add grafyx/server.py tests/test_subclasses.py
git commit -m "feat: add get_subclasses MCP tool"
```

---

### Task 4: Add `get_module_dependencies` method to `graph.py`

**Files:**
- Modify: `grafyx/graph.py` (add method after `get_subclasses`, before `_is_test_path`)

**Step 1: Write the failing test**

Create `tests/test_module_deps.py`:

```python
"""Tests for get_module_dependencies in grafyx.graph module."""

from grafyx.graph import CodebaseGraph
from unittest.mock import MagicMock


def _make_mock_graph_with_modules():
    """Build a mock CodebaseGraph with module-level import data."""
    graph = MagicMock(spec=CodebaseGraph)
    graph.original_path = "/project"
    graph._project_path = "/project"

    # Simulate _forward_import_index: file -> [files it imports]
    graph._forward_import_index = {
        "/project/api/routes.py": ["/project/services/user_service.py", "/project/models/user.py"],
        "/project/api/middleware.py": ["/project/services/auth_service.py"],
        "/project/services/user_service.py": ["/project/models/user.py", "/project/utils/helpers.py"],
        "/project/services/auth_service.py": ["/project/models/user.py"],
        "/project/models/user.py": ["/project/utils/helpers.py"],
        "/project/utils/helpers.py": [],
    }

    # Simulate get_all_files to list all known files
    graph.get_all_files.return_value = [
        {"path": "/project/api/routes.py", "language": "python"},
        {"path": "/project/api/middleware.py", "language": "python"},
        {"path": "/project/services/user_service.py", "language": "python"},
        {"path": "/project/services/auth_service.py", "language": "python"},
        {"path": "/project/models/user.py", "language": "python"},
        {"path": "/project/utils/helpers.py", "language": "python"},
    ]

    # translate_path is identity in this mock
    graph.translate_path = lambda p: p

    # Use the real method
    graph.get_module_dependencies = CodebaseGraph.get_module_dependencies.__get__(graph)
    return graph


class TestGetModuleDependencies:
    def test_basic_module_graph(self):
        graph = _make_mock_graph_with_modules()
        result = graph.get_module_dependencies()
        assert "modules" in result
        assert "edges" in result
        # Should have 4 modules: api, services, models, utils
        assert len(result["modules"]) == 4
        assert "api" in result["modules"]
        assert "services" in result["modules"]

    def test_edge_counts(self):
        graph = _make_mock_graph_with_modules()
        result = graph.get_module_dependencies()
        edges = result["edges"]
        # api -> services should exist (routes imports user_service, middleware imports auth_service)
        api_to_services = next((e for e in edges if e["from"] == "api" and e["to"] == "services"), None)
        assert api_to_services is not None
        assert api_to_services["import_count"] == 2

    def test_depends_on_and_depended_on_by(self):
        graph = _make_mock_graph_with_modules()
        result = graph.get_module_dependencies()
        modules = result["modules"]
        # utils should be depended on by services and models
        assert "services" in modules["utils"]["depended_on_by"]
        assert "models" in modules["utils"]["depended_on_by"]
        # utils depends on nothing
        assert modules["utils"]["depends_on"] == []

    def test_module_filter(self):
        graph = _make_mock_graph_with_modules()
        result = graph.get_module_dependencies(module_path="api")
        # Should still return full graph but we verify api is present
        assert "api" in result["modules"]

    def test_file_counts(self):
        graph = _make_mock_graph_with_modules()
        result = graph.get_module_dependencies()
        assert result["modules"]["api"]["file_count"] == 2
        assert result["modules"]["services"]["file_count"] == 2
        assert result["modules"]["models"]["file_count"] == 1
        assert result["modules"]["utils"]["file_count"] == 1

    def test_self_imports_excluded(self):
        """Imports within the same module should not appear as edges."""
        graph = _make_mock_graph_with_modules()
        # Add an intra-module import
        graph._forward_import_index["/project/api/routes.py"].append("/project/api/middleware.py")
        result = graph.get_module_dependencies()
        # No edge from api -> api
        self_edges = [e for e in result["edges"] if e["from"] == e["to"]]
        assert len(self_edges) == 0

    def test_root_files_grouped(self):
        """Files not in any subdirectory should be grouped under '.'."""
        graph = _make_mock_graph_with_modules()
        graph.get_all_files.return_value.append(
            {"path": "/project/main.py", "language": "python"}
        )
        graph._forward_import_index["/project/main.py"] = ["/project/api/routes.py"]
        result = graph.get_module_dependencies()
        assert "." in result["modules"]
```

**Step 2: Run test to verify it fails**

Run (in WSL): `source ~/grafyx-venv/bin/activate && cd "/mnt/c/Kişisel Projelerim/Grafyx" && pytest tests/test_module_deps.py -v`
Expected: FAIL — `CodebaseGraph` has no `get_module_dependencies` method

**Step 3: Implement `get_module_dependencies` in `graph.py`**

Add this method to `CodebaseGraph` class, right after `get_subclasses` and before `_is_test_path`:

```python
    def get_module_dependencies(
        self, module_path: str = "", depth: int = 1,
    ) -> dict:
        """Show how directories/packages depend on each other.

        Aggregates file-level imports into module-level (directory) dependencies.

        Args:
            module_path: Optional filter — only show edges involving this module.
            depth: Subdirectory grouping depth (1 = top-level dirs only).

        Returns:
            Dict with modules, edges, and module_count.
        """
        with self._lock:
            all_files = self.get_all_files(max_results=2000)
            project_root = self.original_path.replace("\\", "/").rstrip("/") + "/"

            # Assign each file to a module (directory) at the given depth
            def _get_module(file_path: str) -> str:
                fp = file_path.replace("\\", "/")
                if fp.startswith(project_root):
                    rel = fp[len(project_root):]
                else:
                    rel = fp.lstrip("/")
                parts = rel.split("/")
                if len(parts) <= 1:
                    return "."  # Root-level files
                return "/".join(parts[:depth])

            # Build set of known files and their modules
            file_to_module: dict[str, str] = {}
            module_files: dict[str, int] = {}  # module -> file count
            for f in all_files:
                fpath = f.get("path", "").replace("\\", "/")
                if not fpath:
                    continue
                mod = _get_module(fpath)
                file_to_module[fpath] = mod
                module_files[mod] = module_files.get(mod, 0) + 1

            # Also index files from _forward_import_index (may have paths
            # not returned by get_all_files if they are import targets)
            for fpath in self._forward_import_index:
                fp = fpath.replace("\\", "/")
                if fp not in file_to_module:
                    mod = _get_module(fp)
                    file_to_module[fp] = mod
                for target in self._forward_import_index[fpath]:
                    tp = target.replace("\\", "/")
                    if tp not in file_to_module:
                        file_to_module[tp] = _get_module(tp)

            # Aggregate edges: (from_module, to_module) -> count
            edge_counts: dict[tuple[str, str], int] = {}
            for source_file, targets in self._forward_import_index.items():
                src_fp = source_file.replace("\\", "/")
                src_mod = file_to_module.get(src_fp, _get_module(src_fp))
                for target_file in targets:
                    tgt_fp = target_file.replace("\\", "/")
                    tgt_mod = file_to_module.get(tgt_fp, _get_module(tgt_fp))
                    # Skip self-imports (within same module)
                    if src_mod == tgt_mod:
                        continue
                    key = (src_mod, tgt_mod)
                    edge_counts[key] = edge_counts.get(key, 0) + 1

            # Build module detail map
            all_modules: set[str] = set(module_files.keys())
            for src, tgt in edge_counts:
                all_modules.add(src)
                all_modules.add(tgt)

            # Apply module_path filter
            if module_path:
                filtered_edges = {
                    k: v for k, v in edge_counts.items()
                    if k[0] == module_path or k[1] == module_path
                }
                relevant_modules = {module_path}
                for src, tgt in filtered_edges:
                    relevant_modules.add(src)
                    relevant_modules.add(tgt)
                all_modules = relevant_modules
                edge_counts = filtered_edges

            modules: dict[str, dict] = {}
            for mod in sorted(all_modules):
                depends_on = sorted({tgt for (src, tgt), _ in edge_counts.items() if src == mod})
                depended_on_by = sorted({src for (src, tgt), _ in edge_counts.items() if tgt == mod})
                modules[mod] = {
                    "file_count": module_files.get(mod, 0),
                    "depends_on": depends_on,
                    "depended_on_by": depended_on_by,
                }

            edges = [
                {"from": src, "to": tgt, "import_count": count}
                for (src, tgt), count in sorted(edge_counts.items(), key=lambda x: -x[1])
            ]

            return {
                "project_path": self.original_path,
                "module_count": len(modules),
                "modules": modules,
                "edges": edges,
            }
```

**Step 4: Run tests to verify they pass**

Run (in WSL): `source ~/grafyx-venv/bin/activate && cd "/mnt/c/Kişisel Projelerim/Grafyx" && pytest tests/test_module_deps.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add grafyx/graph.py tests/test_module_deps.py
git commit -m "feat: add get_module_dependencies method to CodebaseGraph"
```

---

### Task 5: Add `get_module_dependencies` MCP tool to `server.py`

**Files:**
- Modify: `grafyx/server.py` (add Tool 14 after Tool 13)

**Step 1: Write the failing test**

Add to `tests/test_module_deps.py`:

```python
import sys
import pytest

try:
    import watchdog  # noqa: F401
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

needs_watchdog = pytest.mark.skipif(not HAS_WATCHDOG, reason="watchdog not installed")


@needs_watchdog
class TestGetModuleDepsTool:
    """Test the MCP tool wrapper in server.py."""

    @patch("grafyx.server._graph")
    def test_tool_returns_modules(self, mock_graph):
        mock_graph.initialized = True
        mock_graph.get_module_dependencies.return_value = {
            "project_path": "/project",
            "module_count": 2,
            "modules": {
                "api": {"file_count": 2, "depends_on": ["services"], "depended_on_by": []},
                "services": {"file_count": 2, "depends_on": [], "depended_on_by": ["api"]},
            },
            "edges": [{"from": "api", "to": "services", "import_count": 3}],
        }

        from grafyx.server import get_module_dependencies
        result = get_module_dependencies.fn()
        assert result["module_count"] == 2
        assert "api" in result["modules"]
        mock_graph.get_module_dependencies.assert_called_once_with("", 1)

    @patch("grafyx.server._graph")
    def test_tool_with_filter(self, mock_graph):
        mock_graph.initialized = True
        mock_graph.get_module_dependencies.return_value = {
            "project_path": "/project",
            "module_count": 1,
            "modules": {"api": {"file_count": 2, "depends_on": [], "depended_on_by": []}},
            "edges": [],
        }

        from grafyx.server import get_module_dependencies
        result = get_module_dependencies.fn(module_path="api", depth=2)
        mock_graph.get_module_dependencies.assert_called_once_with("api", 2)
```

Add `from unittest.mock import patch` to the top of the file if not already present.

**Step 2: Run test to verify it fails**

Run (in WSL): `source ~/grafyx-venv/bin/activate && cd "/mnt/c/Kişisel Projelerim/Grafyx" && pytest tests/test_module_deps.py::TestGetModuleDepsTool -v`
Expected: FAIL — `get_module_dependencies` tool not defined in server.py

**Step 3: Add the MCP tool wrapper to `server.py`**

Append after Tool 13 (`get_subclasses`) in `grafyx/server.py`:

```python


# ── Tool 14: get_module_dependencies ──


@mcp.tool
def get_module_dependencies(module_path: str = "", depth: int = 1) -> dict:
    """Show how directories/packages depend on each other at the module level.

    Aggregates file-level imports into package-level dependencies. Use this
    to understand which packages would be affected by refactoring a module.

    Args:
        module_path: Optional filter to show only edges involving this module.
        depth: Subdirectory grouping depth (1 = top-level directories).
    """
    graph = _ensure_initialized()
    try:
        result = graph.get_module_dependencies(module_path, depth)
        return truncate_response(result)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_module_dependencies: {e}")
```

**Step 4: Run all tests to verify everything passes**

Run (in WSL): `source ~/grafyx-venv/bin/activate && cd "/mnt/c/Kişisel Projelerim/Grafyx" && pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add grafyx/server.py tests/test_module_deps.py
git commit -m "feat: add get_module_dependencies MCP tool"
```

---

### Task 6: Final integration verification

**Step 1: Run the full test suite**

Run (in WSL): `source ~/grafyx-venv/bin/activate && cd "/mnt/c/Kişisel Projelerim/Grafyx" && pytest tests/ -v --tb=short`
Expected: All tests PASS, no regressions

**Step 2: Verify tool count**

Run (in WSL): `source ~/grafyx-venv/bin/activate && python -c "from grafyx.server import mcp; print(f'Tools: {len(mcp._tool_manager._tools)}')"`
Expected: `Tools: 14`

**Step 3: Final commit (if any fixes were needed)**

```bash
git add -A
git commit -m "chore: finalize new tools integration"
```
