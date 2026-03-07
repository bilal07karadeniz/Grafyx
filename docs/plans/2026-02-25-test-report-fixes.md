# Test Report Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 4 bugs and accuracy gaps identified in the Eddy project test report, raising Grafyx from 8.3/10 to ~9.2/10 overall.

**Architecture:** Six independent fixes targeting the three subsystems: graph engine (4 fixes), search engine (1 fix), and import resolution (1 fix). Each fix is self-contained with its own test file, touching 1-2 source files. All fixes are additive (no breaking changes to existing APIs or data structures).

**Tech Stack:** Python 3.12, pytest, unittest.mock, graph-sitter (via `safe_get_attr`)

**Run commands in WSL:**
- Activate: `source ~/grafyx-venv/bin/activate`
- Test: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/ -v`
- Single test: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_file.py::TestClass::test_method -v`

---

## Task 1: Fix Import Alias Tracking in Symbol Index

**Problem:** `from utils import User as AppUser` extracts `{"User"}` but loses the alias `"AppUser"`. When code later calls `AppUser.method()`, the caller index can't link it back. Same for `import X as Y`.

**Impact:** MEDIUM — undercounts callers, causes false "unused" reports for symbols accessed through aliases.

**Files:**
- Modify: `grafyx/graph/_indexes.py:722-780` (`_extract_symbol_names_from_import`)
- Modify: `grafyx/graph/_indexes.py:626-633` (symbol index population in `_build_import_index`)
- Test: `tests/test_alias_tracking.py` (new)

### Step 1: Write the failing tests

Create `tests/test_alias_tracking.py`:

```python
"""Tests for import alias tracking in the symbol index.

Covers:
- _extract_symbol_names_from_import returns both original AND alias names
- _file_symbol_imports stores alias mappings
- Aliased symbols are findable via _build_imported_names
"""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph


class TestExtractSymbolNamesWithAliases:
    """_extract_symbol_names_from_import should return both original and alias."""

    def test_python_from_import_with_alias_returns_both(self):
        """from models import User as AppUser → {User, AppUser}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "from models import User as AppUser"
        )
        assert "User" in names
        assert "AppUser" in names

    def test_python_from_import_multiple_aliases(self):
        """from utils import A as X, B as Y, C → {A, X, B, Y, C}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "from utils import A as X, B as Y, C"
        )
        assert names == {"A", "X", "B", "Y", "C"}

    def test_python_from_import_no_alias_unchanged(self):
        """from auth import create_token, verify → {create_token, verify}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "from auth import create_token, verify"
        )
        assert names == {"create_token", "verify"}

    def test_ts_named_import_with_alias_returns_both(self):
        """import { Config as AppConfig } from './config' → {Config, AppConfig}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "import { Config as AppConfig } from './config'"
        )
        assert "Config" in names
        assert "AppConfig" in names

    def test_ts_named_import_multiple_aliases(self):
        """import { A as X, B } from './mod' → {A, X, B}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "import { A as X, B } from './mod'"
        )
        assert names == {"A", "X", "B"}


class TestImportedNamesIncludesAliases:
    """_build_imported_names should include alias names so dead code
    detection doesn't flag aliased re-exports as unused."""

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._lock = MagicMock()
        graph._external_packages = set()
        graph.translate_path = lambda p: str(p) if p else ""
        graph._is_ignored_file_path = lambda p: False
        graph._build_imported_names = CodebaseGraph._build_imported_names.__get__(graph)
        graph._extract_symbol_names_from_import = CodebaseGraph._extract_symbol_names_from_import
        return graph

    def test_alias_appears_in_imported_names(self):
        """If file imports 'from X import store_chunks as store_chunks_in_qdrant',
        both 'store_chunks' and 'store_chunks_in_qdrant' should be in imported names."""
        graph = self._make_graph()

        imp = MagicMock()
        imp.source = "from rag.chunking import store_chunks as store_chunks_in_qdrant"
        f = MagicMock()
        f.imports = [imp]
        codebase = MagicMock()
        codebase.files = [f]
        graph._codebases = {"python": codebase}

        names = graph._build_imported_names()
        assert "store_chunks" in names
        assert "store_chunks_in_qdrant" in names
```

### Step 2: Run tests to verify they fail

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_alias_tracking.py -v`
Expected: FAIL — `AppUser` not in names, `store_chunks_in_qdrant` not in names.

### Step 3: Fix `_extract_symbol_names_from_import` to include aliases

In `grafyx/graph/_indexes.py`, modify the TS/JS named import section (line ~744-748):

**Current** (line 744-745):
```python
            for token in content.split(","):
                clean = token.strip().split(" as ")[0].strip()
```

**New:**
```python
            for token in content.split(","):
                parts = token.strip().split(" as ")
                clean = parts[0].strip()
                if clean and clean.isidentifier() and clean not in NOISE:
                    names.add(clean)
                # Also add the alias name so callers using the alias are tracked
                if len(parts) > 1:
                    alias = parts[1].strip()
                    if alias and alias.isidentifier() and alias not in NOISE:
                        names.add(alias)
            return names
```

Note: The original code after the loop already has `if clean and clean.isidentifier() and clean not in NOISE: names.add(clean)` — we need to restructure the loop to add BOTH original and alias inside the loop, then remove the duplicate check after. The full replacement for lines 742-748:

```python
        if "{" in imp_str and "}" in imp_str:
            content = imp_str.split("{", 1)[1].split("}", 1)[0]
            for token in content.split(","):
                parts = token.strip().split(" as ")
                original = parts[0].strip()
                if original and original.isidentifier() and original not in NOISE:
                    names.add(original)
                if len(parts) > 1:
                    alias = parts[1].strip()
                    if alias and alias.isidentifier() and alias not in NOISE:
                        names.add(alias)
            return names
```

Then modify the Python `from X import A as B` section (line ~771-772):

**Current:**
```python
            for token in names_part.split(","):
                clean = token.strip().split(" as ")[0].strip()
```

**New:**
```python
            for token in names_part.split(","):
                parts = token.strip().split(" as ")
                clean = parts[0].strip()
                if "." in clean:
                    clean = clean.rsplit(".", 1)[-1]
                if clean and clean.isidentifier() and clean not in NOISE:
                    names.add(clean)
                if len(parts) > 1:
                    alias = parts[1].strip()
                    if alias and alias.isidentifier() and alias not in NOISE:
                        names.add(alias)
            return names
```

### Step 4: Run tests to verify they pass

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_alias_tracking.py -v`
Expected: PASS

### Step 5: Run full test suite for regressions

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/ -v`
Expected: All existing tests pass. The `test_python_from_import_with_alias` test in `test_symbol_imports.py` (line 25-30) asserts `names == {"User"}` — this WILL BREAK because we now also return `"AppUser"`. Fix it:

In `tests/test_symbol_imports.py:25-30`, change:
```python
    def test_python_from_import_with_alias(self):
        """from module import Foo as Bar → {Foo, Bar}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "from models import User as AppUser"
        )
        assert names == {"User", "AppUser"}
```

And in `tests/test_symbol_imports.py:53-58` (TS alias test), change:
```python
    def test_ts_named_imports_with_alias(self):
        """import { Foo as Bar } from './module' → {Foo, Bar}"""
        names = CodebaseGraph._extract_symbol_names_from_import(
            "import { Config as AppConfig } from './config'"
        )
        assert names == {"Config", "AppConfig"}
```

### Step 6: Run full test suite again

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/ -v`
Expected: ALL PASS

### Step 7: Commit

```bash
git add tests/test_alias_tracking.py grafyx/graph/_indexes.py tests/test_symbol_imports.py
git commit -m "fix: track import aliases in symbol index (User as AppUser → both tracked)"
```

---

## Task 2: Fix Local Module Disambiguation in Import Resolution

**Problem:** `from config import X` in `load_tests/worker.py` resolves to `backend/app/api/voice/config.py` instead of `load_tests/config.py`. The suffix-based resolver picks whichever suffix matches first without considering the importing file's directory.

**Impact:** LOW-MEDIUM — false module dependency edges (e.g., `load_tests → backend`).

**Files:**
- Modify: `grafyx/graph/_indexes.py:536-607` (`_build_import_index`, Phase 2 resolution)
- Test: `tests/test_local_module_resolution.py` (new)

### Step 1: Write the failing tests

Create `tests/test_local_module_resolution.py`:

```python
"""Tests for local module preference in import resolution.

When multiple files match a module name suffix (e.g., config.py exists in
both load_tests/ and backend/app/api/voice/), the resolver should prefer
the file closest to the importing file's directory.
"""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph


class TestLocalModuleResolution:
    """Import resolution should prefer same-directory modules."""

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._project_path = "/project"
        graph.original_path = "/project"
        graph._lock = MagicMock()
        graph._external_packages = set()
        graph.translate_path = lambda p: str(p) if p else ""
        graph._build_import_index = CodebaseGraph._build_import_index.__get__(graph)
        graph._extract_module_from_import = CodebaseGraph._extract_module_from_import
        graph._extract_symbol_names_from_import = CodebaseGraph._extract_symbol_names_from_import
        graph._is_ignored_file_path = lambda p: False
        return graph

    def test_same_dir_config_preferred_over_distant(self):
        """load_tests/worker.py importing 'config' should resolve to
        load_tests/config.py, not backend/app/api/voice/config.py."""
        graph = self._make_graph()

        # load_tests/worker.py does 'from config import Settings'
        worker = MagicMock()
        worker.filepath = "/project/load_tests/worker.py"
        worker.path = "/project/load_tests/worker.py"
        imp = MagicMock()
        imp.source = "from config import Settings"
        worker.imports = [imp]

        local_config = MagicMock()
        local_config.filepath = "/project/load_tests/config.py"
        local_config.path = "/project/load_tests/config.py"
        local_config.imports = []

        distant_config = MagicMock()
        distant_config.filepath = "/project/backend/app/api/voice/config.py"
        distant_config.path = "/project/backend/app/api/voice/config.py"
        distant_config.imports = []

        codebase = MagicMock()
        codebase.files = [worker, local_config, distant_config]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Should resolve to local config, not distant
        local_importers = graph._import_index.get("/project/load_tests/config.py", [])
        distant_importers = graph._import_index.get("/project/backend/app/api/voice/config.py", [])
        assert "/project/load_tests/worker.py" in local_importers
        assert "/project/load_tests/worker.py" not in distant_importers

    def test_unambiguous_module_unaffected(self):
        """When only one file matches (e.g., 'database'), resolve as before."""
        graph = self._make_graph()

        main = MagicMock()
        main.filepath = "/project/app/main.py"
        main.path = "/project/app/main.py"
        imp = MagicMock()
        imp.source = "from database import get_db"
        main.imports = [imp]

        db_file = MagicMock()
        db_file.filepath = "/project/app/database.py"
        db_file.path = "/project/app/database.py"
        db_file.imports = []

        codebase = MagicMock()
        codebase.files = [main, db_file]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        assert "/project/app/main.py" in graph._import_index.get("/project/app/database.py", [])
```

### Step 2: Run tests to verify they fail

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_local_module_resolution.py -v`
Expected: FAIL — first test fails because `worker.py` resolves to distant config.

### Step 3: Implement local module preference in suffix matching

In `grafyx/graph/_indexes.py`, modify `_build_import_index` Phase 1 to build a suffix→**list** of paths instead of suffix→single path (line 528-532):

**Current:**
```python
                        parts = fp.split("/")
                        for i in range(len(parts)):
                            suffix = "/".join(parts[i:])
                            if suffix not in suffix_to_path:
                                suffix_to_path[suffix] = fp
```

**New:**
```python
                        parts = fp.split("/")
                        for i in range(len(parts)):
                            suffix = "/".join(parts[i:])
                            if suffix not in suffix_to_path:
                                suffix_to_path[suffix] = [fp]
                            else:
                                suffix_to_path[suffix].append(fp)
```

Then modify the Phase 2 resolution (lines 598-607) to pick the closest match when multiple candidates exist:

**Current:**
```python
                            target = None
                            for candidate in same_lang:
                                target = suffix_to_path.get(candidate)
                                if target:
                                    break
                            if target is None:
                                for candidate in cross_lang:
                                    target = suffix_to_path.get(candidate)
                                    if target:
                                        break
```

**New — add a helper inside `_build_import_index`, just before Phase 2 (after line 534):**

```python
            def _pick_closest(candidates: list[str], importer: str) -> str:
                """From multiple candidate files, pick the one closest to importer.

                Closeness = length of shared directory prefix. This ensures
                'from config import X' in load_tests/worker.py resolves to
                load_tests/config.py over backend/app/api/voice/config.py.
                """
                if len(candidates) == 1:
                    return candidates[0]
                imp_parts = importer.rsplit("/", 1)[0].split("/") if "/" in importer else []
                best = candidates[0]
                best_shared = -1
                for cand in candidates:
                    cand_parts = cand.rsplit("/", 1)[0].split("/") if "/" in cand else []
                    shared = 0
                    for a, b in zip(imp_parts, cand_parts):
                        if a == b:
                            shared += 1
                        else:
                            break
                    if shared > best_shared:
                        best_shared = shared
                        best = cand
                return best
```

Then replace the resolution block (lines 598-607):

```python
                            target = None
                            for candidate in same_lang:
                                matches = suffix_to_path.get(candidate)
                                if matches:
                                    target = _pick_closest(matches, fpath)
                                    break
                            if target is None:
                                for candidate in cross_lang:
                                    matches = suffix_to_path.get(candidate)
                                    if matches:
                                        target = _pick_closest(matches, fpath)
                                        break
```

### Step 4: Run tests to verify they pass

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_local_module_resolution.py -v`
Expected: PASS

### Step 5: Run full test suite for regressions

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/ -v`
Expected: ALL PASS. Existing `test_import_index.py` tests should still pass because they use unambiguous paths.

**IMPORTANT:** Check `test_import_index.py` — if any test directly inspects `suffix_to_path` as a dict of strings (not lists), update it to match the new list type. Scan for `suffix_to_path` usage in tests.

### Step 6: Commit

```bash
git add tests/test_local_module_resolution.py grafyx/graph/_indexes.py
git commit -m "fix: prefer same-directory modules in import resolution"
```

---

## Task 3: Improve Instance Method Call Tracking for Dead Code Detection

**Problem:** `self.execute()` and `service.send_message()` calls via `self.field` attribute chains are not tracked, causing false "unused" reports on methods like `ToolExecutor.execute` and `EddyService.send_message`.

**Root cause:** Pass 3 (`_augment_index_with_local_var_types`) only resolves typed locals/params within a single function. It misses calls through `self.field` where the field type is defined at the class level (e.g., `self.executor: ToolExecutor` as a class attribute, then `self.executor.execute()` in a method).

**Impact:** MEDIUM — false positives in `get_unused_symbols` for core methods called through class attributes.

**Files:**
- Modify: `grafyx/graph/_indexes.py:293-415` (add Pass 3b: class attribute type tracking)
- Test: `tests/test_class_attr_tracking.py` (new)

### Step 1: Write the failing tests

Create `tests/test_class_attr_tracking.py`:

```python
"""Tests for class attribute type tracking in caller index.

Covers:
- self.field.method() calls where field type is declared in __init__ or class body
- Should add trusted caller entries so methods aren't falsely flagged as unused
"""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph
from grafyx.utils import safe_get_attr


class TestClassAttributeTypeTracking:
    """Methods called via self.typed_field.method() should be tracked."""

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._lock = MagicMock()
        graph._caller_index = {}
        graph._class_method_names = {
            "ToolExecutor": {"execute", "validate"},
            "OrchestratorService": {"run_tools"},
        }
        graph._file_class_methods = {}
        graph._class_defined_in = {}
        graph._external_packages = set()
        graph.translate_path = lambda p: str(p) if p else ""
        graph._is_ignored_file_path = lambda p: False
        graph._codebases = {}
        return graph

    def test_self_field_method_call_tracked(self):
        """self.executor.execute() should create a caller for ToolExecutor.execute
        when __init__ has self.executor = ToolExecutor(...)."""
        graph = self._make_graph()

        # Simulate a class with __init__ that assigns self.executor = ToolExecutor(...)
        # and a method that calls self.executor.execute()
        init_method = MagicMock()
        init_method.name = "__init__"
        init_method.filepath = "/project/service.py"
        init_method.source = (
            "def __init__(self):\n"
            "    self.executor = ToolExecutor(config)\n"
        )

        run_method = MagicMock()
        run_method.name = "run_tools"
        run_method.filepath = "/project/service.py"
        run_method.source = (
            "def run_tools(self, tools):\n"
            "    result = self.executor.execute(tools)\n"
            "    return result\n"
        )
        run_method.function_calls = []

        cls = MagicMock()
        cls.name = "OrchestratorService"
        cls.filepath = "/project/service.py"
        cls.methods = [init_method, run_method]

        codebase = MagicMock()
        codebase.functions = []
        codebase.classes = [cls]
        graph._codebases = {"python": codebase}

        # Bind the method
        graph._augment_index_with_class_attr_types = (
            CodebaseGraph._augment_index_with_class_attr_types.__get__(graph)
        )
        graph._augment_index_with_class_attr_types()

        # execute should now have a caller entry from run_tools
        callers = graph._caller_index.get("execute", [])
        assert any(
            c["name"] == "run_tools" and c.get("_trusted")
            for c in callers
        ), f"Expected trusted caller from run_tools, got: {callers}"

    def test_typed_annotation_field(self):
        """self.executor: ToolExecutor in __init__ should also work."""
        graph = self._make_graph()

        init_method = MagicMock()
        init_method.name = "__init__"
        init_method.filepath = "/project/service.py"
        init_method.source = (
            "def __init__(self, executor: ToolExecutor):\n"
            "    self.executor: ToolExecutor = executor\n"
        )

        use_method = MagicMock()
        use_method.name = "process"
        use_method.filepath = "/project/service.py"
        use_method.source = (
            "def process(self):\n"
            "    self.executor.validate(data)\n"
        )
        use_method.function_calls = []

        cls = MagicMock()
        cls.name = "MyService"
        cls.filepath = "/project/service.py"
        cls.methods = [init_method, use_method]

        codebase = MagicMock()
        codebase.functions = []
        codebase.classes = [cls]
        graph._codebases = {"python": codebase}

        graph._class_method_names["MyService"] = {"process", "__init__"}
        graph._augment_index_with_class_attr_types = (
            CodebaseGraph._augment_index_with_class_attr_types.__get__(graph)
        )
        graph._augment_index_with_class_attr_types()

        callers = graph._caller_index.get("validate", [])
        assert any(
            c["name"] == "process" and c.get("_trusted")
            for c in callers
        ), f"Expected trusted caller from process, got: {callers}"
```

### Step 2: Run tests to verify they fail

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_class_attr_tracking.py -v`
Expected: FAIL — `AttributeError: _augment_index_with_class_attr_types` doesn't exist yet.

### Step 3: Implement `_augment_index_with_class_attr_types` (Pass 3b)

Add the new method to `IndexBuilderMixin` in `grafyx/graph/_indexes.py`, after `_augment_index_with_local_var_types` (after line ~415):

```python
    # --- Pass 3b: Class Attribute Type Resolution ---

    def _augment_index_with_class_attr_types(self) -> None:
        """Scan __init__ methods for self.field = ClassName(...) assignments,
        then link self.field.method() calls in sibling methods (Pass 3b).

        This extends Pass 3 to handle cross-method attribute access patterns:
            class Service:
                def __init__(self):
                    self.executor = ToolExecutor(config)   # <- type captured here
                def run(self):
                    self.executor.execute(tools)           # <- caller linked here

        Must be called AFTER _build_caller_index (Pass 1-3) since it reads
        _class_method_names and writes to _caller_index.
        """
        all_class_names: set[str] = set(self._class_method_names.keys())
        if not all_class_names:
            return

        # Regex: self.field = ClassName( — captures field name and class name
        self_assign_re = re.compile(
            r'self\.(\w+)\s*(?::\s*\w[\w\[\], |]*\s*)?=\s*([A-Z]\w*)\s*\('
        )
        # Regex: self.field: ClassName = — typed annotation with assignment
        self_typed_re = re.compile(
            r'self\.(\w+)\s*:\s*([A-Z]\w*)\s*='
        )
        # Regex: self.field.method( — captures field name and method name
        self_field_call_re = re.compile(r'self\.(\w+)\.(\w+)\s*\(')

        additions = 0
        for _lang, codebase in self._codebases.items():
            try:
                for cls in codebase.classes:
                    cls_file = self.translate_path(
                        str(safe_get_attr(cls, "filepath", ""))
                    )
                    if self._is_ignored_file_path(cls_file):
                        continue
                    cls_name = safe_get_attr(cls, "name", "")
                    methods = safe_get_attr(cls, "methods", [])
                    if not methods:
                        continue

                    # Phase A: scan all methods for self.field type assignments
                    field_types: dict[str, str] = {}
                    for method in methods:
                        source = safe_str(safe_get_attr(method, "source", ""))
                        if not source:
                            continue
                        for m in self_assign_re.finditer(source):
                            field_name, type_name = m.group(1), m.group(2)
                            if type_name in all_class_names:
                                field_types[field_name] = type_name
                        for m in self_typed_re.finditer(source):
                            field_name, type_name = m.group(1), m.group(2)
                            if type_name in all_class_names:
                                field_types[field_name] = type_name

                    if not field_types:
                        continue

                    # Phase B: scan all methods for self.field.method() calls
                    for method in methods:
                        caller_name = safe_get_attr(method, "name", "")
                        if not caller_name:
                            continue
                        source = safe_str(safe_get_attr(method, "source", ""))
                        if not source:
                            continue
                        caller_file = self.translate_path(
                            str(safe_get_attr(method, "filepath", ""))
                        )

                        for m in self_field_call_re.finditer(source):
                            field_name, method_name = m.group(1), m.group(2)
                            target_cls = field_types.get(field_name)
                            if not target_cls:
                                continue
                            if method_name not in self._class_method_names.get(target_cls, set()):
                                continue
                            if method_name not in self._caller_index:
                                self._caller_index[method_name] = []
                            entry = {
                                "name": caller_name,
                                "file": caller_file,
                                "_trusted": True,
                            }
                            if cls_name:
                                entry["class"] = cls_name
                            if not any(
                                e["name"] == caller_name and e["file"] == caller_file
                                for e in self._caller_index[method_name]
                            ):
                                self._caller_index[method_name].append(entry)
                                additions += 1
            except Exception as e:
                logger.debug("Error scanning class attr types for %s: %s", _lang, e)
        if additions:
            logger.debug("Class attr type scan added %d caller entries", additions)
```

Then call it from `_build_caller_index` at the end (line ~157), after the existing Pass 3:

```python
        # Third pass: resolve method calls through typed locals/params (FastAPI DI, local instances)
        self._augment_index_with_local_var_types()
        # Pass 3b: resolve method calls through self.field class attribute types
        self._augment_index_with_class_attr_types()
```

### Step 4: Run tests to verify they pass

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_class_attr_tracking.py -v`
Expected: PASS

### Step 5: Run full test suite

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/ -v`
Expected: ALL PASS

### Step 6: Commit

```bash
git add tests/test_class_attr_tracking.py grafyx/graph/_indexes.py
git commit -m "feat: track self.field.method() calls via class attribute types (Pass 3b)"
```

---

## Task 4: Add Test File Penalty to Search Scoring

**Problem:** `find_related_code` for "voice chat WebRTC Daily transport" ranked `load_tests/webrtc_client.py` above production voice code. Test files are scored identically to production files.

**Impact:** MEDIUM — search results are less useful when test files dominate.

**Files:**
- Modify: `grafyx/search/_scoring.py:347-374` (path scoring section)
- Test: `tests/test_search_test_penalty.py` (new)

### Step 1: Write the failing tests

Create `tests/test_search_test_penalty.py`:

```python
"""Tests for test file scoring penalty in search.

Production files should score higher than test files with equivalent
name/docstring matches, because users searching for functionality
typically want the implementation, not the test.
"""

from grafyx.search._scoring import ScoringMixin


class TestTestFilePenalty:
    """Test files should receive a scoring penalty vs production files."""

    def setup_method(self):
        self.scorer = ScoringMixin()

    def test_production_file_scores_higher(self):
        """Same function name in test vs production file: production wins."""
        prod_score = self.scorer._score_match(
            query_tokens=["webrtc", "client"],
            query_lower="webrtc client",
            name="WebRTCClient",
            docstring="WebRTC client for voice communication",
            file_path="backend/app/voice/webrtc_client.py",
        )
        test_score = self.scorer._score_match(
            query_tokens=["webrtc", "client"],
            query_lower="webrtc client",
            name="WebRTCClient",
            docstring="WebRTC client for voice communication",
            file_path="load_tests/webrtc_client.py",
        )
        assert prod_score > test_score, (
            f"Production score ({prod_score:.3f}) should be > test score ({test_score:.3f})"
        )

    def test_test_directory_penalized(self):
        """Files in tests/ directory should get penalized."""
        prod_score = self.scorer._score_match(
            query_tokens=["auth", "handler"],
            query_lower="auth handler",
            name="authenticate",
            docstring="Handle authentication",
            file_path="src/auth/handler.py",
        )
        test_score = self.scorer._score_match(
            query_tokens=["auth", "handler"],
            query_lower="auth handler",
            name="authenticate",
            docstring="Handle authentication",
            file_path="tests/test_auth/test_handler.py",
        )
        assert prod_score > test_score

    def test_non_test_files_unpenalized(self):
        """Non-test files in various directories should not be penalized."""
        score1 = self.scorer._score_match(
            query_tokens=["process", "data"],
            query_lower="process data",
            name="process_data",
            docstring="Process incoming data",
            file_path="src/processing/data.py",
        )
        score2 = self.scorer._score_match(
            query_tokens=["process", "data"],
            query_lower="process data",
            name="process_data",
            docstring="Process incoming data",
            file_path="lib/processing/data.py",
        )
        # Both non-test, should score identically
        assert abs(score1 - score2) < 0.01
```

### Step 2: Run tests to verify they fail

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_search_test_penalty.py -v`
Expected: FAIL — scores are equal for test vs production.

### Step 3: Add test file penalty to `_score_match`

In `grafyx/search/_scoring.py`, add a test file penalty in the **Multiplicative Adjustments** section (after the concept coverage bonus, ~line 478, before the soft cap):

```python
        # --- Test File Penalty ---
        # Production code should rank above test code for discovery queries.
        # Files in test/tests/load_tests directories or with test_ prefix are
        # penalized 0.85x to push them below equivalent production matches.
        if file_path:
            path_norm = file_path.replace("\\", "/").lower()
            path_parts = path_norm.split("/")
            _test_dirs = {"test", "tests", "test_", "load_tests", "__tests__",
                          "spec", "specs", "__mocks__"}
            if any(p in _test_dirs or p.startswith("test_") for p in path_parts):
                final *= 0.85
```

### Step 4: Run tests to verify they pass

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_search_test_penalty.py -v`
Expected: PASS

### Step 5: Run full test suite

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/ -v`
Expected: ALL PASS. Existing search tests use simple paths like "main.py", "utils.py" — no test directories.

### Step 6: Commit

```bash
git add tests/test_search_test_penalty.py grafyx/search/_scoring.py
git commit -m "feat: penalize test files 0.85x in search ranking"
```

---

## Task 5: Improve Re-Export Tracking for `__init__.py`

**Problem:** When `package/__init__.py` re-exports symbols (`from .impl import ServiceClass`), files importing `from package import ServiceClass` aren't linked to `ServiceClass` as callers. This causes false "unused" reports.

**Root cause:** `_build_import_index` line 612-615 skips `__init__.py` → sub-package imports entirely, preventing the re-export relationship from being indexed. The `__init__.py`'s own exports are never propagated to the import index.

**Impact:** LOW — only affects packages that use `__init__.py` re-export patterns and where the re-exported symbols have ambiguous names.

**Files:**
- Modify: `grafyx/graph/_indexes.py:609-615` (skip logic) and `~636` (post-processing)
- Test: `tests/test_reexport_tracking.py` (new)

### Step 1: Write the failing tests

Create `tests/test_reexport_tracking.py`:

```python
"""Tests for __init__.py re-export tracking.

When package/__init__.py re-exports a symbol from a submodule,
files importing from the package should be linked to the original
defining file in the import index.
"""

from unittest.mock import MagicMock
from grafyx.graph import CodebaseGraph


class TestReExportTracking:

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._project_path = "/project"
        graph.original_path = "/project"
        graph._lock = MagicMock()
        graph._external_packages = set()
        graph.translate_path = lambda p: str(p) if p else ""
        graph._build_import_index = CodebaseGraph._build_import_index.__get__(graph)
        graph._extract_module_from_import = CodebaseGraph._extract_module_from_import
        graph._extract_symbol_names_from_import = CodebaseGraph._extract_symbol_names_from_import
        graph._is_ignored_file_path = lambda p: False
        return graph

    def test_reexport_links_consumer_to_init(self):
        """Files importing from package/__init__.py should appear in
        _import_index for the __init__.py file itself."""
        graph = self._make_graph()

        # package/__init__.py re-exports from .impl
        init_file = MagicMock()
        init_file.filepath = "/project/package/__init__.py"
        init_file.path = "/project/package/__init__.py"
        init_imp = MagicMock()
        init_imp.source = "from .impl import ServiceClass"
        init_file.imports = [init_imp]

        impl_file = MagicMock()
        impl_file.filepath = "/project/package/impl.py"
        impl_file.path = "/project/package/impl.py"
        impl_file.imports = []

        # Consumer imports from the package
        consumer = MagicMock()
        consumer.filepath = "/project/app/main.py"
        consumer.path = "/project/app/main.py"
        consumer_imp = MagicMock()
        consumer_imp.source = "from package import ServiceClass"
        consumer.imports = [consumer_imp]

        codebase = MagicMock()
        codebase.files = [init_file, impl_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Consumer should import the __init__.py (which is how package imports resolve)
        init_importers = graph._import_index.get("/project/package/__init__.py", [])
        assert "/project/app/main.py" in init_importers, (
            f"Consumer should be in init importers, got: {init_importers}"
        )

    def test_init_self_import_still_skipped(self):
        """__init__.py importing its own submodule should still be skipped
        in the import index (prevents self-loops)."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/package/__init__.py"
        init_file.path = "/project/package/__init__.py"
        init_imp = MagicMock()
        init_imp.source = "from package.impl import ServiceClass"
        init_file.imports = [init_imp]

        impl_file = MagicMock()
        impl_file.filepath = "/project/package/impl.py"
        impl_file.path = "/project/package/impl.py"
        impl_file.imports = []

        codebase = MagicMock()
        codebase.files = [init_file, impl_file]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # __init__.py should NOT appear as an importer of impl.py
        impl_importers = graph._import_index.get("/project/package/impl.py", [])
        assert "/project/package/__init__.py" not in impl_importers
```

### Step 2: Run tests to verify they fail

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_reexport_tracking.py -v`
Expected: First test likely PASSES already (since `from package import X` resolves to `package/__init__.py` via suffix matching). Second test should PASS (the skip logic exists). **If both pass, this task is already working and can be skipped.** If the first test fails, proceed with the fix.

### Step 3: Add re-export propagation (only if needed)

If the `from .impl import` relative import in `__init__.py` is being skipped (line 708-710 skips relative imports), add a post-processing step after the import index is built (after line 638):

```python
            # Propagate __init__.py re-exports: if __init__.py imports from
            # a sibling file (relative import), consumers of the __init__.py
            # should also be considered importers of the sibling file.
            # This fixes Level 3 caller filtering for re-exported symbols.
            for target_path, importers in list(index.items()):
                if not target_path.endswith("/__init__.py"):
                    continue
                # Find what the __init__.py itself imports (from forward index)
                init_targets = forward.get(target_path, [])
                init_dir = target_path[: -len("/__init__.py")]
                for sub_target in init_targets:
                    if sub_target.startswith(init_dir + "/"):
                        # Propagate: all importers of __init__.py are also
                        # indirect importers of the sub-target
                        if sub_target not in index:
                            index[sub_target] = []
                        for imp in importers:
                            if imp not in index[sub_target]:
                                index[sub_target].append(imp)
```

**Note:** Since `__init__.py` relative imports are currently skipped entirely (line 708-710), the `__init__.py` won't have forward index entries for its re-exports. The actual fix requires ALSO handling relative imports in `_extract_module_from_import`. This is a bigger change — **consider deferring to a separate task if scope grows too large**.

**Minimal fix alternative:** Instead of resolving relative imports, modify `_get_class_importer_files` (in `_callers.py`) to also check the `__init__.py`'s OWN importers transitively. The existing code at lines 190-198 already does this partially — verify it works and add a test.

### Step 4: Run tests and iterate

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_reexport_tracking.py -v`

### Step 5: Run full test suite

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/ -v`

### Step 6: Commit

```bash
git add tests/test_reexport_tracking.py grafyx/graph/_indexes.py
git commit -m "fix: propagate __init__.py re-exports to import index"
```

---

## Task 6: Add `_build_imported_names` Alias Support

**Problem:** `_build_imported_names()` (used by dead code detection, line 957+) extracts symbol names from imports but currently only gets pre-alias names. If `from X import foo as bar` is used, only `"foo"` appears in the set, so `"bar"` could be falsely flagged as unused if it's re-exported under that alias.

**Impact:** LOW — this is already partially addressed by Task 1's `_extract_symbol_names_from_import` fix. But `_build_imported_names` has its OWN parsing logic (doesn't call `_extract_symbol_names_from_import`). It needs the same alias treatment.

**Files:**
- Modify: `grafyx/graph/_indexes.py:957+` (`_build_imported_names`)
- Test: Covered by `tests/test_alias_tracking.py::TestImportedNamesIncludesAliases` (from Task 1)

### Step 1: Read and verify `_build_imported_names` parsing

Read `grafyx/graph/_indexes.py` starting at line 957 to see if it has its own parsing or calls `_extract_symbol_names_from_import`.

### Step 2: If `_build_imported_names` has its own parsing, update it

Look for `split(" as ")` patterns that discard aliases. Update them to keep both original and alias, similar to Task 1's fix.

The method likely iterates all imports and extracts tokens. Find lines like:
```python
clean = token.strip().split(" as ")[0].strip()
```
And change to:
```python
parts = token.strip().split(" as ")
clean = parts[0].strip()
if clean and clean.isidentifier() and clean not in NOISE:
    names.add(clean)
if len(parts) > 1:
    alias = parts[1].strip()
    if alias and alias.isidentifier() and alias not in NOISE:
        names.add(alias)
```

### Step 3: Run the Task 1 tests to verify

Run: `cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_alias_tracking.py::TestImportedNamesIncludesAliases -v`
Expected: PASS

### Step 4: Commit (combined with Task 1 if done in same session)

```bash
git add grafyx/graph/_indexes.py
git commit -m "fix: _build_imported_names also tracks import aliases"
```

---

## Execution Order & Dependencies

```
Task 1 (alias tracking) ──→ Task 6 (imported_names aliases) [same file, same concept]
Task 2 (local module)   ──→ independent
Task 3 (class attrs)    ──→ independent
Task 4 (search penalty) ──→ independent
Task 5 (re-exports)     ──→ depends on Task 2 if suffix_to_path changed to lists
```

**Recommended order:** Tasks 1+6 → Task 2 → Task 3 → Task 4 → Task 5

**Parallelizable:** Tasks 3 and 4 can run in parallel (different files). Tasks 1+6 and 2 touch the same file (`_indexes.py`) so should be sequential.

---

## Verification Checklist

After all tasks, run:

1. `python -m pytest tests/ -v` — all tests pass
2. `python -m pytest tests/ -v --tb=short` — no warnings
3. Manual smoke test on a real project (if available): confirm `get_unused_symbols` has fewer false positives, `find_related_code` ranks production > test files, `get_module_dependencies` shows correct edges for projects with same-name config modules
