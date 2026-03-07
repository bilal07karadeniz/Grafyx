# ML Accuracy Boost Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Push Grafyx tool accuracy from 68/100 to 91/100 by replacing all hardcoded heuristics with 6 ML models (~28M params, ~57MB on disk).

**Architecture:** Three phases — (1) data quality fixes to graph engine, (2) train and integrate 4 MLPs for classification tasks, (3) train and integrate Mamba bi-encoder + cross-encoder for semantic code search. All inference is numpy-only (with optional cupy GPU). Every model has a graceful fallback.

**Tech Stack:** Python 3.12+, numpy (inference), PyTorch (training), sentencepiece (tokenizer training), graph-sitter, FastMCP

**Design Doc:** `docs/plans/2026-03-07-ml-accuracy-boost-design.md`

---

## Phase 1: Data Quality Fixes

### Task 1: `__init__.py` Re-export Resolution in Import Index

**Files:**
- Modify: `grafyx/graph/_indexes.py:827-1050` (the `_build_import_index` method)
- Test: `tests/test_init_reexport_resolution.py` (create)

The import index currently can't trace imports through `__init__.py` re-export patterns (lazy `__getattr__`, `__all__`, explicit re-exports). When code does `from app.services.auth import func`, and `auth/__init__.py` uses `__getattr__` for lazy loading, the import resolves to `__init__.py` instead of the actual source module.

**Step 1: Write failing tests**

```python
# tests/test_init_reexport_resolution.py
"""Tests for __init__.py re-export resolution in import index."""
import textwrap
import pytest
from grafyx.graph._indexes import IndexBuilderMixin


class FakeFunc:
    def __init__(self, name, file, source=""):
        self.name = name
        self.file = file
        self.source_code = source
        self.function_calls = []


class FakeFile:
    def __init__(self, path, source="", imports=None):
        self.path = path
        self.file_path = path
        self.source_code = source
        self.imports = imports or []


class FakeImport:
    def __init__(self, module, names=None):
        self.module = module
        self.imported_names = names or []


class TestInitReexportResolution:
    """Test that import index follows __init__.py re-exports to actual modules."""

    def test_explicit_reexport_resolved(self):
        """from pkg import X where pkg/__init__.py has 'from .module import X'
        should resolve to pkg/module.py, not pkg/__init__.py."""
        # Setup: consumer.py imports from pkg, __init__.py re-exports from module.py
        # After building import index, consumer.py should appear as importer of module.py
        pass  # Implement with actual mixin test

    def test_getattr_lazy_import_resolved(self):
        """from pkg import X where pkg/__init__.py uses __getattr__ lazy loading
        should resolve to the actual source module."""
        pass

    def test_all_export_resolved(self):
        """from pkg import X where pkg/__init__.py has __all__ = ['X']
        and 'from .module import X' should resolve to module.py."""
        pass

    def test_direct_assignment_resolved(self):
        """from pkg import X where pkg/__init__.py has X = module.X
        should resolve to the module defining X."""
        pass

    def test_non_reexport_stays_at_init(self):
        """Functions defined directly in __init__.py (not re-exported)
        should resolve to __init__.py itself."""
        pass
```

**Step 2: Run tests to verify they fail**

Run: `source ~/grafyx-venv/bin/activate && cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_init_reexport_resolution.py -v`

**Step 3: Implement re-export resolution**

Add a new method `_resolve_init_reexports()` to `IndexBuilderMixin` in `grafyx/graph/_indexes.py`. This method should:

1. After `_build_import_index()` completes, scan all `__init__.py` files in the project
2. For each `__init__.py`, detect re-export patterns:
   - Parse `from .submodule import name` statements → map name to submodule
   - Parse `__all__ = [...]` declarations
   - Parse `__getattr__` body for lazy import dicts (regex: `["'](\w+)["']\s*:\s*["'](\.\w+)["']`)
   - Parse direct assignments `X = submodule.X`
3. Build a mapping: `(init_file, exported_name) → actual_source_file`
4. When a file imports `from pkg import X` and it resolves to `pkg/__init__.py`, check the mapping to find the actual source module
5. Update `_import_index` and `_file_symbol_imports` to point to the actual source

Call `_resolve_init_reexports()` at the end of `_build_import_index()`.

**Step 4: Run tests to verify they pass**

Run: `source ~/grafyx-venv/bin/activate && cd "/mnt/c/Kişisel Projelerim/Grafyx" && python -m pytest tests/test_init_reexport_resolution.py -v`

**Step 5: Run existing tests to check for regressions**

Run: `python -m pytest tests/ -v --tb=short`

**Step 6: Commit**

```bash
git add grafyx/graph/_indexes.py tests/test_init_reexport_resolution.py
git commit -m "feat: resolve __init__.py re-exports in import index

Follow lazy __getattr__, __all__, explicit re-exports, and direct
assignments through __init__.py to actual source modules. Fixes
broken reverse dependency tracking and imported_by completeness."
```

---

### Task 2: Extract Receiver Tokens from Call Sites

**Files:**
- Modify: `grafyx/graph/_indexes.py:760-821` (the `_index_calls_from` method)
- Modify: `grafyx/graph/_callers.py:56-167` (update caller entry format)
- Test: `tests/test_receiver_extraction.py` (create)

The caller index already stores `_receivers` (set of receiver expressions) at line 813-815. However, we need to ensure:
1. The receiver token is consistently extracted for ALL call sites (not just some)
2. A `has_dot_syntax` boolean is stored per entry
3. The data is accessible by downstream ML models

**Step 1: Write failing tests**

```python
# tests/test_receiver_extraction.py
"""Tests for receiver token extraction in caller index."""
import textwrap


class TestReceiverExtraction:
    """Test that call sites store receiver info for ML disambiguation."""

    def test_dot_syntax_call_stores_receiver(self):
        """db.refresh(obj) should store receiver='db', has_dot_syntax=True."""
        pass

    def test_standalone_call_no_receiver(self):
        """process_data() should store receiver=None, has_dot_syntax=False."""
        pass

    def test_self_method_call_stores_self(self):
        """self.validate() should store receiver='self', has_dot_syntax=True."""
        pass

    def test_chained_call_stores_immediate_receiver(self):
        """self.db.query() should store receiver='db' (immediate), has_dot_syntax=True."""
        pass

    def test_receiver_available_in_get_callers(self):
        """get_callers() results should include receiver_token and has_dot_syntax fields."""
        pass
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement receiver extraction enhancement**

In `grafyx/graph/_indexes.py`, method `_index_calls_from()` (line 760):
- The current code already extracts `call_receivers` at lines 790-815
- Ensure every caller entry dict has:
  - `"_receivers"`: set of receiver expressions (already exists)
  - `"_has_dot_syntax"`: bool — True if any receiver expression contains a dot or is non-empty
- In `grafyx/graph/_callers.py`, method `get_callers()` (line 56):
  - Include `receiver_token` and `has_dot_syntax` in the returned caller dicts
  - Extract the immediate receiver (last token before the dot) from `_receivers`

**Step 4: Run tests, verify pass**

**Step 5: Run existing tests for regressions**

**Step 6: Commit**

```bash
git add grafyx/graph/_indexes.py grafyx/graph/_callers.py tests/test_receiver_extraction.py
git commit -m "feat: expose receiver tokens and dot-syntax flag in caller entries

Ensure all caller index entries include receiver_token and
has_dot_syntax fields for downstream ML caller disambiguation."
```

---

### Task 3: Remove All Hardcoded Caps

**Files:**
- Modify: `grafyx/server/_tools_graph.py:344` (dep graph `[:20]` cap)
- Modify: `grafyx/server/_tools_introspection.py:482` (class context `[:20]` cap)
- Modify: `grafyx/server/_tools_introspection.py:333` (imported_by `[:30]` cap)
- Modify: `grafyx/server/_hints.py:38` (MAX_HINTS = 3)
- Test: `tests/test_cap_removal.py` (create)

**Step 1: Write failing tests**

```python
# tests/test_cap_removal.py
"""Tests that hardcoded caps are removed or made dynamic."""

class TestDynamicLimits:

    def test_dep_graph_returns_all_dependents(self):
        """get_dependency_graph should not truncate at 20 files."""
        pass

    def test_class_context_returns_all_usages(self):
        """get_class_context cross_file_usages not truncated at 20."""
        pass

    def test_imported_by_returns_all(self):
        """get_file_context imported_by not truncated at 30."""
        pass

    def test_hints_count_dynamic(self):
        """Hints count should not be hardcoded at 3."""
        pass
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement dynamic limits**

- `_tools_graph.py:344`: Change `[:20]` to return all results. The tool's `max_results` parameter (if it has one) or a sensible proportional limit should control truncation.
- `_tools_introspection.py:482`: Change `[:20]` to return all. The output format already handles large lists.
- `_tools_introspection.py:333`: Change `[:30]` to return all importers.
- `_hints.py:38`: Change `MAX_HINTS = 3` to a dynamic value. For now, set to 5 as default but make it configurable. Later M4 will rank hints by importance.

**Step 4: Run all tests**

**Step 5: Commit**

```bash
git add grafyx/server/_tools_graph.py grafyx/server/_tools_introspection.py grafyx/server/_hints.py tests/test_cap_removal.py
git commit -m "feat: remove hardcoded caps from dep graph, class context, hints

Replace [:20], [:30], MAX_HINTS=3 with dynamic limits. Downstream
ML rankers will control what to show instead of arbitrary slicing."
```

---

## Phase 2: MLP Models (M1-M4)

### Task 4: Shared ML Infrastructure

**Files:**
- Create: `grafyx/ml_inference.py`
- Create: `ml/data_common/repos.json`
- Create: `ml/data_common/clone_repos.py`
- Create: `ml/data_common/extract_symbols.py`

**Step 1: Create `grafyx/ml_inference.py`**

Shared model loading, lazy init, and GPU detection for all models.

```python
"""Shared ML model infrastructure. Lazy loading, GPU detection, numpy inference."""
import os
import numpy as np
from pathlib import Path
from typing import Any

_MODEL_DIR = Path(__file__).parent / "search" / "model"


def _get_backend():
    """Auto-detect numpy (CPU) or cupy (GPU)."""
    try:
        import cupy as cp
        return cp
    except ImportError:
        return np


xp = _get_backend()


class MLPModel:
    """Generic MLP with numpy/cupy inference. Supports any layer count."""

    def __init__(self, weights_path: str | Path):
        self._weights_path = Path(weights_path)
        self._layers: list[tuple[Any, Any]] = []  # [(weight, bias), ...]
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not self._weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self._weights_path}")
        data = np.load(self._weights_path)
        i = 0
        while f"W{i}" in data:
            W = xp.asarray(data[f"W{i}"])
            b = xp.asarray(data[f"b{i}"])
            self._layers.append((W, b))
            i += 1
        self._loaded = True

    def predict(self, features: np.ndarray) -> float:
        """Forward pass. Returns sigmoid output."""
        self._ensure_loaded()
        x = xp.asarray(features, dtype=xp.float32)
        for i, (W, b) in enumerate(self._layers):
            x = x @ W + b
            if i < len(self._layers) - 1:  # ReLU for hidden layers
                x = xp.maximum(x, 0)
        # Sigmoid for output
        logit = float(x.item() if hasattr(x, 'item') else x)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))

    @property
    def is_available(self) -> bool:
        return self._weights_path.exists()


# Lazy singletons
_models: dict[str, MLPModel] = {}


def get_model(name: str) -> MLPModel | None:
    """Get a named model, or None if weights don't exist."""
    if name not in _models:
        path = _MODEL_DIR / f"{name}_weights.npz"
        model = MLPModel(path)
        _models[name] = model
    m = _models[name]
    return m if m.is_available else None
```

**Step 2: Create `ml/data_common/repos.json`**

List of 50+ OSS repos with licenses for training data generation:

```json
{
  "python": [
    {"repo": "tiangolo/fastapi", "license": "MIT"},
    {"repo": "pallets/flask", "license": "BSD-3"},
    {"repo": "django/django", "license": "BSD-3"},
    {"repo": "encode/starlette", "license": "BSD-3"},
    {"repo": "langchain-ai/langchain", "license": "MIT"},
    {"repo": "run-llama/llama_index", "license": "MIT"},
    {"repo": "sqlalchemy/sqlalchemy", "license": "MIT"},
    {"repo": "pydantic/pydantic", "license": "MIT"},
    {"repo": "celery/celery", "license": "BSD-3"},
    {"repo": "encode/httpx", "license": "BSD-3"},
    {"repo": "pallets/click", "license": "BSD-3"},
    {"repo": "tiangolo/typer", "license": "MIT"},
    {"repo": "Textualize/rich", "license": "MIT"},
    {"repo": "Textualize/textual", "license": "MIT"},
    {"repo": "psf/requests", "license": "Apache-2.0"},
    {"repo": "aio-libs/aiohttp", "license": "Apache-2.0"},
    {"repo": "python-attrs/attrs", "license": "MIT"},
    {"repo": "marshmallow-code/marshmallow", "license": "MIT"},
    {"repo": "tortoise/tortoise-orm", "license": "Apache-2.0"},
    {"repo": "tiangolo/sqlmodel", "license": "MIT"},
    {"repo": "Dramatiq/dramatiq", "license": "LGPL-3.0"},
    {"repo": "rq/rq", "license": "BSD-2"},
    {"repo": "pytest-dev/pytest", "license": "MIT"},
    {"repo": "HypothesisWorks/hypothesis", "license": "MPL-2.0"},
    {"repo": "sanic-org/sanic", "license": "MIT"}
  ],
  "typescript": [
    {"repo": "vercel/next.js", "license": "MIT"},
    {"repo": "facebook/react", "license": "MIT"},
    {"repo": "vuejs/core", "license": "MIT"},
    {"repo": "sveltejs/svelte", "license": "MIT"},
    {"repo": "trpc/trpc", "license": "MIT"},
    {"repo": "prisma/prisma", "license": "Apache-2.0"}
  ]
}
```

**Step 3: Create `ml/data_common/clone_repos.py`**

Script to clone all repos into a local cache directory.

```python
"""Clone OSS repos for training data generation."""
import json
import subprocess
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / ".repo_cache"
REPOS_FILE = Path(__file__).parent / "repos.json"


def clone_all(max_depth: int = 1):
    """Shallow clone all repos."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    repos = json.loads(REPOS_FILE.read_text())
    for lang, repo_list in repos.items():
        for entry in repo_list:
            repo = entry["repo"]
            name = repo.split("/")[-1]
            dest = CACHE_DIR / lang / name
            if dest.exists():
                print(f"  Skip {repo} (exists)")
                continue
            print(f"  Clone {repo}...")
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", str(max_depth),
                 f"https://github.com/{repo}.git", str(dest)],
                capture_output=True,
            )


if __name__ == "__main__":
    clone_all()
```

**Step 4: Create `ml/data_common/extract_symbols.py`**

Script to extract functions, classes, imports, and source from all cloned repos using graph-sitter.

```python
"""Extract symbols from cloned repos for training data generation."""
import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).parent.parent / ".repo_cache"


def extract_from_repo(repo_path: Path) -> list[dict[str, Any]]:
    """Extract all symbols from a repo using graph-sitter."""
    from graph_sitter import CodebaseGraph
    try:
        graph = CodebaseGraph(str(repo_path))
        symbols = []
        for lang, funcs in graph.functions.items():
            for func in funcs:
                symbols.append({
                    "type": "function",
                    "name": getattr(func, "name", ""),
                    "file": getattr(func, "file_path", ""),
                    "docstring": getattr(func, "docstring", "") or "",
                    "source": getattr(func, "source_code", "") or "",
                    "params": [getattr(p, "name", "") for p in getattr(func, "parameters", [])],
                    "decorators": [str(d) for d in getattr(func, "decorators", [])],
                    "class_name": getattr(func, "class_name", None),
                    "calls": [getattr(c, "name", "") for c in getattr(func, "function_calls", [])],
                    "language": lang,
                })
        for lang, classes in graph.classes.items():
            for cls in classes:
                symbols.append({
                    "type": "class",
                    "name": getattr(cls, "name", ""),
                    "file": getattr(cls, "file_path", ""),
                    "docstring": getattr(cls, "docstring", "") or "",
                    "source": getattr(cls, "source_code", "") or "",
                    "methods": [getattr(m, "name", "") for m in getattr(cls, "methods", [])],
                    "base_classes": [str(b) for b in getattr(cls, "base_classes", [])],
                    "language": lang,
                })
        return symbols
    except Exception as e:
        print(f"  Error extracting {repo_path}: {e}")
        return []


def extract_all(output_path: Path | None = None) -> list[dict]:
    """Extract symbols from all cached repos."""
    all_symbols = []
    for lang_dir in sorted(CACHE_DIR.iterdir()):
        if not lang_dir.is_dir():
            continue
        for repo_dir in sorted(lang_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            print(f"  Extracting {lang_dir.name}/{repo_dir.name}...")
            symbols = extract_from_repo(repo_dir)
            for s in symbols:
                s["repo"] = f"{lang_dir.name}/{repo_dir.name}"
            all_symbols.extend(symbols)
    if output_path:
        output_path.write_text(json.dumps(all_symbols, indent=2))
    print(f"  Total: {len(all_symbols)} symbols from {len(list(CACHE_DIR.rglob('*')))} repos")
    return all_symbols


if __name__ == "__main__":
    extract_all(Path(__file__).parent.parent / "all_symbols.json")
```

**Step 5: Commit**

```bash
git add grafyx/ml_inference.py ml/data_common/
git commit -m "feat: add shared ML inference module and training data infrastructure

MLPModel class with numpy/cupy inference, lazy model loading,
GPU auto-detection. Training infra: repo list, clone script,
symbol extraction pipeline."
```

---

### Task 5: M1 — Relevance Ranker v2 (Data + Training)

**Files:**
- Create: `ml/relevance_ranker_v2/features.py` (42-feature extraction)
- Create: `ml/relevance_ranker_v2/generate_data.py` (500K examples)
- Create: `ml/relevance_ranker_v2/train.py`
- Create: `ml/relevance_ranker_v2/stress_test.py`

**Step 1: Create feature extraction (42 features)**

`ml/relevance_ranker_v2/features.py` — extends existing 33 features with 9 new ones:

```python
"""42-feature extraction for Relevance Ranker v2."""
import re
import numpy as np
from collections import Counter

FEATURE_COUNT = 42


def extract_features(
    query_tokens: list[str],
    query_lower: str,
    name: str,
    docstring: str,
    file_path: str,
    token_weights: dict[str, float] | None = None,
    source_tokens: set[str] | None = None,
    # New v2 params:
    is_dunder: bool = False,
    is_init_file: bool = False,
    is_method: bool = False,
    is_class: bool = False,
    receiver_call_ratio: float = 0.0,
    source_token_entropy: float = 0.0,
    source_unique_token_ratio: float = 0.0,
    embedding_similarity: float = 0.0,
    caller_count_normalized: float = 0.0,
) -> np.ndarray:
    """Extract 42 features for a (query, symbol) pair."""
    vec = np.zeros(FEATURE_COUNT, dtype=np.float32)

    # Features 0-32: Same as v1 (copy logic from grafyx/search/_relevance.py:69-214)
    # ... (reuse existing _extract_features logic)

    # Features 33-41: New v2 features
    vec[33] = float(is_dunder)
    vec[34] = float(is_init_file)
    vec[35] = float(is_method)
    vec[36] = float(is_class)
    vec[37] = receiver_call_ratio
    vec[38] = source_token_entropy
    vec[39] = source_unique_token_ratio
    vec[40] = embedding_similarity
    vec[41] = caller_count_normalized

    return vec
```

**Step 2: Create data generation script**

`ml/relevance_ranker_v2/generate_data.py`:
- Load existing 200K training data from `ml/relevance_ranker/data/`
- Recompute with 42 features (add v2 features to each example)
- Generate 300K new examples:
  - 30K `__getattr__` hard negatives (dunder + random query → label 0)
  - 20K dunder method negatives
  - 20K `__init__.py` function negatives
  - 30K high-source-entropy negatives
  - 50K exact-match positives (query = function name → label 1)
  - 150K cross-project diversity (from cloned repos)
- Split: 80% train / 10% val / 10% test
- Output: `ml/relevance_ranker_v2/data/{train,val,test}.jsonl`

**Step 3: Create training script**

`ml/relevance_ranker_v2/train.py`:
- Architecture: 42 → 128 → 64 → 1 (same structure as v1, wider input)
- Loss: BCEWithLogitsLoss
- Optimizer: Adam, lr=1e-3
- Epochs: 40, early stopping patience=8
- Dropout: 0.2 → 0.1
- Export: `relevance_weights_v2.npz` with pre-transposed weight matrices (for numpy inference)

**Step 4: Create stress test**

`ml/relevance_ranker_v2/stress_test.py`:
- Include all original 522 stress test cases
- Add new cases from evaluation:
  - `("JWT authentication token verification", "__getattr__")` → must score < 0.3
  - `("JWT authentication token verification", "create_access_token")` → must score > 0.7
  - `("file upload S3 storage", "__getattr__")` → must score < 0.3
  - `("file upload S3 storage", "upload_file")` → must score > 0.7
  - `("password hash bcrypt", "verify_password")` → must score > 0.7
  - `("WebSocket transcript", "get_geo_from_ip")` → must score < 0.3
- Target: 95%+ on stress test

**Step 5: Train the model**

Run: `cd ml/relevance_ranker_v2 && python generate_data.py && python train.py && python stress_test.py`

**Step 6: Copy weights to production**

```bash
cp ml/relevance_ranker_v2/model/relevance_weights_v2.npz grafyx/search/model/
```

**Step 7: Commit**

```bash
git add ml/relevance_ranker_v2/ grafyx/search/model/relevance_weights_v2.npz
git commit -m "feat: train Relevance Ranker v2 with 42 features

9 new features: is_dunder, is_init_file, is_method, is_class,
receiver_call_ratio, source_token_entropy, source_unique_token_ratio,
embedding_similarity, caller_count_normalized. Trained on 500K
examples with __getattr__ hard negatives."
```

---

### Task 6: M2 — Caller Disambiguator (Data + Training)

**Files:**
- Create: `ml/caller_disambiguator/features.py` (25-feature extraction)
- Create: `ml/caller_disambiguator/generate_data.py`
- Create: `ml/caller_disambiguator/train.py`
- Create: `ml/caller_disambiguator/stress_test.py`

**Step 1: Create 25-feature extraction**

`ml/caller_disambiguator/features.py`:

Features per (call_site, candidate_callee) pair:
1. `receiver_token_overlap_class_name` — Jaccard of receiver tokens vs class name tokens
2. `receiver_char_bigram_sim_class_name` — char bigram similarity
3. `caller_imports_callee_module` — bool
4. `caller_imports_callee_package` — bool
5. `file_path_distance` — normalized shared prefix length
6. `same_directory` — bool
7. `same_top_package` — bool
8. `has_dot_syntax` — bool (obj.method() vs method())
9. `receiver_is_self` — bool
10. `method_uniqueness` — 1 / count_classes_with_method
11. `callee_is_method` — bool (has self/cls param)
12. `callee_is_standalone` — bool
13. `same_language` — bool
14. `receiver_type_known` — bool (from type annotation)
15. `receiver_type_matches` — bool (if known, matches callee class?)
16. `callee_param_count` — normalized
17. `arg_count_matches_params` — bool
18. `callee_has_decorator` — bool
19. `receiver_name_length` — normalized
20. `method_name_commonness` — frequency across all classes
21. `caller_complexity` — LOC normalized
22. `callee_is_property` — bool
23. `callee_is_classmethod` — bool
24. `callee_is_abstractmethod` — bool
25. `receiver_is_common_pattern` — frequency in training data ("db", "app", "req")

**Step 2: Generate training data (200K examples)**

`ml/caller_disambiguator/generate_data.py`:

Strategy: Scan type-annotated Python code where ground truth is known.

```
For each file in cloned repos:
  For each function/method with source code:
    Find all method calls with dot syntax: obj.method()
    If obj has a type annotation (var: Type = ...):
      - POSITIVE: (call_site_features, Type.method) → label 1
      - NEGATIVE: (call_site_features, OtherClass.method) → label 0
        for each other class that has a method with the same name
    If obj is from constructor (obj = ClassName(...)):
      - Same positive/negative as above
    If call is standalone (no dot syntax):
      - POSITIVE: (call_site_features, standalone_function) → label 1
      - NEGATIVE: (call_site_features, SomeClass.method_with_same_name) → label 0
```

Sources: FastAPI, Pydantic, httpx, SQLModel, Rich, Textual (all well-typed).

**Step 3: Train the model**

Architecture: 25 → 64 → 32 → 1
Loss: BCEWithLogitsLoss
Same training hyperparams as M1.

**Step 4: Stress test with evaluation cases**

Must correctly handle:
- `db.refresh()` → NOT a call to standalone `refresh()` endpoint
- `self.cache.get()` → call to cache class's `get()`, not standalone `get()`
- `process_data()` (no dot) → IS a call to standalone `process_data()`

**Step 5: Train and export**

Run: `cd ml/caller_disambiguator && python generate_data.py && python train.py && python stress_test.py`

```bash
cp ml/caller_disambiguator/model/caller_disambig_weights.npz grafyx/search/model/
git add ml/caller_disambiguator/ grafyx/search/model/caller_disambig_weights.npz
git commit -m "feat: train Caller Disambiguator MLP (25 features)

Binary classifier for P(call_site targets candidate_callee). Trained
on 200K examples from type-annotated Python codebases. Replaces
4-level heuristic filter in get_callers()."
```

---

### Task 7: M3 — Source Token Filter (Data + Training)

**Files:**
- Create: `ml/source_token_filter/features.py` (15 features)
- Create: `ml/source_token_filter/generate_data.py`
- Create: `ml/source_token_filter/train.py`

**Step 1: Create 15-feature extraction**

Features per (query_token, function) pair — see design doc Section 3 M3.

**Step 2: Generate 200K training data**

Strategy: For each function in cloned repos, for each token in the source:
- POSITIVE if token appears in function name, docstring, or parameter names
- NEGATIVE if token only appears in import statements, string literals, `__getattr__` bodies, or comments

**Step 3: Train, stress test, export**

Architecture: 15 → 32 → 16 → 1

Must correctly filter: tokens inside `__getattr__` lazy import dicts → label 0.

```bash
cp ml/source_token_filter/model/source_filter_weights.npz grafyx/search/model/
git add ml/source_token_filter/ grafyx/search/model/source_filter_weights.npz
git commit -m "feat: train Source Token Filter MLP (15 features)

Predicts P(token is semantically relevant to function). Prevents
source token contamination from __getattr__ bodies, string literals,
and incidental mentions."
```

---

### Task 8: M4 — Symbol Importance Ranker (Data + Training)

**Files:**
- Create: `ml/symbol_importance/features.py` (18 features)
- Create: `ml/symbol_importance/generate_data.py`
- Create: `ml/symbol_importance/train.py`

**Step 1: Create 18-feature extraction**

Features per symbol — see design doc Section 3 M4.

**Step 2: Generate 100K training data**

Strategy: For each symbol in cloned repos, compute structural features. Label generation uses multi-signal proxy:
- Mentioned in README/docs → high importance (0.3 weight)
- Is API entry point → high importance (0.25)
- Referenced in >10 files → high importance (0.15)
- Is base class with subclasses → high (0.1)
- Cross-package callers → high (0.1)
- Exported in `__all__` → high (0.1)

Normalize combined score to [0, 1].

**Step 3: Train, stress test, export**

Architecture: 18 → 32 → 16 → 1

```bash
cp ml/symbol_importance/model/symbol_importance_weights.npz grafyx/search/model/
git add ml/symbol_importance/ grafyx/search/model/symbol_importance_weights.npz
git commit -m "feat: train Symbol Importance Ranker MLP (18 features)

Predicts symbol importance for navigation hints and result ranking.
Replaces caller_count heuristic. Trained on 100K symbols from 50+
OSS repos with multi-signal importance labels."
```

---

### Task 9: Integrate M1 (Relevance Ranker v2) into Search

**Files:**
- Modify: `grafyx/search/_relevance.py` (update to 42 features, load v2 weights)
- Modify: `grafyx/search/searcher.py` (pass new feature params)
- Test: `tests/test_relevance_v2.py` (create)

**Step 1: Write failing tests**

```python
# tests/test_relevance_v2.py
"""Tests for Relevance Ranker v2 integration."""

class TestRelevanceV2:

    def test_dunder_method_scores_low(self):
        """__getattr__ should score low for any non-dunder query."""
        pass

    def test_init_file_function_penalized(self):
        """Functions in __init__.py should score lower than same function elsewhere."""
        pass

    def test_exact_match_scores_highest(self):
        """Query matching function name exactly should score > 0.8."""
        pass

    def test_42_features_extracted(self):
        """Feature vector should have 42 dimensions."""
        pass

    def test_graceful_fallback_to_v1(self):
        """If v2 weights not found, fall back to v1 scorer."""
        pass
```

**Step 2: Update `_relevance.py`**

- Change `_FEATURE_COUNT = 33` to `_FEATURE_COUNT = 42`
- Add new feature extraction in `_extract_features()` after line 208
- Update `ml_score_match()` to accept and pass new params: `is_dunder`, `is_init_file`, `is_method`, `is_class`, `receiver_call_ratio`, `source_token_entropy`, `source_unique_token_ratio`, `embedding_similarity`, `caller_count_normalized`
- Load `relevance_weights_v2.npz` instead of `relevance_weights.npz`, with fallback to v1

**Step 3: Update `searcher.py`**

In the `search()` method (line 219-225), compute and pass the new params:
- `is_dunder = name.startswith("__") and name.endswith("__")`
- `is_init_file = "__init__" in func_file`
- `is_method = class_name is not None`
- `is_class = False` (for function scoring) / `True` (for class scoring)
- `receiver_call_ratio`, `source_token_entropy`, `source_unique_token_ratio`, `caller_count_normalized` — compute from graph data

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add grafyx/search/_relevance.py grafyx/search/searcher.py tests/test_relevance_v2.py
git commit -m "feat: integrate Relevance Ranker v2 (42 features) into search

New features: is_dunder, is_init_file, is_method, is_class,
receiver_call_ratio, source_token_entropy, source_unique_token_ratio,
embedding_similarity, caller_count_normalized. Falls back to v1."
```

---

### Task 10: Integrate M2 (Caller Disambiguator) into Graph Engine

**Files:**
- Modify: `grafyx/graph/_callers.py:56-167` (replace 4-level heuristic with M2)
- Create: `grafyx/graph/_caller_features.py` (feature extraction for M2)
- Modify: `grafyx/graph/_analysis.py` (use M2 for unused code detection)
- Test: `tests/test_caller_disambig_ml.py` (create)

**Step 1: Write failing tests**

```python
# tests/test_caller_disambig_ml.py
"""Tests for ML-based caller disambiguation."""

class TestCallerDisambiguationML:

    def test_db_refresh_not_matched_to_standalone_refresh(self):
        """db.refresh() should NOT be matched to standalone refresh() function."""
        pass

    def test_self_method_matched_correctly(self):
        """self.validate() should match the same class's validate method."""
        pass

    def test_standalone_call_matched_to_function(self):
        """process_data() should match the standalone function, not class methods."""
        pass

    def test_fallback_to_heuristic_without_model(self):
        """If M2 weights not found, fall back to 4-level heuristic."""
        pass
```

**Step 2: Create `grafyx/graph/_caller_features.py`**

Feature extraction for M2 that works within the graph engine context. Computes the 25 features for each (call_site, candidate) pair using data available in the graph.

**Step 3: Modify `get_callers()` in `_callers.py`**

Replace the 4-level filtering (lines 119-163) with:
```python
# Try ML disambiguation first
model = get_model("caller_disambig")
if model is not None:
    features = extract_caller_features(caller_entry, callee_info, self)
    score = model.predict(features)
    if score < 0.5:  # Learned threshold from training
        continue  # Skip this caller
else:
    # Fallback: existing 4-level heuristic (keep current code)
    ...
```

**Step 4: Run all tests including existing caller tests**

Run: `python -m pytest tests/test_caller_disambig_ml.py tests/test_class_context.py tests/test_dependency_graph.py -v`

**Step 5: Commit**

```bash
git add grafyx/graph/_callers.py grafyx/graph/_caller_features.py grafyx/graph/_analysis.py tests/test_caller_disambig_ml.py
git commit -m "feat: integrate Caller Disambiguator MLP into get_callers()

Replace 4-level heuristic filter with ML binary classifier.
Graceful fallback to heuristic if model weights not available.
Fixes db.refresh() phantom callers and call graph cross-contamination."
```

---

### Task 11: Integrate M3 (Source Token Filter) and M4 (Symbol Importance)

**Files:**
- Create: `grafyx/search/_source_filter.py` (M3 inference integration)
- Modify: `grafyx/search/_source_index.py:246` (use M3 in `_source_score_for`)
- Modify: `grafyx/server/_hints.py:44` (use M4 in `compute_hints`)
- Test: `tests/test_source_filter.py` (create)
- Test: `tests/test_symbol_importance.py` (create)

**Step 1: Create `grafyx/search/_source_filter.py`**

```python
"""Source Token Filter — M3 integration."""
import numpy as np
from grafyx.ml_inference import get_model


def filter_source_tokens(
    query_tokens: list[str],
    function_name: str,
    function_source: str,
    function_file: str,
    function_docstring: str = "",
) -> dict[str, float]:
    """Return quality-filtered source token weights.

    For each query token found in source, predict P(semantically relevant).
    Returns {token: relevance_score} dict.
    """
    model = get_model("source_filter")
    if model is None:
        # Fallback: all source tokens equally weighted
        return {t: 1.0 for t in query_tokens}

    results = {}
    for token in query_tokens:
        if token.lower() not in function_source.lower():
            results[token] = 0.0
            continue
        features = _extract_token_features(token, function_name, function_source,
                                            function_file, function_docstring)
        results[token] = model.predict(features)
    return results
```

**Step 2: Integrate M3 into `_source_index.py`**

In `_source_score_for()` (line 246), before computing the source score, run each matching token through the filter. Multiply source contribution by filter score.

**Step 3: Integrate M4 into `_hints.py`**

In `compute_hints()` (line 44), replace the caller_count/method_count scoring with:
```python
model = get_model("symbol_importance")
if model is not None:
    importance = model.predict(extract_importance_features(symbol, graph))
else:
    importance = caller_count  # fallback
```

Sort candidates by importance score, return top-N.

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add grafyx/search/_source_filter.py grafyx/search/_source_index.py grafyx/server/_hints.py tests/test_source_filter.py tests/test_symbol_importance.py
git commit -m "feat: integrate Source Token Filter (M3) and Symbol Importance (M4)

M3 filters contaminated source tokens (prevents __getattr__ body
from polluting search). M4 ranks symbols by learned importance
for navigation hints. Both with graceful fallbacks."
```

---

## Phase 3: Mamba Semantic Search Models (M5 + M6)

### Task 12: BPE Tokenizer Training + Pure Python Encoder

**Files:**
- Create: `ml/code_search_encoder/train_tokenizer.py`
- Create: `grafyx/search/_tokenizer.py` (pure Python BPE, no runtime deps)
- Test: `tests/test_tokenizer.py`

**Step 1: Train BPE tokenizer**

`ml/code_search_encoder/train_tokenizer.py`:
- Collect code text from CodeSearchNet + cloned repos
- Train sentencepiece BPE with 16K vocab
- Export: `merges.json` (merge rules) and `vocab.json` (token → id mapping)

**Step 2: Implement pure Python BPE encoder**

`grafyx/search/_tokenizer.py`:

```python
"""Pure Python BPE tokenizer for code search. No runtime dependencies beyond json."""
import json
import re
from pathlib import Path

_MODEL_DIR = Path(__file__).parent / "model"


class CodeTokenizer:
    """BPE tokenizer for code and natural language queries."""

    def __init__(self):
        self._merges: list[tuple[str, str]] = []
        self._vocab: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._loaded = False
        self.pad_id = 0
        self.unk_id = 1
        self.sep_id = 2
        self.cls_id = 3

    def _ensure_loaded(self):
        if self._loaded:
            return
        merges_path = _MODEL_DIR / "bpe_merges.json"
        vocab_path = _MODEL_DIR / "bpe_vocab.json"
        if not merges_path.exists() or not vocab_path.exists():
            raise FileNotFoundError("BPE tokenizer files not found")
        self._merges = [tuple(m) for m in json.loads(merges_path.read_text())]
        self._vocab = json.loads(vocab_path.read_text())
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._loaded = True

    def encode(self, text: str, max_length: int = 256) -> list[int]:
        """Encode text to token IDs."""
        self._ensure_loaded()
        # Pre-tokenize: split on whitespace and punctuation
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        all_ids = [self.cls_id]
        for word in words:
            chars = list(word)
            # Apply BPE merges
            for a, b in self._merges:
                i = 0
                while i < len(chars) - 1:
                    if chars[i] == a and chars[i + 1] == b:
                        chars[i] = a + b
                        del chars[i + 1]
                    else:
                        i += 1
            for token in chars:
                all_ids.append(self._vocab.get(token, self.unk_id))
            if len(all_ids) >= max_length - 1:
                break
        # Pad or truncate
        all_ids = all_ids[:max_length]
        while len(all_ids) < max_length:
            all_ids.append(self.pad_id)
        return all_ids

    @property
    def vocab_size(self) -> int:
        self._ensure_loaded()
        return len(self._vocab)

    @property
    def is_available(self) -> bool:
        return (_MODEL_DIR / "bpe_merges.json").exists()
```

**Step 3: Write tests**

```python
# tests/test_tokenizer.py
class TestCodeTokenizer:
    def test_encode_python_function_name(self):
        """get_current_user should tokenize into meaningful subwords."""
        pass

    def test_encode_natural_query(self):
        """'handle user login' should tokenize cleanly."""
        pass

    def test_max_length_respected(self):
        pass

    def test_unknown_tokens_get_unk_id(self):
        pass
```

**Step 4: Commit**

```bash
git add ml/code_search_encoder/train_tokenizer.py grafyx/search/_tokenizer.py grafyx/search/model/bpe_merges.json grafyx/search/model/bpe_vocab.json tests/test_tokenizer.py
git commit -m "feat: BPE tokenizer for code search (16K vocab, pure Python)"
```

---

### Task 13: Mamba Forward Pass in Numpy

**Files:**
- Create: `grafyx/search/_mamba.py` (Mamba SSM inference in numpy)
- Test: `tests/test_mamba_inference.py`

**Step 1: Implement Mamba block in numpy**

`grafyx/search/_mamba.py`:

```python
"""Mamba (Selective State Space Model) inference in numpy/cupy."""
import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np


def selective_scan(x, A, B, C, D, delta):
    """Mamba selective scan. Pure numpy, no dependencies.

    Args:
        x: (seq_len, d_inner) — input sequence
        A: (d_inner, d_state) — state transition (log space)
        B: (seq_len, d_state) — input-dependent input matrix
        C: (seq_len, d_state) — input-dependent output matrix
        D: (d_inner,) — skip connection
        delta: (seq_len, d_inner) — input-dependent step size

    Returns:
        y: (seq_len, d_inner) — output sequence
    """
    seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # Discretize A
    dA = xp.exp(delta[:, :, None] * A[None, :, :])  # (seq_len, d_inner, d_state)
    dB = delta[:, :, None] * B[:, None, :]  # (seq_len, d_inner, d_state)

    h = xp.zeros((d_inner, d_state), dtype=x.dtype)
    ys = xp.zeros_like(x)

    for t in range(seq_len):
        h = dA[t] * h + dB[t] * x[t, :, None]
        ys[t] = (h * C[t, None, :]).sum(axis=-1) + D * x[t]

    return ys


class MambaBlock:
    """Single Mamba block with in_proj, SSM, out_proj, and LayerNorm."""

    def __init__(self, weights: dict):
        self.in_proj_w = xp.asarray(weights["in_proj_w"])  # (d_model, 2*d_inner)
        self.in_proj_b = xp.asarray(weights["in_proj_b"])
        self.A_log = xp.asarray(weights["A_log"])  # (d_inner, d_state)
        self.D = xp.asarray(weights["D"])  # (d_inner,)
        self.dt_proj_w = xp.asarray(weights["dt_proj_w"])
        self.dt_proj_b = xp.asarray(weights["dt_proj_b"])
        self.B_proj_w = xp.asarray(weights["B_proj_w"])
        self.C_proj_w = xp.asarray(weights["C_proj_w"])
        self.out_proj_w = xp.asarray(weights["out_proj_w"])
        self.out_proj_b = xp.asarray(weights["out_proj_b"])
        self.norm_w = xp.asarray(weights["norm_w"])
        self.norm_b = xp.asarray(weights["norm_b"])

    def __call__(self, x):
        """Forward pass. x: (seq_len, d_model) → (seq_len, d_model)."""
        # Layer norm
        residual = x
        x = _layer_norm(x, self.norm_w, self.norm_b)

        # In projection → split into x and z (gate)
        xz = x @ self.in_proj_w + self.in_proj_b  # (seq_len, 2*d_inner)
        d_inner = xz.shape[-1] // 2
        x_inner, z = xz[:, :d_inner], xz[:, d_inner:]

        # Compute input-dependent B, C, delta
        B = x_inner @ self.B_proj_w  # (seq_len, d_state)
        C = x_inner @ self.C_proj_w  # (seq_len, d_state)
        delta = _softplus(x_inner @ self.dt_proj_w + self.dt_proj_b)

        # SSM
        A = -xp.exp(self.A_log)  # (d_inner, d_state)
        y = selective_scan(x_inner, A, B, C, self.D, delta)

        # Gate and output projection
        y = y * _silu(z)
        y = y @ self.out_proj_w + self.out_proj_b

        return y + residual


class AttentionBlock:
    """Standard self-attention block for Mamba+Attention hybrid."""

    def __init__(self, weights: dict):
        self.qkv_w = xp.asarray(weights["qkv_w"])  # (d_model, 3*d_model)
        self.qkv_b = xp.asarray(weights["qkv_b"])
        self.out_w = xp.asarray(weights["out_w"])
        self.out_b = xp.asarray(weights["out_b"])
        self.ffn_w1 = xp.asarray(weights["ffn_w1"])
        self.ffn_b1 = xp.asarray(weights["ffn_b1"])
        self.ffn_w2 = xp.asarray(weights["ffn_w2"])
        self.ffn_b2 = xp.asarray(weights["ffn_b2"])
        self.norm1_w = xp.asarray(weights["norm1_w"])
        self.norm1_b = xp.asarray(weights["norm1_b"])
        self.norm2_w = xp.asarray(weights["norm2_w"])
        self.norm2_b = xp.asarray(weights["norm2_b"])
        self.n_heads = weights.get("n_heads", 6)

    def __call__(self, x):
        """x: (seq_len, d_model) → (seq_len, d_model)."""
        # Self-attention
        residual = x
        x = _layer_norm(x, self.norm1_w, self.norm1_b)
        qkv = x @ self.qkv_w + self.qkv_b
        d = x.shape[-1]
        q, k, v = qkv[:, :d], qkv[:, d:2*d], qkv[:, 2*d:]

        # Multi-head attention
        head_dim = d // self.n_heads
        seq_len = x.shape[0]
        q = q.reshape(seq_len, self.n_heads, head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, self.n_heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, self.n_heads, head_dim).transpose(1, 0, 2)

        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(head_dim)
        attn = _softmax(scores)
        out = (attn @ v).transpose(1, 0, 2).reshape(seq_len, d)
        x = residual + out @ self.out_w + self.out_b

        # FFN
        residual = x
        x = _layer_norm(x, self.norm2_w, self.norm2_b)
        x = _silu(x @ self.ffn_w1 + self.ffn_b1) @ self.ffn_w2 + self.ffn_b2
        return x + residual


def _layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return w * (x - mean) / xp.sqrt(var + eps) + b


def _softplus(x):
    return xp.log1p(xp.exp(xp.clip(x, -20, 20)))


def _silu(x):
    return x / (1.0 + xp.exp(-xp.clip(x, -20, 20)))


def _softmax(x):
    e = xp.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
```

**Step 2: Write tests with small random weights**

```python
# tests/test_mamba_inference.py
class TestMambaInference:
    def test_selective_scan_shape(self):
        """Output shape matches input shape."""
        pass

    def test_mamba_block_residual(self):
        """MambaBlock output has same shape as input (residual connection)."""
        pass

    def test_attention_block_shape(self):
        """AttentionBlock output has same shape as input."""
        pass

    def test_deterministic_output(self):
        """Same input produces same output (no randomness in inference)."""
        pass
```

**Step 3: Commit**

```bash
git add grafyx/search/_mamba.py tests/test_mamba_inference.py
git commit -m "feat: Mamba and Attention block inference in numpy

Selective scan SSM, MambaBlock, AttentionBlock — all pure numpy/cupy.
O(n) Mamba inference for code sequences. Attention block for
query-code cross-attention in hybrid model."
```

---

### Task 14: M5 — Code Search Encoder (Architecture + Training)

**Files:**
- Create: `ml/code_search_encoder/model.py` (PyTorch Mamba bi-encoder)
- Create: `ml/code_search_encoder/dataset.py` (data loading)
- Create: `ml/code_search_encoder/generate_synthetic.py`
- Create: `ml/code_search_encoder/generate_semantic_pairs.py` (Claude Haiku)
- Create: `ml/code_search_encoder/download_codesearchnet.py`
- Create: `ml/code_search_encoder/train.py`
- Create: `ml/code_search_encoder/evaluate.py`

**Step 1: Download CodeSearchNet**

`ml/code_search_encoder/download_codesearchnet.py`:
- Download Python + JavaScript subsets from HuggingFace or GitHub release
- Filter to (docstring, code) pairs with docstring length >= 10 words
- Output: `ml/code_search_encoder/data/codesearchnet.jsonl`

**Step 2: Generate synthetic pairs (500K)**

`ml/code_search_encoder/generate_synthetic.py`:
- For each function in cloned repos:
  - Split name: `get_current_user` → query "get current user"
  - Add template variations: "fetch current user", "retrieve active user"
  - Docstring → query: first sentence of docstring
- Hard negatives: same-file different function, keyword overlap different intent

**Step 3: Generate Claude Haiku semantic pairs (125K)**

`ml/code_search_encoder/generate_semantic_pairs.py`:
- For each function with source code in cloned repos (sample ~25K functions)
- Call Claude Haiku API: "Generate 5 natural language search queries for this function"
- Output: `ml/code_search_encoder/data/semantic_pairs.jsonl`
- Estimated cost: $50-80

**Step 4: Create PyTorch Mamba bi-encoder model**

`ml/code_search_encoder/model.py`:

```python
"""Mamba bi-encoder for code search. PyTorch training model."""
import torch
import torch.nn as nn
from mamba_ssm import Mamba  # pip install mamba-ssm


class CodeSearchEncoder(nn.Module):
    """Mamba bi-encoder: maps text to 256-dim embedding."""

    def __init__(self, vocab_size=16384, d_model=384, n_layers=8,
                 d_state=16, d_conv=4, expand=2, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": Mamba(d_model=d_model, d_state=d_state,
                              d_conv=d_conv, expand=expand),
                "norm": nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, seq_len) → embeddings: (batch, embed_dim)."""
        seq_len = input_ids.shape[1]
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(pos)

        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x = layer["mamba"](x)
            x = x + residual

        x = self.final_norm(x)
        # Mean pooling (exclude padding)
        mask = (input_ids != 0).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.projection(x)
```

**Step 5: Create training script**

`ml/code_search_encoder/train.py`:
- Loss: InfoNCE (contrastive) with in-batch negatives
- Batch size: 512
- Learning rate: 5e-4, cosine decay with warmup
- Epochs: 10
- Hard negative mining after epoch 3
- Gradient accumulation if batch doesn't fit in VRAM
- Save best model by validation loss
- Export to numpy: `code_encoder_weights.npz`

**Step 6: Train on RTX 5070**

Run: `cd ml/code_search_encoder && python download_codesearchnet.py && python generate_synthetic.py && python train.py`

Estimated: 2-3 hours.

**Step 7: Evaluate**

`ml/code_search_encoder/evaluate.py`:
- Test queries from evaluation: "JWT authentication", "file upload S3", "password hash bcrypt"
- For each: encode query, compute cosine sim against all code embeddings
- Check if correct function is in top-5

**Step 8: Export and commit**

```bash
python ml/code_search_encoder/export_to_numpy.py
cp ml/code_search_encoder/model/code_encoder_weights.npz grafyx/search/model/
git add ml/code_search_encoder/ grafyx/search/model/code_encoder_weights.npz
git commit -m "feat: train Mamba bi-encoder for semantic code search (16M params)

8-layer Mamba SSM, 384 hidden dim, 256 embedding dim. Trained on
2.5M examples (CodeSearchNet + synthetic + Claude Haiku semantic
pairs). Enables 'handle login' → authenticate_user matching."
```

---

### Task 15: M6 — Cross-Encoder Reranker (Architecture + Training)

**Files:**
- Create: `ml/cross_encoder/model.py` (Mamba+Attention hybrid)
- Create: `ml/cross_encoder/generate_data.py`
- Create: `ml/cross_encoder/train.py`

**Step 1: Create hybrid Mamba+Attention model**

`ml/cross_encoder/model.py`:

```python
"""Mamba+Attention cross-encoder for search result reranking."""
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class CrossEncoderReranker(nn.Module):
    """6 Mamba + 2 Attention layers. Input: [query SEP code] → score."""

    def __init__(self, vocab_size=16384, d_model=384, n_mamba=6,
                 n_attn=2, n_heads=6, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)

        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": Mamba(d_model=d_model, d_state=d_state,
                              d_conv=d_conv, expand=expand),
                "norm": nn.LayerNorm(d_model),
            })
            for _ in range(n_mamba)
        ])

        # Attention layers (for query-code cross-attention)
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, batch_first=True,
            )
            for _ in range(n_attn)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, seq_len) → scores: (batch, 1)."""
        seq_len = input_ids.shape[1]
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(pos)

        for layer in self.mamba_layers:
            residual = x
            x = layer["norm"](x)
            x = layer["mamba"](x)
            x = x + residual

        for layer in self.attn_layers:
            x = layer(x)

        x = self.final_norm(x)
        # CLS pooling (first token)
        cls = x[:, 0, :]
        return self.head(cls)
```

**Step 2: Generate reranking data (500K)**

Format each example as: `[CLS] query_tokens [SEP] code_name code_doc [SEP] [PAD...]`
- Positive: relevant query-code pairs (from M5 training data)
- Negative: irrelevant pairs (random sampling + M5 hard negatives)
- Label: 1.0 (relevant) or 0.0 (irrelevant)

**Step 3: Train**

- Loss: BCEWithLogitsLoss
- Batch size: 256
- Epochs: 15
- Estimated: 1-2 hours on RTX 5070

**Step 4: Export and commit**

```bash
python ml/cross_encoder/export_to_numpy.py
cp ml/cross_encoder/model/cross_encoder_weights.npz grafyx/search/model/
git add ml/cross_encoder/ grafyx/search/model/cross_encoder_weights.npz
git commit -m "feat: train Mamba+Attention cross-encoder reranker (12M params)

6 Mamba + 2 Attention layers. Reranks top-15 search candidates with
full query-code interaction. Trained on 500K contrastive pairs."
```

---

### Task 16: Integrate M5 (Bi-Encoder) into Search Pipeline

**Files:**
- Create: `grafyx/search/_code_encoder.py` (M5 numpy inference)
- Modify: `grafyx/search/searcher.py` (embedding-based retrieval)
- Test: `tests/test_code_encoder.py`

**Step 1: Create `_code_encoder.py`**

```python
"""Code Search Encoder — M5 Mamba bi-encoder inference."""
import numpy as np
from pathlib import Path
from grafyx.search._mamba import MambaBlock
from grafyx.search._tokenizer import CodeTokenizer

_MODEL_DIR = Path(__file__).parent / "model"


class CodeEncoder:
    """Mamba bi-encoder for semantic code search."""

    def __init__(self):
        self._loaded = False
        self._blocks: list[MambaBlock] = []
        self._tokenizer = CodeTokenizer()
        self._embeddings: np.ndarray | None = None  # Pre-computed code embeddings
        self._embedding_names: list[str] = []  # Symbol names for each embedding

    def _ensure_loaded(self):
        if self._loaded:
            return
        weights_path = _MODEL_DIR / "code_encoder_weights.npz"
        if not weights_path.exists():
            raise FileNotFoundError("Code encoder weights not found")
        # Load weights and build blocks...
        self._loaded = True

    def encode(self, text: str) -> np.ndarray:
        """Encode text to 256-dim embedding vector."""
        self._ensure_loaded()
        token_ids = self._tokenizer.encode(text)
        # Forward pass through embedding + Mamba blocks + mean pool + projection
        # ... (numpy implementation using _mamba.py blocks)
        pass

    def build_index(self, symbols: list[dict]):
        """Pre-compute embeddings for all symbols. Call once at init."""
        self._ensure_loaded()
        embeddings = []
        names = []
        for sym in symbols:
            text = f"{sym.get('name', '')} {sym.get('docstring', '')}"
            emb = self.encode(text)
            embeddings.append(emb)
            names.append(sym.get("name", ""))
        self._embeddings = np.stack(embeddings)
        # L2 normalize for cosine similarity via dot product
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._embeddings /= np.maximum(norms, 1e-8)
        self._embedding_names = names

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Semantic search: encode query, cosine sim against index."""
        if self._embeddings is None:
            return []
        q_emb = self.encode(query)
        q_emb /= max(np.linalg.norm(q_emb), 1e-8)
        sims = self._embeddings @ q_emb  # (n_symbols,)
        top_idx = np.argsort(-sims)[:top_k]
        return [(self._embedding_names[i], float(sims[i])) for i in top_idx]

    @property
    def is_available(self) -> bool:
        return (_MODEL_DIR / "code_encoder_weights.npz").exists() and self._tokenizer.is_available
```

**Step 2: Integrate into `searcher.py`**

In the `search()` method, after tokenization:
1. Run M5 bi-encoder to get top-50 semantic candidates
2. Merge with keyword candidates (union, keep max score per symbol)
3. Pass `embedding_similarity` to M1v2 as feature 40
4. Continue with existing pipeline

**Step 3: Write tests, commit**

```bash
git add grafyx/search/_code_encoder.py grafyx/search/searcher.py tests/test_code_encoder.py
git commit -m "feat: integrate Mamba bi-encoder (M5) into search pipeline

Semantic code search: 'handle login' can now find authenticate_user.
Pre-computes embeddings at init, 10ms query encoding, cosine
similarity retrieval. Falls back to keyword-only if unavailable."
```

---

### Task 17: Integrate M6 (Cross-Encoder Reranker) into Search Pipeline

**Files:**
- Create: `grafyx/search/_cross_encoder.py` (M6 numpy inference)
- Modify: `grafyx/search/searcher.py` (reranking step)
- Test: `tests/test_cross_encoder.py`

**Step 1: Create `_cross_encoder.py`**

Load M6 weights (Mamba blocks + Attention blocks), implement forward pass using `_mamba.py` building blocks.

Input: `[CLS] query_tokens [SEP] code_name code_doc [SEP] [PAD...]`
Output: relevance score (sigmoid)

**Step 2: Integrate into `searcher.py`**

After M1v2 scoring and M3 source filtering, take top-15 candidates:
```python
encoder = get_cross_encoder()
if encoder is not None and encoder.is_available:
    reranked = []
    for candidate in top_15:
        score = encoder.score(query, candidate["name"] + " " + candidate.get("docstring", ""))
        reranked.append((candidate, score))
    reranked.sort(key=lambda x: -x[1])
    results = [r[0] for r in reranked[:max_results]]
```

**Step 3: Write tests, commit**

```bash
git add grafyx/search/_cross_encoder.py grafyx/search/searcher.py tests/test_cross_encoder.py
git commit -m "feat: integrate Mamba+Attention cross-encoder (M6) for reranking

Reranks top-15 search candidates with full query-code attention.
~225ms total reranking time. Falls back to M1 scores if unavailable."
```

---

## Final Validation

### Task 18: End-to-End Evaluation

**Files:**
- Create: `tests/test_accuracy_evaluation.py`
- Modify: `docs/grafyx-accuracy-test-prompt.md` (update with expected improvements)

**Step 1: Create automated accuracy tests**

```python
# tests/test_accuracy_evaluation.py
"""End-to-end accuracy tests based on real evaluation results."""

class TestSearchAccuracy:
    """Tests from the Eddy evaluation — search must find correct functions."""

    def test_jwt_auth_finds_create_access_token(self):
        """'JWT authentication token verification' → create_access_token in top-5."""
        pass

    def test_file_upload_finds_s3_service(self):
        """'file upload S3 storage bucket' → S3StorageService in top-5."""
        pass

    def test_getattr_not_in_top_results(self):
        """__getattr__ should NOT appear in top-5 for any non-dunder query."""
        pass

    def test_gibberish_returns_low_confidence(self):
        """'xyzzy foobar qlrmph' → low_confidence=True."""
        pass


class TestCallerAccuracy:
    """Tests for caller disambiguation accuracy."""

    def test_db_refresh_not_matched_to_refresh_endpoint(self):
        """db.refresh() callers should not include the refresh API handler."""
        pass

    def test_call_graph_no_cross_contamination(self):
        """Children of db.refresh() should not include create_access_token."""
        pass
```

**Step 2: Re-run the full evaluation prompt**

Use `docs/grafyx-accuracy-test-prompt.md` on the Eddy project again. Compare scores.

**Step 3: Document results**

Update MEMORY.md with final accuracy numbers and any remaining issues.

**Step 4: Commit**

```bash
git add tests/test_accuracy_evaluation.py
git commit -m "test: add end-to-end accuracy evaluation tests

Based on real evaluation results. Tests search relevance, caller
disambiguation, and overall pipeline accuracy."
```

---

## Summary

| Phase | Tasks | Models | Expected Time |
|-------|-------|--------|---------------|
| Phase 1 | Tasks 1-3 | — | 1-2 days |
| Phase 2 | Tasks 4-11 | M1-M4 (23K params) | 3-5 days |
| Phase 3 | Tasks 12-17 | M5-M6 (28M params) | 5-7 days |
| Validation | Task 18 | — | 1 day |
| **Total** | **18 tasks** | **6 models, ~28M params** | **~2 weeks** |
