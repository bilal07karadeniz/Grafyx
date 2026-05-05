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
    - get_dependency_graph, get_call_graph
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
MAX_HINTS = 5

# Number of features for M4 symbol importance model
IMPORTANCE_FEATURE_COUNT = 18

# Directories to deprioritize when suggesting modules to explore
_DEPRIORITIZED_DIRS = {"tests", "test", "__pycache__", "migrations", "docs", "scripts"}

# Test directory names for feature extraction
_TEST_DIRS = frozenset({"test", "tests", "load_tests", "__tests__", "spec", "specs"})


def _extract_importance_features(
    symbol_name: str,
    symbol_file: str,
    symbol_type: str,
    graph: Any,
) -> "np.ndarray":
    """Extract 18 features for symbol importance prediction (M4).

    Features capture structural signals that indicate how "important" a symbol
    is within the codebase: caller count, cross-file usage, API indicators,
    class size, etc.

    Feature vector:
        [0]  caller_count (normalized by /20)
        [1]  cross_file_caller_count (unique files, normalized by /10)
        [2]  is_exported_in___all__ (placeholder, always 0)
        [3]  is_api_endpoint (file path contains API indicators)
        [4]  is_entry_point (name contains "main")
        [5]  loc_count (placeholder, always 0)
        [6]  param_count (placeholder, always 0)
        [7]  has_docstring (placeholder, always 0)
        [8]  docstring_length (placeholder, always 0)
        [9]  import_count (files importing this file, normalized by /20)
        [10] is_base_class (placeholder, always 0)
        [11] subclass_count (placeholder, always 0)
        [12] method_count (for classes, normalized by /20)
        [13] is_abstract (placeholder, always 0)
        [14] decorator_count (placeholder, always 0)
        [15] is_test_function (test file or test_ prefix)
        [16] file_depth (normalized by /8)
        [17] name_length (normalized by /30)

    Args:
        symbol_name: Name of the symbol.
        symbol_file: File path where the symbol is defined.
        symbol_type: One of "function", "class", "file".
        graph: CodebaseGraph instance (for accessing indexes).

    Returns:
        numpy float32 array of shape (IMPORTANCE_FEATURE_COUNT,).
    """
    import numpy as np

    vec = np.zeros(IMPORTANCE_FEATURE_COUNT, dtype=np.float32)

    # Feature 0: caller_count (normalized)
    callers = getattr(graph, "_caller_index", {}).get(symbol_name, [])
    vec[0] = min(len(callers) / 20.0, 1.0)

    # Feature 1: cross_file_caller_count (normalized)
    unique_files = {c.get("file", "") for c in callers}
    vec[1] = min(len(unique_files) / 10.0, 1.0)

    # Feature 2: is_exported_in___all__ -- not easily checkable, placeholder
    vec[2] = 0.0

    # Feature 3: is_api_endpoint (file path contains API indicators)
    api_indicators = {"router", "app", "api", "endpoint", "route", "view"}
    file_lower = symbol_file.lower() if symbol_file else ""
    vec[3] = float(any(ind in file_lower for ind in api_indicators))

    # Feature 4: is_entry_point
    vec[4] = float("main" in symbol_name.lower() or "__main__" in file_lower)

    # Features 5-8: placeholders (not available from indexes)
    # vec[5] = loc_count, vec[6] = param_count
    # vec[7] = has_docstring, vec[8] = docstring_length

    # Feature 9: import_count (files that import this file)
    importers = getattr(graph, "_import_index", {}).get(symbol_file, [])
    vec[9] = min(len(importers) / 20.0, 1.0)

    # Features 10-11: base class / subclass (placeholders)

    # Feature 12: method_count (for classes)
    methods = getattr(graph, "_class_method_names", {}).get(symbol_name, set())
    vec[12] = min(len(methods) / 20.0, 1.0)

    # Features 13-14: is_abstract, decorator_count (placeholders)

    # Feature 15: is_test_function
    path_parts = (
        symbol_file.replace("\\", "/").lower().split("/") if symbol_file else []
    )
    vec[15] = float(
        any(p in _TEST_DIRS or p.startswith("test_") for p in path_parts)
        or symbol_name.startswith("test_")
    )

    # Feature 16: file_depth (normalized)
    vec[16] = min(len(path_parts) / 8.0, 1.0)

    # Feature 17: name_length (normalized)
    vec[17] = min(len(symbol_name) / 30.0, 1.0)

    return vec


def _score_symbol_importance(
    symbol_name: str,
    symbol_file: str,
    symbol_type: str,
    graph: Any,
) -> float:
    """Score symbol importance using M4 model with heuristic fallback.

    When M4 model weights are available, extracts features and runs the
    MLP predictor. Otherwise falls back to a simple count-based heuristic
    using caller count + import count.

    Args:
        symbol_name: Name of the symbol.
        symbol_file: File path where the symbol is defined.
        symbol_type: One of "function", "class", "file".
        graph: CodebaseGraph instance.

    Returns:
        Importance score in [0.0, 1.0].
    """
    try:
        from grafyx.ml_inference import get_model
        model = get_model("symbol_importance")
    except Exception:
        model = None

    if model is not None:
        features = _extract_importance_features(
            symbol_name, symbol_file, symbol_type, graph,
        )
        return model.predict(features)

    # Fallback: count-based heuristic
    caller_count = len(getattr(graph, "_caller_index", {}).get(symbol_name, []))
    import_count = len(getattr(graph, "_import_index", {}).get(symbol_file, []))
    return min((caller_count + import_count) / 50.0, 1.0)


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
    For "big" top-level dirs (those with subdir_stats data), drill one
    level deeper so hints point at e.g. ``backend/app`` rather than
    plain ``backend`` -- a zoom an LLM can act on instead of re-running
    the same query.
    """
    dir_stats = data.get("directory_stats", {})
    if not dir_stats:
        return []

    subdir_stats = data.get("subdir_stats", {}) or {}

    scored = []
    for dir_name, stats in dir_stats.items():
        # Skip test/build directories -- they're rarely the first thing
        # an LLM should explore when understanding a project
        if dir_name.lower() in _DEPRIORITIZED_DIRS or dir_name.startswith("."):
            continue
        score = stats.get("functions", 0) + stats.get("classes", 0)
        if score > 0:
            scored.append((score, dir_name, stats))

    scored.sort(reverse=True)
    hints = []
    for score, dir_name, stats in scored[:MAX_HINTS]:
        # When we know the top-level dir is large, point at its biggest
        # non-test subdirectory so the agent gets a meaningfully smaller
        # zoom on the next call.
        target_path = dir_name
        target_stats = stats
        subs = subdir_stats.get(dir_name)
        if subs:
            for sub_name, sub_stats in subs.items():
                if sub_name.lower() in _DEPRIORITIZED_DIRS or sub_name.startswith("."):
                    continue
                if sub_stats.get("files", 0) >= 5:
                    target_path = f"{dir_name}/{sub_name}"
                    target_stats = sub_stats
                    break
        hints.append({
            "tool": "get_module_context",
            "args": {"module_path": target_path},
            "reason": (
                f"{target_stats.get('files', 0)} files, "
                f"{target_stats.get('functions', 0)} functions, "
                f"{target_stats.get('classes', 0)} classes"
            ),
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

    # Interleave: top class, then top 2 functions
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

    # Suggest base class exploration
    base_classes = data.get("base_classes", [])
    for base in base_classes[:1]:
        if base and base not in ("object", "ABC", "Protocol"):
            hints.append({
                "tool": "get_class_context",
                "args": {"class_name": base},
                "reason": "parent class",
            })

    # Suggest most-called method.
    # Use disambiguated `get_callers(name, class_name=cls_name)` so the
    # count matches what `get_function_context` would return — the raw
    # `_caller_index` lumps in callers of same-named methods from OTHER
    # classes and reports a count that doesn't survive disambiguation.
    cls_name = data.get("name", "")
    methods = data.get("methods", [])
    scored_methods = []
    for method in methods:
        name = method.get("name", "")
        if name.startswith("__"):
            continue
        try:
            disambiguated = graph.get_callers(name, class_name=cls_name)
            caller_count = len(disambiguated)
        except Exception:
            caller_count = 0
        scored_methods.append((caller_count, name))
    scored_methods.sort(reverse=True)

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
        reason = (
            f"called by this function, has {_score} callers"
            if _score > 0
            else "called by this function"
        )
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
