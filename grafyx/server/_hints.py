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
        hints.append({
            "tool": "get_module_context",
            "args": {"module_path": dir_name},
            "reason": (
                f"{stats.get('files', 0)} files, "
                f"{stats.get('functions', 0)} functions, "
                f"{stats.get('classes', 0)} classes"
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
