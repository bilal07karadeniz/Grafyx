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
