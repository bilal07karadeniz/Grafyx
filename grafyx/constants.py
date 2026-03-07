"""Shared constants used across multiple Grafyx modules.

This file centralizes constant definitions that are referenced by more than
one module (e.g., server tools and graph analysis).  Keeping them here avoids
circular imports and makes it easy to update values in one place.

Contents:
- PYTHON_BUILTINS: Common Python built-in/stdlib names filtered from call
  graphs and function context output.

Why this file exists:
    Both ``_tools_graph.py`` (get_call_graph) and ``_tools_introspection.py``
    (get_function_context) need to filter out noisy builtin calls.  Defining
    the set once here avoids duplication and ensures consistent filtering
    across all tools.

Maintenance notes:
    - When adding names, consider whether they're truly noise in ALL contexts.
      A name like ``get`` is noise for dict.get() but could be a real method
      name in a REST framework -- err on the side of including it since the
      alternative (missing a call) is worse than extra noise.
    - Names here are matched by exact string equality against
      ``FunctionCall.name``, which is the unqualified function/method name.
"""

# ---------------------------------------------------------------------------
# Python builtins and standard library method names to filter from call graphs.
# These names appear so frequently in every codebase that including them in
# call graphs and function context output adds noise without providing
# architectural insight.
#
# Used by:
#   - server/_tools_graph.py: get_call_graph() filters both the calls tree
#     and callers tree (unless include_builtins=True).
#   - server/_tools_introspection.py: get_function_context() filters the
#     "calls" list to show only application-level callees.
# ---------------------------------------------------------------------------
PYTHON_BUILTINS: set[str] = {
    # --- Built-in functions ---
    # Core type constructors and introspection
    "print", "len", "range", "str", "int", "float", "bool", "list", "dict",
    "set", "tuple", "isinstance", "issubclass", "type", "id", "hash", "repr",
    "getattr", "setattr", "hasattr", "delattr", "callable", "super",
    # Iteration and functional patterns
    "enumerate", "zip", "map", "filter", "sorted", "reversed",
    # Aggregation and math
    "any", "all", "min", "max", "sum", "abs", "round",
    # I/O and iteration protocol
    "open", "next", "iter", "vars", "dir", "input",

    # --- Common string methods ---
    # These are method names, not builtins, but they appear on virtually
    # every string operation and provide no architectural insight.
    "join", "split", "strip", "lstrip", "rstrip", "replace", "format",
    "startswith", "endswith", "lower", "upper", "find", "index", "count",
    "encode", "decode", "title", "capitalize", "isdigit", "isalpha",

    # --- Common list/dict/set methods ---
    # Same rationale as string methods -- ubiquitous container operations.
    "append", "extend", "insert", "remove", "pop", "clear", "copy",
    "keys", "values", "items", "get", "update", "add", "discard",
    "sort", "reverse",

    # --- Common I/O methods ---
    "read", "write", "close", "flush", "seek",

    # --- datetime / time methods ---
    # Calendar and timestamp operations show up in many modules but
    # rarely indicate meaningful coupling.
    "now", "utcnow", "today", "timestamp", "isoformat", "strftime", "strptime",
    "total_seconds", "fromtimestamp", "fromisoformat", "date", "time", "timedelta",
    "replace", "astimezone", "combine",

    # --- Logging methods ---
    # Every module calls logger.info/debug/error -- filtering these keeps
    # call graphs focused on business logic.
    "debug", "info", "warning", "error", "critical", "exception",
    "getLogger", "setLevel", "addHandler",

    # --- Path / OS methods ---
    # File-system operations that appear across many modules.
    "exists", "mkdir", "makedirs", "isfile", "isdir", "basename", "dirname",
    "joinpath", "resolve", "absolute", "parent", "stem", "suffix", "name",

    # --- JSON methods ---
    "dumps", "loads", "dump", "load",

    # --- Common async/misc patterns ---
    # asyncio primitives and common async patterns.
    "sleep", "run", "wait", "create_task", "gather", "ensure_future",
}
