"""Type definitions for the graph package's internal data structures.

This module defines TypedDicts that document the exact shape of dict entries
stored in the various indexes and returned by query methods. They serve two
purposes:

    1. **Documentation**: Each field is annotated with its type and semantics,
       making it easy for AI assistants and developers to understand what data
       flows through the system.
    2. **Type-checking**: Static analysis tools (mypy, pyright) can validate
       that code producing/consuming these dicts uses the correct keys.

At runtime, the code uses plain dicts -- these TypedDicts add no overhead.

The types here map to specific data structures:
    - CallerEntry      -> entries in ``_caller_index[callee_name]``
    - FunctionDict     -> items returned by ``get_all_functions()``
    - ClassDict        -> items returned by ``get_all_classes()``
    - FileDict         -> items returned by ``get_all_files()``
    - UnusedFunctionDict -> items returned by ``get_unused_functions()``
    - UnusedClassDict    -> items returned by ``get_unused_classes()``

This module has no mixin class and no dependencies on other graph modules.
"""

from typing import Any, TypedDict


# --- Caller Index Entry ---

class CallerEntry(TypedDict, total=False):
    """An entry in ``_caller_index[callee_name]``.

    Each entry represents one function/method that calls the callee.
    Built by ``IndexBuilderMixin._build_caller_index()`` and its
    augmentation passes (DI patterns, local var types).

    Required keys:
        name: The calling function/method name.
        file: Absolute path to the file containing the caller.

    Optional keys:
        class: The class name if the caller is a method (from parent_class).
        _receivers: Set of source-level receiver expressions (e.g. ``self.db``)
                    extracted by ``_index_calls_from()`` for Level 4 attribute
                    disambiguation in ``CallerQueryMixin.get_callers()``.
        _trusted: If True, this entry was resolved via local variable type
                  analysis (``_augment_index_with_local_var_types``) and should
                  bypass import-graph filtering (Levels 1-3) since we already
                  know the exact target class.
    """
    name: str
    file: str
    # Optional fields (total=False): class, _receivers, _trusted


# --- Query Result Types ---

class FunctionDict(TypedDict, total=False):
    """Dict returned by ``SymbolQueryMixin.get_all_functions()``.

    When ``include_methods=True``, the ``class_name`` field is present
    for methods but absent for top-level functions.
    """
    name: str
    signature: str
    file: str
    language: str
    line: int | None
    docstring: str
    class_name: str  # Only present for methods (when include_methods=True)


class ClassDict(TypedDict, total=False):
    """Dict returned by ``SymbolQueryMixin.get_all_classes()``."""
    name: str
    base_classes: list[str]
    method_count: int
    file: str
    language: str
    line: int | None
    docstring: str


class FileDict(TypedDict, total=False):
    """Dict returned by ``SymbolQueryMixin.get_all_files()``."""
    path: str
    function_count: int
    class_count: int
    import_count: int
    language: str
    docstring: str


# --- Dead Code Detection Result Types ---

class UnusedFunctionDict(TypedDict, total=False):
    """Dict returned by ``AnalysisMixin.get_unused_functions()``.

    ``kind`` distinguishes top-level functions from class methods.
    ``qualified_name`` is ``ClassName.method`` for methods, bare name otherwise.
    """
    name: str
    qualified_name: str
    file: str
    line: int | None
    kind: str  # "function" or "method"
    language: str


class UnusedClassDict(TypedDict, total=False):
    """Dict returned by ``AnalysisMixin.get_unused_classes()``."""
    name: str
    file: str
    line: int | None
    method_count: int
    language: str
