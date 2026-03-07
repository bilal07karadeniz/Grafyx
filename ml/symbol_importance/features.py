"""18-feature extraction for Symbol Importance Ranker.

Given a symbol (function or class), extracts 18 features that indicate
how important it is in the codebase — used for ranking search results
and skeleton output.

Features:
   0: caller_count              (normalized, callers / 50)
   1: cross_file_caller_count   (normalized, cross-file callers / 20)
   2: is_exported_in_all        (bool, listed in __all__)
   3: is_api_endpoint           (bool, has @router/@app decorator)
   4: is_entry_point            (bool, if __name__ == "__main__" or CLI)
   5: loc_count                 (normalized, lines / 200)
   6: param_count               (normalized, params / 10)
   7: has_docstring             (bool)
   8: docstring_length          (normalized, len / 500)
   9: import_count              (normalized, how many files import this / 20)
  10: is_base_class             (bool, has subclasses)
  11: subclass_count            (normalized, subclasses / 10)
  12: method_count              (normalized, for classes, methods / 20)
  13: is_abstract               (bool)
  14: decorator_count           (normalized, decorators / 5)
  15: is_test_function          (bool)
  16: file_depth                (normalized, path depth / 10)
  17: name_length               (normalized, len(name) / 40)
"""

import re
import numpy as np

FEATURE_COUNT = 18

# Decorators that indicate API endpoints
_API_DECORATORS = frozenset({
    "route", "get", "post", "put", "patch", "delete", "head", "options",
    "api_view", "action", "endpoint",
    # Framework-specific
    "app.route", "app.get", "app.post", "app.put", "app.delete",
    "router.get", "router.post", "router.put", "router.delete",
    "bp.route", "blueprint.route",
})

# Decorators that indicate abstract methods
_ABSTRACT_DECORATORS = frozenset({
    "abstractmethod", "abc.abstractmethod",
    "abstractproperty", "abc.abstractproperty",
    "abstractclassmethod", "abstractstaticmethod",
})

_TEST_PREFIXES = ("test_", "Test")
_TEST_DIRS = frozenset({"test", "tests", "load_tests", "__tests__", "spec", "specs"})


def _split_tokens(text: str) -> list[str]:
    if not text:
        return []
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    parts = re.split(r'[^a-zA-Z0-9]+', s.lower())
    return [p for p in parts if len(p) >= 2]


def _is_api_endpoint(decorators: list[str]) -> bool:
    """Check if any decorator indicates an API endpoint."""
    for dec in decorators:
        dec_clean = dec.strip("@").split("(")[0].lower()
        if dec_clean in _API_DECORATORS:
            return True
        # Check for pattern: @app.route, @router.get, etc.
        parts = dec_clean.split(".")
        if len(parts) >= 2 and parts[-1] in {
            "route", "get", "post", "put", "patch", "delete",
            "head", "options",
        }:
            return True
    return False


def _is_entry_point(source: str, file_path: str) -> bool:
    """Check if the symbol is in an entry point context."""
    if not source:
        return False
    # Check for if __name__ == "__main__" pattern
    if '__name__' in source and '__main__' in source:
        return True
    # Check for CLI decorators
    cli_patterns = ["@click.command", "@click.group", "@app.command"]
    for pat in cli_patterns:
        if pat in source:
            return True
    # Check if file is a typical entry point
    if file_path:
        fname = file_path.replace("\\", "/").split("/")[-1]
        if fname in ("__main__.py", "main.py", "cli.py", "manage.py", "app.py"):
            return True
    return False


def _is_abstract(decorators: list[str], base_classes: list[str] | None = None) -> bool:
    """Check if a symbol is abstract."""
    for dec in decorators:
        dec_clean = dec.strip("@").split("(")[0].lower()
        if dec_clean in _ABSTRACT_DECORATORS:
            return True
    if base_classes:
        for base in base_classes:
            if "ABC" in base or "Abstract" in base:
                return True
    return False


def _is_test(name: str, file_path: str) -> bool:
    """Check if a symbol is a test function/class."""
    if any(name.startswith(prefix) for prefix in _TEST_PREFIXES):
        return True
    if file_path:
        parts = file_path.replace("\\", "/").lower().split("/")
        if any(p in _TEST_DIRS for p in parts):
            return True
        fname = parts[-1] if parts else ""
        if fname.startswith("test_") or fname.endswith("_test.py"):
            return True
    return False


def _file_depth(file_path: str) -> int:
    """Count path depth (number of directory levels)."""
    if not file_path:
        return 0
    parts = [p for p in file_path.replace("\\", "/").split("/") if p]
    return max(0, len(parts) - 1)  # Exclude filename


def extract_features(
    name: str,
    file_path: str,
    source: str = "",
    docstring: str = "",
    param_names: list[str] | None = None,
    decorators: list[str] | None = None,
    base_classes: list[str] | None = None,
    methods: list[str] | None = None,
    # Graph-derived features (computed externally)
    caller_count: int = 0,
    cross_file_caller_count: int = 0,
    is_exported_in_all: bool = False,
    import_count: int = 0,
    subclass_count: int = 0,
) -> np.ndarray:
    """Extract 18 features for a symbol."""
    vec = np.zeros(FEATURE_COUNT, dtype=np.float32)

    param_names = param_names or []
    decorators = decorators or []
    base_classes = base_classes or []
    methods = methods or []

    lines = source.split("\n") if source else []

    # 0: caller_count (normalized)
    vec[0] = min(1.0, caller_count / 50.0)

    # 1: cross_file_caller_count (normalized)
    vec[1] = min(1.0, cross_file_caller_count / 20.0)

    # 2: is_exported_in_all (bool)
    vec[2] = 1.0 if is_exported_in_all else 0.0

    # 3: is_api_endpoint (bool)
    vec[3] = 1.0 if _is_api_endpoint(decorators) else 0.0

    # 4: is_entry_point (bool)
    vec[4] = 1.0 if _is_entry_point(source, file_path) else 0.0

    # 5: loc_count (normalized)
    vec[5] = min(1.0, len(lines) / 200.0)

    # 6: param_count (normalized)
    # Exclude 'self' and 'cls' from count
    effective_params = [
        p for p in param_names if p not in ("self", "cls")
    ]
    vec[6] = min(1.0, len(effective_params) / 10.0)

    # 7: has_docstring (bool)
    vec[7] = 1.0 if docstring and docstring.strip() else 0.0

    # 8: docstring_length (normalized)
    vec[8] = min(1.0, len(docstring or "") / 500.0)

    # 9: import_count (normalized)
    vec[9] = min(1.0, import_count / 20.0)

    # 10: is_base_class (bool)
    vec[10] = 1.0 if subclass_count > 0 else 0.0

    # 11: subclass_count (normalized)
    vec[11] = min(1.0, subclass_count / 10.0)

    # 12: method_count (normalized, for classes)
    vec[12] = min(1.0, len(methods) / 20.0)

    # 13: is_abstract (bool)
    vec[13] = 1.0 if _is_abstract(decorators, base_classes) else 0.0

    # 14: decorator_count (normalized)
    vec[14] = min(1.0, len(decorators) / 5.0)

    # 15: is_test_function (bool)
    vec[15] = 1.0 if _is_test(name, file_path) else 0.0

    # 16: file_depth (normalized)
    vec[16] = min(1.0, _file_depth(file_path) / 10.0)

    # 17: name_length (normalized)
    vec[17] = min(1.0, len(name) / 40.0)

    return vec
