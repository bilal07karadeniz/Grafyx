"""Analysis methods for CodebaseGraph.

This module provides AnalysisMixin, which contains the higher-level analysis
tools that go beyond simple lookups to provide actionable insights:

    **Dead code detection** (``get_unused_functions``, ``get_unused_classes``):
        Finds functions and classes with zero inbound callers/references.
        Uses a fast pre-check on ``_caller_index`` (O(1) lookup) before
        falling back to the expensive ``get_callers()`` disambiguation.
        Excludes many categories of implicitly-used code:
        - Python dunder methods and JS/TS lifecycle hooks
        - Framework-decorated handlers (routes, tasks, fixtures)
        - Abstract/Protocol method implementations (interface contracts)
        - Migration files (auto-generated, called by framework)
        - Symbols that appear in import statements elsewhere

    **Subclass tree traversal** (``get_subclasses``):
        Recursively builds a tree of all classes extending a given base class,
        up to a configurable depth.

Mixin: AnalysisMixin
Reads: self._codebases, self._lock, self._caller_index, self._class_method_names,
       self._class_defined_in, self._import_index
Writes: nothing (read-only analysis)
"""

import logging
from typing import Any

from grafyx.utils import (
    extract_base_classes,
    safe_get_attr,
    safe_str,
)

logger = logging.getLogger(__name__)


class AnalysisMixin:
    """Dead code detection and subclass trees.

    This mixin contains the most computationally expensive operations in the
    graph engine. Dead code detection in particular iterates all functions
    with multiple index lookups per function. All methods acquire
    ``self._lock`` for thread safety.

    Reads: _codebases, _lock, _caller_index, _class_method_names,
    _class_defined_in, _import_index
    """

    # Base classes that indicate abstract/protocol patterns --
    # methods defined in subclasses of these are interface contracts (called
    # polymorphically via the parent type) and should not be flagged as dead code.
    _ABSTRACT_BASE_NAMES = frozenset({
        "Protocol", "ABC", "ABCMeta",
    })

    # Base class names that are *not* indicators of framework dispatch. A class
    # extending these is just a plain Python object — its methods can still be
    # genuinely unused. Anything not in this set, when found as an external
    # ancestor (no definition in the codebase), is treated as a framework
    # base whose methods may be polymorphically dispatched.
    _NON_DISPATCHING_BASES = frozenset({
        "object", "type", "Exception", "BaseException",
        "Warning", "DeprecationWarning", "UserWarning", "FutureWarning",
        "RuntimeError", "ValueError", "TypeError", "KeyError", "AttributeError",
        "ImportError", "OSError", "IOError", "NotImplementedError",
        "Generic", "TypedDict", "NamedTuple",
        "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag",
        "tuple", "list", "dict", "set", "frozenset", "str", "bytes", "int",
        "float", "bool", "complex",
    })

    # Top-level functions in files matching these suffixes are likely plugin /
    # extension hooks dispatched dynamically by an external framework
    # (MkDocs, pytest, Sphinx, Discord cogs, etc.). Skip them in unused
    # detection — false positives here would mislead a developer into deleting
    # framework integration code.
    _HOOK_FILE_SUFFIXES = (
        "_hooks.py", "_hook.py",
        "_plugin.py", "_plugins.py",
        "_extension.py", "_extensions.py",
        "conftest.py",
    )

    # ===================================================================
    # Dead Code Detection: Unused Functions
    # ===================================================================

    def _get_class_bases(self, class_name: str) -> set[str]:
        """Return set of direct base class names for a class.

        Iterates all codebases to find the class definition and extracts
        base classes from graph-sitter's ``superclasses`` or ``bases`` attr.
        """
        for _lang, codebase in self._codebases.items():
            try:
                for cls in safe_get_attr(codebase, "classes", []):
                    if safe_get_attr(cls, "name", "") == class_name:
                        bases = safe_get_attr(cls, "superclasses",
                                    safe_get_attr(cls, "bases", []))
                        return {
                            str(b).split(".")[-1].strip()
                            for b in bases
                        }
            except Exception:
                continue
        return set()

    def get_unused_functions(
        self,
        include_tests: bool = False,
        max_results: int = 100,
    ) -> list[dict]:
        """Find functions/methods with zero inbound callers.

        Algorithm (in order of cost, cheapest checks first):
            1. Skip test files, migration files, implicit methods, main().
            2. Skip functions whose name appears in an import statement.
            3. **Fast path**: Check ``_caller_index`` -- if no raw callers exist,
               the function is definitely unused (no disambiguation needed).
            4. **Disambiguation path**: If raw callers exist for a method name,
               run ``get_callers(name, class_name)`` to check if any callers
               survive the 4-level filtering. If none do, it's unused.
            5. **Decorator check**: If still potentially unused, inspect
               graph-sitter's decorator/usage data to catch framework-managed
               functions (routes, fixtures, event handlers, etc.).
            6. **Abstract/Protocol check**: Skip methods that implement an
               interface contract (any ancestor is Protocol/ABC, or the
               method name exists in an ancestor class).

        Returns:
            List of UnusedFunctionDict-shaped dicts, sorted by file + line.
        """
        # Python dunder methods + TS/JS lifecycle methods called implicitly
        # by the runtime, not by user code. These should never be flagged.
        IMPLICIT_METHODS = {
            # Python dunders (called by runtime)
            "__init__", "__new__", "__del__", "__repr__", "__str__",
            "__bytes__", "__format__", "__hash__", "__bool__",
            "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
            "__getattr__", "__getattribute__", "__setattr__", "__delattr__",
            "__get__", "__set__", "__delete__", "__init_subclass__",
            "__class_getitem__", "__call__", "__len__", "__length_hint__",
            "__getitem__", "__setitem__", "__delitem__", "__missing__",
            "__contains__", "__iter__", "__next__", "__reversed__",
            "__enter__", "__exit__", "__aenter__", "__aexit__",
            "__await__", "__aiter__", "__anext__",
            "__add__", "__radd__", "__iadd__", "__sub__", "__rsub__",
            "__mul__", "__rmul__", "__truediv__", "__floordiv__",
            "__mod__", "__pow__", "__and__", "__or__", "__xor__",
            "__neg__", "__pos__", "__abs__", "__invert__",
            "__complex__", "__int__", "__float__", "__index__",
            "__post_init__",
            # unittest lifecycle (called by test runner)
            "setUp", "tearDown", "setUpClass", "tearDownClass",
            "setUpModule", "tearDownModule",
            "asyncSetUp", "asyncTearDown",
            # JS/TS built-in protocol methods
            "constructor", "toString", "valueOf", "toJSON",
            # Middleware dispatch (called by framework)
            "dispatch",
            # React lifecycle
            "render", "componentDidMount", "componentDidUpdate",
            "componentWillUnmount", "shouldComponentUpdate",
            "getDerivedStateFromProps", "getSnapshotBeforeUpdate",
            "componentDidCatch", "getDerivedStateFromError",
            # Angular lifecycle
            "ngOnInit", "ngOnDestroy", "ngOnChanges", "ngDoCheck",
            "ngAfterContentInit", "ngAfterContentChecked",
            "ngAfterViewInit", "ngAfterViewChecked",
            # Vue Options API lifecycle
            "created", "mounted", "updated", "destroyed", "unmounted",
            "beforeCreate", "beforeMount", "beforeUpdate",
            "beforeDestroy", "beforeUnmount", "setup",
        }

        # Decorators that indicate the function is called by a framework, not
        # directly by user code. If any decorator's base name (after stripping
        # @ and module path) matches, the function is skipped.
        FRAMEWORK_DECORATORS = {
            # Python
            "property", "cached_property", "staticmethod", "classmethod",
            "abstractmethod", "overload", "override", "deprecated",
            "route", "get", "post", "put", "delete", "patch",
            "api_view", "action", "websocket",
            "exception_handler", "middleware",
            "fixture", "parametrize", "mark",
            "receiver", "on_event", "listener", "handler", "hook",
            "subscriber", "callback", "lifespan",
            "command", "group", "argument", "option",
            "tool",
            # Pydantic
            "validator", "root_validator",
            "field_validator", "model_validator", "field_serializer",
            "model_serializer", "computed_field",
            # FastAPI
            "depends",
            # Celery
            "task", "shared_task", "periodic_task",
            # Django REST Framework
            "serializer_method",
            # Angular
            "Component", "Injectable", "Directive", "Pipe", "NgModule",
            "Input", "Output", "HostListener", "HostBinding",
            # NestJS
            "Controller", "Module", "Guard", "Middleware",
            "UseGuards", "UseInterceptors",
            # TypeORM / ORMs
            "Entity", "Column", "PrimaryGeneratedColumn",
            "ManyToOne", "OneToMany", "ManyToMany",
            # MobX
            "observable", "action", "computed",
        }

        with self._lock:
            unused: list[dict] = []
            imported_names = self._build_imported_names()

            # Names re-exported via __init__.py / index.ts / index.js.
            # These are public API surface — unused internally but intentionally
            # exported.  Flagged separately from genuinely dead code.
            re_exported_names: set[str] = set()
            for _imp_path, _targets in getattr(self, "_file_symbol_imports", {}).items():
                _norm = _imp_path.replace("\\", "/")
                if (_norm.endswith("/__init__.py")
                        or _norm.endswith("/index.ts")
                        or _norm.endswith("/index.js")):
                    for _tgt, _names in _targets.items():
                        re_exported_names.update(_names)

            all_funcs = self.get_all_functions(
                max_results=5000, include_methods=True
            )

            # Build a direct-parent map once for transitive ancestor lookups.
            # Uses extract_base_classes() which has a multi-attribute + regex
            # fallback chain to handle varying graph-sitter versions reliably.
            # This powers the Protocol/ABC check: if any ancestor is abstract,
            # all methods in the subclass are considered interface implementations.
            _inh: dict[str, set[str]] = {}
            for _l, _cb in self._codebases.items():
                try:
                    for _cls in safe_get_attr(_cb, "classes", []):
                        _cn = safe_get_attr(_cls, "name", "")
                        if not _cn:
                            continue
                        _bases = extract_base_classes(_cls)
                        _inh[_cn] = {b.split(".")[-1].strip() for b in _bases if b}
                except Exception:
                    continue

            def _ancestors(cls: str, depth: int = 8) -> set[str]:
                """BFS over _inh to find all transitive ancestor class names."""
                seen: set[str] = set()
                frontier = list(_inh.get(cls, set()))
                for _ in range(depth):
                    if not frontier:
                        break
                    next_f: list[str] = []
                    for b in frontier:
                        if b not in seen:
                            seen.add(b)
                            next_f.extend(_inh.get(b, set()))
                    frontier = next_f
                return seen

            for func_dict in all_funcs:
                if len(unused) >= max_results:
                    break

                name = func_dict.get("name", "")
                file_path = func_dict.get("file", "")
                class_name = func_dict.get("class_name")

                if not name:
                    continue

                if not include_tests and self._is_test_path(file_path):
                    continue

                if self._is_migration_path(file_path):
                    continue

                if name in IMPLICIT_METHODS:
                    continue

                if name == "main" or file_path.replace("\\", "/").endswith("__main__.py"):
                    continue

                # Skip top-level functions in plugin/hook files — these are
                # dispatched by external frameworks and look unused statically.
                _norm_path = file_path.replace("\\", "/")
                if not class_name and any(
                    _norm_path.endswith(s) for s in self._HOOK_FILE_SUFFIXES
                ):
                    continue

                # Check if this function is explicitly imported by name elsewhere.
                # For top-level functions, also verify they're actually CALLED
                # (not just re-exported). Re-exports inflate imported_names.
                if not class_name and name in imported_names:
                    if self._caller_index.get(name, []):
                        continue
                    # Fall through — imported/re-exported but never called

                # Fast path: O(1) check on _caller_index. If no raw callers
                # exist at all, the function is definitely unused (skip directly
                # to the decorator check). For methods with raw callers, we need
                # the full get_callers() disambiguation to determine if any
                # callers survive filtering for THIS specific class.
                raw_callers = self._caller_index.get(name, [])
                if raw_callers:
                    if class_name:
                        callers = self.get_callers(name, class_name=class_name)
                        if callers:
                            continue
                    else:
                        # Top-level function with any caller -> definitely used
                        continue

                # --- Text search fallback ---
                # When static analysis finds zero callers, scan source files for
                # the symbol name as a call pattern. This catches:
                # - instance.method() where instance type is unknown
                # - callbacks in dicts, getattr(), string registries
                # - Top-level functions used via indirect patterns
                _call_pattern = f".{name}(" if class_name else f"{name}("
                _found_text_ref = False
                for _lang2, _codebase2 in self._codebases.items():
                    if _found_text_ref:
                        break
                    for _f2 in _codebase2.files:
                        _f2_path = self.translate_path(
                            str(safe_get_attr(_f2, "filepath", safe_get_attr(_f2, "path", "")))
                        )
                        if _f2_path == file_path:
                            continue  # Skip the defining file
                        _src2 = safe_str(safe_get_attr(_f2, "source", ""))
                        if _call_pattern in _src2:
                            _found_text_ref = True
                            break
                if _found_text_ref:
                    continue

                # Expensive path: check decorators and usages on potentially
                # unused functions. This requires fetching the full function
                # object from graph-sitter.
                skip = False
                result = self.get_function(
                    f"{class_name}.{name}" if class_name else name
                )
                if result is not None:
                    func_obj = None
                    if isinstance(result, list):
                        for _lang, f, _cn in result:
                            fp = self.translate_path(
                                str(safe_get_attr(f, "filepath", ""))
                            )
                            if fp == file_path:
                                func_obj = f
                                break
                        if func_obj is None and result:
                            func_obj = result[0][1]
                    else:
                        func_obj = result[1]

                    if func_obj:
                        # Check graph-sitter usages (catches dict refs,
                        # re-exports, type annotations, etc.)
                        usages = safe_get_attr(func_obj, "usages", [])
                        if usages:
                            # Filter: ignore usages only from test files
                            if not include_tests:
                                non_test = [
                                    u for u in usages
                                    if not self._is_test_path(
                                        str(safe_get_attr(
                                            safe_get_attr(u, "match", u),
                                            "filepath",
                                            safe_get_attr(u, "file", ""),
                                        ))
                                    )
                                ]
                                if non_test:
                                    skip = True
                            else:
                                skip = True

                        if not skip:
                            decorators = safe_get_attr(func_obj, "decorators", [])
                            if decorators:
                                for d in decorators:
                                    d_full = safe_str(d).strip("@").split("(")[0]
                                    d_base = d_full.split(".")[-1]
                                    if d_base.lower() in FRAMEWORK_DECORATORS:
                                        skip = True
                                        break
                                    # Compound: signal.connect patterns
                                    # e.g., worker_process_init.connect
                                    if d_base.lower() == "connect" and "." in d_full:
                                        skip = True
                                        break

                        # Body-deprecation check: if the function emits a
                        # DeprecationWarning via warnings.warn(), it's
                        # intentionally kept for backwards compatibility.
                        # FastAPI's `generate_operation_id` uses this pattern
                        # rather than the (Python 3.13+) @deprecated decorator.
                        if not skip:
                            _src = safe_str(safe_get_attr(func_obj, "source", ""))
                            if "warnings.warn" in _src and (
                                "DeprecationWarning" in _src
                                or "PendingDeprecationWarning" in _src
                            ):
                                skip = True

                        # Skip methods that implement an abstract/protocol contract.
                        # We check two conditions (transitively through the full MRO):
                        # 1. Any ancestor is a Protocol/ABC -- all its concrete methods
                        #    may be dispatched polymorphically.
                        # 2. The method name is defined in some ancestor -- it's an
                        #    override, likely called via the parent type (factory dispatch,
                        #    registry patterns, etc.).
                        if not skip and class_name:
                            all_ancs = _ancestors(class_name)
                            if all_ancs & self._ABSTRACT_BASE_NAMES:
                                skip = True
                            elif all_ancs:
                                _cdi = getattr(self, "_class_defined_in", {}) or {}
                                for _anc in all_ancs:
                                    if name in self._class_method_names.get(_anc, set()):
                                        skip = True
                                        break
                                    # External (non-codebase, non-builtin) base
                                    # class — likely a framework type whose
                                    # methods may be polymorphically dispatched
                                    # (Starlette Route, Django Model, Pydantic
                                    # BaseModel, etc.). Better to under-flag
                                    # than to mislead someone into deleting an
                                    # override that the framework calls.
                                    if (
                                        _cdi
                                        and _anc not in _cdi
                                        and _anc not in self._NON_DISPATCHING_BASES
                                    ):
                                        skip = True
                                        break

                if skip:
                    continue

                qualified = f"{class_name}.{name}" if class_name else name
                kind = "method" if class_name else "function"

                entry: dict = {
                    "name": name,
                    "qualified_name": qualified,
                    "file": file_path,
                    "line": func_dict.get("line"),
                    "kind": kind,
                    "language": func_dict.get("language", ""),
                }
                if name in re_exported_names:
                    entry["re_exported"] = True
                unused.append(entry)

            unused.sort(key=lambda x: (x.get("file", ""), x.get("line") or 0))
            return unused

    # ===================================================================
    # Dead Code Detection: Unused Classes
    # ===================================================================

    def get_unused_classes(
        self,
        include_tests: bool = False,
        max_results: int = 50,
    ) -> list[dict]:
        """Find classes with zero inbound references.

        A class is considered "used" if any of these are true:
            1. It's subclassed by another class in the project.
            2. Any of its methods have callers (via get_callers() disambiguation).
            3. Its constructor (class name) appears in _caller_index.
            4. It appears in an import statement elsewhere.
            5. It has graph-sitter usages (type annotations, re-exports, etc.).

        Excludes:
            - Test classes (when include_tests=False)
            - Migration files (auto-generated)
            - Exception classes (typically caught, not called)
            - Subclasses of framework bases (Pydantic, SQLAlchemy, etc.)

        Returns:
            List of UnusedClassDict-shaped dicts, sorted by file + line.
        """
        # Classes inheriting from these are framework-managed: their usage
        # is implicit (serialization, ORM queries, middleware dispatch, etc.)
        FRAMEWORK_BASES = {
            # Pydantic
            "BaseModel", "BaseSettings", "GenericModel", "RootModel",
            # SQLAlchemy
            "Base", "DeclarativeBase", "DeclarativeBaseNoMeta",
            # Django
            "Model", "View", "Serializer", "ModelSerializer",
            "Form", "ModelForm", "ModelAdmin", "AppConfig",
            "TestCase", "TransactionTestCase", "SimpleTestCase",
            "LiveServerTestCase",
            "Command", "BaseCommand",
            # Starlette / FastAPI
            "BaseHTTPMiddleware",
            # Flask
            "FlaskForm", "Resource", "MethodView",
            # Marshmallow
            "Schema",
            # Python stdlib / typing
            "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag",
            "TypedDict", "NamedTuple",
            # React
            "Component", "PureComponent",
            # Angular (though Angular mostly uses decorators)
            "Pipe", "Directive",
            # TypeORM
            "BaseEntity",
            # Abstract / Protocol
            "Protocol", "ABC",
            # Testing
            "TestCase",
        }

        with self._lock:
            unused: list[dict] = []
            imported_names = self._build_imported_names()

            all_base_names: set[str] = set()
            all_classes = self.get_all_classes(max_results=2000)
            for cls_dict in all_classes:
                for base in cls_dict.get("base_classes", []):
                    base_name = str(base).split(".")[-1].strip()
                    if base_name:
                        all_base_names.add(base_name)

            for cls_dict in all_classes:
                if len(unused) >= max_results:
                    break

                name = cls_dict.get("name", "")
                file_path = cls_dict.get("file", "")

                if not name:
                    continue

                if not include_tests and self._is_test_path(file_path):
                    continue

                if self._is_migration_path(file_path):
                    continue

                if name.endswith("Error") or name.endswith("Exception"):
                    continue

                # Skip classes that inherit from framework base classes
                base_classes = cls_dict.get("base_classes", [])
                if any(
                    str(b).split(".")[-1].strip() in FRAMEWORK_BASES
                    for b in base_classes
                ):
                    continue

                # Skip classes explicitly imported by name elsewhere
                if name in imported_names:
                    continue

                if name in all_base_names:
                    continue

                method_names = self._class_method_names.get(name, set())
                has_method_callers = False
                for m_name in method_names:
                    if m_name.startswith("__") and m_name.endswith("__"):
                        continue
                    if m_name in self._caller_index:
                        callers = self.get_callers(m_name, class_name=name)
                        if callers:
                            has_method_callers = True
                            break
                if has_method_callers:
                    continue

                if name in self._caller_index:
                    continue

                # Check graph-sitter usages (catches type annotations,
                # Field(default_factory=X), re-exports, etc.)
                cls_result = self.get_class(name)
                if cls_result is not None:
                    cls_obj = cls_result[1] if not isinstance(cls_result, list) else cls_result[0][1]
                    usages = safe_get_attr(cls_obj, "usages", [])
                    if usages:
                        if not include_tests:
                            non_test = [
                                u for u in usages
                                if not self._is_test_path(
                                    str(safe_get_attr(
                                        safe_get_attr(u, "match", u),
                                        "filepath",
                                        safe_get_attr(u, "file", ""),
                                    ))
                                )
                            ]
                            if non_test:
                                continue
                        else:
                            continue

                unused.append({
                    "name": name,
                    "file": file_path,
                    "line": cls_dict.get("line"),
                    "method_count": cls_dict.get("method_count", 0),
                    "language": cls_dict.get("language", ""),
                })

            unused.sort(key=lambda x: (x.get("file", ""), x.get("line") or 0))
            return unused

    # ===================================================================
    # Subclass Tree Traversal
    # ===================================================================

    def get_subclasses(
        self,
        class_name: str,
        depth: int = 3,
        file_path: str | None = None,
    ) -> dict | None:
        """Find all classes that extend the given class, recursively.

        Builds a reverse inheritance map (base_name -> [child_class_dicts])
        and then does a BFS/DFS traversal up to the specified depth.
        Uses a ``visited`` set to handle diamond inheritance safely.

        When the project has multiple classes with the same name (e.g.,
        FastAPI's ``SecurityBase`` exists in both ``security/base.py`` and
        ``openapi/models.py``), pass ``file_path`` to disambiguate. Without
        it, the first match wins and the result includes a ``candidates``
        list and an ``ambiguous`` flag listing all definitions found.

        Args:
            class_name: The base class to find subclasses of.
            depth: How many inheritance levels to traverse (1 = direct only).
            file_path: When multiple classes share ``class_name``, narrows
                the query to the definition in this file. Children are then
                filtered to those whose own imports trace ``class_name`` back
                to this same file (heuristic — children with no resolvable
                import are conservatively included).

        Returns:
            Dict with class info, direct/total counts, a nested
            ``subclasses`` tree, and (if ambiguous) ``candidates`` +
            ``ambiguous: True``. None if the class is not found.
        """
        with self._lock:
            all_classes = self.get_all_classes(max_results=2000)

            # Find ALL classes matching the name to detect ambiguity
            candidates_list = [
                c for c in all_classes if c.get("name") == class_name
            ]
            if not candidates_list:
                return None

            # Pick the target: file_path-disambiguated if provided, else first.
            target = None
            if file_path:
                _norm_fp = file_path.replace("\\", "/")
                for c in candidates_list:
                    if c.get("file", "").replace("\\", "/") == _norm_fp:
                        target = c
                        break
                if target is None:
                    # file_path provided but no match — fall back to first
                    target = candidates_list[0]
            else:
                target = candidates_list[0]

            target_file = target.get("file", "").replace("\\", "/")
            ambiguous = len(candidates_list) > 1
            disambiguate = ambiguous and file_path is not None

            # Resolve which definition of `name` is in scope at child_file by
            # consulting the symbol-import map. Returns the source file path
            # that the child imports `name` from, or None if untracked.
            sym_imports_idx = getattr(self, "_file_symbol_imports", {}) or {}

            def _resolved_base_file(child_file: str, base_name: str) -> str | None:
                cf = child_file.replace("\\", "/")
                imports = sym_imports_idx.get(cf) or sym_imports_idx.get(child_file, {})
                for source, names in imports.items():
                    if base_name in names:
                        return source.replace("\\", "/")
                return None

            # Build reverse map: base_name -> [child_class_dicts]
            # When disambiguating, only include children whose imports
            # resolve `class_name` to target_file (or are untracked).
            children_of: dict[str, list[dict]] = {}
            for cls_dict in all_classes:
                child_file = cls_dict.get("file", "")
                for base in cls_dict.get("base_classes", []):
                    base_name = str(base).split(".")[-1].strip()
                    if not base_name:
                        continue
                    if disambiguate and base_name == class_name:
                        # Skip children whose imports point to a DIFFERENT file
                        rb = _resolved_base_file(child_file, base_name)
                        if rb is not None and rb != target_file:
                            continue
                    children_of.setdefault(base_name, []).append(cls_dict)

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

            result = {
                "class_name": class_name,
                "file": target.get("file", ""),
                "line": target.get("line"),
                "language": target.get("language", ""),
                "direct_subclass_count": direct_count,
                "total_subclass_count": total_count,
                "subclasses": tree,
            }
            if ambiguous:
                result["ambiguous"] = True
                result["candidates"] = [
                    {"file": c.get("file", ""), "line": c.get("line")}
                    for c in candidates_list
                ]
                if not file_path:
                    result["note"] = (
                        f"Multiple classes named '{class_name}' exist "
                        f"({len(candidates_list)} definitions). Pass "
                        "file_path= to disambiguate; current result uses "
                        f"the first match: {target_file}"
                    )
            return result

