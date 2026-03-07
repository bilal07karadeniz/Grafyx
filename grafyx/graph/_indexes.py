"""Index-building methods for CodebaseGraph.

This is the largest and most critical mixin -- it constructs the reverse
indexes that power all query and analysis operations. Called once during
``initialize()`` and again on each ``refresh()``.

IndexBuilderMixin builds and maintains these indexes:

    **Caller index** (``_build_caller_index`` + augmentation passes):
        - ``_caller_index``:       callee_name -> [{name, file, class?, ...}]
        - ``_class_method_names``: class_name -> {method_names}
        - ``_file_class_methods``: file_path -> {class_name: {method_names}}
        - ``_class_defined_in``:   class_name -> {file_paths}

    **Import index** (``_build_import_index``):
        - ``_import_index``:         target_file -> [importer_files]  (reverse)
        - ``_forward_import_index``: source_file -> [imported_files]  (forward)
        - ``_file_symbol_imports``:  importer -> {target_file -> {symbol_names}}

    **Instance index** (``_build_class_instances``):
        - ``_class_instances``: class_name -> [(instance_var_name, file)]

    **External package index** (``_build_external_packages``):
        - ``_external_packages``: set of top-level package names (pip + npm + stdlib)

The caller index is built in five passes:
    1. **Primary pass**: Inverts ``function_calls`` (outgoing) to build the
       reverse caller map. Graph-sitter's ``call_sites`` (incoming) is unreliable,
       so we always use the outgoing direction.
    2. **DI augmentation**: Scans function source for dependency injection
       patterns (``Depends(func)``, ``callback=func``) and adds synthetic
       caller entries for function references passed as arguments.
    3. **Local var type augmentation**: Detects typed local variables and
       parameters (``service: MyService = ...``, ``executor = MyClass(...)``)
       and links ``var.method()`` calls to the correct class. Entries from
       this pass are marked ``_trusted=True`` to bypass import-graph filtering.
    3b. **Class attr type augmentation**: Detects ``self.field = ClassName(...)``
        in class methods and links ``self.field.method()`` calls to the target
        class. Also marked ``_trusted=True``.
    4. **Celery task dispatch**: Detects ``task.delay()`` / ``task.apply_async()``
       patterns by first collecting @task/@shared_task decorated functions,
       then regex-scanning all function sources for dispatch calls.
    5. **Unique-method heuristic**: When ``var.method()`` appears and ``method``
       exists in exactly one class, adds a synthetic caller entry.

Mixin: IndexBuilderMixin
Writes: _caller_index, _class_method_names, _file_class_methods,
        _class_defined_in, _import_index, _forward_import_index,
        _external_packages, _class_instances, _file_symbol_imports
Reads:  self._codebases, self._lock, self._ignore_patterns, self._project_path
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from grafyx.utils import (
    EXTENSION_TO_LANGUAGE,
    safe_get_attr,
    safe_str,
)

logger = logging.getLogger(__name__)


class IndexBuilderMixin:
    """Builds reverse caller, import, and instance indexes from parsed codebases.

    This mixin is called during ``initialize()`` and ``refresh()`` to
    populate all the ``self._*_index`` data structures that power
    ``CallerQueryMixin``, ``SymbolQueryMixin``, and ``AnalysisMixin``.

    Writes to: _caller_index, _class_method_names, _file_class_methods,
    _class_defined_in, _import_index, _forward_import_index,
    _external_packages, _class_instances, _file_symbol_imports
    """

    # ===================================================================
    # Caller Index: who calls what (reverse function_calls)
    # ===================================================================

    def _build_caller_index(self) -> None:
        """Build a reverse caller index from function_calls (Pass 1 of 5).

        Since graph-sitter's ``function_calls`` (outgoing) works reliably
        but ``call_sites`` (incoming) does not, we iterate every function
        once and invert the relationship: if function A calls B, we store
        A as a caller of B.

        Result: ``_caller_index`` maps callee_name -> list of
        ``{name, file, class?, _receivers?}`` dicts.

        Also builds three disambiguation side-indexes consumed by
        ``CallerQueryMixin.get_callers()`` to eliminate false positives:

        - ``_class_method_names``:  class_name -> {method_names}
          Used for Level 1 filtering: if the caller's own class defines
          the same method name, the call is to its own class, not the target.

        - ``_file_class_methods``:  file_path -> {class_name: {method_names}}
          Used for Level 2 filtering: file-level fallback when graph-sitter
          doesn't populate ``parent_class`` on the caller.

        - ``_class_defined_in``:    class_name -> {file_paths}
          Used for Level 3 filtering: builds the import-based allowlist
          of files that could reference a given class.
        """
        index: dict[str, list[dict]] = {}
        class_methods: dict[str, set[str]] = {}
        file_class_meths: dict[str, dict[str, set[str]]] = {}
        class_defined_in: dict[str, set[str]] = {}
        # NOTE: Callers (initialize/refresh) already hold self._lock,
        # but RLock permits re-entrant acquisition for safety.
        with self._lock:
            for lang, codebase in self._codebases.items():
                try:
                    # Index top-level functions
                    for func in codebase.functions:
                        func_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))
                        if self._is_ignored_file_path(func_file):
                            continue
                        self._index_calls_from(func, index)
                    # Index class methods + build class/file method indexes
                    for cls in codebase.classes:
                        cls_name = safe_get_attr(cls, "name", "")
                        cls_file = self.translate_path(
                            str(safe_get_attr(cls, "filepath", ""))
                        )
                        if self._is_ignored_file_path(cls_file):
                            continue
                        # Track which file(s) define each class
                        if cls_name and cls_file:
                            if cls_name not in class_defined_in:
                                class_defined_in[cls_name] = set()
                            class_defined_in[cls_name].add(cls_file)
                        methods = safe_get_attr(cls, "methods", [])
                        if methods:
                            method_names: set[str] = set()
                            for method in methods:
                                self._index_calls_from(method, index)
                                m_name = safe_get_attr(method, "name", "")
                                if m_name:
                                    method_names.add(m_name)
                            if cls_name and method_names:
                                class_methods[cls_name] = method_names
                                if cls_file:
                                    if cls_file not in file_class_meths:
                                        file_class_meths[cls_file] = {}
                                    file_class_meths[cls_file][cls_name] = method_names
                except Exception as e:
                    logger.error("Error building caller index for %s: %s", lang, e)
            self._caller_index = index
            self._class_method_names = class_methods
            self._file_class_methods = file_class_meths
            self._class_defined_in = class_defined_in
        logger.debug("Caller index built: %d entries, %d classes, %d files indexed",
                     len(index), len(class_methods), len(file_class_meths))

        # Second pass: detect DI/callback patterns (function references passed as arguments)
        self._augment_index_with_di_patterns()
        # Third pass: resolve method calls through typed locals/params (FastAPI DI, local instances)
        self._augment_index_with_local_var_types()
        # Pass 3b: resolve method calls through self.field class attribute types
        self._augment_index_with_class_attr_types()
        # Pass 4: detect Celery task dispatch (.delay(), .apply_async())
        self._augment_index_with_celery_tasks()
        # Pass 5: infer method targets from unique method names
        self._augment_index_with_unique_method_calls()

    # --- Pass 2: Dependency Injection / Callback Pattern Augmentation ---

    def _augment_index_with_di_patterns(self) -> None:
        """Scan function sources for identifiers passed as arguments that
        match known function names (Pass 2 of 3).

        Adds synthetic caller entries for patterns like:
            - ``Depends(get_db_session)``   (FastAPI DI)
            - ``callback=on_complete``      (callback keyword arg)
            - ``router.include(my_handler)`` (positional arg)

        Must be called AFTER ``_caller_index`` is fully built from
        ``function_calls`` (Pass 1), since it validates references against
        the set of known function names to avoid false positives from
        variable names that happen to match.
        """
        # Build set of all known function/method names for validation
        known_functions: set[str] = set()
        for _lang, codebase in self._codebases.items():
            try:
                for func in codebase.functions:
                    f_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))
                    if self._is_ignored_file_path(f_file):
                        continue
                    name = safe_get_attr(func, "name", "")
                    if name:
                        known_functions.add(name)
                for cls in codebase.classes:
                    cls_file = self.translate_path(str(safe_get_attr(cls, "filepath", "")))
                    if self._is_ignored_file_path(cls_file):
                        continue
                    methods = safe_get_attr(cls, "methods", [])
                    if methods:
                        for method in methods:
                            name = safe_get_attr(method, "name", "")
                            if name:
                                known_functions.add(name)
            except Exception:
                continue

        # Regex matches two patterns for function references passed as args:
        # Group 1: positional arg -- Depends(func_name, ...) or register(handler)
        # Group 2: keyword arg   -- callback=func_name, or on_success=handler)
        arg_pattern = re.compile(
            r'(?:'
            r'\w+\s*\(\s*(\w+)\s*[,)]'
            r'|'
            r'=\s*(\w+)\s*[,)\n]'
            r')'
        )

        # Built-in names that should never be treated as function references
        skip_names = {"self", "cls", "None", "True", "False", "super",
                       "int", "str", "float", "bool", "list", "dict",
                       "set", "tuple", "type", "object", "bytes"}

        additions = 0
        for _lang, codebase in self._codebases.items():
            try:
                for func in codebase.functions:
                    f_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))
                    if self._is_ignored_file_path(f_file):
                        continue
                    additions += self._scan_di_refs(
                        func, None, known_functions, arg_pattern, skip_names,
                    )
                for cls in codebase.classes:
                    cls_file = self.translate_path(str(safe_get_attr(cls, "filepath", "")))
                    if self._is_ignored_file_path(cls_file):
                        continue
                    cls_name = safe_get_attr(cls, "name", "")
                    methods = safe_get_attr(cls, "methods", [])
                    if methods:
                        for method in methods:
                            additions += self._scan_di_refs(
                                method, cls_name, known_functions,
                                arg_pattern, skip_names,
                            )
            except Exception as e:
                logger.debug("Error scanning DI patterns for %s: %s", _lang, e)
        if additions:
            logger.debug("DI pattern scan added %d caller entries", additions)

    def _scan_di_refs(
        self,
        func: Any,
        caller_class: str | None,
        known_functions: set[str],
        pattern: re.Pattern,
        skip_names: set[str],
    ) -> int:
        """Scan a single function's source for DI/callback references.

        For each identifier found in argument position that matches a known
        function name, adds a synthetic caller entry to ``_caller_index``.
        Returns the number of entries added.
        """
        caller_name = safe_get_attr(func, "name", "")
        if not caller_name:
            return 0
        func_source = safe_str(safe_get_attr(func, "source", ""))
        if not func_source:
            return 0
        caller_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))

        additions = 0
        for match in pattern.finditer(func_source):
            ref_name = match.group(1) or match.group(2)
            if not ref_name or ref_name == caller_name:
                continue
            if ref_name in skip_names:
                continue
            # Must be a known function/method name in the project
            if ref_name not in known_functions:
                continue

            if ref_name not in self._caller_index:
                self._caller_index[ref_name] = []
            entry: dict[str, Any] = {
                "name": caller_name,
                "file": caller_file,
            }
            if caller_class:
                entry["class"] = caller_class
            if not any(
                e["name"] == caller_name and e["file"] == caller_file
                for e in self._caller_index[ref_name]
            ):
                self._caller_index[ref_name].append(entry)
                additions += 1
        return additions

    # --- Pass 3: Local Variable Type Analysis Augmentation ---

    def _augment_index_with_local_var_types(self) -> None:
        """Scan function bodies for local variable type assignments (Pass 3 of 3).

        Detects three patterns that reveal the type of a local variable:
            - ``param: ClassName = ...``  (typed parameter, incl. FastAPI Depends)
            - ``var = ClassName(...)``     (local instantiation)
            - ``var: ClassName = ...``     (typed local variable)

        When the function body calls ``var.method()``, adds a synthetic caller
        entry for ``ClassName.method`` in ``_caller_index`` with
        ``_trusted=True``. Trusted entries bypass import-graph filtering in
        ``get_callers()`` since we already know the exact target class.

        This resolves false positives in dead code detection for:
            - FastAPI DI: ``service: EddyService = Depends(get_eddy_service)``
            - Local instantiation: ``executor = ToolExecutor(args)``
        """
        all_class_names: set[str] = set(self._class_method_names.keys())
        if not all_class_names:
            return

        # Regex: typed param/local -- matches "word: ClassName =" or "word: ClassName,"
        # or "word: ClassName)". The ClassName must start with uppercase (convention
        # for class names in both Python and TS/JS).
        typed_re = re.compile(r'\b(\w+)\s*:\s*([A-Z]\w*)(?:\s*=|\s*[,\)])')
        # Regex: local instantiation -- matches indented "var = ClassName(" lines.
        # Requires leading whitespace to avoid matching module-level assignments
        # (those are handled by _build_class_instances instead).
        assign_re = re.compile(r'^[ \t]+(\w+)\s*=\s*([A-Z]\w*)\s*\(', re.MULTILINE)
        # Regex: method call -- matches "var.method(" to find calls on typed locals
        method_call_re = re.compile(r'\b(\w+)\.(\w+)\s*\(')
        skip_vars = {"self", "cls"}

        additions = 0
        for _lang, codebase in self._codebases.items():
            try:
                for func in codebase.functions:
                    f_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))
                    if self._is_ignored_file_path(f_file):
                        continue
                    additions += self._scan_local_var_types(
                        func, None, all_class_names,
                        typed_re, assign_re, method_call_re, skip_vars,
                    )
                for cls in codebase.classes:
                    cls_file = self.translate_path(str(safe_get_attr(cls, "filepath", "")))
                    if self._is_ignored_file_path(cls_file):
                        continue
                    cls_name = safe_get_attr(cls, "name", "")
                    for method in safe_get_attr(cls, "methods", []):
                        additions += self._scan_local_var_types(
                            method, cls_name, all_class_names,
                            typed_re, assign_re, method_call_re, skip_vars,
                        )
            except Exception as e:
                logger.debug("Error scanning local var types for %s: %s", _lang, e)
        if additions:
            logger.debug("Local var type scan added %d caller entries", additions)

    def _scan_local_var_types(
        self,
        func: Any,
        caller_class: str | None,
        all_class_names: set[str],
        typed_re: re.Pattern,
        assign_re: re.Pattern,
        method_call_re: re.Pattern,
        skip_vars: set[str],
    ) -> int:
        """Scan a single function for local variable type assignments.

        Builds a ``var_name -> ClassName`` map from type annotations and
        instantiations, then matches ``var.method()`` calls to add
        ``_trusted`` caller entries. Returns the number of entries added.
        """
        caller_name = safe_get_attr(func, "name", "")
        if not caller_name:
            return 0
        source = safe_str(safe_get_attr(func, "source", ""))
        if not source:
            return 0
        caller_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))

        # Build var -> class_name mapping from this function's source
        var_types: dict[str, str] = {}

        # 1. Typed params / typed locals: var: ClassName = / var: ClassName,
        for m in typed_re.finditer(source):
            var_name, cls_name = m.group(1), m.group(2)
            if var_name not in skip_vars and cls_name in all_class_names:
                var_types[var_name] = cls_name

        # 2. Local instantiation: var = ClassName(
        for m in assign_re.finditer(source):
            var_name, cls_name = m.group(1), m.group(2)
            if var_name not in skip_vars and cls_name in all_class_names:
                var_types[var_name] = cls_name

        if not var_types:
            return 0

        # Scan method calls var.method( and link to ClassName.method
        additions = 0
        for m in method_call_re.finditer(source):
            var_name, method_name = m.group(1), m.group(2)
            cls_name = var_types.get(var_name)
            if not cls_name:
                continue
            if method_name not in self._class_method_names.get(cls_name, set()):
                continue
            if method_name not in self._caller_index:
                self._caller_index[method_name] = []
            # Mark as trusted: type-resolved, bypasses import-graph filtering
            entry: dict[str, Any] = {"name": caller_name, "file": caller_file, "_trusted": True}
            if caller_class:
                entry["class"] = caller_class
            if not any(
                e["name"] == caller_name and e["file"] == caller_file
                for e in self._caller_index[method_name]
            ):
                self._caller_index[method_name].append(entry)
                additions += 1
        return additions

    # --- Pass 3b: Class Attribute Type Analysis Augmentation ---

    def _augment_index_with_class_attr_types(self) -> None:
        """Resolve method calls through ``self.field`` class attribute types (Pass 3b).

        Scans class methods for two patterns:

        **Phase A** — Build ``field_types`` mapping per class:
            - ``self.field = ClassName(...)``  (assignment in ``__init__`` or other method)
            - ``self.field: ClassName = ...``  (typed assignment)

        **Phase B** — Link ``self.field.method()`` calls to the target class:
            When ``field`` has a known type and ``method`` exists in that type's
            ``_class_method_names``, adds a ``_trusted=True`` caller entry.

        This resolves false positives in dead code detection for methods called
        through class attributes, e.g.:
            - ``self.executor = ToolExecutor(config)`` in ``__init__``
            - ``self.executor.execute(tools)`` in another method
        """
        all_class_names: set[str] = set(self._class_method_names.keys())
        if not all_class_names:
            return

        # Regex: self.field = ClassName( or self.field: ClassName = ...
        # Captures (field_name, ClassName) where ClassName starts with uppercase
        attr_assign_re = re.compile(
            r'self\.(\w+)\s*(?::\s*\w[\w\[\], |]*\s*)?=\s*([A-Z]\w*)\s*\('
        )
        attr_typed_re = re.compile(
            r'self\.(\w+)\s*:\s*([A-Z]\w*)\s*='
        )
        # Regex: self.field.method(
        attr_call_re = re.compile(r'self\.(\w+)\.(\w+)\s*\(')

        additions = 0
        for _lang, codebase in self._codebases.items():
            try:
                for cls in codebase.classes:
                    cls_name = safe_get_attr(cls, "name", "")
                    cls_file = self.translate_path(
                        str(safe_get_attr(cls, "filepath", ""))
                    )
                    if self._is_ignored_file_path(cls_file):
                        continue
                    methods = safe_get_attr(cls, "methods", [])
                    if not methods:
                        continue

                    # Phase A: scan ALL methods for self.field type assignments
                    field_types: dict[str, str] = {}
                    for method in methods:
                        source = safe_str(safe_get_attr(method, "source", ""))
                        if not source:
                            continue
                        for m in attr_assign_re.finditer(source):
                            field_name, type_name = m.group(1), m.group(2)
                            if type_name in all_class_names:
                                field_types[field_name] = type_name
                        for m in attr_typed_re.finditer(source):
                            field_name, type_name = m.group(1), m.group(2)
                            if type_name in all_class_names:
                                field_types[field_name] = type_name

                    if not field_types:
                        continue

                    # Phase B: scan ALL methods for self.field.method() calls
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

                        for m in attr_call_re.finditer(source):
                            field_name, method_name = m.group(1), m.group(2)
                            target_cls = field_types.get(field_name)
                            if not target_cls:
                                continue
                            if method_name not in self._class_method_names.get(target_cls, set()):
                                continue
                            if method_name not in self._caller_index:
                                self._caller_index[method_name] = []
                            entry: dict[str, Any] = {
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

    # --- Pass 4: Celery Task Dispatch Detection ---

    def _augment_index_with_celery_tasks(self) -> None:
        """Detect Celery task invocations via .delay() and .apply_async() (Pass 4).

        Celery tasks are called through a dynamic dispatch registry:
            my_task.delay(args)        -> equivalent to calling my_task()
            my_task.apply_async(args)  -> equivalent to calling my_task()
            my_task.s(args)            -> creates a signature (lazy invocation)
            my_task.si(args)           -> creates an immutable signature

        Algorithm:
            1. Collect all @task/@shared_task/@periodic_task decorated functions.
            2. Build a regex matching task_name.(delay|apply_async|s|si)(
            3. Scan all function/method sources for matches.
            4. Add synthetic caller entries linking the caller to the task.
        """
        task_decorators = {"task", "shared_task", "periodic_task"}
        known_tasks: dict[str, str] = {}  # task_name -> file_path

        for _lang, codebase in self._codebases.items():
            try:
                for func in codebase.functions:
                    f_name = safe_get_attr(func, "name", "")
                    if not f_name:
                        continue
                    f_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))
                    if self._is_ignored_file_path(f_file):
                        continue
                    decorators = safe_get_attr(func, "decorators", [])
                    for d in decorators:
                        d_str = safe_str(d).strip("@").split("(")[0].split(".")[-1]
                        if d_str in task_decorators:
                            known_tasks[f_name] = f_file
                            break
            except Exception:
                continue

        if not known_tasks:
            return

        celery_call_re = re.compile(
            r'\b(' + '|'.join(re.escape(t) for t in known_tasks)
            + r')\.(delay|apply_async|s|si)\s*\('
        )

        additions = 0
        for _lang, codebase in self._codebases.items():
            try:
                for func in codebase.functions:
                    additions += self._scan_celery_calls(
                        func, None, celery_call_re, known_tasks,
                    )
                for cls in codebase.classes:
                    cls_name_str = safe_get_attr(cls, "name", "")
                    cls_file = self.translate_path(str(safe_get_attr(cls, "filepath", "")))
                    if self._is_ignored_file_path(cls_file):
                        continue
                    for method in safe_get_attr(cls, "methods", []):
                        additions += self._scan_celery_calls(
                            method, cls_name_str, celery_call_re, known_tasks,
                        )
            except Exception as e:
                logger.debug("Error scanning Celery tasks for %s: %s", _lang, e)
        if additions:
            logger.debug("Celery task scan added %d caller entries", additions)

    def _scan_celery_calls(
        self,
        func: Any,
        caller_class: str | None,
        celery_call_re: re.Pattern,
        known_tasks: dict[str, str],
    ) -> int:
        """Scan a single function's source for Celery dispatch patterns."""
        caller_name = safe_get_attr(func, "name", "")
        if not caller_name:
            return 0
        caller_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))
        if self._is_ignored_file_path(caller_file):
            return 0
        source = safe_str(safe_get_attr(func, "source", ""))
        if not source:
            return 0

        additions = 0
        for match in celery_call_re.finditer(source):
            task_name = match.group(1)
            if task_name == caller_name:
                continue
            if task_name not in self._caller_index:
                self._caller_index[task_name] = []
            entry: dict[str, Any] = {
                "name": caller_name,
                "file": caller_file,
                "_trusted": True,
            }
            if caller_class:
                entry["class"] = caller_class
            if not any(
                e["name"] == caller_name and e["file"] == caller_file
                for e in self._caller_index[task_name]
            ):
                self._caller_index[task_name].append(entry)
                additions += 1
        return additions

    # --- Pass 5: Unique-Method Heuristic for Untyped Instance Calls ---

    def _augment_index_with_unique_method_calls(self) -> None:
        """Infer method call targets from unique method names (Pass 5).

        When a function calls ``var.method()`` and ``method`` is defined in
        exactly ONE class across the entire project, we can safely assume the
        call targets that class's method. This is the same logic as
        ``get_callers()`` Level 4 but applied proactively during index building.

        Safety guards:
            - Only methods that exist in exactly 1 class are eligible.
            - Dunder methods (__init__, etc.) are always skipped.
            - Common variable names (self, cls, os, etc.) are skipped.
            - Self-calls (caller's class == target class) are skipped.
        """
        # Build unique method -> class mapping
        method_to_classes: dict[str, list[str]] = {}
        for cls_name, methods in self._class_method_names.items():
            for m in methods:
                if m.startswith("__"):
                    continue
                if m not in method_to_classes:
                    method_to_classes[m] = []
                method_to_classes[m].append(cls_name)

        # Keep only truly unique methods (defined in exactly 1 class)
        unique_methods: dict[str, str] = {
            m: classes[0]
            for m, classes in method_to_classes.items()
            if len(classes) == 1
        }
        if not unique_methods:
            return

        method_call_re = re.compile(r'\b(\w+)\.(\w+)\s*\(')
        skip_vars = {
            "self", "cls", "super", "os", "sys", "re", "json", "math",
            "logging", "logger", "log", "print", "str", "int", "float",
            "list", "dict", "set", "tuple", "type", "object", "path",
            "Path", "datetime", "date", "time", "uuid",
        }

        additions = 0
        for _lang, codebase in self._codebases.items():
            try:
                for func in codebase.functions:
                    additions += self._scan_unique_method_calls(
                        func, None, unique_methods, method_call_re, skip_vars,
                    )
                for cls in codebase.classes:
                    cls_name = safe_get_attr(cls, "name", "")
                    cls_file = self.translate_path(
                        str(safe_get_attr(cls, "filepath", ""))
                    )
                    if self._is_ignored_file_path(cls_file):
                        continue
                    for method in safe_get_attr(cls, "methods", []):
                        additions += self._scan_unique_method_calls(
                            method, cls_name, unique_methods,
                            method_call_re, skip_vars,
                        )
            except Exception as e:
                logger.debug("Error scanning unique method calls for %s: %s", _lang, e)
        if additions:
            logger.debug("Unique method call scan added %d caller entries", additions)

    def _scan_unique_method_calls(
        self,
        func: Any,
        caller_class: str | None,
        unique_methods: dict[str, str],
        method_call_re: re.Pattern,
        skip_vars: set[str],
    ) -> int:
        """Scan a single function for calls to unique methods on untyped vars."""
        caller_name = safe_get_attr(func, "name", "")
        if not caller_name:
            return 0
        caller_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))
        if self._is_ignored_file_path(caller_file):
            return 0
        source = safe_str(safe_get_attr(func, "source", ""))
        if not source:
            return 0

        additions = 0
        for m in method_call_re.finditer(source):
            var_name, method_name = m.group(1), m.group(2)
            if var_name in skip_vars:
                continue
            target_cls = unique_methods.get(method_name)
            if not target_cls:
                continue
            # Skip self-calls: caller's own class IS the target class
            if caller_class == target_cls:
                continue
            if method_name not in self._caller_index:
                self._caller_index[method_name] = []
            entry: dict[str, Any] = {
                "name": caller_name,
                "file": caller_file,
                "_trusted": True,
            }
            if caller_class:
                entry["class"] = caller_class
            if not any(
                e["name"] == caller_name and e["file"] == caller_file
                for e in self._caller_index[method_name]
            ):
                self._caller_index[method_name].append(entry)
                additions += 1
        return additions

    # --- Primary Caller Index Helper ---

    def _index_calls_from(self, func: Any, index: dict[str, list[dict]]) -> None:
        """Add all outgoing calls from a single function to the reverse index.

        For each callee in ``func.function_calls``, creates a caller entry
        with the caller's name, file, optional class, and receiver expressions.

        Receiver extraction (``_receivers`` field) is done via regex on the
        function source. For a call like ``self.episodic.add(item)``, the
        receiver ``self.episodic`` is stored alongside the callee ``add``.
        ``CallerQueryMixin.get_callers()`` uses receivers for Level 4
        attribute-based disambiguation.
        """
        caller_name = safe_get_attr(func, "name", "")
        if not caller_name:
            return
        caller_file = self.translate_path(str(safe_get_attr(func, "filepath", "")))

        # Get the class name if this is a method
        parent_class = safe_get_attr(func, "parent_class", None)
        caller_class = safe_get_attr(parent_class, "name", None) if parent_class else None

        function_calls = safe_get_attr(func, "function_calls", [])
        if not function_calls:
            return

        # Pre-extract method call receivers from source for Level 4 filtering.
        # e.g., "self.episodic.add(" -> receiver "self.episodic" for method "add".
        # The regex captures dotted chains like "self.db.session" as receivers
        # and the final ".method(" as the callee. This lets get_callers() check
        # whether the receiver name matches the target class name.
        func_source = safe_str(safe_get_attr(func, "source", ""))
        receivers_by_method: dict[str, set[str]] = {}
        if func_source:
            for match in re.finditer(r'(\w+(?:\.\w+)*)\.(\w+)\s*\(', func_source):
                receiver = match.group(1)
                method = match.group(2)
                if method not in receivers_by_method:
                    receivers_by_method[method] = set()
                receivers_by_method[method].add(receiver)

        for called in function_calls:
            callee_name = safe_get_attr(called, "name", "")
            if not callee_name:
                continue
            if callee_name not in index:
                index[callee_name] = []
            entry: dict[str, Any] = {
                "name": caller_name,
                "file": caller_file,
            }
            if caller_class:
                entry["class"] = caller_class
            # Store receivers for Level 4 attribute disambiguation
            call_receivers = receivers_by_method.get(callee_name)
            if call_receivers:
                entry["_receivers"] = call_receivers
            # Avoid duplicate entries (same caller name + file)
            if not any(
                e["name"] == caller_name and e["file"] == caller_file
                for e in index[callee_name]
            ):
                index[callee_name].append(entry)

    # ===================================================================
    # Import Index: which files import which (bidirectional)
    # ===================================================================

    def _build_import_index(self) -> None:
        """Build bidirectional import indexes: reverse and forward.

        Uses a two-phase approach:
            **Phase 1**: Build a suffix -> full_path lookup table from all known
            files. This allows efficient resolution of import module paths
            (e.g., ``from app.services.auth import ...``) to actual file paths
            without requiring exact directory prefix matching.

            **Phase 2**: For each file, extract imports, resolve the module name
            to a file path using suffix matching, and populate three indexes:
            - ``_import_index``: target_file -> [importer_files] (reverse)
            - ``_forward_import_index``: source_file -> [imported_files] (forward)
            - ``_file_symbol_imports``: importer -> {target -> {symbol_names}}

        The forward index powers ``find_import_cycles()`` (DFS for cycles)
        and ``get_module_dependencies()`` (edge aggregation). The reverse
        index powers ``get_importers()`` and Level 3 caller filtering.
        """
        index: dict[str, list[str]] = {}
        forward: dict[str, list[str]] = {}
        top_level_forward: dict[str, list[str]] = {}
        symbol_imports: dict[str, dict[str, set[str]]] = {}
        with self._lock:
            # Phase 1: Build suffix -> full_path lookup for O(1) resolution.
            # We prefer "path" over "filepath" because in some graph-sitter
            # versions "filepath" can carry a stale or incorrect directory
            # prefix while "path" is always the canonical source.
            suffix_to_path: dict[str, list[str]] = {}
            for lang, codebase in self._codebases.items():
                try:
                    for f in codebase.files:
                        fp = self.translate_path(
                            str(safe_get_attr(f, "path", safe_get_attr(f, "filepath", "")))
                        ).replace("\\", "/")
                        if not fp:
                            continue
                        if self._is_ignored_file_path(fp):
                            continue
                        # Index by every possible suffix of the path so that
                        # an import like "app.services.auth" can match the file
                        # at ".../myproject/app/services/auth.py" via the
                        # suffix "app/services/auth.py".
                        parts = fp.split("/")
                        for i in range(len(parts)):
                            suffix = "/".join(parts[i:])
                            if suffix not in suffix_to_path:
                                suffix_to_path[suffix] = [fp]
                            else:
                                suffix_to_path[suffix].append(fp)
                except Exception:
                    continue

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

            # Build function line ranges per file for scope detection.
            # Used to distinguish top-level imports from function-body
            # (lazy) imports. Lazy imports should not create module-level
            # dependency edges.
            file_func_ranges: dict[str, list[tuple[int, int]]] = {}
            for _lang2, _cb2 in self._codebases.items():
                try:
                    for _func in _cb2.functions:
                        _fp = self.translate_path(
                            str(safe_get_attr(_func, "filepath", ""))
                        ).replace("\\", "/")
                        if not _fp:
                            continue
                        _start = self._extract_line(_func)
                        _src = safe_str(safe_get_attr(_func, "source", ""))
                        if _start and _src:
                            _end = _start + _src.count("\n")
                            if _fp not in file_func_ranges:
                                file_func_ranges[_fp] = []
                            file_func_ranges[_fp].append((_start, _end))
                    for _cls in _cb2.classes:
                        for _meth in safe_get_attr(_cls, "methods", []):
                            _fp = self.translate_path(
                                str(safe_get_attr(_meth, "filepath", ""))
                            ).replace("\\", "/")
                            if not _fp:
                                continue
                            _start = self._extract_line(_meth)
                            _src = safe_str(safe_get_attr(_meth, "source", ""))
                            if _start and _src:
                                _end = _start + _src.count("\n")
                                if _fp not in file_func_ranges:
                                    file_func_ranges[_fp] = []
                                file_func_ranges[_fp].append((_start, _end))
                except Exception:
                    continue

            # Phase 2: for each file, resolve imports to known files
            for lang, codebase in self._codebases.items():
                try:
                    for f in codebase.files:
                        fpath = self.translate_path(
                            str(safe_get_attr(f, "path", safe_get_attr(f, "filepath", "")))
                        ).replace("\\", "/")
                        if not fpath:
                            continue
                        if self._is_ignored_file_path(fpath):
                            continue

                        imports = safe_get_attr(f, "imports", [])
                        if not imports:
                            continue

                        for imp in imports:
                            imp_source = safe_str(safe_get_attr(imp, "source", ""))
                            if not imp_source:
                                imp_source = safe_str(imp)
                            if not imp_source:
                                continue

                            module = self._extract_module_from_import(imp_source)
                            if not module:
                                continue

                            # Skip known external packages (pip + stdlib)
                            top_level = module.split(".")[0].replace("-", "_").lower()
                            if top_level in self._external_packages:
                                continue

                            module_as_path = module.replace(".", "/")

                            # Determine source file language from extension
                            src_ext = ""
                            if "." in fpath:
                                src_ext = "." + fpath.rsplit(".", 1)[-1]
                            src_lang = EXTENSION_TO_LANGUAGE.get(src_ext, "")

                            # Generate candidate file paths from the module name.
                            # Each candidate is tagged with its language so we can
                            # try same-language matches first (e.g., a .py file
                            # importing "app.utils" should prefer "app/utils.py"
                            # over "app/utils.ts").
                            all_candidates = [
                                (module_as_path + ".py", "python"),
                                (module_as_path + "/__init__.py", "python"),
                                (module_as_path + ".ts", "typescript"),
                                (module_as_path + ".tsx", "typescript"),
                                (module_as_path + "/index.ts", "typescript"),
                                (module_as_path + "/index.tsx", "typescript"),
                                (module_as_path + ".js", "javascript"),
                                (module_as_path + ".jsx", "javascript"),
                                (module_as_path + "/index.js", "javascript"),
                                (module_as_path + "/index.jsx", "javascript"),
                            ]

                            # Try same-language candidates first, then cross-language
                            same_lang = [c for c, cl in all_candidates if not src_lang or cl == src_lang]

                            target = None
                            for candidate in same_lang:
                                matches = suffix_to_path.get(candidate)
                                if matches:
                                    target = _pick_closest(matches, fpath)
                                    break
                            if target is None:
                                # Cross-language fallback: only within the same
                                # language family (JS <-> TS is fine, Python <-> TS
                                # is almost always a false positive from coincidental
                                # module name overlap like "config" or "utils").
                                _WEB_LANGS = {"typescript", "javascript"}
                                src_family = "web" if src_lang in _WEB_LANGS else src_lang
                                for cand, cand_lang in all_candidates:
                                    if not src_lang or cand_lang == src_lang:
                                        continue  # Already tried in same_lang pass
                                    cand_family = "web" if cand_lang in _WEB_LANGS else cand_lang
                                    if src_family != cand_family:
                                        continue  # Skip cross-family resolution
                                    matches = suffix_to_path.get(cand)
                                    if matches:
                                        target = _pick_closest(matches, fpath)
                                        break

                            if target:
                                # Skip __init__.py importing its own sub-packages
                                # e.g., mypackage/__init__.py -> mypackage/sub.py
                                if fpath.endswith("/__init__.py"):
                                    init_dir = fpath[: -len("/__init__.py")]
                                    if target.startswith(init_dir + "/"):
                                        continue

                                if target not in index:
                                    index[target] = []
                                if fpath not in index[target]:
                                    index[target].append(fpath)
                                # Forward direction: this file imports target
                                if fpath not in forward:
                                    forward[fpath] = []
                                if target not in forward[fpath]:
                                    forward[fpath].append(target)
                                # Track whether this import is at top-level
                                # or inside a function body (lazy import).
                                imp_line = self._extract_line(imp)
                                is_top_level = True
                                if imp_line and fpath in file_func_ranges:
                                    for _fstart, _fend in file_func_ranges[fpath]:
                                        if _fstart <= imp_line <= _fend:
                                            is_top_level = False
                                            break
                                if is_top_level:
                                    if fpath not in top_level_forward:
                                        top_level_forward[fpath] = []
                                    if target not in top_level_forward[fpath]:
                                        top_level_forward[fpath].append(target)
                                # Symbol-level: track which names this file imports from target
                                sym_names = self._extract_symbol_names_from_import(imp_source)
                                if sym_names:
                                    if fpath not in symbol_imports:
                                        symbol_imports[fpath] = {}
                                    if target not in symbol_imports[fpath]:
                                        symbol_imports[fpath][target] = set()
                                    symbol_imports[fpath][target].update(sym_names)
                except Exception as e:
                    logger.error("Error building import index for %s: %s", lang, e)
            self._import_index = index
            self._forward_import_index = forward
            self._top_level_forward_import_index = top_level_forward
            self._file_symbol_imports = symbol_imports
        logger.debug("Import index built: %d reverse, %d forward, %d symbol-level entries",
                     len(index), len(forward), len(symbol_imports))

    # --- Import Statement Parsing Helpers ---

    @staticmethod
    def _extract_module_from_import(imp_str: str) -> str:
        """Extract the module path from an import statement string.

        Handles both Python and TypeScript/JavaScript import syntax.

        Examples:
            Python:  'from foo.bar import Baz'        -> 'foo.bar'
            Python:  'import foo.bar'                  -> 'foo.bar'
            Python:  'from .utils import helper'       -> '' (relative, skip)
            TS/JS:   "import { Foo } from './bar'"     -> 'bar'
            TS/JS:   "import Foo from './bar'"         -> 'bar'
            TS/JS:   "import * as Foo from 'bar'"      -> 'bar'
            TS/JS:   "export { Foo } from './bar'"     -> 'bar'
        """
        imp_str = imp_str.strip()

        # TS/JS: import/export ... from 'module' or "module".
        # Must check before Python handling because TS/JS also has " from "
        # but with quotes around the module path.
        if " from " in imp_str and ("'" in imp_str or '"' in imp_str):
            from_part = imp_str.split(" from ")[-1].strip()
            raw_module = from_part.strip("'\"`;, ")
            if raw_module:
                # Track whether original had a relative prefix
                was_relative = raw_module.startswith("./") or raw_module.startswith("../")
                # Strip relative path prefixes (resolvable via suffix matching)
                module = raw_module
                while module.startswith("../"):
                    module = module[3:]
                if module.startswith("./"):
                    module = module[2:]
                # Bare specifiers (no ./ or ../ prefix) are npm packages, not
                # local files. E.g., 'sonner', 'react', 'next/router'.
                # Scoped packages (@scope/pkg) are also external but start
                # with '@', so we let them through the prefix check.
                if not was_relative and not module.startswith("@"):
                    return ""
                # Strip file extensions if already present
                for ext in (".ts", ".tsx", ".js", ".jsx"):
                    if module.endswith(ext):
                        module = module[: -len(ext)]
                        break
                return module

        # TS/JS: require('module')
        if "require(" in imp_str:
            try:
                start = imp_str.index("require(") + 8
                end = imp_str.index(")", start)
                raw_module = imp_str[start:end].strip("'\"")
                was_relative = raw_module.startswith("./") or raw_module.startswith("../")
                module = raw_module
                while module.startswith("../"):
                    module = module[3:]
                if module.startswith("./"):
                    module = module[2:]
                # Bare specifiers are npm packages -- skip
                if not was_relative:
                    return ""
                return module
            except ValueError:
                pass

        # Python: skip relative imports
        if imp_str.startswith("from .") or imp_str.startswith("."):
            return ""
        # Python: from foo.bar import Baz
        if imp_str.startswith("from "):
            parts = imp_str[5:].split(" import ")
            if parts:
                return parts[0].strip()
        # Python: import foo.bar
        if imp_str.startswith("import "):
            module = imp_str[7:].split(" as ")[0].split(",")[0].strip()
            return module
        return ""

    @staticmethod
    def _extract_symbol_names_from_import(imp_str: str) -> set[str]:
        """Extract the specific symbol names imported from a single import statement.

        Returns the set of symbol names (including aliases), or an empty set
        for wildcard / bare imports. When an alias is present, both the
        original name and the alias are returned.

        Examples:
            'from foo import A, B, C'              -> {'A', 'B', 'C'}
            'from foo import A as X'               -> {'A', 'X'}
            'import foo'                            -> set()  (bare import)
            "import { A, B } from './foo'"          -> {'A', 'B'}
            "import { A as X } from './foo'"        -> {'A', 'X'}
            "import Foo from './bar'"               -> {'Foo'}
            "import * as Foo from './bar'"          -> set()  (wildcard)
        """
        imp_str = imp_str.strip()
        NOISE = {"import", "from", "as", "export", "default", "type",
                 "const", "let", "var", "require", "module"}
        names: set[str] = set()

        # TS/JS named imports: import { A, B } from 'X'
        if "{" in imp_str and "}" in imp_str:
            content = imp_str.split("{", 1)[1].split("}", 1)[0]
            for token in content.split(","):
                parts = token.strip().split(" as ")
                clean = parts[0].strip()
                if clean and clean.isidentifier() and clean not in NOISE:
                    names.add(clean)
                if len(parts) > 1:
                    alias = parts[1].strip()
                    if alias and alias.isidentifier() and alias not in NOISE:
                        names.add(alias)
            return names

        # TS/JS default import: import Foo from './bar'
        if " from " in imp_str and ("'" in imp_str or '"' in imp_str):
            between = imp_str.split(" from ")[0]
            for prefix in ("import ", "export "):
                if between.startswith(prefix):
                    between = between[len(prefix):]
                    break
            between = between.strip()
            if "* as " in between or "*" in between:
                return set()  # Wildcard
            if between.startswith("type "):
                between = between[5:].strip()
            if between and between.isidentifier() and between not in NOISE:
                names.add(between)
            return names

        # Python: from X import A, B, C
        if " import " in imp_str and imp_str.startswith("from "):
            names_part = imp_str.split(" import ", 1)[1]
            if names_part.strip() == "*":
                return set()  # Wildcard
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

        # Python: import foo.bar -- no specific symbols
        return set()

    # ===================================================================
    # Class Instance Index: module-level singleton detection
    # ===================================================================

    def _build_class_instances(self) -> None:
        """Detect module-level singleton patterns like ``cache = MyClass()``.

        Scans each file's raw source (not AST) for unindented lines matching::

            variable_name = ClassName(...)

        where ``ClassName`` is a class defined in the project.  Results are
        stored in ``self._class_instances`` so that the MCP tool
        ``get_class_context`` can search for usages by *instance name* in
        addition to the class name -- e.g., finding ``db.query()`` when the
        user asks about the ``Database`` class.

        Only matches lines at column 0 (no leading whitespace) to capture
        module-level singletons. Local instantiations inside functions are
        handled by ``_augment_index_with_local_var_types`` instead.
        """
        # Collect all known class names from the project
        all_class_names: set[str] = set(self._class_method_names.keys())
        for _lang, codebase in self._codebases.items():
            try:
                for cls in safe_get_attr(codebase, "classes", []):
                    name = safe_get_attr(cls, "name", "")
                    if name:
                        all_class_names.add(name)
            except Exception:
                continue

        if not all_class_names:
            return

        # Regex: matches module-level "var = ClassName(" or "var: Type = ClassName("
        # at column 0 (no indentation). The optional type annotation group
        # handles patterns like "cache: RedisCache = RedisCache(...)".
        # Variable name must start lowercase (convention for instances).
        cls_pattern = re.compile(
            r'^([a-z_][a-z0-9_]*)\s*(?::\s*\w[\w\[\], |]*\s*)?=\s*([A-Z]\w*)\s*\(',
            re.MULTILINE,
        )

        instances: dict[str, list[tuple[str, str]]] = {}
        for _lang, codebase in self._codebases.items():
            try:
                for f in codebase.files:
                    fpath = self.translate_path(
                        str(safe_get_attr(f, "filepath", safe_get_attr(f, "path", "")))
                    )
                    if not fpath:
                        continue
                    try:
                        with open(fpath, "r", errors="replace") as fh:
                            source = fh.read()
                    except (OSError, IOError):
                        continue
                    for m in cls_pattern.finditer(source):
                        var_name = m.group(1)
                        cls_name = m.group(2)
                        if cls_name in all_class_names:
                            if cls_name not in instances:
                                instances[cls_name] = []
                            instances[cls_name].append((var_name, fpath))
            except Exception as e:
                logger.debug("Error scanning class instances in %s: %s", _lang, e)

        self._class_instances = instances
        logger.debug("Class instances index built: %d classes with instances",
                     len(instances))

    # ===================================================================
    # External Package Index: pip/npm/stdlib names to skip
    # ===================================================================

    def _build_external_packages(self) -> None:
        """Scan project manifests for external package names (pip + npm + stdlib).

        Populates ``self._external_packages`` with top-level package names
        so ``_build_import_index`` can skip them -- they're not local files
        and would create false-positive import edges between local modules
        and libraries like ``json``, ``requests``, or ``react``.

        Sources scanned (all optional, skipped if not found):
            - Python stdlib: ``sys.stdlib_module_names`` (Python 3.10+)
            - requirements.txt / requirements-dev.txt
            - pyproject.toml (dependencies + optional-dependencies)
            - setup.cfg
            - package.json (dependencies, devDependencies, peerDependencies)
        """
        pkgs: set[str] = set()
        root = Path(self._project_path)

        # -- Python stdlib --
        # sys.stdlib_module_names available since Python 3.10
        stdlib = getattr(sys, "stdlib_module_names", set())
        pkgs.update(stdlib)

        # -- pip packages from requirements.txt --
        for req_file in ("requirements.txt", "requirements-dev.txt",
                         "requirements_dev.txt"):
            req_path = root / req_file
            if req_path.is_file():
                try:
                    for line in req_path.read_text(errors="ignore").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("-"):
                            continue
                        # "qdrant-client>=1.0" -> "qdrant_client"
                        name = re.split(r"[>=<!\[;@\s]", line)[0].strip()
                        if name:
                            pkgs.add(name.replace("-", "_").lower())
                except Exception:
                    pass

        # -- pip packages from pyproject.toml (dependencies + optional) --
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            try:
                text = pyproject.read_text(errors="ignore")
                # Simple regex extraction of quoted package names -- avoids
                # requiring a toml parsing library. Matches any quoted string
                # that looks like a package specifier (name + optional extras/version).
                for m in re.finditer(
                    r'"([a-zA-Z0-9_-]+)(?:\[.*?\])?(?:[>=<!\s].*?)?"', text
                ):
                    name = m.group(1)
                    if len(name) > 1:
                        pkgs.add(name.replace("-", "_").lower())
            except Exception:
                pass

        # -- pip packages from setup.cfg --
        setup_cfg = root / "setup.cfg"
        if setup_cfg.is_file():
            try:
                for line in setup_cfg.read_text(errors="ignore").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("["):
                        continue
                    name = re.split(r"[>=<!\[;@\s]", line)[0].strip()
                    if name and name[0].isalpha():
                        pkgs.add(name.replace("-", "_").lower())
            except Exception:
                pass

        # -- npm packages from package.json --
        pkg_json = root / "package.json"
        if not pkg_json.is_file():
            # Check common subdirectories
            for sub in ("frontend", "client", "web", "app"):
                candidate = root / sub / "package.json"
                if candidate.is_file():
                    pkg_json = candidate
                    break
        if pkg_json.is_file():
            try:
                data = json.loads(pkg_json.read_text(errors="ignore"))
                for section in ("dependencies", "devDependencies",
                                "peerDependencies"):
                    for name in data.get(section, {}):
                        # "@scope/pkg" -> "pkg", "sonner" -> "sonner"
                        clean = name.split("/")[-1] if "/" in name else name
                        pkgs.add(clean.lower())
            except Exception:
                pass

        self._external_packages = pkgs
        logger.debug("External packages index: %d entries", len(pkgs))

    # ===================================================================
    # Imported Names: all symbols referenced in import statements
    # ===================================================================

    def _build_imported_names(self) -> set[str]:
        """Build set of all symbol names that appear in import statements.

        Used by dead code detection to exclude functions and classes that
        are explicitly imported somewhere in the codebase. If a symbol
        appears in an import statement, it's likely used (even if we can't
        trace the call chain through the caller index).

        Handles both Python and TS/JS syntax:
            Python:  ``from X import A, B, C``  -> {A, B, C}
            TS/JS:   ``import { A, B } from 'X'``  -> {A, B}
            TS/JS:   ``import A from 'X'``  -> {A}
            TS/JS:   ``import * as A from 'X'``  -> {A}
            TS/JS:   ``export { A } from 'X'``  -> {A}
        """
        NOISE = {
            "import", "from", "as", "export", "default",
            "type", "const", "let", "var", "require", "module",
        }
        names: set[str] = set()
        for _lang, codebase in self._codebases.items():
            try:
                for f in codebase.files:
                    for imp in safe_get_attr(f, "imports", []):
                        imp_str = safe_str(safe_get_attr(imp, "source", ""))
                        if not imp_str:
                            imp_str = safe_str(imp)
                        if not imp_str:
                            continue

                        # TS/JS named imports: import { A, B } from 'X'
                        # Also handles: export { A, B } from 'X'
                        if "{" in imp_str and "}" in imp_str:
                            content = imp_str.split("{", 1)[1].split("}", 1)[0]
                            for token in content.split(","):
                                parts = token.strip().split(" as ")
                                clean = parts[0].strip()
                                if clean and clean.isidentifier() and clean not in NOISE:
                                    names.add(clean)
                                if len(parts) > 1:
                                    alias = parts[1].strip()
                                    if alias and alias.isidentifier() and alias not in NOISE:
                                        names.add(alias)
                            continue

                        # TS/JS default/namespace: import X from 'Y'
                        # or: import * as X from 'Y'
                        if " from " in imp_str and ("'" in imp_str or '"' in imp_str):
                            between = imp_str.split(" from ")[0]
                            # Strip "import " or "export " prefix
                            for prefix in ("import ", "export "):
                                if between.startswith(prefix):
                                    between = between[len(prefix):]
                                    break
                            between = between.strip()
                            if "* as " in between:
                                between = between.split("* as ")[-1].strip()
                            # Could be "type X" in TS
                            if between.startswith("type "):
                                between = between[5:].strip()
                            if between and between.isidentifier() and between not in NOISE:
                                names.add(between)
                            continue

                        # Python: from X import A, B, C
                        if " import " in imp_str:
                            names_part = imp_str.split(" import ", 1)[1]
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
            except Exception:
                continue
        return names
