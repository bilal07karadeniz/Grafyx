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

    **Circular import detection** (``find_import_cycles``):
        DFS with gray/black coloring on the forward import graph to detect
        cycles like A -> B -> C -> A. Normalizes cycles by rotating to the
        lexicographically smallest element to avoid reporting the same cycle
        starting from different nodes.

    **Subclass tree traversal** (``get_subclasses``):
        Recursively builds a tree of all classes extending a given base class,
        up to a configurable depth.

    **Module dependency aggregation** (``get_module_dependencies``):
        Aggregates file-level imports into directory/package-level edges,
        with optional filtering by module path and debug mode showing
        sample file-to-file edges for each module edge.

Mixin: AnalysisMixin
Reads: self._codebases, self._lock, self._caller_index, self._class_method_names,
       self._forward_import_index, self._class_defined_in, self._import_index
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
    """Dead code detection, cycle finding, subclass trees, and module dependencies.

    This mixin contains the most computationally expensive operations in the
    graph engine. Dead code detection in particular iterates all functions
    with multiple index lookups per function. All methods acquire
    ``self._lock`` for thread safety.

    Reads: _codebases, _lock, _caller_index, _class_method_names,
    _forward_import_index, _class_defined_in, _import_index
    """

    # Base classes that indicate abstract/protocol patterns --
    # methods defined in subclasses of these are interface contracts (called
    # polymorphically via the parent type) and should not be flagged as dead code.
    _ABSTRACT_BASE_NAMES = frozenset({
        "Protocol", "ABC", "ABCMeta",
    })

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
            "abstractmethod", "overload", "override",
            "route", "get", "post", "put", "delete", "patch",
            "api_view", "action", "websocket",
            "exception_handler", "middleware",
            "fixture", "parametrize", "mark",
            "receiver", "on_event", "listener", "handler", "hook",
            "subscriber", "callback", "lifespan",
            "command", "group", "argument", "option",
            "tool",
            # Celery
            "task", "shared_task", "periodic_task",
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

                # Check if this function is explicitly imported by name elsewhere
                if not class_name and name in imported_names:
                    continue

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
                                for _anc in all_ancs:
                                    if name in self._class_method_names.get(_anc, set()):
                                        skip = True
                                        break

                if skip:
                    continue

                qualified = f"{class_name}.{name}" if class_name else name
                kind = "method" if class_name else "function"

                unused.append({
                    "name": name,
                    "qualified_name": qualified,
                    "file": file_path,
                    "line": func_dict.get("line"),
                    "kind": kind,
                    "language": func_dict.get("language", ""),
                })

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
            "BaseModel", "BaseSettings", "GenericModel",
            # SQLAlchemy
            "Base", "DeclarativeBase", "DeclarativeBaseNoMeta",
            # Django
            "Model", "View", "Serializer", "ModelSerializer",
            "Form", "ModelForm", "ModelAdmin", "AppConfig",
            # Starlette / FastAPI
            "BaseHTTPMiddleware",
            # Marshmallow
            "Schema",
            # React
            "Component", "PureComponent",
            # Angular (though Angular mostly uses decorators)
            "Pipe", "Directive",
            # TypeORM
            "BaseEntity",
            # Abstract / Protocol
            "Protocol", "ABC",
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
    # Circular Import Detection
    # ===================================================================

    def find_import_cycles(self, max_chain_length: int = 10) -> list[list[str]]:
        """Detect circular import chains using DFS with gray/black coloring.

        Uses the standard 3-color DFS algorithm on ``_forward_import_index``:
            - WHITE (0): unvisited
            - GRAY (1): on the current DFS path (back-edge target = cycle)
            - BLACK (2): fully explored

        Cycles are normalized by rotating to the lexicographically smallest
        element so the same cycle starting from different nodes is only
        reported once.

        Returns:
            List of cycles, each a list of file paths where ``first == last``.
            Sorted by length ascending (shortest cycles = most actionable).
        """
        with self._lock:
            WHITE, GRAY, BLACK = 0, 1, 2
            color: dict[str, int] = {}
            path_stack: list[str] = []
            cycles: list[list[str]] = []
            seen_cycle_keys: set[str] = set()

            for node in self._forward_import_index:
                color.setdefault(node, WHITE)

            def _dfs(node: str) -> None:
                color[node] = GRAY
                path_stack.append(node)

                for neighbor in self._forward_import_index.get(node, []):
                    if color.get(neighbor, WHITE) == GRAY:
                        cycle_start = path_stack.index(neighbor)
                        cycle = path_stack[cycle_start:] + [neighbor]
                        if len(cycle) - 1 <= max_chain_length:
                            chain = cycle[:-1]
                            min_idx = chain.index(min(chain))
                            normalized = chain[min_idx:] + chain[:min_idx]
                            cycle_key = "->".join(normalized)
                            if cycle_key not in seen_cycle_keys:
                                seen_cycle_keys.add(cycle_key)
                                cycles.append(cycle)
                    elif color.get(neighbor, WHITE) == WHITE:
                        _dfs(neighbor)

                path_stack.pop()
                color[node] = BLACK

            for node in list(self._forward_import_index.keys()):
                if color.get(node, WHITE) == WHITE:
                    _dfs(node)

            cycles.sort(key=len)
            return cycles

    # ===================================================================
    # Subclass Tree Traversal
    # ===================================================================

    def get_subclasses(self, class_name: str, depth: int = 3) -> dict | None:
        """Find all classes that extend the given class, recursively.

        Builds a reverse inheritance map (base_name -> [child_class_dicts])
        and then does a BFS/DFS traversal up to the specified depth.
        Uses a ``visited`` set to handle diamond inheritance safely.

        Args:
            class_name: The base class to find subclasses of.
            depth: How many inheritance levels to traverse (1 = direct only).

        Returns:
            Dict with class info, direct/total counts, and a nested
            ``subclasses`` tree. None if the class is not found.
        """
        with self._lock:
            all_classes = self.get_all_classes(max_results=2000)

            # Check that the target class exists
            target = None
            for cls_dict in all_classes:
                if cls_dict.get("name") == class_name:
                    target = cls_dict
                    break
            if target is None:
                return None

            # Build reverse map: base_name -> [child_class_dicts]
            children_of: dict[str, list[dict]] = {}
            for cls_dict in all_classes:
                for base in cls_dict.get("base_classes", []):
                    base_name = str(base).split(".")[-1].strip()
                    if base_name:
                        if base_name not in children_of:
                            children_of[base_name] = []
                        children_of[base_name].append(cls_dict)

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

            return {
                "class_name": class_name,
                "file": target.get("file", ""),
                "line": target.get("line"),
                "language": target.get("language", ""),
                "direct_subclass_count": direct_count,
                "total_subclass_count": total_count,
                "subclasses": tree,
            }

    # ===================================================================
    # Module-Level Dependency Aggregation
    # ===================================================================

    def get_module_dependencies(
        self, module_path: str = "", depth: int = 1, debug: bool = False,
    ) -> dict:
        """Show how directories/packages depend on each other.

        Aggregates file-level imports from ``_forward_import_index`` into
        module-level (directory) dependencies. Each file is assigned to a
        "module" based on its directory path truncated to the given depth.

        For example, at depth=1, ``src/services/auth.py`` and
        ``src/services/users.py`` both belong to module ``src``.
        At depth=2, they belong to ``src/services``.

        Args:
            module_path: Optional filter -- only show edges involving this
                module (or its sub-modules). Depth is auto-adjusted so the
                target module is visible as a distinct node.
            depth: Subdirectory grouping depth (1 = top-level dirs only).
            debug: If True, include ``debug_file_edges`` showing up to 5
                sample file-to-file imports for each module edge.

        Returns:
            Dict with ``modules`` (each with depends_on/depended_on_by),
            ``edges`` (sorted by import_count descending), and ``module_count``.
        """
        with self._lock:
            all_files = self.get_all_files(max_results=2000)
            project_root = self.original_path.replace("\\", "/").rstrip("/") + "/"

            # Auto-adjust depth so module_path is visible as a distinct module
            if module_path:
                mp_clean = module_path.strip("/")
                min_depth = len(mp_clean.split("/"))
                if depth < min_depth:
                    depth = min_depth

            # Assign each file to a module (directory) at the given depth.
            # Strips the project root prefix and truncates the directory path
            # to `depth` components. Files at the project root get module ".".
            def _get_module(file_path: str) -> str:
                fp = file_path.replace("\\", "/")
                if fp.startswith(project_root):
                    rel = fp[len(project_root):]
                else:
                    rel = fp.lstrip("/")
                parts = rel.split("/")
                # Strip the filename (last component) -- we want directory grouping.
                # Without this, a file at exactly `depth` directory levels (e.g.,
                # "backend/app/utils.py" at depth=3) would include the filename in
                # its module name ("backend/app/utils.py"), breaking self-import
                # deduplication and producing false cross-module edges.
                dir_parts = parts[:-1]
                if not dir_parts:
                    return "."  # Root-level files
                return "/".join(dir_parts[:depth])

            # Build set of known files and their modules
            file_to_module: dict[str, str] = {}
            module_files: dict[str, int] = {}  # module -> file count
            for f in all_files:
                fpath = f.get("path", "").replace("\\", "/")
                if not fpath:
                    continue
                mod = _get_module(fpath)
                file_to_module[fpath] = mod
                module_files[mod] = module_files.get(mod, 0) + 1

            # Also index files from _forward_import_index (may have paths
            # not returned by get_all_files if they are import targets)
            for fpath in self._forward_import_index:
                fp = fpath.replace("\\", "/")
                if fp not in file_to_module:
                    mod = _get_module(fp)
                    file_to_module[fp] = mod
                for target in self._forward_import_index[fpath]:
                    tp = target.replace("\\", "/")
                    if tp not in file_to_module:
                        file_to_module[tp] = _get_module(tp)

            # Aggregate edges: (from_module, to_module) -> count
            # Use top-level-only forward index to avoid ghost edges from
            # lazy imports inside function bodies.
            _fwd_idx = getattr(self, '_top_level_forward_import_index', None)
            if not _fwd_idx:
                _fwd_idx = self._forward_import_index
            edge_counts: dict[tuple[str, str], int] = {}
            for source_file, targets in _fwd_idx.items():
                src_fp = source_file.replace("\\", "/")
                src_mod = file_to_module.get(src_fp, _get_module(src_fp))
                for target_file in targets:
                    tgt_fp = target_file.replace("\\", "/")
                    tgt_mod = file_to_module.get(tgt_fp, _get_module(tgt_fp))
                    # Skip self-imports (within same module)
                    if src_mod == tgt_mod:
                        continue
                    key = (src_mod, tgt_mod)
                    edge_counts[key] = edge_counts.get(key, 0) + 1

            # Build module detail map
            all_modules: set[str] = set(module_files.keys())
            for src, tgt in edge_counts:
                all_modules.add(src)
                all_modules.add(tgt)

            # Apply module_path filter (exact match or prefix with /)
            if module_path:
                mp_clean = module_path.strip("/")

                def _matches_filter(mod: str) -> bool:
                    return mod == mp_clean or mod.startswith(mp_clean + "/")

                filtered_edges = {
                    k: v for k, v in edge_counts.items()
                    if _matches_filter(k[0]) or _matches_filter(k[1])
                }
                relevant_modules = set()
                # Include any modules matching the filter even if they have no edges
                for mod in all_modules:
                    if _matches_filter(mod):
                        relevant_modules.add(mod)
                for src, tgt in filtered_edges:
                    relevant_modules.add(src)
                    relevant_modules.add(tgt)
                all_modules = relevant_modules
                edge_counts = filtered_edges

            modules: dict[str, dict] = {}
            for mod in sorted(all_modules):
                depends_on = sorted({tgt for (src, tgt), _ in edge_counts.items() if src == mod})
                depended_on_by = sorted({src for (src, tgt), _ in edge_counts.items() if tgt == mod})
                modules[mod] = {
                    "file_count": module_files.get(mod, 0),
                    "depends_on": depends_on,
                    "depended_on_by": depended_on_by,
                }

            edges = [
                {"from": src, "to": tgt, "import_count": count}
                for (src, tgt), count in sorted(edge_counts.items(), key=lambda x: -x[1])
            ]

            result: dict = {
                "project_path": self.original_path,
                "module_count": len(modules),
                "modules": modules,
                "edges": edges,
            }

            if debug:
                # Show up to 5 file-level edges per module edge to aid diagnosis
                # of false positives (e.g., services->api edges that don't exist).
                file_edge_samples: dict[str, list[dict]] = {}
                for source_file, targets in _fwd_idx.items():
                    src_fp = source_file.replace("\\", "/")
                    src_mod = file_to_module.get(src_fp, _get_module(src_fp))
                    for tgt_file in targets:
                        tgt_fp = tgt_file.replace("\\", "/")
                        tgt_mod = file_to_module.get(tgt_fp, _get_module(tgt_fp))
                        if src_mod == tgt_mod:
                            continue
                        if module_path:
                            mp_clean = module_path.strip("/")
                            if not (src_mod == mp_clean or src_mod.startswith(mp_clean + "/")
                                    or tgt_mod == mp_clean or tgt_mod.startswith(mp_clean + "/")):
                                continue
                        edge_key = f"{src_mod} \u2192 {tgt_mod}"
                        if edge_key not in file_edge_samples:
                            file_edge_samples[edge_key] = []
                        if len(file_edge_samples[edge_key]) < 5:
                            file_edge_samples[edge_key].append({
                                "from_file": src_fp,
                                "to_file": tgt_fp,
                            })
                result["debug_file_edges"] = file_edge_samples

            return result
