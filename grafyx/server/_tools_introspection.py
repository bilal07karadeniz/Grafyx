"""MCP Tools for deep symbol introspection: function, file, and class context.

This module provides the three "context" tools that let an AI assistant
drill into specific symbols in the codebase:

- **get_function_context** (Tool 2) -- signature, params, callers, callees,
  dependencies, and optional source code for a single function or method.
- **get_file_context** (Tool 3) -- all functions, classes, imports, and
  reverse-import relationships for a single file.
- **get_class_context** (Tool 4) -- methods, properties, base classes,
  and cross-file usage analysis for a single class.

All three tools share the same pattern:
    1. Call ``_ensure_initialized()`` to get the graph.
    2. Look up the symbol via ``graph.get_function/get_file/get_class``.
    3. Extract attributes using ``safe_get_attr()`` for graph-sitter
       version safety.
    4. Enrich with cross-references (callers, importers, usages).
    5. Return a truncated dict to stay within context-window limits.

Registered on the shared ``mcp`` instance from ``_state.py``.
"""

from fastmcp.exceptions import ToolError

from grafyx.constants import PYTHON_BUILTINS
from grafyx.server._hints import compute_hints
from grafyx.server._resolution import filter_by_detail
from grafyx.server._state import _ensure_initialized, _find_reference_lines, mcp
from grafyx.utils import (
    EXTENSION_TO_LANGUAGE,
    extract_base_classes,
    format_class_summary,
    format_file_summary,
    format_function_signature,
    safe_get_attr,
    safe_str,
    truncate_response,
)


# Web-family languages share a parser and call into each other natively,
# so a TS file calling a JS function (or vice versa) is NOT a cross-
# language jump. Anything outside this set crossing into Python (or
# vice versa) is treated as cross-language and filtered conservatively.
_WEB_LANGS = frozenset({"typescript", "javascript"})


def _filter_cross_language_callers(
    callers: list[dict], target_lang: str,
) -> list[dict]:
    """Drop weak cross-language callers that are likely name-collision FPs.

    A frontend file calling a same-named backend function only makes
    sense via an HTTP/API client wrapper, which always shows up as a
    method call (e.g. ``api.login(...)``) — never as a bare ``login()``
    in another language. We keep cross-language callers ONLY when they
    invoke through a receiver (``has_dot_syntax=True``), because a bare
    ``login()`` in TypeScript is by definition calling the TypeScript
    ``login`` symbol, not the Python one.
    """
    if not target_lang:
        return callers
    target_family = "web" if target_lang in _WEB_LANGS else target_lang
    out: list[dict] = []
    for c in callers:
        cf = c.get("file", "")
        if not cf or "." not in cf:
            out.append(c)
            continue
        ext = "." + cf.rsplit(".", 1)[-1].lower()
        cl = EXTENSION_TO_LANGUAGE.get(ext, "")
        cf_family = "web" if cl in _WEB_LANGS else cl
        if not cl or cf_family == target_family:
            out.append(c)
            continue
        # Cross-language: require dot syntax (an actual method/property
        # call on a receiver object). Bare same-name calls almost
        # always resolve to a local function in the caller's language.
        if c.get("has_dot_syntax", False):
            out.append(c)
    return out


# --- Tool 2: get_function_context ---


@mcp.tool
def get_function_context(
    function_name: str,
    detail: str = "summary",
    include_hints: bool = True,
    file_path: str = "",
) -> dict:
    """Get comprehensive context for a function: signature, parameters,
    callers, callees, dependencies, and docstring.

    Supports 'ClassName.method_name' syntax (e.g., 'ToolExecutor.execute').
    For two top-level functions sharing the same name in different
    files (where ``ClassName.method`` doesn't apply), pass ``file_path``
    to pick one. If multiple matches remain, returns a disambiguation list.

    Use this when you need to understand what a function does, who calls it,
    and what its blast radius is for modifications.

    Args:
        function_name: Name of the function to look up.
        detail: Level of detail: "signatures", "summary" (default), or "full".
        include_hints: If True, append navigation suggestions.
        file_path: Optional path filter to disambiguate same-named
            top-level functions in different files. Matched by
            substring against each candidate's filepath, so a partial
            path like ``"api/agents.py"`` is enough.
    """
    graph = _ensure_initialized()
    try:
        result = graph.get_function(function_name)
        if result is None:
            return {"found": False, "message": f"Function '{function_name}' not found in the codebase."}

        # --- Disambiguation ---
        # graph.get_function() returns a list of (lang, func, cls_name) tuples
        # when multiple functions share the same name (e.g., "execute" exists
        # in both ToolExecutor and Database).  We present all matches so the
        # AI can re-call with "ClassName.method_name" to disambiguate.
        if isinstance(result, list):
            # Apply file_path filter first when provided. This lets a
            # caller select between two same-named top-level functions
            # in different files — the ``ClassName.method`` form alone
            # can't disambiguate those (both have cls_name=None).
            if file_path and len(result) > 1:
                file_path_norm = file_path.replace("\\", "/")
                filtered = [
                    item for item in result
                    if file_path_norm in str(
                        safe_get_attr(item[1], "filepath", "")
                    ).replace("\\", "/")
                ]
                if len(filtered) >= 1:
                    result = filtered
            if isinstance(result, list) and len(result) > 1:
                disambiguation = []
                for lang, func, cls_name in result:
                    qualified = f"{cls_name}.{safe_get_attr(func, 'name', '?')}" if cls_name else safe_get_attr(func, "name", "?")
                    disambiguation.append({
                        "qualified_name": qualified,
                        "file": graph.translate_path(str(safe_get_attr(func, "filepath", ""))),
                        "signature": format_function_signature(func),
                        "line": graph.get_line_number(func),
                    })
                return truncate_response({
                    "ambiguous": True,
                    "message": (
                        f"Multiple functions named '{function_name}' found. "
                        f"Use 'ClassName.method_name' syntax — or pass "
                        f"``file_path=`` to pick by file — to disambiguate."
                    ),
                    "matches": disambiguation,
                })
            # Single match -- unpack from the list
            lang, func, cls_name = result[0]
        else:
            # Direct (lang, func, cls_name) tuple -- only one match
            lang, func, cls_name = result

        # --- Build the context dict ---
        func_name_str = safe_get_attr(func, "name", function_name)
        context = {
            "name": func_name_str,
            "class": cls_name,
            "qualified_name": f"{cls_name}.{func_name_str}" if cls_name else func_name_str,
            "signature": format_function_signature(func),
            "file": graph.translate_path(str(safe_get_attr(func, "filepath", ""))),
            "line": graph.get_line_number(func),
            "language": lang,
            "is_async": safe_get_attr(func, "is_async", False),
            "docstring": safe_str(safe_get_attr(func, "docstring", "")) or None,
            "parameters": [],
            "return_type": safe_str(safe_get_attr(func, "return_type", "")) or None,
            "decorators": [],
            "calls": [],
            "called_by": [],
            "dependencies": [],
        }

        # --- Parameters ---
        # Extract parameter names, types, and defaults.  graph-sitter
        # sometimes returns "TypePlaceholder" for unresolved types --
        # we normalize those to None so the AI doesn't see a confusing name.
        params = safe_get_attr(func, "parameters", [])
        if params:
            for p in params:
                p_type = safe_str(safe_get_attr(p, "type", "")) or None
                # graph-sitter returns "TypePlaceholder" for unresolved types;
                # replace with None so consumers don't see a confusing name.
                if p_type and "placeholder" in p_type.lower():
                    p_type = None
                context["parameters"].append({
                    "name": safe_get_attr(p, "name", "?"),
                    "type": p_type,
                    "default": safe_str(safe_get_attr(p, "default", "")) or None,
                })

        # --- Decorators ---
        decorators = safe_get_attr(func, "decorators", [])
        if decorators:
            context["decorators"] = [safe_str(d) for d in decorators]

        # --- Outgoing calls (what this function calls) ---
        # Builtins (len, append, etc.) are filtered out to reduce noise.
        # Calls are deduplicated and counted so the AI sees
        # {"name": "emit", "call_count": 3} instead of three separate entries.
        function_calls = safe_get_attr(func, "function_calls", [])
        if function_calls:
            call_counts: dict[str, dict] = {}  # name -> {file, count}
            for fc in function_calls:
                fc_name = safe_get_attr(fc, "name", "?")
                if fc_name in PYTHON_BUILTINS:
                    continue
                if fc_name in call_counts:
                    call_counts[fc_name]["count"] += 1
                else:
                    fc_file = graph.translate_path(graph.get_filepath_from_obj(fc))
                    call_counts[fc_name] = {"name": fc_name, "file": fc_file, "count": 1}
            for entry in call_counts.values():
                call_entry: dict = {"name": entry["name"], "file": entry["file"]}
                # Only include call_count when > 1 to keep output compact
                if entry["count"] > 1:
                    call_entry["call_count"] = entry["count"]
                context["calls"].append(call_entry)

        # --- Incoming calls (who calls this function) ---
        # Uses the reverse caller index built by CodebaseGraph.  The
        # class_name parameter filters out methods with the same name in
        # OTHER classes (e.g., Database.connect calling "execute" won't
        # appear in ToolExecutor.execute's callers).
        # Cross-language callers (e.g. a TS file appearing as caller of
        # a Python route) are filtered to require dot-syntax — a bare
        # ``login()`` in TS calls the TS ``login``, not the Python one.
        func_name = safe_get_attr(func, "name", function_name)
        callers = graph.get_callers(func_name, class_name=cls_name)
        callers = _filter_cross_language_callers(callers, lang)
        if callers:
            context["called_by"] = callers

        # --- Dependencies ---
        # graph-sitter "dependencies" are import-level references that
        # this function uses (modules, classes, etc.).
        deps = safe_get_attr(func, "dependencies", [])
        if deps:
            for d in deps:
                context["dependencies"].append({
                    "name": safe_str(safe_get_attr(d, "name", str(d))),
                    "kind": (
                        type(d).__name__.lower()
                        if hasattr(d, "__class__")
                        else "unknown"
                    ),
                })

        # --- Source code (always extracted, filtered by detail level) ---
        source = safe_str(safe_get_attr(func, "source", "")) or None
        if source:
            context["source"] = source

        # Apply detail-level filtering (strips fields based on detail)
        context = filter_by_detail(context, detail, "function")

        # Compute navigation hints for exploration
        if include_hints:
            hints = compute_hints(graph, "function", context)
            if hints:
                context["suggested_next"] = hints

        return truncate_response(context)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_function_context: {e}")


# --- Tool 3: get_file_context ---


@mcp.tool
def get_file_context(file_path: str, detail: str = "summary", include_hints: bool = True) -> dict:
    """Get comprehensive context for a file: functions, classes, imports,
    and relationships to other files.

    Use this when you need to understand what a file contains and how it
    connects to the rest of the codebase.

    Args:
        file_path: Path to the file to inspect.
        detail: Level of detail: "signatures", "summary" (default), or "full".
        include_hints: If True, append navigation suggestions.
    """
    graph = _ensure_initialized()
    try:
        result = graph.get_file(file_path)
        if result is None:
            return {"found": False, "message": f"File '{file_path}' not found in the codebase."}

        lang, file_obj = result
        summary = format_file_summary(file_obj)

        context = {
            "path": graph.translate_path(summary.get("path", file_path)),
            "language": lang,
            "function_count": summary.get("function_count", 0),
            "class_count": summary.get("class_count", 0),
            "import_count": summary.get("import_count", 0),
            "functions": [],
            "classes": [],
            "imports": [],
        }

        # --- Functions defined in this file ---
        functions = safe_get_attr(file_obj, "functions", [])
        if functions:
            for f in functions:
                context["functions"].append({
                    "name": safe_get_attr(f, "name", "?"),
                    "signature": format_function_signature(f),
                    "line": graph.get_line_number(f),
                    "docstring": safe_str(safe_get_attr(f, "docstring", "")) or None,
                })

        # --- Object literal methods (TS/JS arrow functions) ---
        file_path_for_olm = graph.translate_path(
            str(safe_get_attr(file_obj, "filepath", safe_get_attr(file_obj, "path", "")))
        )
        for olm in getattr(graph, "_object_literal_methods", []):
            if olm["file"] == file_path_for_olm:
                context["functions"].append({
                    "name": f"{olm['parent']}.{olm['name']}",
                    "signature": f"{olm['name']}: (...) => ...",
                    "line": olm["line"],
                    "docstring": None,
                })
        # Update function count to include OLMs
        context["function_count"] = len(context["functions"])

        # --- Classes defined in this file ---
        classes = safe_get_attr(file_obj, "classes", [])
        if classes:
            for c in classes:
                cls_summary = format_class_summary(c)
                cls_summary["line"] = graph.get_line_number(c)

                # Override truncated docstring with full version.
                # format_class_summary() uses _first_line() for compact display,
                # but in file context we want the full class docstring so the AI
                # can understand the class's purpose without a follow-up call.
                full_doc = safe_str(safe_get_attr(c, "docstring", ""))
                if full_doc and full_doc.strip():
                    cls_summary["docstring"] = full_doc.strip()

                # --- Methods and properties ---
                # We separate @property-decorated methods into a "properties"
                # list, and real methods into "methods".  This gives the AI a
                # clearer picture of the class's interface.
                methods = safe_get_attr(c, "methods", [])
                method_list = []
                property_names: set[str] = set()
                if methods:
                    for m in methods:
                        m_name = safe_get_attr(m, "name", "?")
                        decorators = safe_get_attr(m, "decorators", [])
                        dec_strs = [safe_str(d).strip("@") for d in decorators] if decorators else []
                        # Detect @property, @cached_property, @x.setter, @x.deleter
                        is_prop = any(
                            d in ("property", "cached_property") or d.endswith(".setter") or d.endswith(".deleter")
                            for d in dec_strs
                        )
                        if is_prop:
                            property_names.add(m_name)
                        else:
                            method_list.append({
                                "name": m_name,
                                "signature": format_function_signature(m),
                                "line": graph.get_line_number(m),
                            })
                if method_list:
                    cls_summary["methods"] = method_list

                # Merge graph-sitter's native "properties" attribute with
                # the @property-decorated methods we detected above.  Some
                # graph-sitter versions expose .properties natively; others
                # only have them as decorated methods.
                native_props = safe_get_attr(c, "properties", [])
                if native_props:
                    for p in native_props:
                        pn = safe_str(safe_get_attr(p, "name", str(p)))
                        if pn:
                            property_names.add(pn)
                cls_summary["property_count"] = len(property_names)
                if property_names:
                    cls_summary["properties"] = sorted(property_names)
                context["classes"].append(cls_summary)

        # --- Imports (deduplicated, preserving order) ---
        imports = safe_get_attr(file_obj, "imports", [])
        if imports:
            seen: set[str] = set()
            unique_imports: list[str] = []
            for imp in imports:
                imp_str = safe_str(imp)
                if imp_str and imp_str not in seen:
                    seen.add(imp_str)
                    unique_imports.append(imp_str)
            context["imports"] = unique_imports

        # --- Reverse imports: who depends on this file ---
        # graph.get_importers() returns files that import this file.
        file_path_for_lookup = graph.translate_path(
            str(safe_get_attr(file_obj, "filepath", safe_get_attr(file_obj, "path", "")))
        )
        importers = graph.get_importers(file_path_for_lookup)
        if importers:
            context["imported_by_count"] = len(importers)
            context["imported_by"] = importers

        # --- Source code (always extracted, filtered by detail level) ---
        source = safe_str(safe_get_attr(file_obj, "source", "")) or None
        if source:
            context["source"] = source

        # Apply detail-level filtering
        context = filter_by_detail(context, detail, "file")

        # Compute navigation hints
        if include_hints:
            hints = compute_hints(graph, "file", context)
            if hints:
                context["suggested_next"] = hints

        return truncate_response(context)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_file_context: {e}")


# --- Tool 4: get_class_context ---


@mcp.tool
def get_class_context(class_name: str, detail: str = "summary", include_hints: bool = True) -> dict:
    """Get comprehensive context for a class: methods, properties,
    inheritance chain, and usages throughout the codebase.

    Use this when you need to understand a class's interface and where it is used.

    Args:
        class_name: Name of the class to look up.
        detail: Level of detail: "signatures", "summary" (default), or "full".
        include_hints: If True, append navigation suggestions.
    """
    graph = _ensure_initialized()
    try:
        result = graph.get_class(class_name)
        if result is None:
            return {"found": False, "message": f"Class '{class_name}' not found in the codebase."}

        # Handle both single-result and multi-result forms.
        # When multiple classes share a name, we take the first match.
        if isinstance(result, list):
            lang, cls = result[0]
        else:
            lang, cls = result

        cls_file = graph.resolve_path(str(safe_get_attr(cls, "filepath", "")))

        context = {
            "name": safe_get_attr(cls, "name", class_name),
            "file": cls_file,
            "line": graph.get_line_number(cls),
            "language": lang,
            "docstring": safe_str(safe_get_attr(cls, "docstring", "")) or None,
            "base_classes": [],
            "methods": [],
            "properties": [],
            "cross_file_usages": [],
            "internal_usage_count": 0,
            "dependencies": [],
        }

        # --- Base classes ---
        # extract_base_classes() has a multi-attribute fallback chain:
        # tries .base_classes, .superclasses, then regex on source code.
        # This handles different graph-sitter versions gracefully.
        bases = extract_base_classes(cls)
        if bases:
            context["base_classes"] = bases

        # --- Methods and properties ---
        # Separate @property-decorated methods into a dedicated properties
        # list so the AI sees the class's data interface clearly.
        methods = safe_get_attr(cls, "methods", [])
        property_names: set[str] = set()
        if methods:
            for m in methods:
                m_name = safe_get_attr(m, "name", "?")
                decorators = safe_get_attr(m, "decorators", [])
                decorator_strs = [safe_str(d).strip("@") for d in decorators] if decorators else []
                is_property = any(
                    d in ("property", "cached_property") or d.endswith(".setter") or d.endswith(".deleter")
                    for d in decorator_strs
                )
                if is_property:
                    property_names.add(m_name)
                else:
                    context["methods"].append({
                        "name": m_name,
                        "signature": format_function_signature(m),
                        "line": graph.get_line_number(m),
                        "is_async": safe_get_attr(m, "is_async", False),
                        "docstring": safe_str(safe_get_attr(m, "docstring", "")) or None,
                    })

        # Merge graph-sitter's native properties with @property-detected ones
        props = safe_get_attr(cls, "properties", [])
        if props:
            for p in props:
                p_name = safe_str(safe_get_attr(p, "name", str(p)))
                if p_name:
                    property_names.add(p_name)
        if property_names:
            context["properties"] = [{"name": n} for n in sorted(property_names)]

        # ================================================================
        # Cross-file usage detection -- three complementary strategies
        # ================================================================
        # Finding all usages of a class is surprisingly hard.  graph-sitter
        # usages are often incomplete (especially for classes used only via
        # imports or as type hints).  We combine three strategies to get
        # comprehensive coverage:
        #
        #   Strategy 1: graph-sitter .usages attribute (AST-level references)
        #   Strategy 2: import index (files that import this class's module)
        #   Strategy 3: caller index (files calling unique methods of this class)
        #
        # Each strategy adds files NOT already found by earlier strategies.

        # --- Strategy 1: graph-sitter usages ---
        # These are direct AST-level references found by graph-sitter.
        # Grouped by file, with max 3 line numbers per file.
        usages = safe_get_attr(cls, "usages", [])
        if usages:
            cross_file_by_path: dict[str, list] = {}
            internal_count = 0
            for u in usages:
                u_file = graph.resolve_path(graph.get_filepath_from_obj(u))
                u_line = graph.get_line_number(u)
                # Skip usages with no resolved file path
                if not u_file:
                    internal_count += 1
                    continue
                # Skip same-file (internal) usages -- not interesting for
                # understanding cross-module coupling.
                if u_file == cls_file:
                    internal_count += 1
                    continue
                if u_file not in cross_file_by_path:
                    cross_file_by_path[u_file] = []
                if len(cross_file_by_path[u_file]) < 3:  # Max 3 lines per file
                    cross_file_by_path[u_file].append(u_line)
            context["internal_usage_count"] = internal_count
            for u_file, lines in list(cross_file_by_path.items()):
                filtered_lines = [l for l in lines if l is not None]
                if filtered_lines:
                    context["cross_file_usages"].append({
                        "file": u_file,
                        "lines": filtered_lines,
                    })

        # --- Strategy 2: import index fallback ---
        # Use the import index to find files that import this class's module.
        # Runs ALWAYS (not just when Strategy 1 is empty) because graph-sitter
        # usages can miss references the import index catches (e.g., type
        # hints, re-exports, dynamic usage patterns).
        cls_name_str = safe_get_attr(cls, "name", class_name)
        # Collect singleton instance names for this class (e.g.,
        # "transcript_cache" for TranscriptCache) so we can find files
        # that reference the class through its instance variable.
        instance_names = [
            inst_name for inst_name, _inst_file
            in graph._class_instances.get(cls_name_str, [])
        ]
        if cls_file:
            # Track which files we've already added to avoid duplicates
            found_files: set[str] = {u["file"] for u in context["cross_file_usages"]}
            importers = graph.get_importers(cls_file)
            for imp_file in importers:
                imp_file_norm = graph.resolve_path(imp_file)
                if not imp_file_norm or imp_file_norm == cls_file or imp_file_norm in found_files:
                    continue
                # Lightweight text search for the class name and its
                # singleton instances in the importing file.
                lines = _find_reference_lines(imp_file_norm, cls_name_str)
                for inst_name in instance_names:
                    inst_lines = _find_reference_lines(imp_file_norm, inst_name)
                    for ln in inst_lines:
                        if ln not in lines:
                            lines.append(ln)
                lines.sort()
                if lines:
                    context["cross_file_usages"].append({
                        "file": imp_file_norm,
                        "lines": lines[:5],
                    })
                found_files.add(imp_file_norm)

        # --- Strategy 3: caller index for unique methods ---
        # For classes with uniquely-named methods (methods that don't exist
        # on any other class), check the reverse caller index.  If "emit"
        # only exists on EventBus, then any file calling "emit" is an
        # EventBus user.  Methods like "execute" that exist on 5+ classes
        # are skipped because callers can't be reliably attributed.
        # Also runs always -- adds only files not yet found.
        found_files_s2: set[str] = {u["file"] for u in context["cross_file_usages"]}
        usage_files: set[str] = set()
        methods = safe_get_attr(cls, "methods", [])
        if methods:
            for method in methods:
                m_name = safe_get_attr(method, "name", "")
                # Skip dunder methods -- they're too common to be useful
                if not m_name or m_name.startswith("__"):
                    continue
                # Skip methods whose name exists in OTHER classes --
                # these cause cross-class contamination in caller results.
                other_classes = sum(
                    1 for cls_n, meths in graph._class_method_names.items()
                    if cls_n != cls_name_str and m_name in meths
                )
                if other_classes > 0:
                    continue
                # This method name is unique to this class -- its callers
                # are definitely users of this class.
                callers = graph.get_callers(m_name)
                for caller in callers:
                    c_file = graph.resolve_path(caller.get("file", ""))
                    if c_file and c_file != cls_file and c_file not in found_files_s2:
                        usage_files.add(c_file)
        for uf in sorted(usage_files):
            if uf in found_files_s2:
                continue
            # Add line numbers via text search (same approach as Strategy 2)
            lines = _find_reference_lines(uf, cls_name_str)
            for inst_name in instance_names:
                inst_lines = _find_reference_lines(uf, inst_name)
                for ln in inst_lines:
                    if ln not in lines:
                        lines.append(ln)
            lines.sort()
            if lines:
                context["cross_file_usages"].append({
                    "file": uf,
                    "lines": lines[:5],
                })

        # --- Strategy 4: factory-pattern callers ---
        # When a class is consumed via a factory (e.g. ``coord = await
        # get_coordinator()`` rather than ``coord = WorkerCoordinator()``),
        # the class name never appears at the call site. Strategies 1-3
        # therefore miss every consumer. The caller-index augmentation
        # built ``_factory_return_types`` (factory_fn_name -> class_name)
        # in v0.2.2 to fix the *method-resolution* side; here we use the
        # same mapping to surface the *files* that use the class.
        factory_map = getattr(graph, "_factory_return_types", {})
        if factory_map:
            factory_names = [
                fn for fn, ret_cls in factory_map.items()
                if ret_cls == cls_name_str
            ]
            if factory_names:
                already_listed: set[str] = {
                    u["file"] for u in context["cross_file_usages"]
                }
                factory_files: set[str] = set()
                for fn_name in factory_names:
                    for caller in graph.get_callers(fn_name):
                        c_file = graph.resolve_path(caller.get("file", ""))
                        if c_file and c_file != cls_file and c_file not in already_listed:
                            factory_files.add(c_file)
                for ff in sorted(factory_files):
                    # Light reference-line lookup for both the factory
                    # function name and the class name itself.
                    lines = _find_reference_lines(ff, cls_name_str)
                    for fn_name in factory_names:
                        more = _find_reference_lines(ff, fn_name)
                        for ln in more:
                            if ln not in lines:
                                lines.append(ln)
                    lines.sort()
                    if lines:
                        context["cross_file_usages"].append({
                            "file": ff,
                            "lines": lines[:5],
                        })
                    else:
                        # Caller is real even when text search misses
                        # (text search may miss class-name occurrences
                        # in renamed imports). Record without lines.
                        context["cross_file_usages"].append({
                            "file": ff,
                            "lines": [],
                        })

        # --- Dependencies ---
        deps = safe_get_attr(cls, "dependencies", [])
        if deps:
            for d in deps:
                context["dependencies"].append({
                    "name": safe_str(safe_get_attr(d, "name", str(d))),
                })

        # --- Source code (always extracted, filtered by detail level) ---
        src = safe_str(safe_get_attr(cls, "source", ""))
        if src:
            context["source"] = src

        # Apply detail-level filtering
        context = filter_by_detail(context, detail, "class")

        # Compute navigation hints
        if include_hints:
            hints = compute_hints(graph, "class", context)
            if hints:
                context["suggested_next"] = hints

        return truncate_response(context)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in get_class_context: {e}")
