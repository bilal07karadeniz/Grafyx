"""Caller query and method-class resolution for CodebaseGraph.

This module provides CallerQueryMixin, which queries the reverse caller
index built by ``IndexBuilderMixin`` with sophisticated multi-level
disambiguation filtering.

The core problem: when multiple classes define a method with the same name
(e.g., ``Database.execute`` and ``ToolExecutor.execute``), a naive reverse
index lookup for "execute" returns callers of *all* execute methods. The
4-level filtering pipeline in ``get_callers()`` progressively eliminates
false positives:

    **Level 1 - Class-level**: If the caller's own class defines the same
    method name, the call is to its own class, not the target.

    **Level 2 - File-level**: Fallback when graph-sitter doesn't populate
    ``parent_class``. If the caller's file defines any other class with
    this method name, the call is probably to that local class.

    **Level 3 - Import-level**: For ambiguous method names, only keep
    callers from files that import the target class's defining module.
    Uses ``_get_class_importer_files()`` to build an allowlist.

    **Level 4 - Attribute-level**: Checks the source-level receiver
    expression. If ``self.short_term.add()`` is the call, and the receiver
    tokens ("short_term") don't match the target class ("EpisodicMemory"),
    the caller is filtered out.

Mixin: CallerQueryMixin
Reads: self._caller_index, self._class_method_names, self._file_class_methods,
       self._class_defined_in, self._import_index, self._lock
Writes: nothing (read-only queries)
"""

import logging
from typing import Any

from grafyx.utils import split_tokens

logger = logging.getLogger(__name__)


def _extract_immediate_receiver(receivers: set[str] | None) -> str | None:
    """Extract the most informative receiver token from a set of receiver expressions.

    For 'self.db' -> 'db'
    For 'self.episodic' -> 'episodic'
    For bare 'self' -> None
    For 'response' -> 'response'
    For None/empty -> None
    """
    if not receivers:
        return None
    best = None
    best_depth = -1
    for recv in receivers:
        parts = recv.split(".")
        # Remove 'self' prefix
        if parts and parts[0] == "self":
            parts = parts[1:]
        if not parts:
            continue
        # The immediate receiver is the first part after self
        token = parts[0]
        depth = len(parts)
        if depth > best_depth:
            best = token
            best_depth = depth
    return best


def _format_caller_output(caller: dict) -> dict:
    """Format a caller entry for public output.

    Strips internal ``_``-prefixed keys and exposes ``receiver_token``
    and ``has_dot_syntax`` as public fields.
    """
    out = {k: v for k, v in caller.items() if not k.startswith("_")}
    out["receiver_token"] = _extract_immediate_receiver(caller.get("_receivers"))
    out["has_dot_syntax"] = caller.get("_has_dot_syntax", False)
    return out


class CallerQueryMixin:
    """Query the caller index with disambiguation and resolve method ownership.

    This mixin is the primary interface for answering "who calls X?" queries.
    It reads the indexes built by ``IndexBuilderMixin`` and applies the
    4-level disambiguation pipeline described in the module docstring.

    Reads: _caller_index, _class_method_names, _file_class_methods,
    _class_defined_in, _import_index, _lock
    """

    # --- Caller Lookup with Disambiguation ---

    def get_callers(self, function_name: str, class_name: str | None = None) -> list[dict]:
        """Return list of {name, file} dicts for functions that call function_name.

        Args:
            function_name: The name of the called function/method.
            class_name: If provided, activate the 4-level disambiguation
                pipeline to filter out callers that are actually calling
                a *different* class's method with the same name.

        When ``class_name`` is None, returns all raw callers (no filtering).
        Internal metadata fields (prefixed with ``_``) are always stripped
        from the returned dicts. Two derived fields are added:
            - ``receiver_token``: the immediate receiver variable name
              (e.g., ``"db"`` from ``self.db.query()``), or None.
            - ``has_dot_syntax``: True if the call used dot syntax.

        Filtering levels (applied only when class_name is set):
            1. **Class-level**: caller's own class has this method -> skip.
            2. **File-level**: caller's file defines another class with it -> skip.
            3. **Import-level**: caller's file doesn't import the target class -> skip.
            4. **Attribute-level**: receiver tokens don't match class name -> skip.

        Trusted entries (``_trusted=True``, from local var type analysis) bypass
        Levels 1-3 since we already resolved the exact target class statically.
        """
        with self._lock:
            callers = list(self._caller_index.get(function_name, []))
            if not class_name or not callers:
                return [_format_caller_output(c) for c in callers]

            # Determine if this method name is ambiguous: defined in 2+ classes.
            # Only when ambiguous do we need the expensive Level 3 (import) and
            # Level 4 (attribute) filtering. Non-ambiguous methods (e.g., a unique
            # "process_payment") can skip these checks entirely.
            ambiguous = sum(
                1 for cls_n, meths in self._class_method_names.items()
                if function_name in meths
            ) >= 2

            allowed_files: set[str] | None = None
            if ambiguous:
                allowed_files = self._get_class_importer_files(class_name)

            # Pre-compute class name tokens for Level 4 receiver matching.
            # split_tokens("EpisodicMemory") -> {"episodic", "memory"} so we
            # can check if any receiver token overlaps with the class name.
            class_name_tokens: set[str] | None = None
            if ambiguous and class_name:
                class_name_tokens = set(split_tokens(class_name))

            filtered = []
            for caller in callers:
                caller_class = caller.get("class")
                caller_file = caller.get("file", "")

                # Same class -> always keep
                if caller_class == class_name:
                    filtered.append(caller)
                    continue

                # Trusted entries (from _augment_index_with_local_var_types) are
                # type-resolved -- we already know which class the call targets.
                # Skip import-graph filtering (Levels 1-3) for them.
                _trusted = caller.get("_trusted", False)

                # Level 1: caller's own class has this method -> skip
                if caller_class and not _trusted:
                    if function_name in self._class_method_names.get(caller_class, set()):
                        continue

                # Level 2: file-level fallback -- if caller's file defines
                # ANY other class with this method, the call is likely local.
                if caller_file and not _trusted:
                    file_classes = self._file_class_methods.get(caller_file, {})
                    has_local = any(
                        function_name in meths
                        for cls_n, meths in file_classes.items()
                        if cls_n != class_name
                    )
                    if has_local:
                        continue

                # Level 3: import-based -- for ambiguous method names,
                # only keep callers from files that import the target class's module.
                if allowed_files is not None and caller_file and not _trusted:
                    if caller_file not in allowed_files:
                        continue

                # Level 4: source-based attribute analysis -- when the caller's
                # source shows it calls .method() on a receiver whose name
                # tokens don't match the target class, filter it out.
                # e.g., self.short_term.add() -> "short_term" doesn't match
                # "EpisodicMemory" -> filter.
                if class_name_tokens is not None:
                    call_receivers = caller.get("_receivers")
                    if call_receivers:
                        any_match = False
                        all_have_context = True
                        for receiver in call_receivers:
                            recv_tokens = set(split_tokens(receiver))
                            recv_tokens.discard("self")
                            if not recv_tokens:
                                # Bare self.method() -- can't determine target
                                all_have_context = False
                                break
                            if recv_tokens & class_name_tokens:
                                any_match = True
                                break
                        if all_have_context and not any_match:
                            continue

                filtered.append(caller)
            return [_format_caller_output(c) for c in filtered]

    # --- Import-Based Allowlist for Level 3 Filtering ---

    def _get_class_importer_files(self, class_name: str) -> set[str]:
        """Get all files that could legitimately reference a class.

        Returns the union of:
            1. Files where the class is defined.
            2. Files that import any of those defining files.
            3. Files that import the package ``__init__.py`` / ``index.ts``
               of the defining directory (handles re-exports).

        The result is used as an allowlist for Level 3 filtering: if a
        caller file isn't in this set, it can't be calling *this* class's
        method and should be filtered out.
        """
        defining_files = self._class_defined_in.get(class_name, set())
        allowed = set(defining_files)
        for def_file in defining_files:
            # Files that import the defining module
            importers = self._import_index.get(def_file, [])
            allowed.update(importers)
            # Also check if the defining file is in a package -- files importing
            # the package __init__.py might re-export the class.
            def_norm = def_file.replace("\\", "/")
            dir_path = def_norm.rsplit("/", 1)[0] if "/" in def_norm else ""
            if dir_path:
                for init_name in ("__init__.py", "index.ts", "index.js"):
                    init_path = dir_path + "/" + init_name
                    init_importers = self._import_index.get(init_path, [])
                    allowed.update(init_importers)
        return allowed

    # --- Method-to-Class Resolution ---

    def resolve_method_class(self, method_name: str, file_path: str) -> str | None:
        """Given a method name and file, determine which class it belongs to.

        Returns the class name if exactly one class in the file defines
        this method. Returns None if ambiguous (multiple classes define it)
        or not found. Used by the MCP server to auto-detect class context
        when only a method name is provided.
        """
        with self._lock:
            file_classes = self._file_class_methods.get(file_path, {})
            candidates = [
                cls_name for cls_name, methods in file_classes.items()
                if method_name in methods
            ]
            return candidates[0] if len(candidates) == 1 else None
