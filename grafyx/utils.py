"""Shared utilities for Grafyx MCP server."""

import json
import os
import re
from pathlib import Path
from typing import Any


# Language extension mappings
LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py", ".pyi"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
}

# Reverse map: extension -> language
EXTENSION_TO_LANGUAGE: dict[str, str] = {}
for _lang, _exts in LANGUAGE_EXTENSIONS.items():
    for _ext in _exts:
        EXTENSION_TO_LANGUAGE[_ext] = _lang

# Directories/patterns to ignore during scanning
DEFAULT_IGNORE_PATTERNS: list[str] = [
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "egg-info",
    ".eggs",
    ".next",
    ".nuxt",
    "coverage",
    ".coverage",
    ".nyc_output",
]


def detect_languages(project_path: str, ignore_dirs: list[str] | None = None) -> list[str]:
    """Walk the project tree and detect which programming languages are present.

    Only returns languages that have at least 2 source files (to filter noise).
    Respects ignore patterns to skip irrelevant directories.

    Args:
        project_path: Absolute path to the project root.
        ignore_dirs: Additional directory names to skip. Merged with DEFAULT_IGNORE_PATTERNS.

    Returns:
        List of detected language names, e.g. ["python", "typescript"].
    """
    ignore_set = set(DEFAULT_IGNORE_PATTERNS)
    if ignore_dirs:
        ignore_set.update(ignore_dirs)

    language_file_counts: dict[str, int] = {}

    for root, dirs, files in os.walk(project_path):
        # Filter out ignored directories (modifying dirs in-place to prevent os.walk from descending)
        dirs[:] = [d for d in dirs if d not in ignore_set and not d.startswith('.')]

        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext in EXTENSION_TO_LANGUAGE:
                lang = EXTENSION_TO_LANGUAGE[ext]
                language_file_counts[lang] = language_file_counts.get(lang, 0) + 1

    # Only return languages with >= 2 source files
    detected = [lang for lang, count in language_file_counts.items() if count >= 2]

    # Graph-sitter only supports 'python' and 'typescript'.
    # JavaScript is parsed by the TypeScript parser, so merge JS → TS.
    if "javascript" in detected:
        detected.remove("javascript")
        if "typescript" not in detected:
            detected.append("typescript")

    return detected


def truncate_response(data: Any, max_chars: int = 50_000) -> Any:
    """Truncate a response if its JSON representation exceeds max_chars.

    Prevents oversized responses that would blow up AI context windows.
    If truncation occurs, a '_truncated' key is added to dict responses,
    or a note is appended to string responses.

    Args:
        data: The response data (dict, list, or str).
        max_chars: Maximum allowed characters in JSON-serialized form.

    Returns:
        The original data if under limit, or a truncated version with a note.
    """
    serialized = json.dumps(data, default=str)
    if len(serialized) <= max_chars:
        return data

    if isinstance(data, dict):
        data["_truncated"] = True
        data["_truncated_note"] = (
            f"Response truncated. Original size: {len(serialized)} chars. "
            f"Use more specific queries or set max_results/max_depth to reduce output."
        )
        # Try to trim list values
        trimmed = {}
        for key, value in data.items():
            if key.startswith("_"):
                continue
            if isinstance(value, list) and len(value) > 10:
                trimmed[key] = value[:10]
                trimmed[f"_{key}_total"] = len(value)
        data.update(trimmed)
        return data
    elif isinstance(data, list):
        return {
            "items": data[:20],
            "_truncated": True,
            "_total": len(data),
            "_truncated_note": "Response truncated. Use max_results to limit output.",
        }
    elif isinstance(data, str):
        return data[:max_chars] + "\n\n[... truncated]"

    return data


def format_function_signature(func: Any) -> str:
    """Format a graph-sitter Function object into a human-readable signature.

    Args:
        func: A graph-sitter Function object.

    Returns:
        String like 'async def func_name(param1: type1, param2: type2) -> ReturnType'
    """
    try:
        prefix = "async def" if safe_get_attr(func, "is_async", False) else "def"
        name = safe_get_attr(func, "name", "unknown")

        # Build parameter string
        params = safe_get_attr(func, "parameters", [])
        param_parts = []
        if params:
            for p in params:
                p_name = safe_get_attr(p, "name", "?")
                p_type = safe_get_attr(p, "type", None)
                p_default = safe_get_attr(p, "default", None)
                part = p_name
                if p_type:
                    part += f": {p_type}"
                if p_default is not None:
                    part += f" = {p_default}"
                param_parts.append(part)

        param_str = ", ".join(param_parts)

        return_type = safe_get_attr(func, "return_type", None)
        ret_str = f" -> {return_type}" if return_type else ""

        return f"{prefix} {name}({param_str}){ret_str}"
    except Exception:
        return f"def {safe_get_attr(func, 'name', 'unknown')}(...)"


def extract_base_classes(cls: Any) -> list[str]:
    """Extract base class names from a graph-sitter Class object.

    Tries multiple attribute names to handle graph-sitter version differences,
    with a regex fallback on the class source code.

    Args:
        cls: A graph-sitter Class object.

    Returns:
        List of base class name strings.
    """
    # Try direct attribute access in priority order
    for attr_name in ("base_classes", "parent_classes", "parent_class_names"):
        bases = safe_get_attr(cls, attr_name, None)
        if bases is not None:
            result = [str(b) for b in bases if str(b).strip()]
            if result:
                return result

    # Try callable: superclasses()
    superclasses_fn = safe_get_attr(cls, "superclasses", None)
    if callable(superclasses_fn):
        try:
            bases = superclasses_fn()
            if bases:
                result = [str(b) for b in bases if str(b).strip()]
                if result:
                    return result
        except Exception:
            pass

    # Fallback: parse the class source code
    source = safe_str(safe_get_attr(cls, "source", ""))
    if source:
        match = re.match(r"class\s+\w+\s*\(([^)]+)\)\s*:", source)
        if match:
            bases_str = match.group(1)
            bases = [b.strip() for b in bases_str.split(",") if b.strip()]
            # Filter out bare "object" base class (implicit in Python 3)
            bases = [b for b in bases if b not in ("object",)]
            if bases:
                return bases

    return []


def format_class_summary(cls: Any) -> dict:
    """Format a graph-sitter Class object into a summary dict.

    Args:
        cls: A graph-sitter Class object.

    Returns:
        Dict with keys: name, file, base_classes, method_count, property_count, docstring.
    """
    methods = safe_get_attr(cls, "methods", [])
    properties = safe_get_attr(cls, "properties", [])
    base_classes = extract_base_classes(cls)
    docstring = safe_get_attr(cls, "docstring", "")

    return {
        "name": safe_get_attr(cls, "name", "unknown"),
        "base_classes": base_classes,
        "method_count": len(list(methods)) if methods else 0,
        "property_count": len(list(properties)) if properties else 0,
        "docstring": _first_line(safe_str(docstring)) if docstring else None,
    }


def format_file_summary(file_obj: Any) -> dict:
    """Format a graph-sitter File object into a summary dict.

    Args:
        file_obj: A graph-sitter File/SourceFile object.

    Returns:
        Dict with keys: path, function_count, class_count, import_count.
    """
    functions = safe_get_attr(file_obj, "functions", [])
    classes = safe_get_attr(file_obj, "classes", [])
    imports = safe_get_attr(file_obj, "imports", [])

    return {
        "path": str(safe_get_attr(file_obj, "path", safe_get_attr(file_obj, "filepath", "unknown"))),
        "function_count": len(list(functions)) if functions else 0,
        "class_count": len(list(classes)) if classes else 0,
        "import_count": len(list(imports)) if imports else 0,
    }


def build_directory_tree(files: list[str], project_path: str, max_depth: int = 3) -> str:
    """Build an ASCII directory tree from a list of file paths.

    Files deeper than max_depth are not shown individually, but their
    directories are preserved with a file count summary.

    Args:
        files: List of file paths (relative or absolute).
        project_path: Project root for making paths relative.
        max_depth: Maximum directory depth to show individual files.
                   Directories beyond this depth show a summary instead.

    Returns:
        ASCII tree string.
    """
    tree: dict = {}
    # Track file counts and first few filenames for directories beyond max_depth
    deep_dir_counts: dict[str, int] = {}
    deep_dir_preview: dict[str, list[str]] = {}  # first 3 filenames per deep dir
    project = Path(project_path)

    for file_path in files:
        try:
            rel_path = Path(file_path).relative_to(project)
        except ValueError:
            rel_path = Path(file_path)

        parts = list(rel_path.parts)

        if len(parts) > max_depth + 1:
            # File is deeper than max_depth.
            # Add directory structure up to max_depth+1 level,
            # and count files for a summary node.
            dir_parts = parts[:max_depth + 1]
            dir_key = "/".join(dir_parts)
            deep_dir_counts[dir_key] = deep_dir_counts.get(dir_key, 0) + 1
            # Track first 3 filenames for preview
            if dir_key not in deep_dir_preview:
                deep_dir_preview[dir_key] = []
            if len(deep_dir_preview[dir_key]) < 3:
                deep_dir_preview[dir_key].append(parts[-1])

            # Insert the directory structure into the tree
            current = tree
            for part in dir_parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
        else:
            # File is within max_depth, add it normally
            current = tree
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

    # Insert preview files + summary nodes for deep directories
    for dir_key, count in deep_dir_counts.items():
        dir_parts = dir_key.split("/")
        current = tree
        for part in dir_parts:
            if part not in current:
                current[part] = {}
            current = current[part]
        # Show first 3 files as preview
        preview = deep_dir_preview.get(dir_key, [])
        for fname in preview:
            current[fname] = {}
        remaining = count - len(preview)
        if remaining > 0:
            summary_key = f"... ({remaining} more files)"
            current[summary_key] = {}

    lines = []
    _build_tree_lines(tree, lines, "")
    return "\n".join(lines) if lines else "(empty project)"


def _build_tree_lines(tree: dict, lines: list[str], prefix: str) -> None:
    """Recursively build tree lines."""
    items = sorted(tree.items())
    for i, (name, subtree) in enumerate(items):
        is_last = i == len(items) - 1
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        lines.append(f"{prefix}{connector}{name}")

        if subtree:
            extension = "    " if is_last else "\u2502   "
            _build_tree_lines(subtree, lines, prefix + extension)


def safe_str(value: Any) -> str:
    """Convert any value to a plain string. Handles graph-sitter objects
    like PyCommentGroup that aren't plain strings."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return ""


def safe_get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute from an object.

    Args:
        obj: The object to read from.
        attr: Attribute name.
        default: Default value if attribute doesn't exist or is None.

    Returns:
        The attribute value, or default.
    """
    try:
        value = getattr(obj, attr, default)
        return value if value is not None else default
    except Exception:
        return default


def _first_line(text: str) -> str:
    """Return the first non-empty line of text."""
    for line in text.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith('"""') and not line.startswith("'''"):
            return line
    return text.strip()[:100] if text else ""


def split_tokens(text: str) -> list[str]:
    """Split text into lowercase tokens, handling camelCase and snake_case.

    'getUserData' -> ['get', 'user', 'data']
    'get_user_data' -> ['get', 'user', 'data']
    'HTTPResponse' -> ['http', 'response']

    Args:
        text: The identifier or text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    # First, split on underscores, hyphens, dots, spaces, slashes
    parts = re.split(r'[_\-.\s/\\]+', text)

    tokens = []
    for part in parts:
        if not part:
            continue
        # Split camelCase: insert boundary between lowercase-uppercase and uppercase-uppercase-lowercase
        # e.g., "getUserData" -> "get User Data", "HTTPResponse" -> "HTTP Response"
        sub_parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
        sub_parts = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', sub_parts)
        for token in sub_parts.split():
            if token:
                tokens.append(token.lower())

    return tokens
