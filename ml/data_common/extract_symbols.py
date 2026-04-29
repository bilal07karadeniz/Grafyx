"""Extract symbols from cloned repos for training data generation."""
import json
import sys
from pathlib import Path
from typing import Any

# Patch py_mini_racer compatibility (JSEvalException was renamed in newer versions)
try:
    import py_mini_racer._types as _pmt
    if not hasattr(_pmt, "JSEvalException"):
        _pmt.JSEvalException = type("JSEvalException", (Exception,), {})
except ImportError:
    pass

CACHE_DIR = Path(__file__).parent.parent / ".repo_cache"


def _safe_str(obj) -> str:
    """Safely convert any object to string."""
    try:
        s = str(obj)
        return s if isinstance(s, str) else ""
    except Exception:
        return ""


def _safe_str_list(objs) -> list[str]:
    """Safely convert a list of objects to strings."""
    return [_safe_str(o) for o in (objs or [])]


def _safe_name_list(objs) -> list[str]:
    """Get .name from a list of objects safely."""
    result = []
    for o in (objs or []):
        name = getattr(o, "name", None)
        if name:
            result.append(str(name))
    return result


def extract_from_repo(repo_path: Path, language: str) -> list[dict[str, Any]]:
    """Extract all symbols from a repo using graph-sitter."""
    from graph_sitter import Codebase
    try:
        graph = Codebase(str(repo_path), language=language)
        symbols = []
        for func in graph.functions:
            # Get class name safely
            parent = getattr(func, "parent_class", None)
            class_name = getattr(parent, "name", None) if parent else None

            symbols.append({
                "type": "function",
                "name": str(getattr(func, "name", "") or ""),
                "file": str(getattr(func, "filepath", "") or getattr(func, "file_path", "") or ""),
                "docstring": str(getattr(func, "docstring", "") or ""),
                "source": str(getattr(func, "source", "") or getattr(func, "source_code", "") or "")[:3000],
                "params": _safe_name_list(getattr(func, "parameters", [])),
                "decorators": _safe_str_list(getattr(func, "decorators", [])),
                "class_name": str(class_name) if class_name else None,
                "calls": _safe_name_list(getattr(func, "function_calls", [])),
                "language": language,
            })
        for cls in graph.classes:
            supers = getattr(cls, "superclasses", None) or getattr(cls, "base_classes", [])
            symbols.append({
                "type": "class",
                "name": str(getattr(cls, "name", "") or ""),
                "file": str(getattr(cls, "filepath", "") or getattr(cls, "file_path", "") or ""),
                "docstring": str(getattr(cls, "docstring", "") or ""),
                "source": str(getattr(cls, "source", "") or getattr(cls, "source_code", "") or "")[:3000],
                "methods": _safe_name_list(getattr(cls, "methods", [])),
                "base_classes": _safe_str_list(supers),
                "language": language,
            })
        return symbols
    except Exception as e:
        print(f"  Error extracting {repo_path}: {e}", file=sys.stderr)
        return []


def extract_all(output_path: Path | None = None) -> list[dict]:
    """Extract symbols from all cached repos."""
    all_symbols = []
    for lang_dir in sorted(CACHE_DIR.iterdir()):
        if not lang_dir.is_dir():
            continue
        language = lang_dir.name  # "python" or "typescript"
        for repo_dir in sorted(lang_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            print(f"  Extracting {lang_dir.name}/{repo_dir.name}...")
            symbols = extract_from_repo(repo_dir, language)
            for s in symbols:
                s["repo"] = f"{lang_dir.name}/{repo_dir.name}"
            all_symbols.extend(symbols)
            print(f"    -> {len(symbols)} symbols")
    if output_path:
        output_path.write_text(json.dumps(all_symbols, indent=2))
    print(f"  Total: {len(all_symbols)} symbols")
    return all_symbols


if __name__ == "__main__":
    extract_all(Path(__file__).parent.parent / "all_symbols.json")
