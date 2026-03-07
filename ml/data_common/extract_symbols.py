"""Extract symbols from cloned repos for training data generation."""
import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).parent.parent / ".repo_cache"


def extract_from_repo(repo_path: Path) -> list[dict[str, Any]]:
    """Extract all symbols from a repo using graph-sitter."""
    from graph_sitter import CodebaseGraph
    try:
        graph = CodebaseGraph(str(repo_path))
        symbols = []
        for lang, funcs in graph.functions.items():
            for func in funcs:
                symbols.append({
                    "type": "function",
                    "name": getattr(func, "name", ""),
                    "file": getattr(func, "file_path", ""),
                    "docstring": getattr(func, "docstring", "") or "",
                    "source": getattr(func, "source_code", "") or "",
                    "params": [getattr(p, "name", "") for p in getattr(func, "parameters", [])],
                    "decorators": [str(d) for d in getattr(func, "decorators", [])],
                    "class_name": getattr(func, "class_name", None),
                    "calls": [getattr(c, "name", "") for c in getattr(func, "function_calls", [])],
                    "language": lang,
                })
        for lang, classes in graph.classes.items():
            for cls in classes:
                symbols.append({
                    "type": "class",
                    "name": getattr(cls, "name", ""),
                    "file": getattr(cls, "file_path", ""),
                    "docstring": getattr(cls, "docstring", "") or "",
                    "source": getattr(cls, "source_code", "") or "",
                    "methods": [getattr(m, "name", "") for m in getattr(cls, "methods", [])],
                    "base_classes": [str(b) for b in getattr(cls, "base_classes", [])],
                    "language": lang,
                })
        return symbols
    except Exception as e:
        print(f"  Error extracting {repo_path}: {e}")
        return []


def extract_all(output_path: Path | None = None) -> list[dict]:
    """Extract symbols from all cached repos."""
    all_symbols = []
    for lang_dir in sorted(CACHE_DIR.iterdir()):
        if not lang_dir.is_dir():
            continue
        for repo_dir in sorted(lang_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            print(f"  Extracting {lang_dir.name}/{repo_dir.name}...")
            symbols = extract_from_repo(repo_dir)
            for s in symbols:
                s["repo"] = f"{lang_dir.name}/{repo_dir.name}"
            all_symbols.extend(symbols)
    if output_path:
        output_path.write_text(json.dumps(all_symbols, indent=2))
    print(f"  Total: {len(all_symbols)} symbols")
    return all_symbols


if __name__ == "__main__":
    extract_all(Path(__file__).parent.parent / "all_symbols.json")
