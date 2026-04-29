#!/usr/bin/env python3
"""Extract symbols from real Python codebases using ast module.

Parses all .py files in cloned repos and extracts:
- Functions: name, file, docstring, params, decorators, line count
- Classes: name, file, docstring, methods, base classes, decorators
- Methods: name, class, file, docstring, params, decorators
- Imports: file-to-file import graph
- Call sites: method calls with receiver expressions

Output: JSON file with all extracted symbols for training data generation.
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any


def _get_docstring(node: ast.AST) -> str:
    """Extract docstring from a function or class node."""
    try:
        return ast.get_docstring(node) or ""
    except Exception:
        return ""


def _get_decorator_names(node: ast.AST) -> list[str]:
    """Extract decorator names from a function or class."""
    names = []
    for dec in getattr(node, "decorator_list", []):
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(f"{_get_full_name(dec.value)}.{dec.attr}")
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                names.append(dec.func.id)
            elif isinstance(dec.func, ast.Attribute):
                names.append(f"{_get_full_name(dec.func.value)}.{dec.func.attr}")
    return names


def _get_full_name(node: ast.AST) -> str:
    """Get the full dotted name of a node (e.g., 'self.db.session')."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_get_full_name(node.value)}.{node.attr}"
    return ""


def _get_param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Extract parameter names from a function."""
    params = []
    for arg in node.args.args:
        params.append(arg.arg)
    for arg in node.args.kwonlyargs:
        params.append(arg.arg)
    if node.args.vararg:
        params.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        params.append(f"**{node.args.kwarg.arg}")
    return params


def _count_lines(node: ast.AST) -> int:
    """Count lines in a function/class body."""
    if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
        return node.end_lineno - node.lineno + 1
    return 0


def extract_from_file(filepath: str, repo_name: str) -> dict:
    """Extract all symbols from a single Python file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except Exception:
        return {"functions": [], "classes": [], "imports": [], "calls": [], "source": ""}

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return {"functions": [], "classes": [], "imports": [], "calls": [], "source": ""}

    rel_path = os.path.relpath(filepath, os.path.dirname(os.path.dirname(filepath)))
    rel_path = rel_path.replace("\\", "/")

    functions = []
    classes = []
    imports = []
    calls = []

    # Extract imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"type": "import", "module": alias.name,
                                "alias": alias.asname})
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({"type": "from", "module": module,
                                "name": alias.name, "alias": alias.asname})

    # Extract top-level functions and classes
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_data = {
                "name": node.name,
                "file": rel_path,
                "repo": repo_name,
                "docstring": _get_docstring(node),
                "params": _get_param_names(node),
                "decorators": _get_decorator_names(node),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "line_count": _count_lines(node),
                "is_method": False,
                "class_name": "",
            }
            functions.append(func_data)

            # Extract call sites from function body
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    call_info = _extract_call(child)
                    if call_info:
                        call_info["caller"] = node.name
                        call_info["caller_file"] = rel_path
                        call_info["caller_class"] = ""
                        calls.append(call_info)

        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_data = {
                        "name": item.name,
                        "file": rel_path,
                        "repo": repo_name,
                        "docstring": _get_docstring(item),
                        "params": _get_param_names(item),
                        "decorators": _get_decorator_names(item),
                        "is_async": isinstance(item, ast.AsyncFunctionDef),
                        "line_count": _count_lines(item),
                        "is_method": True,
                        "class_name": node.name,
                    }
                    methods.append(method_data)
                    functions.append(method_data)

                    # Extract calls from methods
                    for child in ast.walk(item):
                        if isinstance(child, ast.Call):
                            call_info = _extract_call(child)
                            if call_info:
                                call_info["caller"] = item.name
                                call_info["caller_file"] = rel_path
                                call_info["caller_class"] = node.name
                                calls.append(call_info)

            bases = []
            for base in node.bases:
                bases.append(_get_full_name(base))

            cls_data = {
                "name": node.name,
                "file": rel_path,
                "repo": repo_name,
                "docstring": _get_docstring(node),
                "methods": [m["name"] for m in methods],
                "method_details": methods,
                "base_classes": bases,
                "decorators": _get_decorator_names(node),
                "line_count": _count_lines(node),
            }
            classes.append(cls_data)

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "calls": calls,
        "source": source[:5000],  # First 5K chars for source token extraction
    }


def _extract_call(node: ast.Call) -> dict | None:
    """Extract call site info from an ast.Call node."""
    if isinstance(node.func, ast.Attribute):
        receiver = _get_full_name(node.func.value)
        method = node.func.attr
        return {
            "method": method,
            "receiver": receiver,
            "has_dot_syntax": True,
            "arg_count": len(node.args) + len(node.keywords),
        }
    elif isinstance(node.func, ast.Name):
        return {
            "method": node.func.id,
            "receiver": "",
            "has_dot_syntax": False,
            "arg_count": len(node.args) + len(node.keywords),
        }
    return None


def extract_repo(repo_path: str) -> dict:
    """Extract all symbols from a repository."""
    repo_name = os.path.basename(repo_path)
    all_functions = []
    all_classes = []
    all_imports = {}  # file -> list of imports
    all_calls = []
    all_sources = {}  # file -> source snippet

    py_files = list(Path(repo_path).rglob("*.py"))
    # Skip venv, .git, node_modules, etc.
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv",
                 ".tox", ".eggs", "dist", "build", ".mypy_cache"}

    for py_file in py_files:
        parts = py_file.parts
        if any(p in skip_dirs for p in parts):
            continue

        data = extract_from_file(str(py_file), repo_name)
        all_functions.extend(data["functions"])
        all_classes.extend(data["classes"])
        rel_path = os.path.relpath(str(py_file),
                                   os.path.dirname(repo_path)).replace("\\", "/")
        all_imports[rel_path] = data["imports"]
        all_calls.extend(data["calls"])
        if data["source"]:
            all_sources[rel_path] = data["source"]

    return {
        "repo": repo_name,
        "functions": all_functions,
        "classes": all_classes,
        "imports": all_imports,
        "calls": all_calls,
        "sources": all_sources,
        "stats": {
            "files": len(py_files),
            "functions": len(all_functions),
            "classes": len(all_classes),
            "calls": len(all_calls),
        },
    }


def main():
    repos_dir = Path(__file__).parent / "real_repos"
    if not repos_dir.exists():
        print(f"No repos dir at {repos_dir}")
        sys.exit(1)

    all_data = []
    total_stats = {"files": 0, "functions": 0, "classes": 0, "calls": 0}

    for repo_dir in sorted(repos_dir.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name.startswith("."):
            continue
        print(f"Extracting {repo_dir.name}...", end=" ", flush=True)
        data = extract_repo(str(repo_dir))
        all_data.append(data)
        for k in total_stats:
            total_stats[k] += data["stats"][k]
        print(f"{data['stats']['functions']} funcs, {data['stats']['classes']} classes, "
              f"{data['stats']['calls']} calls")

    print(f"\nTotal: {total_stats['files']} files, {total_stats['functions']} functions, "
          f"{total_stats['classes']} classes, {total_stats['calls']} calls")

    out_path = Path(__file__).parent / "real_symbols.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=1, ensure_ascii=False)
    print(f"Saved to {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
