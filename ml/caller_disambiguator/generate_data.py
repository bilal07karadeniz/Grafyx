"""Generate 200K training examples for Caller Disambiguator.

Strategy: Scan type-annotated Python code for ground truth.
For each method call with dot syntax where type is known:
  - POSITIVE: (features, correct_class.method) -> 1
  - NEGATIVE: (features, wrong_class.method)   -> 0

Also generates synthetic examples from extracted symbols to cover
edge cases (self.method, standalone vs method, same-name methods).
"""

import ast
import json
import random
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features, _split_tokens, FEATURE_COUNT

DATA_DIR = Path(__file__).parent / "data"
SYMBOLS_FILE = Path(__file__).parent.parent / "all_symbols.json"
CACHE_DIR = Path(__file__).parent.parent / ".repo_cache"


# ── Type annotation extraction ────────────────────────────────────


class TypeAnnotationVisitor(ast.NodeVisitor):
    """Extract local variable type annotations and function call sites."""

    def __init__(self, source: str, file_path: str):
        self.source = source
        self.file_path = file_path
        self.lines = source.split("\n")
        # Map variable name -> annotated type
        self.var_types: dict[str, str] = {}
        # Collected call sites: (receiver, method, arg_count, type_annotation)
        self.call_sites: list[dict] = []
        # Imports in this file
        self.imports: list[str] = []
        # Current function LOC
        self.current_func_loc = 0

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Capture type annotations: `x: SomeType = ...`."""
        if isinstance(node.target, ast.Name) and isinstance(node.annotation, ast.Name):
            self.var_types[node.target.id] = node.annotation.id
        elif isinstance(node.target, ast.Name) and isinstance(node.annotation, ast.Attribute):
            self.var_types[node.target.id] = ast.dump(node.annotation)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract parameter type annotations."""
        self.current_func_loc = node.end_lineno - node.lineno + 1 if node.end_lineno else 0
        for arg in node.args.args:
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                self.var_types[arg.arg] = arg.annotation.id
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call):
        """Extract method call sites with dot syntax."""
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            receiver = self._extract_receiver(node.func.value)
            if receiver:
                type_ann = self.var_types.get(receiver, "")
                self.call_sites.append({
                    "receiver": receiver,
                    "method": method_name,
                    "arg_count": len(node.args),
                    "type_annotation": type_ann,
                    "has_dot_syntax": True,
                    "file": self.file_path,
                    "caller_loc": self.current_func_loc,
                })
        self.generic_visit(node)

    def _extract_receiver(self, node: ast.expr) -> str | None:
        """Get the receiver name from an attribute access."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._extract_receiver(node.value)
            if base:
                return f"{base}.{node.attr}"
        return None


def extract_call_sites_from_file(file_path: Path) -> list[dict]:
    """Parse a Python file and extract typed method calls."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    visitor = TypeAnnotationVisitor(source, str(file_path))
    visitor.visit(tree)

    # Only keep call sites where we know the type
    typed_calls = [
        cs for cs in visitor.call_sites if cs["type_annotation"]
    ]
    for cs in typed_calls:
        cs["imports"] = visitor.imports
    return typed_calls


# ── Example generation ────────────────────────────────────────────


def _build_method_index(
    symbols: list[dict],
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Build method_name -> [symbol] index and frequency map."""
    method_index: dict[str, list[dict]] = defaultdict(list)
    name_counts: Counter = Counter()

    for sym in symbols:
        if sym["type"] == "function":
            name = sym["name"]
            method_index[name].append(sym)
            name_counts[name] += 1

    total = sum(name_counts.values()) or 1
    name_freq = {name: count / total for name, count in name_counts.items()}
    return dict(method_index), name_freq


def _make_example(
    call_site: dict,
    callee: dict,
    method_count: int,
    name_freq: float,
    label: int,
) -> dict:
    """Build a training example from a call site and callee candidate."""
    decorators = callee.get("decorators", [])
    features = extract_features(
        receiver_text=call_site.get("receiver", ""),
        method_name=call_site.get("method", ""),
        caller_file=call_site.get("file", ""),
        caller_imports=call_site.get("imports", []),
        arg_count=call_site.get("arg_count", 0),
        has_dot_syntax=call_site.get("has_dot_syntax", False),
        caller_loc=call_site.get("caller_loc", 0),
        callee_name=callee.get("name", ""),
        callee_class_name=callee.get("class_name", "") or "",
        callee_file=callee.get("file", ""),
        callee_module=_extract_module(callee.get("file", "")),
        callee_package=_extract_package(callee.get("file", "")),
        callee_param_count=len(callee.get("params", [])),
        callee_is_method=callee.get("class_name") is not None,
        callee_decorators=decorators,
        method_count_with_same_name=method_count,
        method_name_frequency=name_freq,
        receiver_type_annotation=call_site.get("type_annotation", ""),
    )
    return {"features": features.tolist(), "label": label}


def _extract_module(file_path: str) -> str:
    """Extract module name from file path (last component without .py)."""
    if not file_path:
        return ""
    parts = file_path.replace("\\", "/").split("/")
    name = parts[-1] if parts else ""
    return name.replace(".py", "").replace(".ts", "").replace(".js", "")


def _extract_package(file_path: str) -> str:
    """Extract top-level package from file path."""
    if not file_path:
        return ""
    parts = file_path.replace("\\", "/").split("/")
    # Skip common prefixes like 'src', 'lib'
    for p in parts:
        if p not in ("src", "lib", "app", ""):
            return p
    return parts[0] if parts else ""


def generate_from_typed_calls(
    symbols: list[dict],
    method_index: dict[str, list[dict]],
    name_freq: dict[str, float],
    target: int = 100_000,
    rng: random.Random | None = None,
) -> list[dict]:
    """Generate examples from type-annotated call sites in repos."""
    rng = rng or random.Random(42)
    examples: list[dict] = []

    # Scan cloned repos for typed calls
    if not CACHE_DIR.exists():
        print("  WARNING: .repo_cache not found, skipping typed call extraction")
        return examples

    all_call_sites: list[dict] = []
    py_files = list(CACHE_DIR.rglob("*.py"))
    rng.shuffle(py_files)

    print(f"  Scanning {len(py_files)} Python files for typed calls...")
    for i, py_file in enumerate(py_files[:5000]):  # Cap at 5K files
        sites = extract_call_sites_from_file(py_file)
        all_call_sites.extend(sites)
        if (i + 1) % 1000 == 0:
            print(f"    processed {i + 1} files, found {len(all_call_sites)} typed calls")

    print(f"  Found {len(all_call_sites)} typed call sites")

    # For each typed call, create positive (matching class) + negative (wrong class)
    for cs in all_call_sites:
        method = cs["method"]
        type_name = cs["type_annotation"]

        candidates = method_index.get(method, [])
        if not candidates:
            continue

        # Find correct match: callee whose class_name matches type annotation
        correct = [
            c for c in candidates
            if c.get("class_name")
            and _tokens_overlap(c["class_name"], type_name)
        ]
        wrong = [
            c for c in candidates
            if c.get("class_name")
            and not _tokens_overlap(c["class_name"], type_name)
        ]

        mc = len(candidates)
        freq = name_freq.get(method, 0.0)

        for callee in correct:
            examples.append(_make_example(cs, callee, mc, freq, label=1))
        for callee in wrong[:3]:  # Limit negatives per call site
            examples.append(_make_example(cs, callee, mc, freq, label=0))

        if len(examples) >= target:
            break

    return examples[:target]


def _tokens_overlap(a: str, b: str) -> bool:
    """Check if two strings share any tokens."""
    ta = set(_split_tokens(a))
    tb = set(_split_tokens(b))
    return bool(ta & tb)


def generate_synthetic(
    symbols: list[dict],
    method_index: dict[str, list[dict]],
    name_freq: dict[str, float],
    target: int = 100_000,
    rng: random.Random | None = None,
) -> list[dict]:
    """Generate synthetic examples from symbol data."""
    rng = rng or random.Random(43)
    examples: list[dict] = []

    methods = [s for s in symbols if s["type"] == "function" and s.get("class_name")]
    standalones = [s for s in symbols if s["type"] == "function" and not s.get("class_name")]
    classes = [s for s in symbols if s["type"] == "class"]

    if not methods:
        print("  WARNING: No methods found in symbols")
        return examples

    # Class name -> methods
    class_methods: dict[str, list[dict]] = defaultdict(list)
    for m in methods:
        class_methods[m["class_name"]].append(m)

    # 1. self.method() -> correct class (positives)
    print("  Generating self.method positives...")
    for _ in range(target // 4):
        m = rng.choice(methods)
        cs = {
            "receiver": "self",
            "method": m["name"],
            "file": m["file"],
            "imports": [],
            "arg_count": max(0, len(m.get("params", [])) - 1),
            "has_dot_syntax": True,
            "caller_loc": rng.randint(10, 100),
            "type_annotation": m["class_name"],
        }
        mc = len(method_index.get(m["name"], []))
        freq = name_freq.get(m["name"], 0.0)
        examples.append(_make_example(cs, m, mc, freq, label=1))

    # 2. self.method() -> wrong class (negatives)
    print("  Generating self.method negatives...")
    for _ in range(target // 4):
        m = rng.choice(methods)
        wrong_m = rng.choice(methods)
        if wrong_m["class_name"] == m["class_name"]:
            continue
        cs = {
            "receiver": "self",
            "method": m["name"],
            "file": m["file"],
            "imports": [],
            "arg_count": max(0, len(m.get("params", [])) - 1),
            "has_dot_syntax": True,
            "caller_loc": rng.randint(10, 100),
            "type_annotation": m["class_name"],
        }
        mc = len(method_index.get(m["name"], []))
        freq = name_freq.get(m["name"], 0.0)
        examples.append(_make_example(cs, wrong_m, mc, freq, label=0))

    # 3. db.method() -> correct class (typed receiver positives)
    print("  Generating typed receiver positives...")
    for _ in range(target // 6):
        if not classes:
            break
        cls = rng.choice(classes)
        cls_methods_list = class_methods.get(cls["name"], [])
        if not cls_methods_list:
            continue
        m = rng.choice(cls_methods_list)
        receiver_name = _make_receiver_name(cls["name"])
        cs = {
            "receiver": receiver_name,
            "method": m["name"],
            "file": rng.choice(symbols)["file"],
            "imports": [_extract_module(cls["file"])],
            "arg_count": max(0, len(m.get("params", [])) - 1),
            "has_dot_syntax": True,
            "caller_loc": rng.randint(10, 150),
            "type_annotation": cls["name"],
        }
        mc = len(method_index.get(m["name"], []))
        freq = name_freq.get(m["name"], 0.0)
        examples.append(_make_example(cs, m, mc, freq, label=1))

    # 4. standalone function call confused with method (negatives)
    print("  Generating standalone vs method negatives...")
    for _ in range(target // 6):
        if not standalones:
            break
        func = rng.choice(standalones)
        m = rng.choice(methods)
        if func["name"] != m["name"]:
            continue
        # Call site calls standalone, but candidate is the method
        cs = {
            "receiver": "",
            "method": func["name"],
            "file": func["file"],
            "imports": [_extract_module(func["file"])],
            "arg_count": len(func.get("params", [])),
            "has_dot_syntax": False,
            "caller_loc": rng.randint(5, 80),
            "type_annotation": "",
        }
        mc = len(method_index.get(func["name"], []))
        freq = name_freq.get(func["name"], 0.0)
        examples.append(_make_example(cs, m, mc, freq, label=0))

    # 5. Same-name method on different classes (disambiguation)
    print("  Generating same-name disambiguation...")
    ambiguous_names = [
        name for name, syms in method_index.items()
        if len([s for s in syms if s.get("class_name")]) >= 2
    ]
    for _ in range(target // 6):
        if not ambiguous_names:
            break
        name = rng.choice(ambiguous_names)
        candidates = [s for s in method_index[name] if s.get("class_name")]
        if len(candidates) < 2:
            continue
        correct = rng.choice(candidates)
        wrong = rng.choice([c for c in candidates if c["class_name"] != correct["class_name"]])
        receiver_name = _make_receiver_name(correct["class_name"])
        cs = {
            "receiver": receiver_name,
            "method": name,
            "file": correct["file"],
            "imports": [_extract_module(correct["file"])],
            "arg_count": max(0, len(correct.get("params", [])) - 1),
            "has_dot_syntax": True,
            "caller_loc": rng.randint(10, 120),
            "type_annotation": correct["class_name"],
        }
        mc = len(candidates)
        freq = name_freq.get(name, 0.0)
        examples.append(_make_example(cs, correct, mc, freq, label=1))
        examples.append(_make_example(cs, wrong, mc, freq, label=0))

    return examples[:target]


def _make_receiver_name(class_name: str) -> str:
    """Generate a plausible receiver variable name from a class name.

    E.g., 'DatabaseSession' -> 'db_session' or 'session'.
    """
    tokens = _split_tokens(class_name)
    if not tokens:
        return class_name.lower()
    if len(tokens) == 1:
        return tokens[0]
    # Use last 1-2 tokens as variable name
    return "_".join(tokens[-2:])


def generate_all(target_count: int = 200_000):
    """Generate all training data for caller disambiguation."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not SYMBOLS_FILE.exists():
        print(f"ERROR: {SYMBOLS_FILE} not found.")
        print("Run: python -m ml.data_common.extract_symbols")
        return

    symbols = json.loads(SYMBOLS_FILE.read_text())
    print(f"Loaded {len(symbols)} symbols")

    method_index, name_freq = _build_method_index(symbols)
    print(f"  {len(method_index)} unique method/function names")

    rng = random.Random(42)
    all_examples: list[dict] = []

    # Phase 1: From typed call sites in real code
    print("\n[Phase 1] Extracting from typed call sites...")
    typed_examples = generate_from_typed_calls(
        symbols, method_index, name_freq,
        target=target_count // 2, rng=rng,
    )
    print(f"  Generated {len(typed_examples)} typed examples")
    all_examples.extend(typed_examples)

    # Phase 2: Synthetic examples
    remaining = target_count - len(all_examples)
    print(f"\n[Phase 2] Generating {remaining} synthetic examples...")
    synthetic_examples = generate_synthetic(
        symbols, method_index, name_freq,
        target=remaining, rng=rng,
    )
    print(f"  Generated {len(synthetic_examples)} synthetic examples")
    all_examples.extend(synthetic_examples)

    # Shuffle and split
    rng.shuffle(all_examples)
    all_examples = all_examples[:target_count]

    n = len(all_examples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    splits = {
        "train": all_examples[:train_end],
        "val": all_examples[train_end:val_end],
        "test": all_examples[val_end:],
    }

    print(f"\nTotal examples: {n}")
    for split_name, split_data in splits.items():
        path = DATA_DIR / f"{split_name}.jsonl"
        pos = sum(1 for ex in split_data if ex["label"] == 1)
        neg = len(split_data) - pos
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        print(f"  {split_name}: {len(split_data)} (pos={pos}, neg={neg}) -> {path}")


if __name__ == "__main__":
    generate_all()
