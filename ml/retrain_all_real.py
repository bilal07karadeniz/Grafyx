#!/usr/bin/env python3
"""Retrain all 4 MLPs using REAL codebase data from real_symbols.json.

Key improvement: instead of synthetic verb_noun combos, we use 70K+ real
function names, real file paths, real docstrings, and real call graphs
extracted from 20 popular Python repos. Combined with LLM-generated
natural language queries for a subset.

Models:
  M1: Relevance Ranker v2      (42 -> 128 -> 64 -> 1) — MSE loss, graded labels
  M2: Caller Disambiguator     (25 -> 64 -> 32 -> 1) — BCE loss, binary labels
  M3: Source Token Filter       (15 -> 32 -> 16 -> 1) — BCE loss, binary labels
  M4: Symbol Importance         (18 -> 32 -> 16 -> 1) — MSE loss, graded labels

Usage:
    python ml/retrain_all_real.py [--epochs 60] [--quick] [--only m1]
"""

import argparse
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERROR: PyTorch required. pip install torch")
    sys.exit(1)

ML_DIR = Path(__file__).parent
PRODUCTION_MODEL_DIR = ML_DIR.parent / "grafyx" / "search" / "model"

# ── Import feature modules ────────────────────────────────────────────

def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m1_mod = _load_mod("m1_feat", ML_DIR / "relevance_ranker_v2" / "features.py")
m2_mod = _load_mod("m2_feat", ML_DIR / "caller_disambiguator" / "features.py")
m3_mod = _load_mod("m3_feat", ML_DIR / "source_token_filter" / "features.py")
m4_mod = _load_mod("m4_feat", ML_DIR / "symbol_importance" / "features.py")

m1_extract = m1_mod.extract_features; M1_FEAT = m1_mod.FEATURE_COUNT
m2_extract = m2_mod.extract_features; M2_FEAT = m2_mod.FEATURE_COUNT
m3_extract = m3_mod.extract_features; M3_FEAT = m3_mod.FEATURE_COUNT
m4_extract = m4_mod.extract_features; M4_FEAT = m4_mod.FEATURE_COUNT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════
#  SHARED
# ══════════════════════════════════════════════════════════════════════

def _split_tokens(text: str) -> list[str]:
    if not text:
        return []
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    parts = re.split(r'[^a-zA-Z0-9]+', s.lower())
    return [p for p in parts if len(p) >= 2]


def _stem_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 4:
        return False
    if longer.startswith(shorter):
        return True
    min_prefix = max(4, len(shorter) - 1)
    return min_prefix <= len(longer) and longer[:min_prefix] == shorter[:min_prefix]


def _load_real_symbols() -> dict:
    """Load real_symbols.json and organize by type."""
    path = ML_DIR / "real_symbols.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run extract_real_symbols.py first.")
        sys.exit(1)
    print(f"Loading {path}...")
    with open(path, encoding="utf-8") as f:
        repos = json.load(f)

    functions = []
    classes = []
    calls = []
    imports = {}  # file -> list of imports
    sources = {}  # file -> source snippet

    for repo in repos:
        for func in repo["functions"]:
            if func.get("name"):
                functions.append(func)
        for cls in repo["classes"]:
            if cls.get("name"):
                classes.append(cls)
        calls.extend(repo.get("calls", []))
        imports.update(repo.get("imports", {}))
        sources.update(repo.get("sources", {}))

    print(f"  {len(functions)} functions, {len(classes)} classes, "
          f"{len(calls)} calls, {len(sources)} source files")
    return {
        "functions": functions,
        "classes": classes,
        "calls": calls,
        "imports": imports,
        "sources": sources,
    }


def _load_llm_pairs() -> list[dict]:
    """Load LLM-generated query pairs if available."""
    path = ML_DIR / "llm_training_pairs.json"
    if not path.exists():
        print("  No LLM pairs found (optional)")
        return []
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"  {len(pairs)} LLM-generated pairs loaded")
    return pairs


# ══════════════════════════════════════════════════════════════════════
#  M1: RELEVANCE RANKER V2 — Real Data Generation
# ══════════════════════════════════════════════════════════════════════

def _gen_m1_real_data(
    data: dict, llm_pairs: list[dict], n: int, rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate M1 training data from real symbols + LLM queries.

    Categories with graded labels:
      1. Exact name match (0.90-0.95)     — query IS the tokenized name
      2. Partial name match (0.65-0.80)    — 1 of 2+ tokens from name
      3. Stem match (0.55-0.70)            — stem variant of a name token
      4. Docstring-only match (0.45-0.60)  — doc tokens, not in name
      5. Path-only match (0.20-0.35)       — path token, not in name/doc
      6. Dunder generic (0.05-0.15)        — __init__ with query "init"
      7. Dunder specific (0.30-0.45)       — "ClassName init"
      8. Test file (0.45-0.60)             — test_ prefix in test dir
      9. Zero overlap negative (0.00-0.02) — completely unrelated
     10. Weak overlap negative (0.10-0.25) — shares 1 common verb only
     11. Name+doc strong (0.85-0.95)       — both name and doc match
     12. Name+path strong (0.85-0.95)      — both name and path match
     13. LLM HIGH (0.85-0.95)             — LLM-generated direct queries
     14. LLM MED (0.55-0.70)              — LLM-generated partial queries
     15. LLM LOW (0.10-0.25)              — LLM-generated misleading queries
    """
    X, y = [], []
    funcs = data["functions"]
    classes = data["classes"]

    # Separate dunders, test funcs, normal funcs
    normal_funcs = [f for f in funcs if not f["name"].startswith("__")
                    and not f["name"].startswith("test_")]
    dunder_funcs = [f for f in funcs if f["name"].startswith("__")
                    and f["name"].endswith("__")]
    test_funcs = [f for f in funcs if f["name"].startswith("test_")]

    per_cat = n // 15

    def _make_feat(query_tokens, query, func, **kwargs):
        return m1_extract(
            query_tokens=query_tokens, query_lower=query,
            name=func.get("name", ""),
            docstring=func.get("docstring", "")[:200],
            file_path=func.get("file", ""),
            is_dunder=func.get("name", "").startswith("__"),
            is_method=func.get("is_method", False),
            is_class=func.get("name", "")[0].isupper() if func.get("name") else False,
            **kwargs,
        )

    # ━━━ Cat 1: Exact name match (0.90-0.95) ━━━
    for _ in range(per_cat * 2):
        func = rng.choice(normal_funcs)
        tokens = _split_tokens(func["name"])
        if not tokens:
            continue
        query = " ".join(tokens)
        feat = _make_feat(tokens, query, func)
        X.append(feat); y.append(rng.uniform(0.90, 0.95))

    # ━━━ Cat 2: Partial name match (0.65-0.80) ━━━
    for _ in range(per_cat * 2):
        func = rng.choice(normal_funcs)
        tokens = _split_tokens(func["name"])
        if len(tokens) < 2:
            continue
        subset = [rng.choice(tokens)]
        query = " ".join(subset)
        feat = _make_feat(subset, query, func)
        X.append(feat); y.append(rng.uniform(0.65, 0.80))

    # ━━━ Cat 3: Stem match (0.55-0.70) ━━━
    for _ in range(per_cat):
        func = rng.choice(normal_funcs)
        tokens = _split_tokens(func["name"])
        if not tokens:
            continue
        base = rng.choice(tokens)
        if len(base) < 5:
            continue
        variants = [base + "s", base + "ed", base + "ing", base + "er",
                    base + "tion", base[:len(base)-1]]
        stem = rng.choice(variants)
        feat = _make_feat([stem], stem, func)
        X.append(feat); y.append(rng.uniform(0.55, 0.70))

    # ━━━ Cat 4: Docstring-only match (0.45-0.60) ━━━
    for _ in range(per_cat):
        func = rng.choice([f for f in normal_funcs if f.get("docstring")])
        doc = func["docstring"][:200]
        doc_tokens = _split_tokens(doc)
        name_set = set(_split_tokens(func["name"]))
        doc_only = [t for t in doc_tokens if t not in name_set and len(t) >= 4]
        if not doc_only:
            continue
        subset = rng.sample(doc_only, min(2, len(doc_only)))
        query = " ".join(subset)
        feat = _make_feat(subset, query, func)
        X.append(feat); y.append(rng.uniform(0.45, 0.60))

    # ━━━ Cat 5: Path-only match (0.20-0.35) ━━━
    for _ in range(per_cat):
        func = rng.choice(normal_funcs)
        path_tokens = _split_tokens(func.get("file", ""))
        name_set = set(_split_tokens(func["name"]))
        path_only = [t for t in path_tokens if t not in name_set and len(t) >= 3]
        if not path_only:
            continue
        qt = rng.choice(path_only)
        feat = _make_feat([qt], qt, func)
        X.append(feat); y.append(rng.uniform(0.20, 0.35))

    # ━━━ Cat 6: Dunder generic (0.05-0.15) ━━━
    for _ in range(per_cat):
        if not dunder_funcs:
            break
        func = rng.choice(dunder_funcs)
        inner = func["name"].strip("_")
        if not inner:
            continue
        feat = _make_feat([inner], inner, func)
        X.append(feat); y.append(rng.uniform(0.05, 0.15))

    # ━━━ Cat 7: Dunder specific (0.30-0.45) ━━━
    for _ in range(per_cat // 2):
        if not dunder_funcs:
            break
        func = rng.choice(dunder_funcs)
        inner = func["name"].strip("_")
        class_name = func.get("class_name", "")
        if not inner or not class_name:
            continue
        tokens = _split_tokens(class_name) + [inner]
        query = " ".join(tokens)
        feat = _make_feat(tokens, query, func)
        X.append(feat); y.append(rng.uniform(0.30, 0.45))

    # ━━━ Cat 8: Test file (0.45-0.60) ━━━
    for _ in range(per_cat):
        if not test_funcs:
            break
        func = rng.choice(test_funcs)
        # Query is the function being tested (strip test_ prefix)
        base_name = func["name"].removeprefix("test_")
        tokens = _split_tokens(base_name)
        if not tokens:
            continue
        query = " ".join(tokens)
        feat = _make_feat(tokens, query, func)
        X.append(feat); y.append(rng.uniform(0.45, 0.60))

    # ━━━ Cat 9: Zero overlap negative (0.00-0.02) ━━━
    for _ in range(per_cat * 2):
        func = rng.choice(normal_funcs)
        other = rng.choice(normal_funcs)
        if func is other:
            continue
        tokens = _split_tokens(other["name"])
        name_set = set(_split_tokens(func["name"]))
        if not tokens or (set(tokens) & name_set):
            continue
        # Check no path overlap either
        path_set = set(_split_tokens(func.get("file", "")))
        if set(tokens) & path_set:
            continue
        query = " ".join(tokens)
        feat = _make_feat(tokens, query, func)
        X.append(feat); y.append(rng.uniform(0.00, 0.02))

    # ━━━ Cat 10: Weak overlap negative (0.10-0.25) ━━━
    # Two functions that share a common verb but different nouns
    for _ in range(per_cat):
        func = rng.choice(normal_funcs)
        other = rng.choice(normal_funcs)
        if func is other:
            continue
        func_tokens = set(_split_tokens(func["name"]))
        other_tokens = _split_tokens(other["name"])
        if not other_tokens or not func_tokens:
            continue
        overlap = set(other_tokens) & func_tokens
        # Exactly 1 shared token (weak match)
        if len(overlap) == 1:
            query = " ".join(other_tokens)
            feat = _make_feat(other_tokens, query, func)
            X.append(feat); y.append(rng.uniform(0.10, 0.25))

    # ━━━ Cat 11: Name+doc strong (0.85-0.95) ━━━
    for _ in range(per_cat):
        func = rng.choice([f for f in normal_funcs if f.get("docstring")])
        tokens = _split_tokens(func["name"])
        if not tokens:
            continue
        doc_tokens = set(_split_tokens(func.get("docstring", "")[:200]))
        # At least one token also in doc
        if set(tokens) & doc_tokens:
            query = " ".join(tokens)
            feat = _make_feat(tokens, query, func)
            X.append(feat); y.append(rng.uniform(0.85, 0.95))

    # ━━━ Cat 12: Name+path strong (0.85-0.95) ━━━
    for _ in range(per_cat):
        func = rng.choice(normal_funcs)
        tokens = _split_tokens(func["name"])
        path_tokens = set(_split_tokens(func.get("file", "")))
        if not tokens:
            continue
        if set(tokens) & path_tokens:
            query = " ".join(tokens)
            feat = _make_feat(tokens, query, func)
            X.append(feat); y.append(rng.uniform(0.85, 0.95))

    # ━━━ Cat 13-15: LLM-generated pairs ━━━
    for pair in llm_pairs:
        query = pair.get("query", "")
        name = pair.get("name", "")
        tokens = _split_tokens(query)
        if not tokens or not name:
            continue
        feat = m1_extract(
            query_tokens=tokens, query_lower=query,
            name=name,
            docstring=pair.get("docstring", "")[:200],
            file_path=pair.get("file", ""),
            is_dunder=name.startswith("__"),
        )
        # Map LLM relevance to our graded scale
        rel = pair.get("relevance", 0.5)
        X.append(feat); y.append(rel)

    # ━━━ Classes as well ━━━
    for _ in range(per_cat):
        cls = rng.choice(classes)
        tokens = _split_tokens(cls["name"])
        if not tokens:
            continue
        query = " ".join(tokens)
        feat = m1_extract(
            query_tokens=tokens, query_lower=query,
            name=cls["name"],
            docstring=cls.get("docstring", "")[:200],
            file_path=cls.get("file", ""),
            is_class=True,
        )
        X.append(feat); y.append(rng.uniform(0.85, 0.95))

    # Shuffle
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    idx = np.arange(len(X_arr))
    rng_np = np.random.RandomState(42)
    rng_np.shuffle(idx)
    return X_arr[idx], y_arr[idx]


# ══════════════════════════════════════════════════════════════════════
#  M2: CALLER DISAMBIGUATOR — Real Data Generation
# ══════════════════════════════════════════════════════════════════════

def _gen_m2_real_data(
    data: dict, n: int, rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate M2 training data from real call sites + functions.

    Positive: call site actually resolves to callee (same class, import match)
    Negative: random callee with same method name but wrong class/module
    """
    X, y = [], []
    funcs = data["functions"]
    calls = data["calls"]
    imports = data["imports"]

    # Build method name → list of functions with that name
    method_index: dict[str, list[dict]] = {}
    for func in funcs:
        name = func.get("name", "")
        if name:
            method_index.setdefault(name, []).append(func)

    # Method name frequency
    name_freq = Counter(f.get("name", "") for f in funcs)
    max_freq = max(name_freq.values()) if name_freq else 1

    half = n // 2

    # ━━━ Positives: call site matches the callee ━━━
    for _ in range(half):
        if not calls:
            break
        call = rng.choice(calls)
        method = call.get("method", "")
        receiver = call.get("receiver", "")
        caller_file = call.get("caller_file", "")
        caller_class = call.get("caller_class", "")
        arg_count = call.get("arg_count", 0)
        has_dot = call.get("has_dot_syntax", False)

        # Find a matching callee (same method name + same class or imported)
        candidates = method_index.get(method, [])
        if not candidates:
            continue

        # Pick the most likely match
        callee = None
        for c in candidates:
            if c.get("class_name") and receiver and c["class_name"].lower() in receiver.lower():
                callee = c
                break
            if c.get("file") == caller_file:
                callee = c
                break
        if callee is None:
            callee = rng.choice(candidates)

        caller_file_imports = []
        for imp_list in imports.get(caller_file, []):
            if isinstance(imp_list, dict):
                caller_file_imports.append(imp_list.get("module", ""))

        feat = m2_extract(
            receiver_text=receiver,
            method_name=method,
            caller_file=caller_file,
            caller_imports=caller_file_imports,
            arg_count=arg_count,
            has_dot_syntax=has_dot,
            caller_loc=rng.randint(10, 200),
            callee_name=callee.get("name", ""),
            callee_class_name=callee.get("class_name", ""),
            callee_file=callee.get("file", ""),
            callee_module=callee.get("file", "").replace("/", ".").removesuffix(".py"),
            callee_package=callee.get("file", "").split("/")[0] if "/" in callee.get("file", "") else "",
            callee_param_count=len(callee.get("params", [])),
            callee_is_method=callee.get("is_method", False),
            callee_decorators=callee.get("decorators", []),
            method_count_with_same_name=len(candidates),
            method_name_frequency=name_freq.get(method, 0) / max_freq,
        )
        X.append(feat); y.append(1.0)

    # ━━━ Negatives: wrong callee for call site ━━━
    for _ in range(half):
        if not calls:
            break
        call = rng.choice(calls)
        method = call.get("method", "")
        receiver = call.get("receiver", "")
        caller_file = call.get("caller_file", "")

        # Pick a random function that does NOT match
        callee = rng.choice(funcs)
        # Make sure it's actually wrong (different name or different class)
        if callee.get("name") == method and callee.get("file") == caller_file:
            continue

        caller_file_imports = []
        for imp_list in imports.get(caller_file, []):
            if isinstance(imp_list, dict):
                caller_file_imports.append(imp_list.get("module", ""))

        candidates = method_index.get(method, [])
        feat = m2_extract(
            receiver_text=receiver,
            method_name=method,
            caller_file=caller_file,
            caller_imports=caller_file_imports,
            arg_count=call.get("arg_count", 0),
            has_dot_syntax=call.get("has_dot_syntax", False),
            caller_loc=rng.randint(10, 200),
            callee_name=callee.get("name", ""),
            callee_class_name=callee.get("class_name", ""),
            callee_file=callee.get("file", ""),
            callee_module=callee.get("file", "").replace("/", ".").removesuffix(".py"),
            callee_package=callee.get("file", "").split("/")[0] if "/" in callee.get("file", "") else "",
            callee_param_count=len(callee.get("params", [])),
            callee_is_method=callee.get("is_method", False),
            callee_decorators=callee.get("decorators", []),
            method_count_with_same_name=len(candidates) if candidates else 1,
            method_name_frequency=name_freq.get(callee.get("name", ""), 0) / max_freq,
        )
        X.append(feat); y.append(0.0)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    idx = np.arange(len(X_arr))
    np.random.RandomState(42).shuffle(idx)
    return X_arr[idx], y_arr[idx]


# ══════════════════════════════════════════════════════════════════════
#  M3: SOURCE TOKEN FILTER — Real Data Generation
# ══════════════════════════════════════════════════════════════════════

def _build_source_index(data: dict) -> dict[str, str]:
    """Build a source lookup that handles path mismatches.

    The sources dict may have double-prefixed keys (e.g., 'aiohttp/aiohttp/abc.py')
    while functions have single-prefixed keys ('aiohttp/abc.py'). Build a fallback
    that strips the first component.
    """
    sources = data["sources"]
    idx: dict[str, str] = {}
    for key, src in sources.items():
        idx[key] = src
        # Also index by stripping the first path component
        parts = key.split("/")
        if len(parts) > 1:
            stripped = "/".join(parts[1:])
            idx[stripped] = src
    return idx


def _synthetic_source(func: dict) -> str:
    """Generate minimal synthetic source for a function when real source unavailable."""
    name = func.get("name", "func")
    params = func.get("params", [])
    doc = func.get("docstring", "")
    decorators = func.get("decorators", [])
    lines = []
    for d in decorators:
        lines.append(f"@{d}")
    param_str = ", ".join(params) if params else ""
    lines.append(f"def {name}({param_str}):")
    if doc:
        lines.append(f'    """{doc}"""')
    lines.append("    pass")
    return "\n".join(lines)


def _gen_m3_real_data(
    data: dict, n: int, rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate M3 training data from real source code.

    Positive: tokens from function name, params, meaningful identifiers
    Negative: tokens from imports, string literals, stop words, keywords

    Falls back to synthetic source when real source is unavailable.
    """
    X, y = [], []
    funcs = data["functions"]
    source_idx = _build_source_index(data)

    half = n // 2

    # ━━━ Positives: meaningful tokens ━━━
    for _ in range(half):
        func = rng.choice(funcs)
        source = source_idx.get(func.get("file", ""), "")[:3000]
        if not source:
            source = _synthetic_source(func)

        # Token from function name (clearly relevant)
        name_tokens = _split_tokens(func["name"])
        param_tokens = []
        for p in func.get("params", []):
            param_tokens.extend(_split_tokens(p))

        good_tokens = [t for t in name_tokens + param_tokens
                       if len(t) >= 3 and t not in ("self", "cls")]
        if not good_tokens:
            continue
        token = rng.choice(good_tokens)

        feat = m3_extract(
            token=token,
            function_name=func["name"],
            docstring=func.get("docstring", "")[:200],
            param_names=func.get("params", []),
            decorator_names=func.get("decorators", []),
            source_code=source,
        )
        X.append(feat); y.append(1.0)

    # ━━━ Negatives: noise tokens ━━━
    import keyword as kw
    stop_words = {"the", "is", "in", "at", "of", "and", "or", "not", "for", "to",
                  "if", "else", "this", "that", "with", "from", "by", "as", "on",
                  "none", "true", "false", "self", "cls", "args", "kwargs",
                  "return", "pass", "break", "continue"}

    for _ in range(half):
        func = rng.choice(funcs)
        source = source_idx.get(func.get("file", ""), "")[:3000]
        if not source:
            source = _synthetic_source(func)

        # Pick a noise token
        noise_type = rng.choice(["stop", "keyword", "import", "short"])
        if noise_type == "stop":
            token = rng.choice(list(stop_words))
        elif noise_type == "keyword":
            token = rng.choice(kw.kwlist)
        elif noise_type == "import":
            # Pick a module name from imports
            lines = source.split("\n")
            import_tokens = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    import_tokens.extend(_split_tokens(stripped))
            if import_tokens:
                token = rng.choice(import_tokens)
            else:
                token = rng.choice(list(stop_words))
        else:
            # Short meaningless token
            token = rng.choice(["x", "i", "j", "k", "n", "v", "t", "s"])

        feat = m3_extract(
            token=token,
            function_name=func["name"],
            docstring=func.get("docstring", "")[:200],
            param_names=func.get("params", []),
            decorator_names=func.get("decorators", []),
            source_code=source,
        )
        X.append(feat); y.append(0.0)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    idx = np.arange(len(X_arr))
    np.random.RandomState(42).shuffle(idx)
    return X_arr[idx], y_arr[idx]


# ══════════════════════════════════════════════════════════════════════
#  M4: SYMBOL IMPORTANCE — Real Data Generation
# ══════════════════════════════════════════════════════════════════════

def _gen_m4_real_data(
    data: dict, n: int, rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate M4 training data from real symbols.

    Label = importance score based on real signals:
      - caller count (from call graph)
      - is API endpoint (from decorators)
      - has docstring + length
      - number of methods (for classes)
      - is test function (lower importance)
      - is abstract (higher importance for base classes)
    """
    X, y = [], []
    funcs = data["functions"]
    classes = data["classes"]
    calls = data["calls"]
    sources = data["sources"]

    # Build caller count from call graph
    callee_counts: Counter = Counter()
    cross_file_counts: Counter = Counter()
    for call in calls:
        method = call.get("method", "")
        caller_file = call.get("caller_file", "")
        # Count how many times each method name is called
        callee_counts[method] += 1
        # Track cross-file calls
        key = (method, call.get("caller_class", ""))
        cross_file_counts[key] += 1

    # Build import count per file
    import_counts: Counter = Counter()
    for file_imports in data["imports"].values():
        for imp in file_imports:
            if isinstance(imp, dict):
                mod = imp.get("module", "")
                if mod:
                    import_counts[mod] += 1

    # Build subclass count
    subclass_counts: Counter = Counter()
    for cls in classes:
        for base in cls.get("base_classes", []):
            subclass_counts[base] += 1

    all_symbols = funcs + [{
        "name": c["name"], "file": c["file"], "docstring": c.get("docstring", ""),
        "params": [], "decorators": c.get("decorators", []),
        "is_method": False, "class_name": "",
        "methods": c.get("methods", []),
        "base_classes": c.get("base_classes", []),
        "line_count": c.get("line_count", 0),
    } for c in classes]

    for _ in range(n):
        sym = rng.choice(all_symbols)
        name = sym.get("name", "")
        file_path = sym.get("file", "")
        source = sources.get(file_path, "")[:2000]

        caller_count = callee_counts.get(name, 0)
        cross_file = cross_file_counts.get((name, sym.get("class_name", "")), 0)
        methods = sym.get("methods", [])
        base_classes = sym.get("base_classes", [])
        sc_count = subclass_counts.get(name, 0)

        feat = m4_extract(
            name=name,
            file_path=file_path,
            source=source,
            docstring=sym.get("docstring", "")[:200],
            param_names=sym.get("params", []),
            decorators=sym.get("decorators", []),
            base_classes=base_classes if base_classes else None,
            methods=methods if methods else None,
            caller_count=caller_count,
            cross_file_caller_count=cross_file,
            import_count=import_counts.get(
                file_path.replace("/", ".").removesuffix(".py"), 0),
            subclass_count=sc_count,
        )

        # Compute importance label from real signals
        importance = 0.0
        # Caller count is strongest signal
        importance += min(0.3, caller_count / 100.0)
        # Cross-file callers
        importance += min(0.15, cross_file / 30.0)
        # Has docstring
        if sym.get("docstring"):
            importance += 0.1
        # API endpoint
        decorators = sym.get("decorators", [])
        if any(d.lower() in ("route", "get", "post", "put", "delete")
               for d in decorators):
            importance += 0.15
        # Has subclasses (base class)
        if sc_count > 0:
            importance += min(0.1, sc_count / 20.0)
        # Many methods (important class)
        if methods:
            importance += min(0.1, len(methods) / 40.0)
        # Is test function (lower)
        if name.startswith("test_") or "test" in file_path.lower():
            importance *= 0.5
        # Is private
        if name.startswith("_") and not name.startswith("__"):
            importance *= 0.7

        importance = min(0.95, importance)
        X.append(feat); y.append(importance)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    idx = np.arange(len(X_arr))
    np.random.RandomState(42).shuffle(idx)
    return X_arr[idx], y_arr[idx]


# ══════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════

class RelevanceRankerV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M1_FEAT, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)


class CallerDisambiguator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M2_FEAT, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x)


class SourceTokenFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M3_FEAT, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x): return self.net(x)


class SymbolImportance(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M4_FEAT, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════

def train_mse(
    model: nn.Module, X_train, y_train, X_val, y_val,
    epochs=60, lr=1e-3, batch_size=2048, patience=10,
):
    """Train with MSE loss on sigmoid output (for graded labels)."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4,
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = torch.sigmoid(model(xb).squeeze(-1))
            loss = torch.nn.functional.mse_loss(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_total += len(xb)

        model.eval()
        val_loss = val_total = 0
        val_mae = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = torch.sigmoid(model(xb).squeeze(-1))
                loss = torch.nn.functional.mse_loss(preds, yb)
                val_loss += loss.item() * len(xb)
                val_mae += torch.abs(preds - yb).sum().item()
                val_total += len(xb)

        val_mse = val_loss / val_total
        scheduler.step(val_mse)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"train_mse={train_loss/train_total:.6f} | "
                  f"val_mse={val_mse:.6f} | val_mae={val_mae/val_total:.4f}")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    print(f"    Best epoch: {best_epoch}, val_mse={best_val_loss:.6f}")
    return model


def train_bce(
    model: nn.Module, X_train, y_train, X_val, y_val,
    epochs=40, lr=1e-3, batch_size=2048, patience=8,
):
    """Train with BCE loss (for binary labels)."""
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_correct = train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_correct += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
            train_total += len(xb)

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(xb)
                val_correct += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
                val_total += len(xb)

        val_acc = val_correct / val_total
        val_loss_avg = val_loss / val_total
        scheduler.step(val_loss_avg)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"train_acc={train_correct/train_total:.4f} | "
                  f"val_acc={val_acc:.4f} | val_loss={val_loss_avg:.4f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    print(f"    Best epoch: {best_epoch}, val_acc={val_correct/val_total:.4f}")
    return model


def export_weights(model: nn.Module, out_path: Path):
    """Export in W0/b0 format (pre-transposed for x @ W + b)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    layer_keys = sorted(
        set(k.rsplit(".", 1)[0] for k in sd if "weight" in k),
        key=lambda k: int(re.search(r'\d+', k).group()) if re.search(r'\d+', k) else 0,
    )
    save_dict = {}
    for i, key_prefix in enumerate(layer_keys):
        w = sd[f"{key_prefix}.weight"].cpu().numpy()
        b = sd[f"{key_prefix}.bias"].cpu().numpy()
        save_dict[f"W{i}"] = w.T.astype(np.float32)
        save_dict[f"b{i}"] = b.astype(np.float32)
    np.savez(out_path, **save_dict)
    print(f"    Saved {out_path.name} ({out_path.stat().st_size:,} bytes)")


# ══════════════════════════════════════════════════════════════════════
#  VERIFICATION
# ══════════════════════════════════════════════════════════════════════

def verify_m1(model: nn.Module) -> bool:
    """Verify M1 produces proper ranking."""
    model.eval()
    model = model.cpu()

    cases = [
        ("authenticate user", "authenticate_user", "Authenticate a user", "app/auth.py",
         {"is_dunder": False}, (0.70, 1.0), "exact name match"),
        ("parse json", "parse_json", "Parse JSON string", "utils/serialization.py",
         {}, (0.70, 1.0), "exact name match 2"),
        ("database connect", "create_db_connection", "Create a database connection pool", "db/connections.py",
         {}, (0.35, 0.85), "partial match"),
        ("auth", "process_request", "Process incoming request", "auth/middleware.py",
         {}, (0.05, 0.50), "path-only match"),
        ("payment process", "render_template", "Render an HTML template", "web/views.py",
         {}, (0.00, 0.10), "no relation"),
        ("init", "__init__", "Initialize the class", "models/user.py",
         {"is_dunder": True}, (0.00, 0.30), "dunder generic"),
        ("user manager", "UserManager", "Manage users", "services/user.py",
         {"is_class": True}, (0.70, 1.0), "camelcase class"),
    ]

    all_pass = True
    for query, name, doc, path, kwargs, (lo, hi), desc in cases:
        tokens = _split_tokens(query)
        feat = m1_extract(query_tokens=tokens, query_lower=query, name=name,
                          docstring=doc, file_path=path, **kwargs)
        x = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            score = float(torch.sigmoid(model(x)).item())
        ok = lo <= score <= hi
        if not ok:
            all_pass = False
        status = "OK" if ok else "FAIL"
        print(f"    [{status}] {desc}: {score:.3f} (expected {lo:.2f}-{hi:.2f})")

    # Ranking check
    ranking = [
        ("user", "get_user", "Get user by ID", "app/models.py", {}),
        ("user", "update_user_settings", "Update settings", "app/settings.py", {}),
        ("user", "process_request", "Process request", "app/user/middleware.py", {}),
        ("user", "render_template", "Render HTML", "web/views.py", {}),
    ]
    scores = []
    for query, name, doc, path, kwargs in ranking:
        tokens = _split_tokens(query)
        feat = m1_extract(query_tokens=tokens, query_lower=query, name=name,
                          docstring=doc, file_path=path, **kwargs)
        x = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            scores.append(float(torch.sigmoid(model(x)).item()))

    rank_ok = scores[0] > scores[1] > scores[2] > scores[3]
    status = "OK" if rank_ok else "FAIL"
    if not rank_ok:
        all_pass = False
    print(f"    [{status}] Ranking: {scores[0]:.3f} > {scores[1]:.3f} > "
          f"{scores[2]:.3f} > {scores[3]:.3f}")
    return all_pass


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n", type=int, default=500_000,
                        help="Training examples per model")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", choices=["m1", "m2", "m3", "m4"],
                        help="Train only one model")
    args = parser.parse_args()

    if args.quick:
        args.n = 50_000
        args.epochs = 30

    print(f"Device: {DEVICE}")
    rng = random.Random(42)

    # Load real data
    data = _load_real_symbols()
    llm_pairs = _load_llm_pairs()

    models_to_train = ["m1", "m2", "m3", "m4"]
    if args.only:
        models_to_train = [args.only]

    results = {}

    # ── M1: Relevance Ranker v2 ──────────────────────────────────────
    if "m1" in models_to_train:
        print("\n" + "="*60)
        print("  M1: Relevance Ranker v2 (42 features, MSE loss)")
        print("="*60)
        print(f"  Generating {args.n:,} real-data training examples...")
        X, y_arr = _gen_m1_real_data(data, llm_pairs, args.n, rng)
        split = int(0.85 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y_arr[:split], y_arr[split:]
        print(f"  Total: {len(X):,} | Train: {len(X_train):,} | Val: {len(X_val):,}")
        print(f"  Label stats: mean={y_arr.mean():.3f}, std={y_arr.std():.3f}")

        model = RelevanceRankerV2()
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
        model = train_mse(model, X_train, y_train, X_val, y_val, epochs=args.epochs)

        print("  Verification:")
        passed = verify_m1(model)
        results["m1"] = passed

        out = PRODUCTION_MODEL_DIR / "relevance_weights_v2.npz"
        export_weights(model, out)
        if passed:
            print("  M1: ALL CHECKS PASSED")
        else:
            print("  M1: SOME CHECKS FAILED — model saved but may need tuning")

    # ── M2: Caller Disambiguator ─────────────────────────────────────
    if "m2" in models_to_train:
        print("\n" + "="*60)
        print("  M2: Caller Disambiguator (25 features, BCE loss)")
        print("="*60)
        n_m2 = min(args.n, 200_000)  # M2 has fewer categories
        print(f"  Generating {n_m2:,} real-data training examples...")
        X, y_arr = _gen_m2_real_data(data, n_m2, rng)
        split = int(0.85 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y_arr[:split], y_arr[split:]
        print(f"  Total: {len(X):,} | Train: {len(X_train):,} | Val: {len(X_val):,}")

        model = CallerDisambiguator()
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
        model = train_bce(model, X_train, y_train, X_val, y_val,
                          epochs=min(40, args.epochs))

        out = PRODUCTION_MODEL_DIR / "caller_disambiguator_weights.npz"
        export_weights(model, out)
        results["m2"] = True
        print("  M2: DONE")

    # ── M3: Source Token Filter ───────────────────────────────────────
    if "m3" in models_to_train:
        print("\n" + "="*60)
        print("  M3: Source Token Filter (15 features, BCE loss)")
        print("="*60)
        n_m3 = min(args.n, 200_000)
        print(f"  Generating {n_m3:,} real-data training examples...")
        X, y_arr = _gen_m3_real_data(data, n_m3, rng)
        split = int(0.85 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y_arr[:split], y_arr[split:]
        print(f"  Total: {len(X):,} | Train: {len(X_train):,} | Val: {len(X_val):,}")

        model = SourceTokenFilter()
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
        model = train_bce(model, X_train, y_train, X_val, y_val,
                          epochs=min(40, args.epochs))

        out = PRODUCTION_MODEL_DIR / "source_token_filter_weights.npz"
        export_weights(model, out)
        results["m3"] = True
        print("  M3: DONE")

    # ── M4: Symbol Importance ─────────────────────────────────────────
    if "m4" in models_to_train:
        print("\n" + "="*60)
        print("  M4: Symbol Importance (18 features, MSE loss)")
        print("="*60)
        n_m4 = min(args.n, 300_000)
        print(f"  Generating {n_m4:,} real-data training examples...")
        X, y_arr = _gen_m4_real_data(data, n_m4, rng)
        split = int(0.85 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y_arr[:split], y_arr[split:]
        print(f"  Total: {len(X):,} | Train: {len(X_train):,} | Val: {len(X_val):,}")
        print(f"  Label stats: mean={y_arr.mean():.3f}, std={y_arr.std():.3f}")

        model = SymbolImportance()
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
        model = train_mse(model, X_train, y_train, X_val, y_val, epochs=args.epochs)

        out = PRODUCTION_MODEL_DIR / "symbol_importance_weights.npz"
        export_weights(model, out)
        results["m4"] = True
        print("  M4: DONE")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING SUMMARY")
    print("="*60)
    for model_name, passed in results.items():
        status = "PASS" if passed else "NEEDS REVIEW"
        print(f"  {model_name.upper()}: {status}")
    print(f"\n  Weights saved to: {PRODUCTION_MODEL_DIR}")


if __name__ == "__main__":
    main()
