"""Generate 500K training examples for Relevance Ranker v2.

Uses symbols extracted by data_common/extract_symbols.py to build
(query, symbol) pairs with positive/negative labels.

Categories:
  - Exact match positives (50K)
  - Dunder hard negatives (30K)
  - __init__.py negatives (20K)
  - Random mismatches (200K)
  - Substring match positives (100K)
  - Docstring-based positives (100K)

Split: 80/10/10 train/val/test, written as JSONL.
"""

import json
import math
import random
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features, _split_tokens, FEATURE_COUNT

DATA_DIR = Path(__file__).parent / "data"
SYMBOLS_FILE = Path(__file__).parent.parent / "all_symbols.json"

# Domains for diverse synthetic query construction
DOMAINS = [
    "auth", "database", "cache", "api", "middleware", "logging",
    "config", "testing", "deployment", "monitoring", "search",
    "notification", "payment", "file", "email", "queue", "scheduler",
    "websocket", "streaming", "validation", "serialization", "migration",
    "security", "encryption", "compression", "pagination", "filtering",
    "sorting", "aggregation", "transformation", "routing", "templating",
    "session", "cookie", "oauth", "jwt", "rbac", "audit", "backup",
    "restore", "export", "import", "webhook", "callback", "retry",
    "circuit_breaker", "health_check", "metrics", "tracing",
]

# Semantic query templates (verb + domain noun)
VERBS = [
    "get", "set", "create", "delete", "update", "find", "process",
    "handle", "validate", "check", "send", "receive", "parse", "build",
    "init", "load", "save", "fetch", "compute", "format", "register",
    "dispatch", "execute", "run", "start", "stop", "close", "open",
    "connect", "disconnect", "serialize", "deserialize", "encode",
    "decode", "encrypt", "decrypt", "compress", "decompress",
]

NOUNS = [
    "user", "request", "response", "connection", "session", "token",
    "message", "event", "task", "job", "config", "setting", "handler",
    "middleware", "router", "endpoint", "model", "schema", "field",
    "query", "result", "error", "exception", "logger", "cache",
    "database", "transaction", "migration", "template", "view",
    "controller", "service", "client", "server", "worker", "pool",
    "manager", "factory", "builder", "adapter", "proxy", "decorator",
]


def _compute_source_entropy(source: str) -> float:
    """Compute Shannon entropy of source token distribution."""
    tokens = _split_tokens(source[:2000])
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _compute_unique_token_ratio(source: str) -> float:
    """Compute ratio of unique tokens to total tokens in source."""
    tokens = _split_tokens(source[:2000])
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _make_example(
    symbol: dict,
    query_tokens: list[str],
    query_lower: str,
    label: int,
) -> dict:
    """Build a single training example with all 42 features."""
    name = symbol.get("name", "")
    source = symbol.get("source", "")

    features = extract_features(
        query_tokens=query_tokens,
        query_lower=query_lower,
        name=name,
        docstring=symbol.get("docstring", ""),
        file_path=symbol.get("file", ""),
        is_dunder=name.startswith("__") and name.endswith("__"),
        is_init_file="__init__" in symbol.get("file", ""),
        is_method=symbol.get("class_name") is not None,
        is_class=symbol.get("type") == "class",
        source_token_entropy=_compute_source_entropy(source),
        source_unique_token_ratio=_compute_unique_token_ratio(source),
    )
    return {"features": features.tolist(), "label": label}


def generate_all(target_count: int = 500_000):
    """Generate training data from extracted symbols."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not SYMBOLS_FILE.exists():
        print(f"ERROR: {SYMBOLS_FILE} not found.")
        print("Run: python -m ml.data_common.extract_symbols   (or equivalent)")
        return

    symbols = json.loads(SYMBOLS_FILE.read_text())
    functions = [s for s in symbols if s["type"] == "function"]
    classes = [s for s in symbols if s["type"] == "class"]
    dunders = [
        s for s in functions
        if s["name"].startswith("__") and s["name"].endswith("__")
    ]
    init_funcs = [s for s in functions if "__init__" in s.get("file", "")]
    doc_funcs = [f for f in functions if f.get("docstring")]
    all_symbols = functions + classes

    print(f"Loaded {len(symbols)} symbols "
          f"({len(functions)} funcs, {len(classes)} classes, "
          f"{len(dunders)} dunders, {len(doc_funcs)} with docs)")

    examples: list[dict] = []
    rng = random.Random(42)

    # ── 1. Exact match positives (50K) ────────────────────────────
    print("  [1/6] Exact match positives (50K)...")
    for _ in range(50_000):
        sym = rng.choice(functions)
        tokens = _split_tokens(sym["name"])
        if tokens:
            query = " ".join(tokens)
            examples.append(_make_example(sym, tokens, query, label=1))

    # ── 2. Dunder hard negatives (30K) ────────────────────────────
    print("  [2/6] Dunder hard negatives (30K)...")
    if dunders:
        for _ in range(30_000):
            sym = rng.choice(dunders)
            query_sym = rng.choice(functions)
            tokens = _split_tokens(query_sym["name"])
            if tokens:
                examples.append(
                    _make_example(sym, tokens, " ".join(tokens), label=0)
                )
    else:
        print("    (no dunders found, skipping)")

    # ── 3. __init__.py file negatives (20K) ───────────────────────
    print("  [3/6] __init__.py negatives (20K)...")
    if init_funcs:
        for _ in range(20_000):
            sym = rng.choice(init_funcs)
            query_sym = rng.choice(functions)
            tokens = _split_tokens(query_sym["name"])
            if tokens:
                examples.append(
                    _make_example(sym, tokens, " ".join(tokens), label=0)
                )
    else:
        print("    (no __init__ funcs found, skipping)")

    # ── 4. Random mismatch negatives (200K) ───────────────────────
    print("  [4/6] Random mismatch negatives (200K)...")
    neg_count = 0
    attempts = 0
    while neg_count < 200_000 and attempts < 600_000:
        attempts += 1
        sym = rng.choice(all_symbols)
        query_sym = rng.choice(all_symbols)
        tokens_q = _split_tokens(query_sym["name"])
        tokens_s = set(_split_tokens(sym["name"]))
        if tokens_q and not (set(tokens_q) & tokens_s):
            examples.append(
                _make_example(sym, tokens_q, " ".join(tokens_q), label=0)
            )
            neg_count += 1
    print(f"    generated {neg_count} in {attempts} attempts")

    # ── 5. Substring / partial match positives (100K) ─────────────
    print("  [5/6] Substring match positives (100K)...")
    for _ in range(100_000):
        sym = rng.choice(functions)
        tokens = _split_tokens(sym["name"])
        if len(tokens) >= 2:
            k = rng.randint(1, len(tokens))
            subset = tokens[:k]
            examples.append(
                _make_example(sym, subset, " ".join(subset), label=1)
            )
        elif tokens:
            examples.append(
                _make_example(sym, tokens, " ".join(tokens), label=1)
            )

    # ── 6. Docstring-based positives (100K) ───────────────────────
    print("  [6/6] Docstring positives (100K)...")
    if doc_funcs:
        for _ in range(100_000):
            sym = rng.choice(doc_funcs)
            doc_tokens = _split_tokens(sym["docstring"][:200])
            if len(doc_tokens) >= 3:
                k = min(4, len(doc_tokens))
                subset = rng.sample(doc_tokens, k)
                examples.append(
                    _make_example(sym, subset, " ".join(subset), label=1)
                )
    else:
        print("    (no documented funcs found, skipping)")

    # ── 7. Semantic query positives & negatives (bonus) ───────────
    #    Uses verb+noun combos matched against function names that
    #    contain the same tokens.
    print("  [bonus] Semantic verb+noun examples...")
    name_index: dict[str, list[dict]] = {}
    for sym in all_symbols:
        for tok in _split_tokens(sym["name"]):
            name_index.setdefault(tok, []).append(sym)

    sem_count = 0
    for _ in range(50_000):
        verb = rng.choice(VERBS)
        noun = rng.choice(NOUNS)
        query = f"{verb} {noun}"
        tokens = [verb, noun]
        # Positive: symbol whose name contains at least one query token
        candidates = name_index.get(verb, []) + name_index.get(noun, [])
        if candidates:
            sym = rng.choice(candidates)
            examples.append(_make_example(sym, tokens, query, label=1))
            sem_count += 1
        # Negative: random symbol with no overlap
        neg_sym = rng.choice(all_symbols)
        neg_name_tokens = set(_split_tokens(neg_sym["name"]))
        if not (set(tokens) & neg_name_tokens):
            examples.append(_make_example(neg_sym, tokens, query, label=0))
            sem_count += 1
    print(f"    generated {sem_count} semantic examples")

    # ── Shuffle and trim to target ────────────────────────────────
    rng.shuffle(examples)
    examples = examples[:target_count]
    print(f"\nTotal examples: {len(examples)}")

    # ── Split: 80/10/10 ──────────────────────────────────────────
    n = len(examples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }

    for split_name, split_data in splits.items():
        path = DATA_DIR / f"{split_name}.jsonl"
        pos = sum(1 for ex in split_data if ex["label"] == 1)
        neg = len(split_data) - pos
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        print(f"  {split_name}: {len(split_data)} examples "
              f"(pos={pos}, neg={neg}) -> {path}")


if __name__ == "__main__":
    generate_all()
