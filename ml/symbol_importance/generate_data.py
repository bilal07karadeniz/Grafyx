"""Generate 100K training examples for Symbol Importance Ranker.

Uses multi-signal proxy labeling to estimate importance:
  - README/docs mention     -> 0.30 weight
  - API endpoint            -> 0.25
  - Referenced in >10 files -> 0.15
  - Base class w/ subclasses-> 0.10
  - Cross-package callers   -> 0.10
  - Exported in __all__     -> 0.10

Symbols with combined score >= 0.5 are labeled as important (1),
otherwise not important (0).

Split: 80/10/10 train/val/test, written as JSONL.
"""

import json
import random
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import (
    extract_features,
    _split_tokens,
    _is_api_endpoint,
    _is_test,
    FEATURE_COUNT,
)

DATA_DIR = Path(__file__).parent / "data"
SYMBOLS_FILE = Path(__file__).parent.parent / "all_symbols.json"
CACHE_DIR = Path(__file__).parent.parent / ".repo_cache"


# ── Proxy importance labeling ─────────────────────────────────────


def _load_readme_mentions(repo_path: Path) -> set[str]:
    """Extract symbol-like names mentioned in README files."""
    mentions: set[str] = set()
    for readme_name in ("README.md", "README.rst", "README.txt", "README"):
        readme_path = repo_path / readme_name
        if readme_path.exists():
            try:
                text = readme_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            # Extract identifiers from code blocks and inline code
            # Backtick code: `some_function()`
            for match in re.finditer(r'`([a-zA-Z_]\w*(?:\.\w+)*)\(?', text):
                name = match.group(1).split(".")[-1]
                mentions.add(name.lower())
            # Also extract from headings and plain text
            for match in re.finditer(r'\b([A-Z][a-zA-Z]+(?:[A-Z][a-z]+)+)\b', text):
                mentions.add(match.group(1).lower())
    return mentions


def _load_all_readme_mentions() -> set[str]:
    """Load README mentions from all cached repos."""
    all_mentions: set[str] = set()
    if not CACHE_DIR.exists():
        return all_mentions
    for lang_dir in CACHE_DIR.iterdir():
        if not lang_dir.is_dir():
            continue
        for repo_dir in lang_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            mentions = _load_readme_mentions(repo_dir)
            all_mentions.update(mentions)
    return all_mentions


def _build_call_graph(symbols: list[dict]) -> dict[str, dict]:
    """Build caller statistics from symbol data.

    Returns: name -> {caller_count, cross_file_count, file_set}
    """
    stats: dict[str, dict] = defaultdict(
        lambda: {"caller_count": 0, "cross_file_callers": set(), "files": set()}
    )

    # Build: who calls whom
    for sym in symbols:
        if sym["type"] != "function":
            continue
        caller_file = sym.get("file", "")
        for call_name in sym.get("calls", []):
            call_name_lower = call_name.lower()
            stats[call_name_lower]["caller_count"] += 1
            if caller_file:
                stats[call_name_lower]["cross_file_callers"].add(caller_file)
                stats[call_name_lower]["files"].add(caller_file)

    return dict(stats)


def _build_subclass_map(symbols: list[dict]) -> dict[str, int]:
    """Build class_name -> subclass_count map."""
    subclass_counts: dict[str, int] = defaultdict(int)
    for sym in symbols:
        if sym["type"] == "class":
            for base in sym.get("base_classes", []):
                base_name = base.split(".")[-1].lower()
                subclass_counts[base_name] += 1
    return dict(subclass_counts)


def _build_import_counts(symbols: list[dict]) -> dict[str, int]:
    """Build name -> count of files that import this name."""
    import_counts: dict[str, set[str]] = defaultdict(set)

    for sym in symbols:
        if sym["type"] != "function":
            continue
        source = sym.get("source", "")
        file_path = sym.get("file", "")
        if not source or not file_path:
            continue
        # Extract import names from source
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                # Extract imported names
                for match in re.finditer(r'\b([a-zA-Z_]\w*)\b', stripped):
                    name = match.group(1).lower()
                    import_counts[name].add(file_path)

    return {name: len(files) for name, files in import_counts.items()}


def _build_all_exports(symbols: list[dict]) -> set[str]:
    """Find symbols exported in __all__."""
    exports: set[str] = set()
    for sym in symbols:
        source = sym.get("source", "")
        if not source:
            continue
        # Look for __all__ = [...] patterns
        match = re.search(r'__all__\s*=\s*\[(.*?)\]', source, re.DOTALL)
        if match:
            content = match.group(1)
            for name_match in re.finditer(r'["\'](\w+)["\']', content):
                exports.add(name_match.group(1).lower())
    return exports


def _compute_importance_score(
    sym: dict,
    readme_mentions: set[str],
    call_stats: dict[str, dict],
    subclass_map: dict[str, int],
    all_exports: set[str],
) -> float:
    """Compute proxy importance score for a symbol.

    Multi-signal scoring:
      - README/docs mention     -> 0.30
      - API endpoint            -> 0.25
      - Referenced in >10 files -> 0.15
      - Base class w/ subclasses-> 0.10
      - Cross-package callers   -> 0.10
      - Exported in __all__     -> 0.10
    """
    score = 0.0
    name = sym.get("name", "")
    name_lower = name.lower()

    # README mention
    if name_lower in readme_mentions:
        score += 0.30

    # API endpoint
    decorators = sym.get("decorators", [])
    if _is_api_endpoint(decorators):
        score += 0.25

    # Referenced in >10 files
    stats = call_stats.get(name_lower, {})
    cross_file_count = len(stats.get("cross_file_callers", set()))
    if cross_file_count > 10:
        score += 0.15

    # Base class with subclasses
    if sym["type"] == "class":
        n_subclasses = subclass_map.get(name_lower, 0)
        if n_subclasses > 0:
            score += 0.10

    # Cross-package callers
    if cross_file_count > 3:
        score += 0.10

    # Exported in __all__
    if name_lower in all_exports:
        score += 0.10

    return score


def _make_example(
    sym: dict,
    call_stats: dict[str, dict],
    subclass_map: dict[str, int],
    import_counts: dict[str, int],
    all_exports: set[str],
    label: int,
) -> dict:
    """Build a training example for a symbol."""
    name = sym.get("name", "")
    name_lower = name.lower()
    stats = call_stats.get(name_lower, {})
    caller_count = stats.get("caller_count", 0)
    cross_file_count = len(stats.get("cross_file_callers", set()))

    features = extract_features(
        name=name,
        file_path=sym.get("file", ""),
        source=sym.get("source", ""),
        docstring=sym.get("docstring", ""),
        param_names=sym.get("params", []),
        decorators=sym.get("decorators", []),
        base_classes=sym.get("base_classes", []),
        methods=sym.get("methods", []),
        caller_count=caller_count,
        cross_file_caller_count=cross_file_count,
        is_exported_in_all=name_lower in all_exports,
        import_count=import_counts.get(name_lower, 0),
        subclass_count=subclass_map.get(name_lower, 0),
    )
    return {"features": features.tolist(), "label": label}


def generate_all(target_count: int = 100_000):
    """Generate training data for symbol importance ranking."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not SYMBOLS_FILE.exists():
        print(f"ERROR: {SYMBOLS_FILE} not found.")
        print("Run: python -m ml.data_common.extract_symbols")
        return

    symbols = json.loads(SYMBOLS_FILE.read_text())
    print(f"Loaded {len(symbols)} symbols")

    # Build auxiliary data
    print("Building call graph...")
    call_stats = _build_call_graph(symbols)
    print(f"  {len(call_stats)} unique callee names")

    print("Building subclass map...")
    subclass_map = _build_subclass_map(symbols)
    print(f"  {len(subclass_map)} base classes with subclasses")

    print("Building import counts...")
    import_counts = _build_import_counts(symbols)
    print(f"  {len(import_counts)} imported names")

    print("Building export list...")
    all_exports = _build_all_exports(symbols)
    print(f"  {len(all_exports)} exported names")

    print("Loading README mentions...")
    readme_mentions = _load_all_readme_mentions()
    print(f"  {len(readme_mentions)} README-mentioned names")

    # Score all symbols
    print("\nScoring symbols for importance...")
    scored: list[tuple[dict, float]] = []
    for sym in symbols:
        importance = _compute_importance_score(
            sym, readme_mentions, call_stats, subclass_map, all_exports
        )
        scored.append((sym, importance))

    # Classify: >= 0.5 is important, < 0.5 is not
    important = [(s, score) for s, score in scored if score >= 0.5]
    not_important = [(s, score) for s, score in scored if score < 0.5]
    # Also separate test functions as definitively not important
    test_funcs = [
        (s, score) for s, score in not_important
        if _is_test(s["name"], s.get("file", ""))
    ]
    regular_not_important = [
        (s, score) for s, score in not_important
        if not _is_test(s["name"], s.get("file", ""))
    ]

    print(f"  Important (score >= 0.5): {len(important)}")
    print(f"  Not important: {len(not_important)} "
          f"(test funcs: {len(test_funcs)}, regular: {len(regular_not_important)})")

    # Generate examples
    rng = random.Random(42)
    examples: list[dict] = []

    # All important symbols as positives
    print("\nGenerating positive examples...")
    for sym, _ in important:
        examples.append(
            _make_example(sym, call_stats, subclass_map, import_counts, all_exports, 1)
        )

    # If we need more positives, duplicate with random noise
    while len([e for e in examples if e["label"] == 1]) < target_count // 3:
        sym, _ = rng.choice(important)
        examples.append(
            _make_example(sym, call_stats, subclass_map, import_counts, all_exports, 1)
        )

    # Negative examples: balanced mix of test funcs + regular
    print("Generating negative examples...")
    neg_target = target_count - len(examples)

    # Test functions (easy negatives)
    for sym, _ in test_funcs[:neg_target // 3]:
        examples.append(
            _make_example(sym, call_stats, subclass_map, import_counts, all_exports, 0)
        )

    # Regular non-important (harder negatives)
    rng.shuffle(regular_not_important)
    remaining = target_count - len(examples)
    for sym, _ in regular_not_important[:remaining]:
        examples.append(
            _make_example(sym, call_stats, subclass_map, import_counts, all_exports, 0)
        )

    # Shuffle and trim
    rng.shuffle(examples)
    examples = examples[:target_count]

    pos_count = sum(1 for e in examples if e["label"] == 1)
    neg_count = len(examples) - pos_count
    print(f"\nTotal examples: {len(examples)} (pos={pos_count}, neg={neg_count})")

    # Split: 80/10/10
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
        print(f"  {split_name}: {len(split_data)} (pos={pos}, neg={neg}) -> {path}")


if __name__ == "__main__":
    generate_all()
