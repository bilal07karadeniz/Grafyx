"""Generate 200K training examples for Source Token Filter.

For each function in cloned repos, for each token in source:
  - POSITIVE: token is in function name, docstring, or param names
    (semantically relevant — should be indexed for search)
  - NEGATIVE: token only appears in imports, string literals,
    __getattr__ bodies, or comments (noise — should be filtered)

Split: 80/10/10 train/val/test, written as JSONL.
"""

import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import (
    extract_features,
    _split_tokens,
    _is_in_import_context,
    _is_in_string_context,
    _is_in_comment_context,
    _is_in_getattr_body,
    FEATURE_COUNT,
)

DATA_DIR = Path(__file__).parent / "data"
SYMBOLS_FILE = Path(__file__).parent.parent / "all_symbols.json"


def _classify_token(
    token: str,
    function_name: str,
    docstring: str,
    param_names: list[str],
    source_code: str,
) -> int | None:
    """Classify a token as positive (1), negative (0), or ambiguous (None).

    Returns:
        1  if token is in function name, docstring, or param names
        0  if token only appears in imports, strings, comments, or getattr
        None if classification is ambiguous
    """
    token_lower = token.lower()
    name_tokens = set(_split_tokens(function_name))
    doc_tokens = set(_split_tokens(docstring)) if docstring else set()
    param_token_set: set[str] = set()
    for p in param_names:
        param_token_set.update(_split_tokens(p))

    # Positive: token appears in name, doc, or params
    if token_lower in name_tokens:
        return 1
    if token_lower in doc_tokens:
        return 1
    if token_lower in param_token_set:
        return 1

    # Negative: token only in noise contexts
    lines = source_code.split("\n") if source_code else []

    in_imports = _is_in_import_context(token, lines)
    in_strings = _is_in_string_context(token, source_code)
    in_comments = _is_in_comment_context(token, lines)
    in_getattr = _is_in_getattr_body(token, source_code)

    if in_imports or in_strings or in_comments or in_getattr:
        return 0

    # Ambiguous: token appears in body logic but not in name/doc/params
    # We treat these as weak negatives for tokens that are just
    # local variables or common patterns
    return None


def _make_example(
    token: str,
    function_name: str,
    docstring: str,
    param_names: list[str],
    decorator_names: list[str],
    source_code: str,
    label: int,
) -> dict:
    """Build a training example."""
    features = extract_features(
        token=token,
        function_name=function_name,
        docstring=docstring,
        param_names=param_names,
        decorator_names=decorator_names,
        source_code=source_code,
    )
    return {"features": features.tolist(), "label": label}


def generate_all(target_count: int = 200_000):
    """Generate training data from extracted symbols."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not SYMBOLS_FILE.exists():
        print(f"ERROR: {SYMBOLS_FILE} not found.")
        print("Run: python -m ml.data_common.extract_symbols")
        return

    symbols = json.loads(SYMBOLS_FILE.read_text())
    functions = [
        s for s in symbols
        if s["type"] == "function" and s.get("source")
    ]
    print(f"Loaded {len(symbols)} symbols, {len(functions)} functions with source")

    rng = random.Random(42)
    rng.shuffle(functions)

    positives: list[dict] = []
    negatives: list[dict] = []
    ambiguous_negatives: list[dict] = []

    print("Extracting (token, function) pairs...")
    for i, func in enumerate(functions):
        name = func["name"]
        docstring = func.get("docstring", "")
        source = func.get("source", "")
        params = func.get("params", [])
        decorators = func.get("decorators", [])

        # Get all unique tokens from source
        source_tokens = set(_split_tokens(source[:3000]))
        if not source_tokens:
            continue

        for token in source_tokens:
            label = _classify_token(token, name, docstring, params, source)

            if label == 1:
                positives.append(
                    _make_example(token, name, docstring, params, decorators, source, 1)
                )
            elif label == 0:
                negatives.append(
                    _make_example(token, name, docstring, params, decorators, source, 0)
                )
            else:
                # Ambiguous -> treat as negative with some probability
                ambiguous_negatives.append(
                    _make_example(token, name, docstring, params, decorators, source, 0)
                )

        if (i + 1) % 5000 == 0:
            print(f"  processed {i + 1}/{len(functions)} functions, "
                  f"pos={len(positives)}, neg={len(negatives)}, "
                  f"ambiguous={len(ambiguous_negatives)}")

        # Early exit if we have enough
        if len(positives) + len(negatives) >= target_count * 2:
            break

    print(f"\nRaw counts: pos={len(positives)}, neg={len(negatives)}, "
          f"ambiguous={len(ambiguous_negatives)}")

    # Balance: use all positives + match with negatives
    # Include some ambiguous negatives (50%) for diversity
    rng.shuffle(ambiguous_negatives)
    extra_neg = ambiguous_negatives[:len(ambiguous_negatives) // 2]

    all_negatives = negatives + extra_neg
    rng.shuffle(all_negatives)

    # Target roughly balanced classes
    n_pos = len(positives)
    n_neg = min(len(all_negatives), n_pos)
    if n_pos > target_count // 2:
        n_pos = target_count // 2
    if n_neg > target_count // 2:
        n_neg = target_count // 2

    rng.shuffle(positives)
    examples = positives[:n_pos] + all_negatives[:n_neg]
    rng.shuffle(examples)
    examples = examples[:target_count]

    print(f"Final: {len(examples)} examples")
    pos_count = sum(1 for e in examples if e["label"] == 1)
    neg_count = len(examples) - pos_count
    print(f"  pos={pos_count}, neg={neg_count}, "
          f"ratio={pos_count / max(1, len(examples)):.2f}")

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
