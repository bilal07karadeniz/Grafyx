"""Evaluate M5 bi-encoder: recall@K, MRR, qualitative examples.

Usage:
    python ml/evaluate_m5.py                              # Evaluate current model
    python ml/evaluate_m5.py --test-data ml/m5_test.json  # Custom test set
    python ml/evaluate_m5.py --compare old_weights.npz    # Compare vs old model
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).parent
PROJECT_DIR = ML_DIR.parent
MODEL_DIR = PROJECT_DIR / "grafyx" / "search" / "model"
DEFAULT_SYMBOLS = ML_DIR / "all_symbols.json"
DEFAULT_TEST = ML_DIR / "m5_test.json"


def _build_symbol_text(sym: dict) -> str:
    """Build text representation for a symbol (matches training)."""
    name = sym.get("name", "")
    doc = (sym.get("docstring") or "")[:200]
    file_path = sym.get("file", "")
    class_name = sym.get("class_name", "")

    parts = [name]
    if class_name:
        parts.append(class_name)
    if doc:
        parts.append(doc)
    if file_path:
        mod = file_path.replace("/", " ").replace("\\", " ").replace(".py", "")
        parts.append(mod)
    return " ".join(parts)


def load_encoder(weights_path: str | None = None):
    """Load CodeEncoder, optionally with custom weights."""
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    from grafyx.search._code_encoder import CodeEncoder

    encoder = CodeEncoder()
    if weights_path:
        # Override weights path
        encoder._ensure_loaded()  # Load default first to init structure
        data = np.load(weights_path)
        # Re-load with custom weights
        encoder._loaded = False
        encoder._blocks = []
        # Monkey-patch the path temporarily
        import grafyx.search._code_encoder as mod
        original_dir = mod._MODEL_DIR
        mod._MODEL_DIR = Path(weights_path).parent
        encoder._ensure_loaded()
        mod._MODEL_DIR = original_dir
    return encoder


def build_index(encoder, symbols: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Build embedding index from symbols."""
    embeddings = []
    index = []
    for sym in symbols:
        text = _build_symbol_text(sym)
        if not text.strip():
            continue
        emb = encoder.encode(text)
        embeddings.append(emb)
        index.append(sym)
    return np.stack(embeddings), index


def evaluate(
    encoder,
    embeddings: np.ndarray,
    index: list[dict],
    test_pairs: list[dict],
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict:
    """Compute recall@K and MRR metrics."""
    # Build name->indices lookup for matching
    name_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, sym in enumerate(index):
        name_to_indices[sym["name"].lower()].append(i)

    reciprocal_ranks = []
    recall_at = {k: 0 for k in ks}
    total = 0
    skipped = 0

    for pair in test_pairs:
        if pair.get("relevance", 1.0) < 0.5:
            continue  # Skip negative pairs

        query = pair["query"]
        target_name = pair["name"].lower()

        # Check if target exists in index
        target_indices = name_to_indices.get(target_name, [])
        if not target_indices:
            skipped += 1
            continue

        # Encode query and compute similarities
        q_emb = encoder.encode(query)
        sims = embeddings @ q_emb
        ranked = np.argsort(-sims)

        # Find rank of first target match
        rank = None
        for r, idx in enumerate(ranked):
            if idx in target_indices:
                rank = r + 1  # 1-indexed
                break

        if rank is None:
            reciprocal_ranks.append(0.0)
            total += 1
            continue

        reciprocal_ranks.append(1.0 / rank)
        for k in ks:
            if rank <= k:
                recall_at[k] += 1
        total += 1

    if total == 0:
        print(f"WARNING: No valid test pairs (skipped {skipped})")
        return {}

    metrics = {
        "MRR": np.mean(reciprocal_ranks),
        "total_queries": total,
        "skipped": skipped,
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall_at[k] / total

    return metrics


def print_qualitative(
    encoder,
    embeddings: np.ndarray,
    index: list[dict],
    test_pairs: list[dict],
    n_samples: int = 20,
):
    """Print top-5 results for sample queries."""
    import random

    rng = random.Random(42)
    positive_pairs = [p for p in test_pairs if p.get("relevance", 1.0) >= 0.5]
    samples = rng.sample(positive_pairs, min(n_samples, len(positive_pairs)))

    print(f"\n{'='*80}")
    print(f"QUALITATIVE EXAMPLES ({len(samples)} queries)")
    print(f"{'='*80}")

    for pair in samples:
        query = pair["query"]
        target = pair["name"]
        q_emb = encoder.encode(query)
        sims = embeddings @ q_emb
        top_idx = np.argsort(-sims)[:5]

        print(f"\n  Query: \"{query}\"")
        print(f"  Target: {target}")
        found = False
        for rank, idx in enumerate(top_idx, 1):
            sym = index[idx]
            score = float(sims[idx])
            marker = " <-- TARGET" if sym["name"].lower() == target.lower() else ""
            if marker:
                found = True
            print(f"    #{rank}: {sym['name']:40s} ({score:.3f}){marker}")
        if not found:
            # Find actual rank
            ranked = np.argsort(-sims)
            for r, idx in enumerate(ranked):
                if index[idx]["name"].lower() == target.lower():
                    print(f"    (target at rank #{r+1}, score={float(sims[idx]):.3f})")
                    break


def main():
    parser = argparse.ArgumentParser(description="Evaluate M5 bi-encoder")
    parser.add_argument("--test-data", type=str, default=str(DEFAULT_TEST),
                        help="Path to test pairs JSON")
    parser.add_argument("--symbols", type=str, default=str(DEFAULT_SYMBOLS),
                        help="Path to all_symbols.json for building index")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to old weights .npz for comparison")
    parser.add_argument("--max-index", type=int, default=50000,
                        help="Max symbols to index (for speed)")
    parser.add_argument("--qualitative", type=int, default=20,
                        help="Number of qualitative examples to show")
    args = parser.parse_args()

    # Ensure grafyx is importable
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    # Load test data
    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"ERROR: {test_path} not found. Run generate_training_data_m5.py --merge first.")
        return

    with open(test_path, encoding="utf-8") as f:
        test_pairs = json.load(f)
    positive_pairs = [p for p in test_pairs if p.get("relevance", 1.0) >= 0.5]
    print(f"Test data: {len(test_pairs)} pairs ({len(positive_pairs)} positive)")

    # Load symbols for index
    symbols_path = Path(args.symbols)
    if not symbols_path.exists():
        print(f"ERROR: {symbols_path} not found.")
        return

    with open(symbols_path, encoding="utf-8") as f:
        all_symbols = json.load(f)

    # Subsample if too large
    if len(all_symbols) > args.max_index:
        import random
        rng = random.Random(42)
        # Ensure test symbols are included
        test_names = {p["name"].lower() for p in test_pairs}
        must_include = [s for s in all_symbols if s["name"].lower() in test_names]
        remainder = [s for s in all_symbols if s["name"].lower() not in test_names]
        rng.shuffle(remainder)
        all_symbols = must_include + remainder[:args.max_index - len(must_include)]

    print(f"Index symbols: {len(all_symbols):,}")

    # Evaluate current model
    print("\n" + "=" * 60)
    print("EVALUATING CURRENT MODEL")
    print("=" * 60)

    from grafyx.search._code_encoder import CodeEncoder
    encoder = CodeEncoder()
    if not encoder.is_available:
        print("ERROR: Model weights not found.")
        return

    print("Building index...")
    t0 = time.time()
    embeddings, index = build_index(encoder, all_symbols)
    print(f"Index built: {len(index):,} symbols in {time.time() - t0:.1f}s")

    print("\nComputing metrics...")
    metrics = evaluate(encoder, embeddings, index, test_pairs)
    if metrics:
        print(f"\n  MRR:        {metrics['MRR']:.4f}")
        for k in [1, 5, 10, 20]:
            key = f"recall@{k}"
            if key in metrics:
                print(f"  Recall@{k:2d}:  {metrics[key]:.4f}")
        print(f"  Queries:    {metrics['total_queries']} (skipped {metrics['skipped']})")

    # Qualitative examples
    if args.qualitative > 0:
        print_qualitative(encoder, embeddings, index, test_pairs, n_samples=args.qualitative)

    # Compare with old model
    if args.compare:
        print("\n" + "=" * 60)
        print(f"COMPARING WITH OLD MODEL: {args.compare}")
        print("=" * 60)

        old_encoder = load_encoder(args.compare)
        if old_encoder.is_available:
            print("Building old index...")
            old_embeddings, old_index = build_index(old_encoder, all_symbols)

            print("Computing old metrics...")
            old_metrics = evaluate(old_encoder, old_embeddings, old_index, test_pairs)

            if old_metrics and metrics:
                print(f"\n{'Metric':<15s} {'Old':>10s} {'New':>10s} {'Delta':>10s}")
                print("-" * 50)
                for key in ["MRR", "recall@1", "recall@5", "recall@10", "recall@20"]:
                    if key in old_metrics and key in metrics:
                        old_val = old_metrics[key]
                        new_val = metrics[key]
                        delta = new_val - old_val
                        sign = "+" if delta >= 0 else ""
                        print(f"  {key:<13s} {old_val:>10.4f} {new_val:>10.4f} {sign}{delta:>9.4f}")
        else:
            print("ERROR: Old model weights not available.")


if __name__ == "__main__":
    main()
