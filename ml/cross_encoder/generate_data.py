"""Generate training data for cross-encoder reranker.

Takes positive pairs from M5's training data and creates hard negatives
by pairing queries with similar-but-wrong code snippets.
"""
import json
import random
from pathlib import Path

M5_DATA_DIR = Path(__file__).parent.parent / "code_search_encoder" / "data"
DATA_DIR = Path(__file__).parent / "data"


def generate(target_count: int = 500_000):
    """Generate reranking training data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load M5 training data
    sources = []
    for f in sorted(M5_DATA_DIR.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                sources.append(json.loads(line))

    if not sources:
        print("ERROR: No M5 training data found. Run M5 data generation first.")
        return

    print(f"Loaded {len(sources)} source examples")

    positives = [s for s in sources if s.get("label", 1) == 1]
    all_codes = [s for s in sources if s.get("code")]

    examples = []

    # Positive examples: relevant (query, code) pairs
    print("Generating positive examples...")
    for p in positives:
        examples.append({
            "query": p["query"],
            "code": p.get("code", ""),
            "name": p.get("name", ""),
            "label": 1,
        })

    # Hard negatives: pair each query with a random code snippet
    print("Generating hard negatives...")
    for p in positives:
        neg = random.choice(all_codes)
        # Make sure it's a different function
        while neg.get("name") == p.get("name"):
            neg = random.choice(all_codes)
        examples.append({
            "query": p["query"],
            "code": neg.get("code", ""),
            "name": neg.get("name", ""),
            "label": 0,
        })

    random.shuffle(examples)
    examples = examples[:target_count]

    # Split
    n = len(examples)
    splits = {
        "train": examples[:int(0.8 * n)],
        "val": examples[int(0.8 * n):int(0.9 * n)],
        "test": examples[int(0.9 * n):],
    }

    for name, data in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        pos = sum(1 for e in data if e["label"] == 1)
        print(f"  {name}: {len(data)} examples ({pos} pos, {len(data)-pos} neg)")


if __name__ == "__main__":
    generate()
