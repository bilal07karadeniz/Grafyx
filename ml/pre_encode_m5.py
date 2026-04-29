"""Pre-encode M5 training data using all CPU cores.

Tokenizes all query/code pairs in parallel and saves as .pt tensors.
This makes training instant-load instead of spending hours on encoding.

Usage:
    python ml/pre_encode_m5.py --train ml/m5_train.json --val ml/m5_val.json
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch

ML_DIR = Path(__file__).parent
PROJECT_DIR = ML_DIR.parent

# These will be set after tokenizer init
_tokenizer = None
MAX_SEQ = 128


def _init_worker():
    """Initialize tokenizer in each worker process."""
    global _tokenizer
    sys.path.insert(0, str(PROJECT_DIR / "grafyx" / "search"))
    from _tokenizer import CodeTokenizer
    _tokenizer = CodeTokenizer()


def _encode_one(text: str) -> list[int]:
    """Encode a single text. Called in worker processes."""
    return _tokenizer.encode(text, max_length=MAX_SEQ)


def _build_symbol_text(sym: dict) -> str:
    """Build text representation for a symbol."""
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


def pre_encode(data_path: str, output_path: str, n_workers: int = 0):
    """Pre-encode a JSON data file into .pt tensor file."""
    if n_workers <= 0:
        n_workers = cpu_count()

    print(f"Loading {data_path}...")
    with open(data_path, encoding="utf-8") as f:
        pairs = json.load(f)

    # Separate positives and negatives
    queries_pos = []
    codes_pos = []
    queries_neg = []
    codes_neg = []

    for p in pairs:
        code_text = _build_symbol_text(p)
        relevance = p.get("relevance", 0.5)
        if relevance > 0.5:
            queries_pos.append(p["query"])
            codes_pos.append(code_text)
        elif relevance < 0.35:
            queries_neg.append(p["query"])
            codes_neg.append(code_text)

    all_texts = queries_pos + codes_pos + queries_neg + codes_neg
    n_pos = len(queries_pos)
    n_neg = len(queries_neg)
    print(f"  {n_pos} positives, {n_neg} negatives, {len(all_texts)} total texts to encode")
    print(f"  Using {n_workers} workers...")

    t0 = time.time()
    with Pool(n_workers, initializer=_init_worker) as pool:
        # Encode all texts in parallel
        all_ids = pool.map(_encode_one, all_texts, chunksize=256)
    elapsed = time.time() - t0
    print(f"  Encoded {len(all_ids)} texts in {elapsed:.1f}s ({len(all_ids)/elapsed:.0f} texts/s)")

    # Split back into components
    idx = 0
    q_pos_ids = all_ids[idx:idx + n_pos]; idx += n_pos
    c_pos_ids = all_ids[idx:idx + n_pos]; idx += n_pos
    q_neg_ids = all_ids[idx:idx + n_neg]; idx += n_neg
    c_neg_ids = all_ids[idx:idx + n_neg]; idx += n_neg

    # Convert to tensors
    data = {
        "q_pos": torch.tensor(q_pos_ids, dtype=torch.long),
        "c_pos": torch.tensor(c_pos_ids, dtype=torch.long),
        "q_neg": torch.tensor(q_neg_ids, dtype=torch.long),
        "c_neg": torch.tensor(c_neg_ids, dtype=torch.long),
    }

    torch.save(data, output_path)
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  Shapes: q_pos={data['q_pos'].shape}, c_pos={data['c_pos'].shape}, "
          f"q_neg={data['q_neg'].shape}, c_neg={data['c_neg'].shape}")


def main():
    parser = argparse.ArgumentParser(description="Pre-encode M5 training data")
    parser.add_argument("--train", type=str, default="ml/m5_train.json")
    parser.add_argument("--val", type=str, default="ml/m5_val.json")
    parser.add_argument("--workers", type=int, default=0, help="0 = all cores")
    args = parser.parse_args()

    t_start = time.time()

    print(f"CPU cores available: {cpu_count()}")
    print()

    train_out = str(Path(args.train).with_suffix(".pt"))
    print(f"=== Encoding training data ===")
    pre_encode(args.train, train_out, n_workers=args.workers)

    print()
    val_out = str(Path(args.val).with_suffix(".pt"))
    print(f"=== Encoding validation data ===")
    pre_encode(args.val, val_out, n_workers=args.workers)

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
