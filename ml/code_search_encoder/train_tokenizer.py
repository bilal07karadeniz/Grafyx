"""Train a BPE tokenizer for code search.

Collects code text from CodeSearchNet and cloned repos, trains sentencepiece
BPE with 16K vocab, and exports merges + vocab as JSON for pure Python inference.

Usage:
    python train_tokenizer.py [--vocab_size 16000] [--output_dir ./model]
"""
import argparse
import json
import tempfile
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / ".repo_cache"


def collect_training_text(max_files: int = 50000) -> list[str]:
    """Collect code snippets and docstrings from cloned repos."""
    texts = []
    for lang_dir in sorted(CACHE_DIR.iterdir()):
        if not lang_dir.is_dir():
            continue
        for repo_dir in sorted(lang_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            # Collect Python files
            for ext in ("*.py", "*.ts", "*.tsx", "*.js", "*.jsx"):
                for f in repo_dir.rglob(ext):
                    try:
                        text = f.read_text(errors="ignore")
                        # Split into chunks of ~500 chars
                        for i in range(0, len(text), 500):
                            chunk = text[i:i+500].strip()
                            if chunk:
                                texts.append(chunk)
                    except Exception:
                        continue
                    if len(texts) >= max_files:
                        break
            if len(texts) >= max_files:
                break
        if len(texts) >= max_files:
            break

    # Also add natural language queries for balanced coverage
    nl_queries = [
        "handle user authentication", "send email notification",
        "validate input data", "database connection pool",
        "process payment transaction", "file upload handler",
        "rate limit middleware", "cache invalidation strategy",
        "error logging and monitoring", "JWT token verification",
        "WebSocket message handler", "REST API endpoint",
        "background task worker", "data serialization",
        "user session management", "role based access control",
        "search index update", "password hashing bcrypt",
        "OAuth2 authorization flow", "GraphQL resolver",
    ]
    texts.extend(nl_queries * 100)  # Repeat to balance

    print(f"Collected {len(texts)} text chunks for tokenizer training")
    return texts


def train_bpe(texts: list[str], vocab_size: int = 16000, output_dir: Path = None):
    """Train sentencepiece BPE and export as JSON."""
    import sentencepiece as spm

    if output_dir is None:
        output_dir = Path(__file__).parent / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write training text to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for text in texts:
            f.write(text + "\n")
        train_file = f.name

    # Train sentencepiece
    model_prefix = str(output_dir / "bpe")
    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        pad_id=0,
        unk_id=1,
        bos_id=2,  # Use as SEP
        eos_id=3,  # Use as CLS
        user_defined_symbols=["[PAD]", "[UNK]", "[SEP]", "[CLS]"],
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="identity",
    )

    # Load trained model and export as JSON
    sp = spm.SentencePieceProcessor()
    sp.load(model_prefix + ".model")

    # Export vocab
    vocab = {}
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        vocab[piece] = i

    vocab_path = output_dir / "bpe_vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2))

    # Export merge rules (extract from sentencepiece model)
    # sentencepiece doesn't directly export merges, so we reconstruct them
    # by iterating through vocab pieces sorted by ID (higher IDs = later merges)
    merges = []
    for i in range(4, sp.get_piece_size()):  # Skip special tokens
        piece = sp.id_to_piece(i)
        if len(piece) >= 2:
            # Try to find the best split point
            for split in range(1, len(piece)):
                left = piece[:split]
                right = piece[split:]
                if left in vocab and right in vocab:
                    merges.append([left, right])
                    break

    merges_path = output_dir / "bpe_merges.json"
    merges_path.write_text(json.dumps(merges, ensure_ascii=False, indent=2))

    print(f"Vocab size: {len(vocab)}")
    print(f"Merge rules: {len(merges)}")
    print(f"Exported to: {output_dir}")

    # Copy to production model dir
    prod_dir = Path(__file__).parent.parent.parent / "grafyx" / "search" / "model"
    if prod_dir.exists():
        import shutil
        shutil.copy2(vocab_path, prod_dir / "bpe_vocab.json")
        shutil.copy2(merges_path, prod_dir / "bpe_merges.json")
        print(f"Copied to production: {prod_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output = Path(args.output_dir) if args.output_dir else None
    texts = collect_training_text()
    train_bpe(texts, args.vocab_size, output)
