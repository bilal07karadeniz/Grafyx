"""Generate programmatic training pairs for M5 bi-encoder + merge all sources.

Three zero-cost generators:
  1. Docstring first-sentence pairs (~8K)
  2. Name-splitting pseudo-queries (~8K)
  3. Docstring paraphrase variants (~4K)

Merge mode combines all data sources, deduplicates, and splits into train/val/test.

Usage:
    python ml/generate_training_data_m5.py                   # Generate Tier 1 pairs
    python ml/generate_training_data_m5.py --merge           # Merge all sources + split
    python ml/generate_training_data_m5.py --symbols all_symbols.json  # Custom symbols file
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ML_DIR = Path(__file__).parent
DEFAULT_SYMBOLS = ML_DIR / "all_symbols.json"
OUTPUT_PROGRAMMATIC = ML_DIR / "m5_programmatic_pairs.json"


# ═══════════════════════════════════════════════════════════════════════
#  Generator 1: Docstring first-sentence pairs
# ═══════════════════════════════════════════════════════════════════════

def _extract_first_sentence(docstring: str) -> str | None:
    """Extract the first meaningful sentence from a docstring."""
    if not docstring:
        return None
    # Strip leading whitespace, quotes, and common prefixes
    doc = docstring.strip().strip('"').strip("'").strip()
    # Take first line or first sentence
    first = doc.split("\n")[0].strip().strip('"').strip("'").strip()
    # Try splitting by period for multi-sentence first lines
    if ". " in first:
        first = first.split(". ")[0] + "."
    # Clean up RST/Sphinx markup
    first = re.sub(r':[\w]+:`([^`]+)`', r'\1', first)  # :func:`foo` -> foo
    first = re.sub(r'``([^`]+)``', r'\1', first)  # ``foo`` -> foo
    first = first.strip()
    # Filter out bad candidates
    if len(first) < 15:
        return None
    if first.lower().startswith(("args:", "returns:", "raises:", "todo", "note:", "see ")):
        return None
    if first.startswith(".."):  # RST directive
        return None
    if ">>>" in first:  # Doctest
        return None
    return first.lower()


def generate_docstring_pairs(symbols: list[dict], max_pairs: int = 8000) -> list[dict]:
    """Generate pairs from docstring first sentences."""
    rng = random.Random(42)
    candidates = []
    for sym in symbols:
        doc = sym.get("docstring", "")
        if not doc:
            continue
        sentence = _extract_first_sentence(doc)
        if sentence:
            candidates.append((sentence, sym))

    rng.shuffle(candidates)
    pairs = []
    seen_queries = set()
    for sentence, sym in candidates:
        if len(pairs) >= max_pairs:
            break
        # Deduplicate by query text
        if sentence in seen_queries:
            continue
        seen_queries.add(sentence)
        pairs.append({
            "query": sentence,
            "name": sym.get("name", ""),
            "file": sym.get("file", ""),
            "docstring": (sym.get("docstring") or "")[:200],
            "class_name": sym.get("class_name") or "",
            "relevance": 0.80,
            "source": "docstring",
        })
    return pairs


# ═══════════════════════════════════════════════════════════════════════
#  Generator 2: Name-splitting pseudo-queries
# ═══════════════════════════════════════════════════════════════════════

def _split_name(name: str) -> list[str]:
    """Split snake_case/camelCase name into tokens."""
    # Handle snake_case
    if "_" in name:
        parts = name.split("_")
    else:
        # Handle camelCase/PascalCase
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)', name)
    return [p.lower() for p in parts if p and len(p) > 1]


def generate_name_split_pairs(symbols: list[dict], max_pairs: int = 8000) -> list[dict]:
    """Generate pairs from splitting symbol names into natural language."""
    rng = random.Random(43)
    candidates = []
    for sym in symbols:
        name = sym.get("name", "")
        if not name or name.startswith("__"):
            continue
        tokens = _split_name(name)
        if len(tokens) < 3:
            continue
        query = " ".join(tokens)
        if len(query) < 8:  # Too short
            continue
        candidates.append((query, sym))

    rng.shuffle(candidates)
    pairs = []
    seen_queries = set()
    for query, sym in candidates:
        if len(pairs) >= max_pairs:
            break
        if query in seen_queries:
            continue
        seen_queries.add(query)
        pairs.append({
            "query": query,
            "name": sym.get("name", ""),
            "file": sym.get("file", ""),
            "docstring": (sym.get("docstring") or "")[:200],
            "class_name": sym.get("class_name") or "",
            "relevance": 0.90,
            "source": "name_split",
        })
    return pairs


# ═══════════════════════════════════════════════════════════════════════
#  Generator 3: Docstring paraphrase variants
# ═══════════════════════════════════════════════════════════════════════

_LEADING_VERBS = [
    "parse", "get", "set", "create", "delete", "update", "check", "validate",
    "convert", "build", "compute", "calculate", "generate", "handle", "process",
    "read", "write", "load", "save", "send", "receive", "open", "close",
    "start", "stop", "run", "execute", "fetch", "find", "search", "filter",
    "add", "remove", "insert", "append", "extract", "transform", "render",
    "return", "returns", "initialize", "configure", "register", "dispatch",
]


def _paraphrase(sentence: str) -> list[str]:
    """Generate simple paraphrase variants of a docstring sentence."""
    variants = []
    lower = sentence.lower().strip().rstrip(".")

    words = lower.split()
    if len(words) < 3:
        return []

    # Variant 1: Remove leading verb (including conjugated forms like "creates", "returns")
    first_word = words[0]
    is_verb = first_word in _LEADING_VERBS
    base_verb = first_word
    if not is_verb and first_word.endswith("s") and first_word[:-1] in _LEADING_VERBS:
        is_verb = True
        base_verb = first_word[:-1]
    if not is_verb and first_word.endswith("es") and first_word[:-2] in _LEADING_VERBS:
        is_verb = True
        base_verb = first_word[:-2]

    if is_verb:
        noun_phrase = " ".join(words[1:])
        noun_phrase = re.sub(r'^(the|a|an)\s+', '', noun_phrase)
        if len(noun_phrase) > 8:
            variants.append(noun_phrase)
        # Also create base verb form: "creates X" -> "create X"
        if first_word != base_verb:
            alt = base_verb + " " + " ".join(words[1:])
            if len(alt) > 8:
                variants.append(alt)

    # Variant 2: Handle -ing forms ("creating X" -> "create X")
    if first_word.endswith("ing") and len(first_word) > 5:
        rest = " ".join(words[1:])
        # Try common -ing -> base transformations
        stem = first_word[:-3]
        if stem + "e" in _LEADING_VERBS:
            variants.append(stem + "e " + rest)
        elif stem in _LEADING_VERBS:
            variants.append(stem + " " + rest)
        # Also just the noun phrase without the -ing verb
        noun_phrase = re.sub(r'^(the|a|an)\s+', '', rest)
        if len(noun_phrase) > 8:
            variants.append(noun_phrase)

    # Variant 3: Extract key noun phrases (remove filler prefixes)
    for prefix in ["function to ", "method to ", "helper to ", "utility to ",
                    "function that ", "method that ", "class that ", "class for ",
                    "a ", "an ", "the ", "base class for ", "abstract class for ",
                    "mixin for ", "wrapper for ", "decorator for "]:
        if lower.startswith(prefix) and len(lower) > len(prefix) + 8:
            remainder = lower[len(prefix):]
            if len(remainder) > 8:
                variants.append(remainder)

    # Variant 4: If sentence has "for" or "of", extract what comes after
    for connector in [" for ", " of ", " to ", " that "]:
        if connector in lower:
            idx = lower.index(connector)
            after = lower[idx + len(connector):]
            after = re.sub(r'^(the|a|an)\s+', '', after)
            if 8 < len(after) < len(lower) - 5:
                variants.append(after)
                break

    return variants


def generate_paraphrase_pairs(symbols: list[dict], max_pairs: int = 4000) -> list[dict]:
    """Generate pairs from paraphrasing docstring sentences."""
    rng = random.Random(44)
    candidates = []
    for sym in symbols:
        doc = sym.get("docstring", "")
        if not doc:
            continue
        sentence = _extract_first_sentence(doc)
        if not sentence:
            continue
        variants = _paraphrase(sentence)
        for v in variants:
            candidates.append((v, sym))

    rng.shuffle(candidates)
    pairs = []
    seen_queries = set()
    for query, sym in candidates:
        if len(pairs) >= max_pairs:
            break
        if query in seen_queries:
            continue
        seen_queries.add(query)
        pairs.append({
            "query": query,
            "name": sym.get("name", ""),
            "file": sym.get("file", ""),
            "docstring": (sym.get("docstring") or "")[:200],
            "class_name": sym.get("class_name") or "",
            "relevance": 0.75,
            "source": "paraphrase",
        })
    return pairs


# ═══════════════════════════════════════════════════════════════════════
#  Merge + Deduplicate + Split
# ═══════════════════════════════════════════════════════════════════════

def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings (token-level)."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def merge_and_split(
    programmatic_path: Path | None = None,
    claude_path: Path | None = None,
    ollama_path: Path | None = None,
    output_dir: Path | None = None,
):
    """Merge all data sources, deduplicate, and create train/val/test splits."""
    out = output_dir or ML_DIR
    all_pairs = []

    # Load all sources
    sources = [
        (programmatic_path or ML_DIR / "m5_programmatic_pairs.json", "programmatic"),
        (claude_path or ML_DIR / "claude_training_pairs.json", "claude"),
        (ollama_path or ML_DIR / "llm_training_pairs.json", "ollama"),
    ]

    for path, source_name in sources:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                pairs = json.load(f)
            # Ensure source tag
            for p in pairs:
                if "source" not in p:
                    p["source"] = source_name
            all_pairs.extend(pairs)
            print(f"  Loaded {len(pairs):,} pairs from {path.name} ({source_name})")
        else:
            print(f"  Skipped {path.name} (not found)")

    if not all_pairs:
        print("ERROR: No data sources found.")
        return

    print(f"\nTotal before dedup: {len(all_pairs):,}")

    # Step 1: Remove exact (query, name) duplicates
    seen_exact = set()
    deduped = []
    for p in all_pairs:
        key = (p["query"].strip().lower(), p["name"].strip().lower())
        if key not in seen_exact:
            seen_exact.add(key)
            deduped.append(p)

    print(f"After exact dedup: {len(deduped):,} (removed {len(all_pairs) - len(deduped):,})")

    # Step 2: Remove near-duplicate queries for same symbol (Jaccard > 0.8)
    by_symbol = defaultdict(list)
    for p in deduped:
        by_symbol[p["name"].lower()].append(p)

    final = []
    removed_near = 0
    for name, pairs in by_symbol.items():
        kept = []
        for p in pairs:
            is_near_dup = False
            for existing in kept:
                if _jaccard_similarity(p["query"], existing["query"]) > 0.8:
                    is_near_dup = True
                    removed_near += 1
                    break
            if not is_near_dup:
                kept.append(p)
        final.extend(kept)

    print(f"After near-dedup: {len(final):,} (removed {removed_near:,})")

    # Step 3: Symbol-level train/val/test split
    rng = random.Random(42)

    # Group by symbol name for symbol-level split
    symbol_names = sorted(set(p["name"].lower() for p in final))
    rng.shuffle(symbol_names)

    n_total = len(symbol_names)
    n_train = int(n_total * 0.85)
    n_val = int(n_total * 0.10)

    train_symbols = set(symbol_names[:n_train])
    val_symbols = set(symbol_names[n_train:n_train + n_val])
    test_symbols = set(symbol_names[n_train + n_val:])

    train_pairs = [p for p in final if p["name"].lower() in train_symbols]
    val_pairs = [p for p in final if p["name"].lower() in val_symbols]
    test_pairs = [p for p in final if p["name"].lower() in test_symbols]

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    rng.shuffle(test_pairs)

    # Report source distribution
    for split_name, split_data in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        source_counts = defaultdict(int)
        for p in split_data:
            source_counts[p.get("source", "unknown")] += 1
        dist = ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items()))
        print(f"  {split_name}: {len(split_data):,} pairs ({dist})")

    # Save splits
    for name, data in [("m5_train", train_pairs), ("m5_val", val_pairs), ("m5_test", test_pairs)]:
        path = out / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print(f"  Saved {path.name}: {len(data):,} pairs")

    # Verify no symbol overlap between train and test
    train_names = {p["name"].lower() for p in train_pairs}
    test_names = {p["name"].lower() for p in test_pairs}
    overlap = train_names & test_names
    if overlap:
        print(f"WARNING: {len(overlap)} symbols appear in both train and test!")
    else:
        print("  No symbol overlap between train and test (OK)")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate M5 training data")
    parser.add_argument("--symbols", type=str, default=str(DEFAULT_SYMBOLS),
                        help="Path to all_symbols.json (flat list)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all sources and create train/val/test splits")
    parser.add_argument("--max-docstring", type=int, default=8000)
    parser.add_argument("--max-namesplit", type=int, default=8000)
    parser.add_argument("--max-paraphrase", type=int, default=4000)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PROGRAMMATIC))
    args = parser.parse_args()

    if args.merge:
        print("=" * 60)
        print("MERGING ALL DATA SOURCES")
        print("=" * 60)
        merge_and_split()
        return

    # Generate Tier 1 programmatic pairs
    print("=" * 60)
    print("GENERATING TIER 1 PROGRAMMATIC PAIRS")
    print("=" * 60)

    symbols_path = Path(args.symbols)
    if not symbols_path.exists():
        print(f"ERROR: {symbols_path} not found. Run extract_symbols.py first.")
        return

    print(f"Loading symbols from {symbols_path}...")
    with open(symbols_path, encoding="utf-8") as f:
        symbols = json.load(f)
    print(f"Loaded {len(symbols):,} symbols")

    # Generate all three types
    print("\n--- Docstring first-sentence pairs ---")
    docstring_pairs = generate_docstring_pairs(symbols, max_pairs=args.max_docstring)
    print(f"  Generated: {len(docstring_pairs):,}")

    print("\n--- Name-split pseudo-queries ---")
    namesplit_pairs = generate_name_split_pairs(symbols, max_pairs=args.max_namesplit)
    print(f"  Generated: {len(namesplit_pairs):,}")

    print("\n--- Docstring paraphrase variants ---")
    paraphrase_pairs = generate_paraphrase_pairs(symbols, max_pairs=args.max_paraphrase)
    print(f"  Generated: {len(paraphrase_pairs):,}")

    # Combine
    all_pairs = docstring_pairs + namesplit_pairs + paraphrase_pairs
    print(f"\nTotal Tier 1 pairs: {len(all_pairs):,}")

    # Quick stats
    by_source = defaultdict(int)
    for p in all_pairs:
        by_source[p["source"]] += 1
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count:,}")

    # Save
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=1, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
