"""Generate training pairs using Claude Haiku 4.5 Batch API.

Selects diverse symbols from all_symbols.json, submits batch prompts to
Claude Haiku 4.5, parses graded responses into training pairs.

Usage:
    python ml/generate_claude_queries.py                       # Submit batch
    python ml/generate_claude_queries.py --poll BATCH_ID       # Poll + collect
    python ml/generate_claude_queries.py --max-symbols 8000    # Custom count
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

ML_DIR = Path(__file__).parent
DEFAULT_SYMBOLS = ML_DIR / "all_symbols.json"
OUTPUT = ML_DIR / "claude_training_pairs.json"
BATCH_STATE_FILE = ML_DIR / ".claude_batch_state.json"


# ═══════════════════════════════════════════════════════════════════════
#  Symbol Selection
# ═══════════════════════════════════════════════════════════════════════

# Common names to skip (too generic for meaningful queries)
SKIP_NAMES = {
    "__init__", "__repr__", "__str__", "__eq__", "__hash__", "__len__",
    "__getitem__", "__setitem__", "__delitem__", "__contains__", "__iter__",
    "__next__", "__enter__", "__exit__", "__call__", "__bool__",
    "setup", "teardown", "setUp", "tearDown", "main", "run",
    "get", "set", "update", "delete", "create", "test",
}


def select_symbols(symbols: list[dict], max_symbols: int = 8000) -> list[dict]:
    """Select diverse symbols for query generation.

    Strategy:
    - At least 1 symbol per repo (ensures coverage)
    - 70% functions, 30% classes
    - Require docstrings
    - Deduplicate by name
    """
    rng = random.Random(42)

    # Group by repo
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for sym in symbols:
        repo = sym.get("repo", "unknown")
        name = sym.get("name", "")
        doc = sym.get("docstring", "")
        if not doc or len(doc) < 20:
            continue
        if name in SKIP_NAMES or name.startswith("_"):
            continue
        by_repo[repo].append(sym)

    # Ensure at least 1 per repo
    selected = []
    seen_names = set()
    for repo, repo_syms in by_repo.items():
        if repo_syms:
            sym = rng.choice(repo_syms)
            if sym["name"] not in seen_names:
                selected.append(sym)
                seen_names.add(sym["name"])

    # Fill remaining with diverse selection (70% funcs, 30% classes)
    functions = [s for s in symbols if s.get("type") == "function"
                 and s.get("docstring") and len(s.get("docstring", "")) >= 20
                 and s["name"] not in SKIP_NAMES and not s["name"].startswith("_")]
    classes = [s for s in symbols if s.get("type") == "class"
               and s.get("docstring") and len(s.get("docstring", "")) >= 20
               and s["name"] not in SKIP_NAMES and not s["name"].startswith("_")]

    rng.shuffle(functions)
    rng.shuffle(classes)

    remaining = max_symbols - len(selected)
    n_funcs = int(remaining * 0.7)
    n_classes = remaining - n_funcs

    for sym in functions:
        if len(selected) >= max_symbols:
            break
        if sym["name"] not in seen_names:
            selected.append(sym)
            seen_names.add(sym["name"])
            n_funcs -= 1
            if n_funcs <= 0:
                break

    for sym in classes:
        if len(selected) >= max_symbols:
            break
        if sym["name"] not in seen_names:
            selected.append(sym)
            seen_names.add(sym["name"])
            n_classes -= 1
            if n_classes <= 0:
                break

    rng.shuffle(selected)
    return selected[:max_symbols]


# ═══════════════════════════════════════════════════════════════════════
#  Prompt Building
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(sym: dict) -> str:
    """Build the prompt for generating graded search queries."""
    name = sym.get("name", "")
    class_name = sym.get("class_name") or ""
    file_path = sym.get("file", "")
    docstring = (sym.get("docstring") or "")[:200]

    display_name = f"{class_name}.{name}" if class_name else name

    return f"""Given this code symbol, generate search queries at 3 relevance levels.

Symbol: {display_name}
File: {file_path}
Docstring: {docstring}

Rules:
- HIGH: What a developer would type to find this. Do NOT include the function name itself.
- MED: Related concept that could lead a developer here.
- LOW: Shares some keywords but is about something completely different.
- Each query: 3-7 words, lowercase, natural developer search language.

Format (exactly 6 lines):
HIGH: <query>
HIGH: <query>
MED: <query>
MED: <query>
LOW: <query>
LOW: <query>"""


# ═══════════════════════════════════════════════════════════════════════
#  Batch API
# ═══════════════════════════════════════════════════════════════════════

def submit_batch(symbols: list[dict]) -> str:
    """Submit a batch of symbol prompts to Claude Haiku 4.5."""
    import anthropic

    client = anthropic.Anthropic()
    requests = []
    for i, sym in enumerate(symbols):
        requests.append({
            "custom_id": f"sym_{i}",
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": build_prompt(sym)}],
            },
        })

    # Batch API limit is 10,000 requests per batch
    if len(requests) > 10000:
        print(f"WARNING: {len(requests)} requests exceeds 10K limit. Truncating.")
        requests = requests[:10000]

    print(f"Submitting batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted: {batch.id}")
    print(f"Status: {batch.processing_status}")

    # Save state for later polling
    state = {
        "batch_id": batch.id,
        "n_symbols": len(symbols),
        "symbols_file": str(DEFAULT_SYMBOLS),
    }
    with open(BATCH_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"State saved to {BATCH_STATE_FILE}")

    return batch.id


def poll_batch(batch_id: str, symbols: list[dict]) -> list[dict]:
    """Poll until batch completes, then collect and parse results."""
    import anthropic

    client = anthropic.Anthropic()

    # Poll for completion
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        print(f"  Status: {status} | "
              f"succeeded={counts.succeeded}, errored={counts.errored}, "
              f"processing={counts.processing}, canceled={counts.canceled}")

        if status == "ended":
            break
        print("  Waiting 60s...")
        time.sleep(60)

    # Collect results
    pairs = []
    n_succeeded = 0
    n_failed = 0

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        idx = int(custom_id.split("_")[1])

        if result.result.type == "succeeded":
            n_succeeded += 1
            text = result.result.message.content[0].text
            sym = symbols[idx] if idx < len(symbols) else None
            if sym:
                parsed = parse_graded_response(text, sym)
                pairs.extend(parsed)
        else:
            n_failed += 1

    print(f"\nBatch results: {n_succeeded} succeeded, {n_failed} failed")
    print(f"Parsed {len(pairs)} training pairs")
    return pairs


# ═══════════════════════════════════════════════════════════════════════
#  Response Parsing
# ═══════════════════════════════════════════════════════════════════════

_RELEVANCE_MAP = {
    "HIGH": 0.90,
    "MED": 0.60,
    "LOW": 0.15,
}


def parse_graded_response(text: str, sym: dict) -> list[dict]:
    """Parse the 6-line graded response into training pairs."""
    pairs = []
    name = sym.get("name", "")
    file_path = sym.get("file", "")
    docstring = (sym.get("docstring") or "")[:200]
    class_name = sym.get("class_name") or ""

    for line in text.strip().split("\n"):
        line = line.strip()
        for level, relevance in _RELEVANCE_MAP.items():
            prefix = f"{level}:"
            if line.upper().startswith(prefix):
                query = line[len(prefix):].strip().strip('"\'').lower()
                # Validate query quality
                if len(query) < 6 or len(query) > 100:
                    continue
                words = query.split()
                if len(words) < 2 or len(words) > 10:
                    continue
                # Skip if query is just the function name
                if query.replace(" ", "_") == name.lower() or query.replace(" ", "") == name.lower():
                    continue
                pairs.append({
                    "query": query,
                    "name": name,
                    "file": file_path,
                    "docstring": docstring,
                    "class_name": class_name,
                    "relevance": relevance,
                    "source": "claude",
                })
                break

    return pairs


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate query/symbol training pairs via Claude Batch API")
    parser.add_argument("--symbols", type=str, default=str(DEFAULT_SYMBOLS),
                        help="Path to all_symbols.json")
    parser.add_argument("--max-symbols", type=int, default=8000,
                        help="Max symbols to generate queries for")
    parser.add_argument("--poll", type=str, default=None,
                        help="Poll an existing batch ID instead of submitting new")
    parser.add_argument("--output", type=str, default=str(OUTPUT))
    args = parser.parse_args()

    # Load symbols
    symbols_path = Path(args.symbols)
    if not symbols_path.exists():
        print(f"ERROR: {symbols_path} not found. Run extract_symbols.py first.")
        return

    print(f"Loading symbols from {symbols_path}...")
    with open(symbols_path, encoding="utf-8") as f:
        all_symbols = json.load(f)
    print(f"Loaded {len(all_symbols):,} symbols")

    if args.poll:
        # Poll mode: use saved symbols selection or re-select
        selected = select_symbols(all_symbols, max_symbols=args.max_symbols)
        print(f"Selected {len(selected):,} symbols (for result parsing)")
        pairs = poll_batch(args.poll, selected)
    else:
        # Submit mode
        selected = select_symbols(all_symbols, max_symbols=args.max_symbols)
        print(f"Selected {len(selected):,} symbols")

        # Show distribution
        by_type = defaultdict(int)
        by_repo = defaultdict(int)
        for s in selected:
            by_type[s.get("type", "unknown")] += 1
            by_repo[s.get("repo", "unknown")] += 1
        print(f"  Types: {dict(by_type)}")
        print(f"  Repos: {len(by_repo)} repos covered")

        batch_id = submit_batch(selected)
        print(f"\nBatch submitted: {batch_id}")
        print(f"Run again with --poll {batch_id} to collect results.")
        print(f"Or wait here for completion...")

        pairs = poll_batch(batch_id, selected)

    if pairs:
        # Save
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=1, ensure_ascii=False)
        print(f"\nSaved {len(pairs):,} pairs to {output_path}")

        # Stats
        by_level = defaultdict(int)
        for p in pairs:
            if p["relevance"] >= 0.8:
                by_level["HIGH"] += 1
            elif p["relevance"] >= 0.4:
                by_level["MED"] += 1
            else:
                by_level["LOW"] += 1
        print(f"  HIGH: {by_level['HIGH']:,}, MED: {by_level['MED']:,}, LOW: {by_level['LOW']:,}")


if __name__ == "__main__":
    main()
