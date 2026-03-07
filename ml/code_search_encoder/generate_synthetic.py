"""Generate synthetic training pairs for code search from cloned repos.

Strategies:
1. Function name splitting: get_current_user -> "get current user"
2. Template variations: "fetch current user", "retrieve active user"
3. Docstring extraction: first sentence as query
4. Hard negatives: same-file different function, keyword overlap different intent
"""
import json
import random
import re
from pathlib import Path

SYMBOLS_FILE = Path(__file__).parent.parent / "all_symbols.json"
DATA_DIR = Path(__file__).parent / "data"

# Verb synonyms for template expansion
VERB_SYNONYMS = {
    "get": ["fetch", "retrieve", "obtain", "load"],
    "set": ["update", "assign", "configure", "modify"],
    "create": ["make", "build", "generate", "initialize"],
    "delete": ["remove", "destroy", "clear", "drop"],
    "check": ["verify", "validate", "test", "ensure"],
    "send": ["transmit", "dispatch", "emit", "post"],
    "find": ["search", "locate", "lookup", "discover"],
    "save": ["store", "persist", "write", "cache"],
    "handle": ["process", "manage", "deal with", "resolve"],
    "parse": ["extract", "decode", "interpret", "read"],
}


def _split_name(name: str) -> list[str]:
    """Split camelCase/snake_case into tokens."""
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return [p.lower() for p in re.split(r'[^a-zA-Z0-9]+', s) if p and len(p) >= 2]


def _make_query_variants(name_tokens: list[str]) -> list[str]:
    """Generate query variants from function name tokens."""
    queries = [" ".join(name_tokens)]

    # Verb synonym substitution
    if name_tokens:
        verb = name_tokens[0]
        rest = " ".join(name_tokens[1:])
        for synonyms in VERB_SYNONYMS.values():
            if verb in synonyms or verb in VERB_SYNONYMS:
                syn_list = VERB_SYNONYMS.get(verb, [])
                for syn in syn_list[:2]:
                    queries.append(f"{syn} {rest}".strip())

    return queries


def generate_pairs(target_count: int = 500_000):
    """Generate synthetic training pairs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not SYMBOLS_FILE.exists():
        print("ERROR: Run extract_symbols.py first")
        return

    symbols = json.loads(SYMBOLS_FILE.read_text())
    functions = [s for s in symbols if s["type"] == "function" and s.get("name")]

    print(f"Loaded {len(functions)} functions")

    pairs = []

    # Strategy 1: Name-based queries (200K)
    print("Strategy 1: Name-based queries...")
    for func in functions:
        tokens = _split_name(func["name"])
        if len(tokens) < 2:
            continue
        for query in _make_query_variants(tokens):
            code_text = f"{func['name']} {func.get('docstring', '')[:200]}"
            pairs.append({
                "query": query,
                "code": code_text,
                "name": func["name"],
                "label": 1,
                "source": "name_split",
            })
        if len(pairs) >= 200_000:
            break

    # Strategy 2: Docstring-based queries (150K)
    print("Strategy 2: Docstring queries...")
    doc_funcs = [f for f in functions if f.get("docstring") and len(f["docstring"]) > 20]
    for func in doc_funcs:
        doc = func["docstring"]
        # First sentence
        first_sent = doc.split(".")[0].strip()
        if len(first_sent.split()) >= 3:
            code_text = f"{func['name']} {doc[:200]}"
            pairs.append({
                "query": first_sent[:100],
                "code": code_text,
                "name": func["name"],
                "label": 1,
                "source": "docstring",
            })
        if len(pairs) >= 350_000:
            break

    # Strategy 3: Hard negatives (150K)
    print("Strategy 3: Hard negatives...")
    by_file = {}
    for func in functions:
        f = func.get("file", "")
        if f not in by_file:
            by_file[f] = []
        by_file[f].append(func)

    neg_count = 0
    for file_funcs in by_file.values():
        if len(file_funcs) < 2:
            continue
        for i, func in enumerate(file_funcs):
            tokens = _split_name(func["name"])
            if len(tokens) < 2:
                continue
            # Pick a different function from same file as negative
            other = file_funcs[(i + 1) % len(file_funcs)]
            code_text = f"{other['name']} {other.get('docstring', '')[:200]}"
            pairs.append({
                "query": " ".join(tokens),
                "code": code_text,
                "name": other["name"],
                "label": 0,
                "source": "hard_negative",
            })
            neg_count += 1
            if neg_count >= 150_000:
                break
        if neg_count >= 150_000:
            break

    random.shuffle(pairs)
    pairs = pairs[:target_count]

    # Split 80/10/10
    n = len(pairs)
    splits = {
        "train": pairs[:int(0.8 * n)],
        "val": pairs[int(0.8 * n):int(0.9 * n)],
        "test": pairs[int(0.9 * n):],
    }

    for name, data in splits.items():
        path = DATA_DIR / f"synthetic_{name}.jsonl"
        with open(path, "w") as f:
            for p in data:
                f.write(json.dumps(p) + "\n")
        print(f"  {name}: {len(data)} pairs")


if __name__ == "__main__":
    generate_pairs()
