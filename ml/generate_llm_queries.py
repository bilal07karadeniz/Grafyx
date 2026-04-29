#!/usr/bin/env python3
"""Generate natural language search queries using a local LLM (Ollama).

For each real function/class extracted from repos, asks the LLM to generate
queries a developer would type to find that symbol. This creates realistic
(query, symbol, relevance_grade) training pairs.

Uses Ollama with Qwen 3.5 8B for fast local inference.
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests required. pip install requests")
    sys.exit(1)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")


def _update_model(m: str):
    global MODEL
    MODEL = m


def _call_ollama(prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
    """Call Ollama API and return the response text.

    Uses think=false to disable thinking mode (official Ollama API parameter).
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "think": False,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  Ollama error: {e}")
        return ""


def generate_queries_for_symbol(symbol: dict, symbol_type: str) -> list[dict]:
    """Generate search queries for a single symbol using LLM.

    Returns list of {query, relevance} pairs.
    """
    name = symbol.get("name", "")
    file_path = symbol.get("file", "")
    docstring = symbol.get("docstring", "")[:200]
    class_name = symbol.get("class_name", "")

    if symbol_type == "function":
        context = f"Function: {name}"
        if class_name:
            context = f"Method: {class_name}.{name}"
        if docstring:
            context += f"\nDocstring: {docstring}"
        context += f"\nFile: {file_path}"
    elif symbol_type == "class":
        methods = symbol.get("methods", [])[:10]
        context = f"Class: {name}"
        if docstring:
            context += f"\nDocstring: {docstring}"
        context += f"\nFile: {file_path}"
        if methods:
            context += f"\nMethods: {', '.join(methods[:8])}"

    prompt = f"""Given this Python code symbol, generate exactly 5 different search queries a developer might type to find this code. Each query should be a natural phrase (2-6 words), NOT the exact function name.

{context}

Output format - exactly 5 lines, one query per line. No numbering, no quotes, no explanation:
"""

    response = _call_ollama(prompt, max_tokens=200, temperature=0.8)
    if not response:
        return []

    queries = []
    for line in response.strip().split("\n"):
        line = line.strip().strip("- ").strip("1234567890.)")
        line = re.sub(r'^[\d]+[\.\)]\s*', '', line)  # Remove numbering
        line = line.strip('"\'')
        if 2 <= len(line.split()) <= 8 and len(line) > 5:
            queries.append({"query": line.lower(), "relevance": 0.85})

    return queries[:5]


def generate_hard_negatives(symbol: dict, all_symbols: list[dict]) -> list[dict]:
    """Generate queries that are related but NOT about this specific symbol.

    These are hard negatives for training: queries that share some keywords
    but should match a different function.
    """
    name = symbol.get("name", "")
    docstring = symbol.get("docstring", "")[:200]

    prompt = f"""Given this Python function: {name}
Docstring: {docstring}

Generate 3 search queries that are SIMILAR but would match a DIFFERENT function (not this one). These should share 1-2 words but be about something else.

Output format - exactly 3 lines, one query per line. No numbering, no explanation:
"""

    response = _call_ollama(prompt, max_tokens=150, temperature=0.9)
    if not response:
        return []

    negatives = []
    for line in response.strip().split("\n"):
        line = line.strip().strip("- ").strip("1234567890.)")
        line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
        line = line.strip('"\'')
        if 2 <= len(line.split()) <= 8 and len(line) > 5:
            negatives.append({"query": line.lower(), "relevance": 0.15})

    return negatives[:3]


def generate_graded_pairs(symbol: dict, all_symbols: list[dict]) -> list[dict]:
    """Generate graded relevance pairs for a symbol.

    Returns pairs at multiple relevance levels:
    - 0.85-0.95: Direct match queries (from LLM)
    - 0.60-0.75: Partial match queries (from LLM)
    - 0.10-0.25: Hard negative queries (from LLM)
    - 0.00-0.05: Random unrelated queries (programmatic)
    """
    name = symbol.get("name", "")
    file_path = symbol.get("file", "")
    docstring = symbol.get("docstring", "")[:200]
    class_name = symbol.get("class_name", "")

    display_name = f"{class_name}.{name}" if class_name else name

    prompt = f"""Given this Python symbol, generate search queries at different relevance levels.

Symbol: {display_name}
File: {file_path}
Docstring: {docstring}

Generate queries in this EXACT format (one per line):
HIGH: <query that directly describes this function>
HIGH: <another direct query>
MED: <query partially related, would also find this>
MED: <another partial query>
LOW: <query that shares keywords but is about something different>
LOW: <another misleading query>
"""

    response = _call_ollama(prompt, max_tokens=250, temperature=0.7)
    if not response:
        return []

    pairs = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("HIGH:"):
            query = line[5:].strip().strip('"\'').lower()
            if len(query) > 5:
                pairs.append({
                    "query": query,
                    "name": name,
                    "file": file_path,
                    "docstring": docstring,
                    "class_name": class_name,
                    "relevance": random.uniform(0.85, 0.95),
                })
        elif line.upper().startswith("MED:"):
            query = line[4:].strip().strip('"\'').lower()
            if len(query) > 5:
                pairs.append({
                    "query": query,
                    "name": name,
                    "file": file_path,
                    "docstring": docstring,
                    "class_name": class_name,
                    "relevance": random.uniform(0.55, 0.70),
                })
        elif line.upper().startswith("LOW:"):
            query = line[4:].strip().strip('"\'').lower()
            if len(query) > 5:
                pairs.append({
                    "query": query,
                    "name": name,
                    "file": file_path,
                    "docstring": docstring,
                    "class_name": class_name,
                    "relevance": random.uniform(0.10, 0.25),
                })

    return pairs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols-file", default=str(Path(__file__).parent / "real_symbols.json"))
    parser.add_argument("--output", default=str(Path(__file__).parent / "llm_training_pairs.json"))
    parser.add_argument("--max-symbols", type=int, default=2000,
                        help="Max symbols to generate queries for")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--test", action="store_true", help="Test with 5 symbols")
    args = parser.parse_args()

    _update_model(args.model)

    # Load symbols
    print(f"Loading symbols from {args.symbols_file}...")
    with open(args.symbols_file, encoding="utf-8") as f:
        repos_data = json.load(f)

    # Collect all functions and classes
    all_functions = []
    all_classes = []
    for repo in repos_data:
        for func in repo["functions"]:
            if not func["name"].startswith("_") or func["name"].startswith("__"):
                all_functions.append(func)
        for cls in repo["classes"]:
            if not cls["name"].startswith("_"):
                all_classes.append(cls)

    print(f"Found {len(all_functions)} functions, {len(all_classes)} classes")

    # Sample diverse symbols
    rng = random.Random(42)

    # Prefer symbols with docstrings (more interesting)
    with_doc = [f for f in all_functions if f.get("docstring")]
    without_doc = [f for f in all_functions if not f.get("docstring")]
    classes_with_doc = [c for c in all_classes if c.get("docstring")]

    max_sym = 5 if args.test else args.max_symbols
    n_func_doc = min(int(max_sym * 0.5), len(with_doc))
    n_func_nodoc = min(int(max_sym * 0.15), len(without_doc))
    n_class = min(int(max_sym * 0.35), len(classes_with_doc))

    selected_funcs = rng.sample(with_doc, n_func_doc) + rng.sample(without_doc, min(n_func_nodoc, len(without_doc)))
    selected_classes = rng.sample(classes_with_doc, min(n_class, len(classes_with_doc)))

    total = len(selected_funcs) + len(selected_classes)
    print(f"Generating queries for {total} symbols ({len(selected_funcs)} funcs, {len(selected_classes)} classes)")
    print(f"Model: {MODEL}")

    # Test connection
    print("Testing Ollama connection...", end=" ", flush=True)
    test = _call_ollama("Say 'OK' in one word.", max_tokens=200)
    if not test:
        print("FAILED - is Ollama running?")
        sys.exit(1)
    print(f"OK ({test[:20]})")

    # Generate pairs
    all_pairs = []
    t0 = time.time()

    for i, func in enumerate(selected_funcs):
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"  [{i}/{total}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

        pairs = generate_graded_pairs(func, all_functions)
        all_pairs.extend(pairs)

    for i, cls in enumerate(selected_classes):
        idx = len(selected_funcs) + i
        if idx % 50 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (total - idx) / rate if rate > 0 else 0
            print(f"  [{idx}/{total}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

        pairs = generate_graded_pairs(
            {"name": cls["name"], "file": cls["file"], "docstring": cls.get("docstring", ""),
             "class_name": ""},
            all_functions,
        )
        all_pairs.extend(pairs)

    elapsed = time.time() - t0
    print(f"\nGenerated {len(all_pairs)} training pairs in {elapsed:.0f}s")
    print(f"  HIGH (0.85+): {sum(1 for p in all_pairs if p['relevance'] > 0.8)}")
    print(f"  MED (0.5-0.8): {sum(1 for p in all_pairs if 0.5 <= p['relevance'] <= 0.8)}")
    print(f"  LOW (<0.5): {sum(1 for p in all_pairs if p['relevance'] < 0.5)}")

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=1, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
