"""Generate semantic (NL query, code) pairs using an LLM.

THIS IS A PLACEHOLDER. The user will decide what LLM approach to use.
Do NOT run this script until the user specifies the approach.

Expected output: ml/code_search_encoder/data/semantic_pairs.jsonl
Each line: {"query": "natural language query", "code": "function code", "name": "func_name"}

Estimated: 25K functions x 5 queries each = 125K pairs.
"""
import json
from pathlib import Path

SYMBOLS_FILE = Path(__file__).parent.parent / "all_symbols.json"
DATA_DIR = Path(__file__).parent / "data"


def generate_semantic_pairs():
    """PLACEHOLDER: Generate semantic pairs.

    The user will specify which LLM approach to use:
    - Claude Haiku API
    - Local model (Llama, Mistral, etc.)
    - Manual curation
    - Other approach

    This script will be updated once the approach is decided.
    """
    print("=" * 60)
    print("SEMANTIC PAIR GENERATION -- PLACEHOLDER")
    print("=" * 60)
    print()
    print("This script needs an LLM to generate natural language")
    print("search queries for code functions.")
    print()
    print("Expected approach:")
    print("  1. Sample ~25K functions from cloned repos")
    print("  2. For each function, generate 5 diverse NL queries")
    print("  3. Output 125K (query, code) pairs")
    print()
    print(f"Output path: {DATA_DIR / 'semantic_pairs.jsonl'}")
    print()
    print("Waiting for user to specify LLM approach...")
    print("Update this script once decided.")


if __name__ == "__main__":
    generate_semantic_pairs()
