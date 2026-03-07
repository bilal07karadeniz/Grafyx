"""Download CodeSearchNet dataset for training code search models.

Downloads Python and JavaScript subsets, filters to (docstring, code) pairs
with meaningful docstrings (>= 10 words).
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def download_and_filter(min_doc_words: int = 10, max_examples: int = 2_000_000):
    """Download CodeSearchNet and filter to quality pairs."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets  (HuggingFace datasets)")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "codesearchnet.jsonl"

    count = 0
    with open(output, "w") as f:
        for lang in ["python", "javascript"]:
            print(f"Downloading CodeSearchNet {lang}...")
            try:
                ds = load_dataset("code_search_net", lang, split="train",
                                  trust_remote_code=True)
            except Exception as e:
                print(f"  Error loading {lang}: {e}")
                continue

            for example in ds:
                doc = example.get("func_documentation_string", "")
                code = example.get("func_code_string", "")
                name = example.get("func_name", "")

                if not doc or not code:
                    continue
                doc_words = doc.strip().split()
                if len(doc_words) < min_doc_words:
                    continue

                f.write(json.dumps({
                    "query": " ".join(doc_words[:50]),  # First 50 words of docstring
                    "code": code[:2000],  # Cap at 2K chars
                    "name": name,
                    "language": lang,
                    "source": "codesearchnet",
                }) + "\n")
                count += 1

                if count >= max_examples:
                    break

            if count >= max_examples:
                break

    print(f"Saved {count} pairs to {output}")


if __name__ == "__main__":
    download_and_filter()
