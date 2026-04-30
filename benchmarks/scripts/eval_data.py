"""Generate / load query→symbol evaluation pairs.

Output schema (one JSON per line in queries/<repo>.jsonl):
    {
      "query": str,                 # natural-language query
      "expected_symbol": str,       # function or class name expected to be top-1
      "expected_file_suffix": str,  # last 2 path components, helps disambiguate
      "source": "codesearchnet" | "manual"
    }

Per-repo eval pairs are extracted from each repo's actual function docstrings
(first sentence -> query, function name -> expected). This gives us a fair test
that's grounded in the repo itself.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval_data" / "queries"
REPOS_DIR = ROOT / "repos"


def _first_sentence(docstring: str) -> str:
    s = docstring.strip().split("\n\n")[0].replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return ""
    return s.split(". ")[0].rstrip(".")


def _extract_pairs_from_repo(repo_dir: Path, max_pairs: int = 200) -> list[dict]:
    """Walk the repo, pick functions with informative docstrings, build query pairs."""
    pairs: list[dict] = []
    for path in repo_dir.rglob("*.py"):
        sp = str(path).replace("\\", "/")
        if "/test" in sp or "/tests/" in sp:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node) or ""
            sentence = _first_sentence(doc)
            if len(sentence.split()) < 5 or len(sentence.split()) > 25:
                continue
            if node.name.startswith("_"):
                continue
            pairs.append({
                "query": sentence,
                "expected_symbol": node.name,
                "expected_file_suffix": "/".join(path.parts[-2:]),
                "source": "manual",
            })
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def build_eval_pairs() -> dict[str, int]:
    """Build per-repo eval JSONL files. Returns {repo: pair_count}."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pins = json.loads((ROOT / "pinned_commits.json").read_text())
    counts: dict[str, int] = {}
    for repo_name in pins:
        repo_dir = REPOS_DIR / repo_name
        if not repo_dir.exists():
            print(f"  skip {repo_name} (not cloned — run setup_repos first)")
            continue
        pairs = _extract_pairs_from_repo(repo_dir, max_pairs=200)
        out = EVAL_DIR / f"{repo_name}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for p in pairs:
                fh.write(json.dumps(p) + "\n")
        counts[repo_name] = len(pairs)
        print(f"  {repo_name}: {len(pairs)} pairs -> {out.name}")
    return counts


def load_eval_pairs(repo_name: str) -> list[dict]:
    path = EVAL_DIR / f"{repo_name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    counts = build_eval_pairs()
    print(f"\nTotal: {sum(counts.values())} pairs across {len(counts)} repos")
