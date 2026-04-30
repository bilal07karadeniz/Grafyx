"""Clone benchmark repos at pinned commits (idempotent).

Usage:
    python -m scripts.setup_repos               # clone all at pinned shas
    python -m scripts.setup_repos --pin         # rewrite pinned_commits.json with current HEAD shas
    python -m scripts.setup_repos --repo django # clone only one
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINS_FILE = ROOT / "pinned_commits.json"
REPOS_DIR = ROOT / "repos"


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    res = subprocess.run(
        cmd, cwd=cwd, check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    return res.stdout.strip()


def _ensure_repo(name: str, url: str, target_sha: str | None) -> str:
    """Clone (or update) repo at target_sha. Returns the actual sha checked out."""
    repo_path = REPOS_DIR / name
    REPOS_DIR.mkdir(exist_ok=True)
    if not repo_path.exists():
        print(f"Cloning {name} from {url}...")
        _run(["git", "clone", "--no-tags", url, str(repo_path)])
    if target_sha and target_sha != "PIN_AT_FIRST_RUN":
        try:
            _run(["git", "cat-file", "-e", target_sha], cwd=repo_path)
        except subprocess.CalledProcessError:
            print(f"  fetching {target_sha[:12]}...")
            _run(["git", "fetch", "origin", target_sha], cwd=repo_path)
        _run(["git", "checkout", "--detach", target_sha], cwd=repo_path)
    return _run(["git", "rev-parse", "HEAD"], cwd=repo_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", action="store_true",
                        help="Rewrite pinned_commits.json with current HEAD shas.")
    parser.add_argument("--repo", help="Operate on only this repo.")
    args = parser.parse_args()

    pins = json.loads(PINS_FILE.read_text())
    selected = {args.repo: pins[args.repo]} if args.repo else pins

    for name, info in selected.items():
        sha = _ensure_repo(name, info["url"], None if args.pin else info["commit"])
        if args.pin or info["commit"] == "PIN_AT_FIRST_RUN":
            pins[name]["commit"] = sha
            print(f"  pinned {name} @ {sha[:12]}")
        else:
            assert sha == info["commit"], f"{name}: HEAD {sha[:12]} != pinned {info['commit'][:12]}"
            print(f"  {name} @ {sha[:12]} [ok]")

    PINS_FILE.write_text(json.dumps(pins, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
