"""Clone OSS repos for training data generation."""
import json
import subprocess
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / ".repo_cache"
REPOS_FILE = Path(__file__).parent / "repos.json"


def clone_all(max_depth: int = 1):
    """Shallow clone all repos."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    repos = json.loads(REPOS_FILE.read_text())
    for lang, repo_list in repos.items():
        for entry in repo_list:
            repo = entry["repo"]
            name = repo.split("/")[-1]
            dest = CACHE_DIR / lang / name
            if dest.exists():
                print(f"  Skip {repo} (exists)")
                continue
            print(f"  Clone {repo}...")
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", str(max_depth),
                 f"https://github.com/{repo}.git", str(dest)],
                capture_output=True,
            )


if __name__ == "__main__":
    clone_all()
