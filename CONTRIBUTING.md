# Contributing to Grafyx

Thanks for your interest! Grafyx is a small, focused MCP server, and contributions are welcome.

## Development setup

Grafyx requires Python 3.12 or 3.13 and effectively Linux at runtime (the underlying [graph-sitter](https://pypi.org/project/graph-sitter/) library does not ship Windows wheels). On Windows, develop and run inside WSL.

```bash
git clone https://github.com/bilal07karadeniz/Grafyx.git
cd Grafyx
python -m venv .venv
source .venv/bin/activate    # On Windows-WSL, same command
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -q
```

The full suite takes about 30-60 seconds. ML model end-to-end tests require the weight files in `grafyx/search/model/` to be present (they ship with the repo).

## Re-training ML models

The trained weights are committed under `grafyx/search/model/`. To retrain:

```bash
# 1. Generate training data (downloads source from popular repos)
python ml/extract_real_symbols.py
python ml/generate_claude_queries.py

# 2. Train the M1-M4 feature MLPs
python ml/retrain_all_real.py
```

Training data files (multi-GB) are gitignored. See `ml/` for individual training scripts.

## Commit style

This repo uses [Conventional Commits](https://www.conventionalcommits.org/). Common prefixes:

- `feat:` user-visible new behavior
- `fix:` bug fixes
- `chore:` build / tooling / cleanup
- `docs:` documentation only
- `test:` tests only
- `refactor:` non-behavioral code changes

## Pull requests

- One logical change per PR.
- Include tests for new behavior.
- The CI workflow runs on Python 3.12 and 3.13 - both must pass.
- Reference any related issue in the PR description.
