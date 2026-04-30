# Grafyx Benchmark Harness

Reproducible evaluation suite for Grafyx tools. Runs against pinned commits of public repos.

## Quick start

```bash
pip install -e ".[embeddings,bench]"
cd benchmarks
python -m scripts.setup_repos          # one-time: clones FastAPI, Django, Home Assistant
python -m scripts.run_all              # full eval suite
```

## Layout

- `pinned_commits.json` — repo → sha pinning
- `scripts/` — eval driver scripts (version controlled)
- `repos/` — cloned repos (gitignored)
- `eval_data/` — generated eval datasets (gitignored)
- `results/<date>/<run_id>/` — per-run JSON + markdown summary (gitignored)

Published reproducible results live in `docs/benchmarks/<version>/`.
