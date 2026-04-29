# Public Release — Grafyx 0.1.0

**Date:** 2026-04-29
**Owner:** Bilal Karadeniz <bilal07karadeniz@gmail.com>
**Target repo:** https://github.com/bilal07karadeniz/Grafyx
**Target package:** `grafyx-mcp` on PyPI

## Goal

Take the current private working tree at `C:\Kişisel Projelerim\Grafyx` (branch
`master`, no remote, ~50 uncommitted changes, ~2.8 GB of ML training artifacts on
disk) and produce a clean public release: `github.com/bilal07karadeniz/Grafyx`
on default branch `main`, plus a published `grafyx-mcp` 0.1.0 package on PyPI
installable via `pip install grafyx-mcp` and `uvx --from grafyx-mcp grafyx`.

## Non-goals

- Refactoring code beyond what the release demands.
- Adding features.
- Code-of-conduct, Dependabot, sponsor pages, PyPI organization — can be added
  post-launch.
- Migrating to a GitHub organization — staying personal account for now.

## Constraints

- `graph-sitter` 0.56.x requires Python `>=3.12, <3.14` and effectively Linux
  (Windows users go through WSL — already documented in README).
- The 11 MB of model weights under `grafyx/search/model/` must ship inside the
  wheel (runtime dependency for ML inference).
- The 2.8 GB under `ml/` (training data, repo cache, `.pt` checkpoints) must
  never be committed. GitHub blocks files >100 MB.
- Pending uncommitted work must be preserved as readable history, not squashed
  into one opaque commit.

## Section 1 — Repo cleanup

### 1.1 `.gitignore` additions

Append a project-specific block to the existing `.gitignore`:

```gitignore
# IDE / editor
.cursor/

# ML training artifacts (never commit — multi-GB)
ml/all_symbols.json
ml/claude_training_pairs.json
ml/llm_training_pairs.json
ml/m5_*.json
ml/m5_programmatic_pairs.json
ml/m5_*.pt
ml/.claude_batch_state.json
ml/.repo_cache/

# Generic ML data extensions
*.pt
*.ckpt
```

**Rationale:** The `ml/` directory is kept in the repo (Q3 = A) for training
script transparency, but its data products (multi-GB) and intermediate caches
must not be tracked.

### 1.2 Files to delete from disk

- `_create_resolution.py`
- `_gen_test.py`
- `_test_b64.txt`
- `_write_test.py`
- `.cursor/` (entire directory)
- All 15 files under `docs/plans/`

**Keep:** `docs/grafyx-accuracy-test-prompt.md` (per Q6c, only the test prompt
stays in `docs/`).

### 1.3 Add `.gitattributes`

```gitattributes
* text=auto eol=lf
*.py text eol=lf
*.md text eol=lf
*.json text eol=lf
*.toml text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
*.txt text eol=lf
*.npz binary
*.pt binary
*.bin binary
*.ckpt binary
```

**Rationale:** Eliminate the CRLF/LF warning storm currently fired on every
diff, and explicitly mark model weight formats as binary so they're never
mangled by line-ending conversion.

### 1.4 LICENSE

Replace line 3:

- Before: `Copyright (c) 2025 Grafyx Contributors`
- After: `Copyright (c) 2026 Bilal Karadeniz`

## Section 2 — Metadata fixes

### 2.1 `pyproject.toml`

```toml
[project]
name = "grafyx-mcp"
version = "0.1.0"
description = "MCP server that gives AI coding assistants real-time codebase understanding via Graph-sitter"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.12,<3.14"
authors = [
    {name = "Bilal Karadeniz", email = "bilal07karadeniz@gmail.com"},
]
keywords = ["mcp", "ai", "codebase", "graph", "tree-sitter", "code-analysis"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries",
]
dependencies = [
    "graph-sitter>=0.56.0",
    "fastmcp>=2.0.0",
    "watchdog>=4.0.0",
]

[project.urls]
Homepage = "https://github.com/bilal07karadeniz/Grafyx"
Repository = "https://github.com/bilal07karadeniz/Grafyx"
Issues = "https://github.com/bilal07karadeniz/Grafyx/issues"
```

`[project.scripts]`, `[project.optional-dependencies]`, `[tool.hatch.build.targets.wheel]`,
and `[build-system]` blocks remain unchanged.

**Note on package data:** the wheel must include `grafyx/search/model/*.npz`
and `grafyx/search/model/*.json`. Hatch includes non-`.py` files inside listed
packages by default, but verify locally with `python -m build` and inspect the
wheel before tagging.

### 2.2 `README.md`

- Find/replace `grafyx-ai/grafyx` → `bilal07karadeniz/Grafyx` (occurs twice:
  Contributing section, plus implicit references).
- Fix the "10 tools" claim — current count is **15** (per CLAUDE.md memory).
- Add a brief "ML-augmented search" subsection under "How It Works" that
  mentions M1 relevance ranker, M3 source filter, M4 importance, M5 bi-encoder,
  and the gibberish detector — so the differentiator isn't invisible.
- Add badges (display once CI + PyPI exist):
  - PyPI version: `https://img.shields.io/pypi/v/grafyx-mcp`
  - CI status: `https://github.com/bilal07karadeniz/Grafyx/actions/workflows/ci.yml/badge.svg`

### 2.3 `grafyx/__init__.py`

Keep `__version__ = "0.1.0"` — single source of truth. `pyproject.toml` keeps
its own `version = "0.1.0"` for now. Future releases bump both manually until a
version-from-VCS scheme is introduced.

## Section 3 — Commit strategy + branch rename

### 3.1 Three logical commits

Stage selectively to produce three readable commits in order:

1. **`chore: remove M6 cross-encoder and obsolete training pipelines`**
   - Deletes: `grafyx/antigravity_proxy.py`, `grafyx/search/_cross_encoder.py`,
     `ml/code_search_encoder/*`, `ml/cross_encoder/*`,
     `tests/test_cross_encoder.py`, `tests/test_module_deps.py`.

2. **`feat: ML search pipeline updates (M1-M5)`**
   - All modifications across `grafyx/search/`, `grafyx/graph/`, `grafyx/server/`,
     `ml/caller_disambiguator/`, `ml/data_common/`, and the modified/new tests.
   - New: `grafyx/search/_gibberish.py`, `grafyx/search/model/` weight files,
     `ml/gibberish_detector/`, `ml/retrain_all_real.py`, `ml/retrain_m1.py`,
     `ml/train_all.py`, `ml/train_m5.py`, `ml/train_m5m6_prototype.py`,
     `ml/evaluate_m5.py`, `ml/extract_real_symbols.py`,
     `ml/generate_claude_queries.py`, `ml/generate_llm_queries.py`,
     `ml/generate_synthetic_symbols.py`, `ml/generate_training_data_m5.py`,
     `ml/pre_encode_m5.py`, plus the new tests
     (`test_error_handling.py`, `test_import_resolution.py`,
     `test_ml_models_e2e.py`, `test_search_gibberish.py`).

3. **`chore: prepare repo for public release`**
   - Section 1 changes (`.gitignore`, `.gitattributes`, LICENSE, file deletions).
   - Section 2 changes (`pyproject.toml`, README).
   - Section 4 additions (CONTRIBUTING, CHANGELOG, `.github/`).

(This design doc is committed in a separate, earlier commit during the
brainstorming phase — it is not part of the three implementation commits above.)

### 3.2 Branch rename

```bash
git branch -m master main
```

Local-only at this point — push happens in Section 6.

## Section 4 — GitHub community files

### 4.1 `CONTRIBUTING.md`

Concise (~60 lines): dev setup steps (with explicit WSL note pulled from the
README troubleshooting section), `pytest` invocation, retraining ML models from
`ml/`, commit message conventions (Conventional Commits — matches existing
history `feat:`/`chore:`/`fix:`).

### 4.2 `CHANGELOG.md`

Keep-a-Changelog format. Single `## [0.1.0] - 2026-04-29` section summarizing:
15 MCP tools, ML-augmented search (M1–M5 + gibberish detector), file watcher,
multi-language (Python / TypeScript / JavaScript), fractal-context features.

### 4.3 `.github/ISSUE_TEMPLATE/bug_report.md`

Standard fields: description, repro steps, expected, actual, environment
(OS, Python version, MCP client, grafyx version), logs.

### 4.4 `.github/ISSUE_TEMPLATE/feature_request.md`

Standard fields: motivation, proposed behavior, alternatives considered.

### 4.5 `.github/PULL_REQUEST_TEMPLATE.md`

Summary, related issue, test plan, checklist.

### 4.6 `.github/workflows/ci.yml`

Trigger: push to `main`, PR to `main`. One job, `ubuntu-latest`, matrix
`python-version: ["3.12", "3.13"]`. Steps:

```yaml
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
  with: { python-version: ${{ matrix.python-version }} }
- run: pip install -e ".[dev]"
- run: pytest -q
```

Linux-only because graph-sitter is Linux-only at runtime.

### 4.7 `.github/workflows/release.yml`

Trigger: push of tag matching `v*`. One job, `ubuntu-latest`, Python 3.12.
Steps:

```yaml
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
  with: { python-version: "3.12" }
- run: pip install build
- run: python -m build
- uses: pypa/gh-action-pypi-publish@release/v1
  with:
    # No password — PyPI Trusted Publishing
    skip-existing: true
```

Job environment: `pypi`. Permissions: `id-token: write`. This is the path that
needs PyPI Trusted Publishing configured (Section 5.2).

### 4.8 Out of scope (explicitly)

- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- Dependabot
- CodeQL

These are easy follow-ups; not blockers for 0.1.0.

## Section 5 — PyPI publish

> **Ordering note:** Sections 5.1–5.3 (account, Trusted Publishing setup) can
> happen any time, including before Section 6. Section 5.4 (tagging the first
> release) **must** happen after Section 6 — the PyPI publish job runs on
> GitHub Actions, so the repo and `release.yml` workflow must already be
> pushed for the workflow to fire.

### 5.1 Account

User confirms a PyPI account at `pypi.org/account/register/` with email
`bilal07karadeniz@gmail.com` verified, 2FA enabled.

### 5.2 Trusted Publishing setup (preferred path)

User adds a Pending Publisher at `pypi.org/manage/account/publishing/` with:

- PyPI Project Name: `grafyx-mcp`
- Owner: `bilal07karadeniz`
- Repository name: `Grafyx`
- Workflow name: `release.yml`
- Environment name: `pypi`

User also creates a GitHub Environment named `pypi` on the repo (Settings →
Environments → New environment). No further config required inside the
environment.

### 5.3 Fallback: API token

If Trusted Publishing fails or user prefers tokens: generate scoped token at
`pypi.org/manage/account/token/`, store as repo secret `PYPI_API_TOKEN`,
update `release.yml` to pass `password: ${{ secrets.PYPI_API_TOKEN }}`.

### 5.4 First release

```bash
git tag -a v0.1.0 -m "Grafyx 0.1.0"
git push origin v0.1.0
```

Verify the release workflow finishes green, then `pip install grafyx-mcp` in a
fresh shell on a clean machine (or fresh venv) and confirm the `grafyx --help`
command runs.

## Section 6 — Push to GitHub

```bash
git remote add origin https://github.com/bilal07karadeniz/Grafyx.git
git push -u origin main
```

GitHub default branch is already `main` on a fresh empty repo. Confirm via
Settings → Branches.

After Section 5 succeeds, the README badges go live automatically.

## Verification checklist

After all sections complete, all of these must hold:

- [ ] `git status` shows clean tree on `main`.
- [ ] `git log --oneline` shows the three new commits on top of existing
      history.
- [ ] `git ls-files | grep -E "(all_symbols|claude_training_pairs|m5_.*\.json|m5_.*\.pt|\.repo_cache|\.cursor|_create_resolution|_gen_test|_test_b64|_write_test)"`
      returns empty.
- [ ] Repo size on GitHub is reasonable (< 50 MB — model weights are ~11 MB).
- [ ] CI workflow passes on the initial push.
- [ ] `pip install grafyx-mcp` works in a fresh venv on Linux.
- [ ] `grafyx --version` prints `grafyx 0.1.0`.
- [ ] README badges render and link correctly.
- [ ] `pip show grafyx-mcp` shows `Author: Bilal Karadeniz`.

## Risks and mitigations

- **Model weight files don't ship in wheel.** Hatch package config is correct,
  but verify with `python -m build && unzip -l dist/*.whl` before the first tag.
- **Trusted Publishing handshake fails on first run.** Re-run the workflow
  after fixing PyPI Pending Publisher fields. Worst case fall back to
  Section 5.3 API token path.
- **Large file in commit history.** If any of the multi-GB files were ever
  staged historically (they shouldn't be — `git log --stat` shows none), use
  `git filter-repo` before the public push. Verify before pushing.
- **`graph-sitter` install on Windows native fails.** Already documented in
  README troubleshooting (WSL workaround). CI runs Linux only.
