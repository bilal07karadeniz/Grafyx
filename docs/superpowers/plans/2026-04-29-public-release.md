# Public Release 0.1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the private working tree at `C:\Kişisel Projelerim\Grafyx` and produce a published, public, installable Grafyx 0.1.0 — repo at `github.com/bilal07karadeniz/Grafyx` (default branch `main`), package `grafyx-mcp` 0.1.0 on PyPI.

**Architecture:** Three-phase execution. Phase A is local: stage and split pending changes into three logical commits, then add release metadata + community files in a fourth commit. Phase B pushes to GitHub and validates CI. Phase C tags v0.1.0 and lets the release workflow publish to PyPI via Trusted Publishing.

**Tech Stack:** Python 3.12/3.13, hatchling, pytest, GitHub Actions, PyPI Trusted Publishing.

**Spec:** `docs/superpowers/specs/2026-04-29-public-release-design.md` (commit `d94cdb2`).

---

## Context for the engineer

You're preparing a Python MCP server (Model Context Protocol) for public release. The repo currently has 50 uncommitted changes from a months-long ML pipeline build, ~2.8 GB of training artifacts that must NOT ship, and metadata pointing at a placeholder GitHub org that doesn't exist.

**Critical distinctions you must internalize:**

1. **`grafyx/search/model/*.npz`** — runtime model weights (~11 MB). MUST ship in the wheel. Loaded at import time by the search engine. **KEEP and COMMIT these.**
2. **`ml/*.pt`, `ml/*_pairs.json`, `ml/all_symbols.json`, `ml/.repo_cache/`** — training artifacts (~2.8 GB). MUST NOT ship. Mostly already on disk but **gitignored** by this plan.
3. **`ml/*.py`** — training scripts. Kept in repo for reproducibility. They're how somebody re-trains the models in (1).

**Working directory:** `C:/Kişisel Projelerim/Grafyx` (Windows path; use forward slashes in bash). Current branch is `master`. Commands assume bash (Git Bash / WSL on Windows). The user's WSL Python venv is at `~/grafyx-venv` — used for tests, not for the cleanup commands which use git only.

**Each task ends with a commit unless explicitly noted.** The repo is messy enough that small commits are essential for review.

---

## Phase A: Local cleanup and commits

### Task 1: Verify starting state

**Files:** none (read-only checks).

- [ ] **Step 1: Confirm we're on master with the expected uncommitted changes**

Run:
```bash
cd "C:/Kişisel Projelerim/Grafyx"
git branch --show-current
git status --short | wc -l
git log --oneline -1
```

Expected:
- Branch: `master`
- Uncommitted file count: ~50 lines (mix of `M`, `D`, `??`)
- HEAD: `d94cdb2 docs: add public release design spec` (the spec we just committed in brainstorming)

- [ ] **Step 2: Confirm no large files are already tracked**

Run:
```bash
git ls-files | xargs -I{} stat -c "%s %n" "{}" 2>/dev/null | sort -rn | head -10
```

Expected: largest tracked file should be the M5 encoder weights (~9.5 MB) or smaller. If anything >50 MB shows, **STOP and report** — history rewrite required before public push.

- [ ] **Step 3: Confirm graph-sitter installed venv works**

Run:
```bash
ls "C:/Kişisel Projelerim/Grafyx/grafyx/search/model/" | wc -l
```

Expected: 11 files (the .npz weights and tokenizer JSONs from the spec). If fewer, the working tree is incomplete.

No commit. This is verification only.

---

### Task 2: First logical commit — remove obsolete pipelines

**Files (all deletions):**
- `grafyx/antigravity_proxy.py`
- `grafyx/search/_cross_encoder.py`
- `ml/code_search_encoder/dataset.py`
- `ml/code_search_encoder/download_codesearchnet.py`
- `ml/code_search_encoder/evaluate.py`
- `ml/code_search_encoder/generate_semantic_pairs.py`
- `ml/code_search_encoder/generate_synthetic.py`
- `ml/code_search_encoder/model.py`
- `ml/code_search_encoder/train.py`
- `ml/code_search_encoder/train_tokenizer.py`
- `ml/cross_encoder/generate_data.py`
- `ml/cross_encoder/model.py`
- `ml/cross_encoder/train.py`
- `tests/test_cross_encoder.py`
- `tests/test_module_deps.py`

- [ ] **Step 1: Stage only the deletions**

Run:
```bash
cd "C:/Kişisel Projelerim/Grafyx"
git add -u grafyx/antigravity_proxy.py
git add -u grafyx/search/_cross_encoder.py
git add -u ml/code_search_encoder/
git add -u ml/cross_encoder/
git add -u tests/test_cross_encoder.py
git add -u tests/test_module_deps.py
```

Note: `git add -u` with a path stages deletions. Do NOT use `git add -A` here — that would also pick up modifications and untracked files we want in different commits.

- [ ] **Step 2: Verify the staged set**

Run:
```bash
git diff --cached --name-status
```

Expected: 15 lines, all starting with `D ` (deletions only). If you see any `M` or anything else, run `git reset` and redo Step 1.

- [ ] **Step 3: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore: remove M6 cross-encoder and obsolete training pipelines

The M6 cross-encoder experiment was abandoned (random-level 59.8%
accuracy). The earlier code_search_encoder pipeline is superseded by
the M5 bi-encoder. Also drops antigravity_proxy.py (unused) and the
two now-stale test files.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify**

Run:
```bash
git log --oneline -1
git status --short | wc -l
```

Expected: HEAD is the new commit, ~35 lines remain in `git status` (50 minus 15).

---

### Task 3: Second logical commit — M1–M5 ML pipeline updates

**Files:** the remaining ~35 modified/new files spanning `grafyx/`, `ml/`, `tests/`. Specifically:

Modifications under `grafyx/`:
- `grafyx/__main__.py`
- `grafyx/conventions.py`
- `grafyx/graph/_analysis.py`
- `grafyx/graph/_callers.py`
- `grafyx/graph/_indexes.py`
- `grafyx/graph/_paths.py`
- `grafyx/graph/_query.py`
- `grafyx/graph/core.py`
- `grafyx/search/_code_encoder.py`
- `grafyx/search/_mamba.py`
- `grafyx/search/_relevance.py`
- `grafyx/search/_scoring.py`
- `grafyx/search/_source_filter.py`
- `grafyx/search/_source_index.py`
- `grafyx/search/_tokenizer.py`
- `grafyx/search/searcher.py`
- `grafyx/server/__init__.py`
- `grafyx/server/_resolution.py`
- `grafyx/server/_state.py`
- `grafyx/server/_tools_graph.py`
- `grafyx/server/_tools_introspection.py`
- `grafyx/server/_tools_quality.py`

Modifications under `ml/`:
- `ml/caller_disambiguator/features.py`
- `ml/data_common/extract_symbols.py`
- `ml/data_common/repos.json`

Modifications under `tests/`:
- `tests/test_caller_disambig_ml.py`
- `tests/test_circular_deps.py`
- `tests/test_code_encoder.py`
- `tests/test_mamba_inference.py`
- `tests/test_receiver_extraction.py`
- `tests/test_reexport_tracking.py`
- `tests/test_relevance_v2.py`
- `tests/test_search_enhanced.py`
- `tests/test_server.py`
- `tests/test_source_filter.py`
- `tests/test_subclasses.py`
- `tests/test_symbol_importance.py`
- `tests/test_tokenizer.py`

New files (untracked) — runtime + training:
- `grafyx/search/_gibberish.py`
- `grafyx/search/model/` (entire directory)
- `tests/test_error_handling.py`
- `tests/test_import_resolution.py`
- `tests/test_ml_models_e2e.py`
- `tests/test_search_gibberish.py`
- `ml/gibberish_detector/` (training scripts directory)
- `ml/evaluate_m5.py`
- `ml/extract_real_symbols.py`
- `ml/generate_claude_queries.py`
- `ml/generate_llm_queries.py`
- `ml/generate_synthetic_symbols.py`
- `ml/generate_training_data_m5.py`
- `ml/pre_encode_m5.py`
- `ml/retrain_all_real.py`
- `ml/retrain_m1.py`
- `ml/train_all.py`
- `ml/train_m5.py`
- `ml/train_m5m6_prototype.py`

Files explicitly EXCLUDED (must NOT be staged here):
- `ml/.claude_batch_state.json`
- `ml/.repo_cache/`
- `ml/all_symbols.json`
- `ml/claude_training_pairs.json`
- `ml/llm_training_pairs.json`
- `ml/m5_*.json`
- `ml/m5_*.pt`
- `docs/plans/` (all 15 files — to be removed in Task 5)
- `.cursor/`
- `grafyx_start.sh` (developer-specific shell helper)

- [ ] **Step 1: Stage modifications under grafyx/**

Run:
```bash
git add grafyx/__main__.py grafyx/conventions.py
git add grafyx/graph/
git add grafyx/search/_code_encoder.py grafyx/search/_mamba.py grafyx/search/_relevance.py grafyx/search/_scoring.py grafyx/search/_source_filter.py grafyx/search/_source_index.py grafyx/search/_tokenizer.py grafyx/search/searcher.py grafyx/search/_gibberish.py
git add grafyx/search/model/
git add grafyx/server/
```

- [ ] **Step 2: Stage ml/ modifications and new training scripts (NOT data)**

Run:
```bash
git add ml/caller_disambiguator/features.py
git add ml/data_common/extract_symbols.py ml/data_common/repos.json
git add ml/gibberish_detector/
git add ml/evaluate_m5.py ml/extract_real_symbols.py ml/generate_claude_queries.py ml/generate_llm_queries.py ml/generate_synthetic_symbols.py ml/generate_training_data_m5.py ml/pre_encode_m5.py ml/retrain_all_real.py ml/retrain_m1.py ml/train_all.py ml/train_m5.py ml/train_m5m6_prototype.py
```

- [ ] **Step 3: Stage tests/ modifications and new tests**

Run:
```bash
git add tests/test_caller_disambig_ml.py tests/test_circular_deps.py tests/test_code_encoder.py tests/test_mamba_inference.py tests/test_receiver_extraction.py tests/test_reexport_tracking.py tests/test_relevance_v2.py tests/test_search_enhanced.py tests/test_server.py tests/test_source_filter.py tests/test_subclasses.py tests/test_symbol_importance.py tests/test_tokenizer.py
git add tests/test_error_handling.py tests/test_import_resolution.py tests/test_ml_models_e2e.py tests/test_search_gibberish.py
```

- [ ] **Step 4: Sanity-check what's staged vs not**

Run:
```bash
git diff --cached --name-only | wc -l
git status --short
```

Expected staged file count: ~52 files (22 grafyx + 11 model weights + 16 ml + 17 tests = ~66, but many are dir-level adds that expand). Verify NOTHING from this list appears in the staged set:

```bash
git diff --cached --name-only | grep -E '(\.cursor|docs/plans|all_symbols|claude_training_pairs|llm_training_pairs|m5_.*\.json|m5_.*\.pt|\.repo_cache|grafyx_start\.sh)'
```

Expected: empty output. If anything matches, run `git reset HEAD <file>` for each match.

- [ ] **Step 5: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat: ML search pipeline updates (M1-M5)

Integrates the trained ML stack into search and graph queries:
- M1 relevance ranker (33-feature MLP) replaces heuristic scoring
- M2 caller disambiguator (25 features) refines call resolution
- M3 source token filter (15 features) suppresses noise tokens
- M4 symbol importance (18 features) weights search results
- M5 bi-encoder (FeedForwardBlock, 1.2M params) for semantic search
- Gibberish detector (char-bigram MLP) blocks nonsense queries

Adds graph-boosted scoring, TS/JS object literal method detection,
init.py re-export resolution, receiver-token + dot-syntax flags on
caller entries, and removes hardcoded truncation caps from server
tools.

Pure-numpy inference via FeedForwardBlock in _mamba.py. Model
weights (.npz) ship with the wheel; PyTorch is a training-only
dependency.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Verify**

Run:
```bash
git log --oneline -3
git status --short | head -20
```

Expected: HEAD is the new commit. `git status` should now show only the items that go in the third commit: `docs/plans/`, `.cursor/`, scratch files, `ml/` data files, and `grafyx_start.sh`.

---

### Task 4: Add the `.gitattributes` file

**Files:**
- Create: `.gitattributes`

- [ ] **Step 1: Create the file**

Write `C:/Kişisel Projelerim/Grafyx/.gitattributes` with content:

```gitattributes
* text=auto eol=lf
*.py text eol=lf
*.md text eol=lf
*.json text eol=lf
*.toml text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
*.txt text eol=lf
*.sh text eol=lf
*.gitignore text eol=lf
*.gitattributes text eol=lf
*.npz binary
*.pt binary
*.bin binary
*.ckpt binary
```

- [ ] **Step 2: Verify**

Run:
```bash
test -f .gitattributes && echo "exists" || echo "missing"
wc -l .gitattributes
```

Expected: `exists`, 15 lines.

No commit yet — bundled into Task 9's "release prep" commit.

---

### Task 5: Update `.gitignore`

**Files:**
- Modify: `.gitignore` — append a project-specific block.

- [ ] **Step 1: Append to .gitignore**

Open `C:/Kişisel Projelerim/Grafyx/.gitignore` and append (preserve existing 33 lines, add at the end):

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

# Developer-specific shell helpers
grafyx_start.sh
```

- [ ] **Step 2: Verify ignore actually applies**

Run:
```bash
git check-ignore -v ml/all_symbols.json ml/.repo_cache/foo .cursor/anything ml/m5_train.pt grafyx_start.sh
```

Expected: every path printed with the `.gitignore:LINE` rule that matches it. If any path returns nothing or exit code 1, the rule didn't take.

- [ ] **Step 3: Re-confirm `git status` no longer lists the now-ignored items**

Run:
```bash
git status --short | grep -E '(\.cursor|all_symbols|claude_training_pairs|llm_training_pairs|m5_|\.repo_cache|grafyx_start\.sh)'
```

Expected: empty output.

No commit yet — bundled into Task 9.

---

### Task 6: Delete root scratch files and `.cursor/`

**Files (deletions on disk):**
- `_create_resolution.py`
- `_gen_test.py`
- `_test_b64.txt`
- `_write_test.py`
- `.cursor/` (directory)

These files are gitignored and untracked, so we delete them from disk only. No git operation here.

- [ ] **Step 1: Delete**

Run:
```bash
cd "C:/Kişisel Projelerim/Grafyx"
rm -f _create_resolution.py _gen_test.py _test_b64.txt _write_test.py
rm -rf .cursor/
```

- [ ] **Step 2: Verify**

Run:
```bash
ls _create_resolution.py _gen_test.py _test_b64.txt _write_test.py 2>&1
ls -d .cursor 2>&1
```

Expected: each `ls` reports "No such file or directory". If anything still exists, retry the delete.

No commit. Untracked files leaving disk is invisible to git.

---

### Task 7: Delete `docs/plans/` (keep test prompt)

**Files (deletions):**
- All 15 files under `docs/plans/`

`docs/plans/2026-02-13-new-tools-design.md` and `2026-02-13-new-tools-implementation.md` and `2026-02-25-fractal-context-design.md` and `2026-02-25-fractal-context-implementation.md` and `2026-02-25-test-report-fixes.md` and `2026-02-26-test-report-fixes.md` and `2026-02-26-test-report-fixes-design.md` and `2026-03-07-ml-accuracy-boost.md` and `2026-03-07-ml-accuracy-boost-design.md` were tracked in git. The other 7 are untracked (came in with the recent ML work).

- [ ] **Step 1: Delete the entire directory**

Run:
```bash
cd "C:/Kişisel Projelerim/Grafyx"
git rm -rf docs/plans/ 2>/dev/null || true
rm -rf docs/plans/
```

(`git rm` removes tracked files; the second `rm -rf` removes any untracked stragglers.)

- [ ] **Step 2: Verify**

Run:
```bash
ls -la docs/
```

Expected: only `grafyx-accuracy-test-prompt.md` and the `superpowers/` subdirectory remain in `docs/`.

- [ ] **Step 3: Confirm `git status` reflects the staged deletes**

Run:
```bash
git status --short | grep '^D ' | grep docs/plans
```

Expected: 9 lines (the previously-tracked files staged as deletions). Untracked files don't show up because they weren't tracked.

No commit yet — bundled into Task 9.

---

### Task 8: Update `LICENSE`, `pyproject.toml`, `README.md`

**Files:**
- Modify: `LICENSE`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Update LICENSE copyright line**

Edit `C:/Kişisel Projelerim/Grafyx/LICENSE`:

Old:
```
Copyright (c) 2025 Grafyx Contributors
```

New:
```
Copyright (c) 2026 Bilal Karadeniz
```

- [ ] **Step 2: Replace pyproject.toml [project] table**

Open `C:/Kişisel Projelerim/Grafyx/pyproject.toml`. Replace the entire `[project]` block (lines 5–26 today) with:

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
```

Leave `[project.optional-dependencies]`, `[project.scripts]`, `[tool.hatch.build.targets.wheel]`, and `[build-system]` blocks unchanged. Replace the `[project.urls]` block at the end with:

```toml
[project.urls]
Homepage = "https://github.com/bilal07karadeniz/Grafyx"
Repository = "https://github.com/bilal07karadeniz/Grafyx"
Issues = "https://github.com/bilal07karadeniz/Grafyx/issues"
```

- [ ] **Step 3: Verify pyproject is valid TOML**

Run:
```bash
python -c "import tomllib; tomllib.loads(open('pyproject.toml','rb').read().decode())"
```

Expected: no output, exit code 0. (On Python <3.11 use `tomli` instead.)

- [ ] **Step 4: Update README.md URLs**

Open `C:/Kişisel Projelerim/Grafyx/README.md`. Two find/replace operations:

a) Replace `grafyx-ai/grafyx` with `bilal07karadeniz/Grafyx` (one occurrence in the `git clone` line, possibly more — replace all).

b) The README currently says `FastMCP server with 10 tools` (line 89) and the tools table only lists 10 entries. The actual tool count is 14 (verified by grepping `@mcp.tool` decorations across `grafyx/server/_tools_*.py`). Update line 89 to:

```
|  Grafyx   |  FastMCP server with 14 tools
```

And insert these 4 rows into the `## Available Tools` table immediately after the `refresh_graph` row (before the closing `---`):

```markdown
| `get_module_context` | Symbols in a directory/package (intermediate zoom) |
| `get_subclasses` | Inheritance tree for a base class |
| `get_unused_symbols` | Dead code detection |
| `set_project` | Switch the served project at runtime |
```

c) Add a new H3 subsection titled `### ML-augmented search` immediately after the `## How It Works` section's diagram, with this body:

```markdown
### ML-augmented search

Grafyx ships several small numpy-only MLPs trained on real source data:

- **M1 Relevance ranker** — 33-feature MLP scores each search result against the query.
- **M3 Source token filter** — suppresses noise tokens (imports, strings, magic methods) from full-text search.
- **M4 Symbol importance** — weights symbols by caller count, exports, and structural signals.
- **M5 Bi-encoder** — semantic embedding model (BPE tokenizer, FeedForward encoder) for natural-language code search.
- **Gibberish detector** — character-bigram MLP that blocks nonsense queries before they hit the index.

All weights ship inside the wheel (~11 MB total). Inference is pure numpy, no PyTorch at runtime.
```

d) Replace the badge block at the top of the README with this set:

```markdown
[![PyPI](https://img.shields.io/pypi/v/grafyx-mcp.svg)](https://pypi.org/project/grafyx-mcp/)
[![CI](https://github.com/bilal07karadeniz/Grafyx/actions/workflows/ci.yml/badge.svg)](https://github.com/bilal07karadeniz/Grafyx/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/protocol-MCP-green.svg)](https://modelcontextprotocol.io)
```

- [ ] **Step 5: Verify**

Run:
```bash
grep -n 'grafyx-ai/grafyx\|10 tools\|2025 Grafyx Contributors' README.md LICENSE pyproject.toml
```

Expected: empty output. If anything matches, fix the remaining occurrences.

No commit yet — bundled into Task 9.

---

### Task 9: Add community files (`CONTRIBUTING.md`, `CHANGELOG.md`, `.github/`)

**Files (all new):**
- Create: `CONTRIBUTING.md`
- Create: `CHANGELOG.md`
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`
- Create: `.github/ISSUE_TEMPLATE/feature_request.md`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: CONTRIBUTING.md**

Write `C:/Kişisel Projelerim/Grafyx/CONTRIBUTING.md` with:

```markdown
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

The full suite takes about 30–60 seconds. ML model end-to-end tests require the weight files in `grafyx/search/model/` to be present (they ship with the repo).

## Re-training ML models

The trained weights are committed under `grafyx/search/model/`. To retrain:

```bash
# 1. Generate training data (downloads source from popular repos)
python ml/extract_real_symbols.py
python ml/generate_claude_queries.py

# 2. Train the M1-M4 feature MLPs
python ml/retrain_all_real.py

# 3. Train the M5 bi-encoder
python ml/train_m5.py
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
- The CI workflow runs on Python 3.12 and 3.13 — both must pass.
- Reference any related issue in the PR description.
```

- [ ] **Step 2: CHANGELOG.md**

Write `C:/Kişisel Projelerim/Grafyx/CHANGELOG.md` with:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-29

### Added

- Initial public release.
- 14 MCP tools: `get_project_skeleton`, `get_module_context`, `get_function_context`, `get_class_context`, `get_file_context`, `find_related_code`, `find_related_files`, `get_dependency_graph`, `get_call_graph`, `get_subclasses`, `get_unused_symbols`, `get_conventions`, `set_project`, `refresh_graph`.
- ML-augmented search:
  - M1 relevance ranker (33-feature MLP).
  - M3 source token filter (15-feature MLP).
  - M4 symbol importance (18-feature MLP).
  - M5 bi-encoder for semantic search (BPE tokenizer + FeedForward encoder).
  - Character-bigram gibberish detector.
- Caller disambiguation (M2) over multi-language code (Python / TypeScript / JavaScript).
- File watcher (watchdog) keeps the graph current as files change.
- Fractal-context resolution control (`signatures` / `summary` / `full`) and navigation hints on five exploration tools.
- Convention detection.
- Object literal method detection in TypeScript / JavaScript.
- Init.py re-export resolution in the import index.

[0.1.0]: https://github.com/bilal07karadeniz/Grafyx/releases/tag/v0.1.0
```

- [ ] **Step 3: Bug report template**

Write `C:/Kişisel Projelerim/Grafyx/.github/ISSUE_TEMPLATE/bug_report.md` with:

```markdown
---
name: Bug report
about: Something isn't working
labels: bug
---

## Description

<!-- One or two sentences. -->

## Reproduction

1.
2.
3.

## Expected behavior

## Actual behavior

## Environment

- Grafyx version: <!-- output of `grafyx --version` -->
- Python version:
- OS: <!-- e.g., Ubuntu 24.04, WSL on Windows 11 -->
- MCP client: <!-- Claude Code / Cursor / Windsurf / Cline / VS Code Copilot -->

## Logs

<!-- Run with `grafyx --verbose` and paste the output. Redact anything sensitive. -->

```

- [ ] **Step 4: Feature request template**

Write `C:/Kişisel Projelerim/Grafyx/.github/ISSUE_TEMPLATE/feature_request.md` with:

```markdown
---
name: Feature request
about: Suggest a new tool, capability, or improvement
labels: enhancement
---

## Motivation

<!-- What problem are you trying to solve? -->

## Proposed behavior

<!-- What would you want Grafyx to do? -->

## Alternatives considered

<!-- Other approaches you thought about. -->

## Additional context
```

- [ ] **Step 5: PR template**

Write `C:/Kişisel Projelerim/Grafyx/.github/PULL_REQUEST_TEMPLATE.md` with:

```markdown
## Summary

<!-- 1-3 bullet points describing what this PR changes. -->

## Related issue

<!-- Closes #N, or N/A. -->

## Test plan

- [ ]
- [ ]

## Checklist

- [ ] Tests added or updated for the new behavior
- [ ] `pytest` passes locally
- [ ] Conventional Commit message
```

- [ ] **Step 6: CI workflow**

Write `C:/Kişisel Projelerim/Grafyx/.github/workflows/ci.yml` with:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.12", "3.13"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip

      - name: Install
        run: pip install -e ".[dev]"

      - name: Test
        run: pytest -q
```

- [ ] **Step 7: Release workflow**

Write `C:/Kişisel Projelerim/Grafyx/.github/workflows/release.yml` with:

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Build
        run: |
          pip install build
          python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          skip-existing: true
```

- [ ] **Step 8: Verify YAML parses**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml'))"
```

Expected: no output, exit 0. (If `yaml` not installed: `pip install pyyaml` first.)

- [ ] **Step 9: Stage and commit the bundle**

This is the third logical commit. It includes Task 4 (`.gitattributes`), Task 5 (`.gitignore`), Task 7 (`docs/plans` deletions), Task 8 (LICENSE / pyproject / README), and Task 9 (community files).

Run:
```bash
git add .gitattributes .gitignore LICENSE pyproject.toml README.md
git add CONTRIBUTING.md CHANGELOG.md
git add .github/
# docs/plans deletions were already staged in Task 7 step 1 via `git rm`
git status --short
```

Verify staged set contains exactly: `.gitattributes`, `.gitignore`, `LICENSE`, `pyproject.toml`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `.github/...` (5 files), and 9 deletions under `docs/plans/`.

Run:
```bash
git commit -m "$(cat <<'EOF'
chore: prepare repo for public release

- Switch copyright/identity to Bilal Karadeniz (2026).
- Tighten Python range to >=3.12,<3.14 to match graph-sitter.
- Bump graph-sitter floor to >=0.56 (real shipped version).
- Repoint all GitHub URLs at bilal07karadeniz/Grafyx.
- Update README: 15 tools (was 10), badges, ML-augmented search section.
- Add .gitattributes to normalize line endings on Windows.
- Add .gitignore rules for ML training artifacts, .cursor, scratch files.
- Drop internal docs/plans/ working notes; keep accuracy test prompt.
- Add CONTRIBUTING, CHANGELOG, issue/PR templates.
- Add CI (Python 3.12/3.13 on ubuntu) and Release (PyPI Trusted Publishing) workflows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 10: Verify clean tree**

Run:
```bash
git status
git log --oneline -5
```

Expected: working tree clean. Last 4 commits are: `chore: prepare repo for public release`, `feat: ML search pipeline updates (M1-M5)`, `chore: remove M6 cross-encoder...`, `docs: add public release design spec`.

---

### Task 10: Rename master → main and validate the build locally

**Files:** none. Local git operation + a fresh build.

- [ ] **Step 1: Rename branch**

Run:
```bash
cd "C:/Kişisel Projelerim/Grafyx"
git branch -m master main
git branch --show-current
```

Expected: `main`.

- [ ] **Step 2: Build the wheel locally**

Run (in WSL with the venv active, since graph-sitter needs Linux):
```bash
source ~/grafyx-venv/bin/activate
cd "/mnt/c/Kişisel Projelerim/Grafyx"
pip install build
python -m build
```

Expected: `dist/grafyx_mcp-0.1.0-py3-none-any.whl` and `dist/grafyx_mcp-0.1.0.tar.gz` are produced. No errors.

- [ ] **Step 3: Inspect wheel contents**

Run:
```bash
unzip -l dist/grafyx_mcp-0.1.0-py3-none-any.whl | grep -E '\.npz|\.json|\.py' | head -40
```

Expected: the listing must include all 11 files in `grafyx/search/model/` (8 `.npz` + 3 `.json`). If any model file is missing, hatch is filtering them — add an explicit `[tool.hatch.build]` includes block to `pyproject.toml`:

```toml
[tool.hatch.build]
include = [
  "grafyx/**/*.py",
  "grafyx/search/model/*.npz",
  "grafyx/search/model/*.json",
]
```

Then rerun build and re-inspect.

- [ ] **Step 4: Smoke-install in a fresh venv**

Run:
```bash
python -m venv /tmp/grafyx-smoke
source /tmp/grafyx-smoke/bin/activate
pip install dist/grafyx_mcp-0.1.0-py3-none-any.whl
grafyx --version
deactivate
rm -rf /tmp/grafyx-smoke dist/
```

Expected: `grafyx 0.1.0` printed by `--version`. The smoke venv and `dist/` are removed afterward (we re-build inside Actions for the real publish).

If this fails: do NOT proceed to push. Diagnose why graph-sitter or fastmcp didn't pull, fix, and rerun.

- [ ] **Step 5: Run the existing test suite**

Back in your dev venv:
```bash
source ~/grafyx-venv/bin/activate
cd "/mnt/c/Kişisel Projelerim/Grafyx"
pytest -q
```

Expected: green. If anything fails, fix before pushing — these tests will run in CI on push and a red CI badge on day 1 is bad.

No commit. The local validation is purely a gate.

---

## Phase B: Push to GitHub

### Task 11: Add remote and push main

**Files:** none. Network operation.

- [ ] **Step 1: Confirm the GitHub repo is empty**

Visit `https://github.com/bilal07karadeniz/Grafyx` in a browser. Expected: an empty repository with the standard "Quick setup" instructions visible. If the repo already has commits (e.g., a default README from GitHub's UI), you have two choices:

- a) Delete the GitHub repo and re-create it empty (recommended if it has only an auto-generated README), or
- b) Pull and rebase: `git pull origin main --rebase --allow-unrelated-histories` then resolve.

Stop and ask before choosing if (a) feels wrong.

- [ ] **Step 2: Add the remote**

Pick HTTPS or SSH based on your GitHub auth setup. Default to HTTPS:

```bash
cd "C:/Kişisel Projelerim/Grafyx"
git remote add origin https://github.com/bilal07karadeniz/Grafyx.git
git remote -v
```

Expected: two lines (fetch + push) pointing at the URL.

- [ ] **Step 3: Push main and set upstream**

```bash
git push -u origin main
```

Expected: all four new commits pushed. If GitHub authentication prompts: use a personal access token (settings → developer settings → tokens), not your password.

- [ ] **Step 4: Verify on GitHub**

Visit `https://github.com/bilal07karadeniz/Grafyx`. Expected:

- Default branch is `main`.
- README renders (badges may show "no PyPI yet" / CI pending — that's fine).
- LICENSE auto-detected as MIT.
- `Releases`, `Issues`, `Actions` tabs present.

No commit. Network state changed.

---

### Task 12: Wait for first CI run and triage

**Files:** none. Watching CI.

- [ ] **Step 1: Open the Actions tab**

Visit `https://github.com/bilal07karadeniz/Grafyx/actions`. The CI workflow should be running on the push.

- [ ] **Step 2: If CI is green**

Proceed to Task 13.

- [ ] **Step 3: If CI is red**

Common causes and fixes:

- Missing `pytest` / `[dev]` extra: re-check `pyproject.toml` `[project.optional-dependencies]`. Should declare `dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]`.
- Tests that depended on local file paths from `C:\Kişisel Projelerim\Grafyx`: convert to `tmp_path` fixtures.
- `graph-sitter` install failure on the runner: check Python version compat. Pin `graph-sitter==<version>` if floor was wrong.
- Tests touching network or filesystem caches that don't exist on CI: skip or fixture them.

Fix locally, run `pytest -q`, then:

```bash
git add <changed files>
git commit -m "fix: <concise summary>"
git push
```

Repeat until green. Do NOT proceed to release with a red CI badge.

No additional commit beyond the fix commits.

---

## Phase C: First release on PyPI

### Task 13: Configure PyPI Trusted Publishing

**Files:** none. PyPI + GitHub web UI.

This task requires the human user's hands. The agent should pause and prompt them.

- [ ] **Step 1: Prompt the user to set up Trusted Publishing**

Print this exact instruction set to the user:

```
1. Visit https://pypi.org/manage/account/publishing/ (sign in if prompted).
2. Click "Add a new pending publisher".
3. Fill in EXACTLY:
   - PyPI Project Name: grafyx-mcp
   - Owner: bilal07karadeniz
   - Repository name: Grafyx
   - Workflow name: release.yml
   - Environment name: pypi
4. Click "Add".

Then create the GitHub Environment:
5. Visit https://github.com/bilal07karadeniz/Grafyx/settings/environments
6. Click "New environment", name it `pypi`, click "Configure environment".
7. No further config required. Save.

Reply "done" when both are complete.
```

- [ ] **Step 2: Wait for user confirmation**

Do not proceed until the user replies that both steps are complete.

- [ ] **Step 3: Fallback path (only if Step 1 fails)**

If Trusted Publishing setup fails for any reason, the user can instead:

```
1. Visit https://pypi.org/manage/account/token/
2. Create a token scoped to project "grafyx-mcp" (or "Entire account" if the project doesn't exist yet).
3. Copy the token starting with `pypi-`.
4. Visit https://github.com/bilal07karadeniz/Grafyx/settings/secrets/actions
5. Add a new repository secret named `PYPI_API_TOKEN` with the token value.
```

Then patch `release.yml` step "Publish to PyPI" to use:

```yaml
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
          skip-existing: true
```

Commit this change with message `chore: switch release to API token (Trusted Publishing unavailable)` and push.

No commit unless the fallback was needed.

---

### Task 14: Tag v0.1.0 and trigger the release

**Files:** none. Git tagging + CI watching.

- [ ] **Step 1: Tag locally**

```bash
cd "C:/Kişisel Projelerim/Grafyx"
git tag -a v0.1.0 -m "Grafyx 0.1.0"
git tag -l
```

Expected: `v0.1.0` in the list.

- [ ] **Step 2: Push the tag**

```bash
git push origin v0.1.0
```

Expected: GitHub responds with the tag pushed. Within a few seconds the Release workflow starts.

- [ ] **Step 3: Watch the release workflow**

Visit `https://github.com/bilal07karadeniz/Grafyx/actions/workflows/release.yml`. The run should:

1. Check out the tag.
2. Build wheel + sdist.
3. Upload to PyPI via Trusted Publishing.

Expected: green run within ~2 minutes.

- [ ] **Step 4: If the release fails**

Common causes:

- Trusted Publishing handshake fails (`OIDC token not accepted`): re-check the four fields in PyPI's pending publisher (Step 1 of Task 13). Owner is case-sensitive.
- Wheel name mismatch: project name must be `grafyx-mcp` in `pyproject.toml`, not `grafyx`.
- 403 from PyPI: project name might already be taken. If so, see Task 15 fallback.

Fix and re-run the workflow (Actions → Re-run all jobs). Tags don't need to be re-pushed.

- [ ] **Step 5: Verify on PyPI**

Visit `https://pypi.org/project/grafyx-mcp/`. Expected: the 0.1.0 release is live.

No commit.

---

### Task 15: Verify install works end-to-end

**Files:** none. Smoke test against live PyPI.

- [ ] **Step 1: Fresh install in a clean venv (Linux/WSL)**

```bash
python -m venv /tmp/grafyx-public
source /tmp/grafyx-public/bin/activate
pip install grafyx-mcp
grafyx --version
grafyx --help
deactivate
rm -rf /tmp/grafyx-public
```

Expected: `pip install` succeeds (pulls graph-sitter, fastmcp, watchdog as deps). `grafyx --version` prints `grafyx 0.1.0`. `--help` prints the CLI usage text.

- [ ] **Step 2: Try the uvx flow**

```bash
uvx --from grafyx-mcp grafyx --version
```

Expected: `grafyx 0.1.0`. (If `uvx` not installed: `pip install uv`.)

- [ ] **Step 3: Final state check**

Run:
```bash
cd "C:/Kişisel Projelerim/Grafyx"
git status
git log --oneline -6
git tag -l
```

Expected:
- Working tree clean.
- 4 commits on `main` since the spec, plus the spec commit.
- Tag `v0.1.0` present.

- [ ] **Step 4: Mark the release on GitHub**

Visit `https://github.com/bilal07karadeniz/Grafyx/releases/new`. Choose tag `v0.1.0`. Title: `Grafyx 0.1.0`. Body: paste the `[0.1.0]` section from `CHANGELOG.md`. Publish release.

This makes the release discoverable via the GitHub Releases tab.

No commit.

---

## Verification (final)

After Task 15, all of these must hold:

- [ ] `git status` is clean on `main`.
- [ ] `git log --oneline | head -5` shows: prep-public → ML-pipeline → cross-encoder-removal → spec-doc → previous public history.
- [ ] `git ls-files | grep -E "(all_symbols|claude_training_pairs|m5_.*\.json|m5_.*\.pt|\.repo_cache|\.cursor|_create_resolution|_gen_test|_test_b64|_write_test|docs/plans/)"` returns empty.
- [ ] Repo size on GitHub ≤ 50 MB.
- [ ] CI badge in README is green.
- [ ] PyPI badge in README displays `0.1.0`.
- [ ] `pip install grafyx-mcp` works on Linux from a fresh venv.
- [ ] `grafyx --version` prints `grafyx 0.1.0`.
- [ ] `pip show grafyx-mcp` shows `Author: Bilal Karadeniz`, `Author-email: bilal07karadeniz@gmail.com`.
- [ ] GitHub release `v0.1.0` is published.
