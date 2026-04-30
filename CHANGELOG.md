# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-30

### Added

- Encoder registry in `grafyx/search/_embeddings.py`. Switch via the
  `GRAFYX_ENCODER` env var. Choices: `jina-v2` (default, Apache-2.0,
  fastembed-native) and `coderankembed` (MIT, 137M, registered via
  `fastembed.add_custom_model` from a custom HF repo).
- Published head-to-head encoder benchmark in
  [`docs/benchmarks/0.2.0/`](docs/benchmarks/0.2.0/) — 278 queries
  across FastAPI + Django, three encoders (tokens-only baseline,
  jina-v2, coderankembed). jina-v2 wins by 12.4 nDCG@10 points
  (0.787 vs 0.663) and is +135% over the tokens-only baseline (0.335).
  Decision rationale, raw JSON, and per-query JSONL all committed.
- New `--encoder tokens` mode in `bench_search.py` lets the harness
  measure source-token search alone (the fallback when fastembed is
  not installed).
- Hosted ONNX-int8 mirror of CodeRankEmbed at
  [`Bilal7Dev/grafyx-coderankembed-onnx`](https://huggingface.co/Bilal7Dev/grafyx-coderankembed-onnx)
  so users opting in via `GRAFYX_ENCODER=coderankembed` get a
  version-pinned artifact (re-host of `mrsladoje/CodeRankEmbed-onnx-int8`,
  attribution in the model card).
- `find_related_code` response now includes `model` and `latency_ms`
  metadata, plus a `degraded` flag and `action_hint` when the encoder is
  unavailable (missing fastembed, build pending, or download failed).
- `CodeSearcher.wait_for_index_ready(timeout)` blocks until the embedding
  index has finished building, used by the benchmark harness.
- Reproducible benchmark harness under `benchmarks/`. Single command:
  `python -m scripts.run_all`. Pinned at FastAPI @ 4f64b8f6, Django @
  02a7d43d, Home Assistant @ 3ed0d8a1. Eval pairs extracted per repo
  from public-function docstrings.
- New optional install extra `bench` for benchmark dependencies.

### Changed

- `find_related_code` semantic retrieval now goes through the
  fastembed-backed `EmbeddingSearcher` exclusively. The M5 Mamba
  bi-encoder hookup has been removed from the search path.
- `EmbeddingSearcher.__init__` consumes the registry (`model_id` arg)
  instead of a hard-coded `model_name`. Caches per-encoder so switching
  does not invalidate the other's vectors.
- `embeddings` extra now pins `fastembed>=0.7.0,<0.10.0` to protect
  against breaking changes in `fastembed.add_custom_model`.

### Removed

- M5 from-scratch bi-encoder retired entirely. Deleted modules:
  `grafyx/search/_code_encoder.py`, `grafyx/search/_tokenizer.py`,
  `grafyx/search/_mamba.py`. Deleted weights:
  `grafyx/search/model/code_encoder_weights.npz`,
  `grafyx/search/model/bpe_merges.json`,
  `grafyx/search/model/bpe_vocab.json`. Deleted training scripts:
  `ml/train_m5.py`, `ml/train_m5m6_prototype.py`, `ml/evaluate_m5.py`,
  `ml/pre_encode_m5.py`, `ml/generate_training_data_m5.py`. Deleted
  tests: `tests/test_code_encoder.py`, `tests/test_tokenizer.py`,
  `tests/test_mamba_inference.py`. Net wheel size reduction: ~5 MB.

## [0.1.1] - 2026-04-29

### Added

- MCP server instructions injected into every connected client's system prompt. Documents when to use vs. avoid Grafyx, known limitations (TypeScript object literals, Celery dynamic dispatch, ML ranker precision), and the recommended pre-edit workflow. Surfaced automatically by all MCP clients (Claude Code, Cursor, Windsurf, Cline, VS Code Copilot) — no per-user setup.

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

[0.2.0]: https://github.com/bilal07karadeniz/Grafyx/releases/tag/v0.2.0
[0.1.1]: https://github.com/bilal07karadeniz/Grafyx/releases/tag/v0.1.1
[0.1.0]: https://github.com/bilal07karadeniz/Grafyx/releases/tag/v0.1.0
