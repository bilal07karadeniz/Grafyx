# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/bilal07karadeniz/Grafyx/releases/tag/v0.1.1
[0.1.0]: https://github.com/bilal07karadeniz/Grafyx/releases/tag/v0.1.0
