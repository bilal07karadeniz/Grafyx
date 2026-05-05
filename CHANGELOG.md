# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-05-05

A large accuracy-focused release covering everything since v0.2.1. Driven
by repeated multi-repo audits against the user's own project plus
FastAPI / Django / Home Assistant. Concrete numbers where measurable:
audit score on the user project went 4.0 → 4.33 / 5; FastAPI 3.7 → 4.36
/ 5 in spot validation.

### Fixed — Import resolution

- **Python relative imports (`from .module import X`) are now tracked.**
  Previously, `_extract_module_from_import` returned `""` for any import
  starting with `"from ."`, so `_build_import_index` silently skipped
  them. A new fast path in Phase 2 calls
  `_resolve_python_relative_import`, which navigates from the importer's
  directory by the number of leading dots (one dot = same package, two
  dots = parent) and resolves via the already-built `suffix_to_path`
  table.  Removes a whole class of name-collision false positives: a
  caller of `from .database import get_db` no longer matches every other
  `get_db` in the project.
- **Submodule imports (`from pkg import submod`) tracked.** Previously
  resolved only to `pkg/__init__.py`, never to `pkg/submod.py` itself,
  so `get_file_context` on the submodule returned an empty `imported_by`.
  A new pass `_augment_with_submodule_imports` walks
  `_file_symbol_imports` after main resolution; for every package import
  where the imported symbol matches a sibling `.py` or `__init__.py` in
  the package directory, the submodule file is added to all three
  indexes.
- **Intra-package imports no longer skipped when the project's own name
  appears in `_external_packages`.** The FastAPI repo declares
  `name = "fastapi"` in `pyproject.toml`, which the regex extractor in
  `_build_external_packages` adds to the external-packages set — so
  Phase 2 was skipping every `from fastapi import …` statement,
  *including ones from inside the* `fastapi/` *package itself*. The skip
  now bails early only when the import is bare (`import X`) OR when no
  local `X/<symbol>.py` file exists. External packages with no local
  shadow are still skipped (verified by `test_circular_deps.py`).
- **`__getattr__` lazy loaders in `__init__.py` are now resolved beyond
  the dict-with-leading-dot form.** v0.2.0 only matched
  `{"X": ".impl"}`; this release also catches inline `from .impl import
  X` bodies, dict entries without a leading dot (`{"X": "impl"}`), and
  `importlib.import_module(f".{name}", __package__)` patterns. When any
  `__getattr__` is defined, a fallback heuristic registers same-named
  submodules of the package as resolvable.

### Fixed — Caller / dependency tracking

- **`get_call_graph` no longer collapses method calls onto same-named
  top-level functions.** Previously, `db.refresh(user)` (SQLAlchemy)
  matched the `refresh` API route handler and the entire token-rotation
  call chain was inserted as if `register_user` invoked it. The tree
  builder now classifies each call site as `dotted` / `bare` / `mixed`
  / `unknown` from the parent function's source, and refuses to recurse
  when call style and candidate set disagree.
- **`get_dependency_graph.depended_on_by` was systematically empty for
  function and method symbols.** Source 3 (caller index) was gated on
  `kind == "class"`, so a function called via a singleton instance —
  e.g. `auth_service.authenticate_user(...)` — had its caller invisibly
  dropped. A new Source 4 queries the reverse caller index directly for
  non-class symbols, passing `class_name=` for methods so same-named
  methods on other classes don't pollute the result. Rated 2/5
  ("actively misleading") in the v0.2.1 audit.
- **`get_class_context.cross_file_usages` now follows factory return
  types** (Strategy 4). Classes consumed exclusively through a factory
  (`coord = await get_coordinator()` rather than direct
  `WorkerCoordinator(...)`) had their entire usage set hidden. Strategy
  4 reuses `_factory_return_types` to look up factories whose return
  type is the target class, then surfaces callers of those factories.
- **`get_class_context` hints now match `get_function_context` caller
  counts.** Hints previously read the raw `_caller_index`, while
  `get_function_context` applies class-name disambiguation. Hints now
  call `get_callers(name, class_name=cls_name)` themselves.
- **Cross-language callers require dot syntax.** A bare `login()` in
  TypeScript is by definition calling the TypeScript `login` — not the
  Python route. Bare cross-language callers are dropped; `await
  api.login(...)` still surfaces.
- **Class-attribute factory pattern resolved** by
  `_augment_index_with_class_attr_types`. Extends the v0.2.1 local-var
  fix to `self.coord = await get_coordinator()`,
  `_coord: WorkerCoordinator = None` class-body declarations, and
  shares the factory map with Pass 3 to avoid recomputation.

### Fixed — `get_subclasses` disambiguation

- **`get_subclasses` now disambiguates same-name classes via
  `file_path=`.** FastAPI has two classes named `SecurityBase`
  (`security/base.py`, the abstract base; `openapi/models.py`, a
  Pydantic model). The previous traversal merged both subtrees,
  inflating the count. The tool now: (1) detects ambiguity and reports
  a `candidates` list with `ambiguous: True`, (2) accepts a
  `file_path` parameter, and (3) when disambiguating, filters child
  classes by tracing `class_name` through `_file_symbol_imports`.

### Fixed — Search quality

- **Gibberish queries can no longer be rescued by a single coincidental
  source-token hit.** Previously, the gibberish-detection gate let the
  query through whenever ANY query token appeared in the source-token
  index — a single coincidence (e.g., `"foobar"` from `"xyzzy foobar
  qlrmph"` matching `test_path_bool_foobar`) was enough to surface
  high-confidence results for nonsense input. The threshold is now ≥2
  distinct token hits; technical-vocabulary queries still survive.
- **`find_related_code` no longer admits files whose only matches come
  from a parent directory name.** Queries like "user authentication
  login flow" used to surface `services/flows/activation_service.py`
  because "flow" matched the `flows/` directory. Matches must now have
  at least one substantive hit (filename token or source token);
  pure-directory matches are dropped, single-token matches are
  soft-penalized 0.6×.

### Fixed — Unused-symbol detection

- **Methods overriding an external-framework base class are no longer
  flagged unused.** `APIRoute.matches` (Starlette base) and
  `FastAPI.build_middleware_stack` were appearing in
  `get_unused_symbols(functions)` because their bases aren't in the
  analyzed codebase. The check now exempts methods whose ancestor chain
  includes any class that is *not* in `_class_defined_in` and *not* in
  a small `_NON_DISPATCHING_BASES` set (`object`, `Exception`, basic
  Python builtins).
- **Top-level functions in plugin/hook files
  (`*_hooks.py`, `*_plugin.py`, `*_extension.py`, `conftest.py`)
  exempted.** MkDocs hooks (`on_config`, `on_files`, `on_nav`, etc.)
  and similar dynamic-dispatch entry points produced 5 false positives
  in a single FastAPI audit run.
- **Functions whose body emits a `DeprecationWarning` via
  `warnings.warn(...)` are exempt.** FastAPI's `generate_operation_id`
  (deprecated public API kept for backwards compat) was surfacing as
  unused. Two new exemptions: (1) the `@deprecated` decorator
  (typing.deprecated, PEP 702), (2) functions whose source contains both
  `warnings.warn` and a `DeprecationWarning` /
  `PendingDeprecationWarning` category.

### Fixed — Language detection

- **`.js` files no longer mis-reported as TypeScript.** graph-sitter
  parses `.js` with the TypeScript parser, so the old `by_language`
  aggregation (keyed on the codebase parser key) labeled every `.js`
  file as `typescript`. A new `_lang_from_path()` static helper on
  `PathMixin` maps file extension → language using
  `EXTENSION_TO_LANGUAGE`. `get_stats`, `get_all_functions`,
  `get_all_classes`, and `get_all_files` now use it, falling back to
  the codebase key only for unknown extensions.

### Fixed — Resolution / detail levels

- **`detail="signatures"` on `get_project_skeleton` now strips
  `file_tree`** in addition to `directory_stats`, `by_language`, and
  `subdir_stats`. `file_tree` is the heaviest field; keeping it made
  signatures barely smaller than summary, which made the level
  pointless.

### Fixed — TypeScript / JavaScript

- **TS/JSX destructured-prop signatures** (`function Comp({ a, b }: Props)`)
  no longer render as `(a: Props, b: Props, …)`.
  `format_function_signature` now extracts the literal parameter list
  from source for `.ts`/`.tsx`/`.js`/`.jsx` files; default values,
  callback types, and generics survive.
- **TypeScript convention detection** no longer reports "0% of
  functions have return type annotations" with confidence 1.0 on
  codebases that clearly use them. The detector now uses the TS-specific
  `): RetType` syntax (with `rfind(")")` to ignore inner parens from
  callback parameter types).

### Fixed — Embedding observability

- **`EmbeddingSearcher` build failures are now surfaced** instead of
  silently retrying forever. A new `_build_error: str | None` is set
  when `build()` raises. `find_related_code` reports
  `degraded_reason: "build_failed"` with a concrete `action_hint` (was
  the misleading `"index_warming_up"`). `wait_for_index_ready` returns
  `False` immediately when set. Build milestones (fingerprint, cache
  hit, model load, vector count) now log at `INFO`.

### Added

- `get_module_context` — `offset` and `limit` parameters for
  paginating large modules; `pagination_hint` block when a module has
  more than 80 files and the caller didn't paginate.
  `total_files`/`total_functions`/`total_classes` always reflect the
  full module.
- `detail="signatures"` on `get_module_context` drops method lists,
  base classes, and function signatures from nested entries — names
  only, so a single response can fit far more files.
- `get_function_context` accepts a `file_path` parameter to
  disambiguate same-named top-level functions in different files
  (e.g. API route `create_agent` vs. service-layer `create_agent`).
  Substring match against the filepath.
- `get_subclasses(class_name, depth=3, file_path=None)` — new
  `file_path` parameter for disambiguating same-name classes.
- `PathMixin._lang_from_path(path) -> str` — file-extension-based
  language detection helper, available on every `CodebaseGraph`
  instance.

### Changed

- `_build_import_index` Phase 2 — external-package skip is now
  conditional on the import shape and presence of a local file (see
  intra-package import fix above).

## [0.2.1] - 2026-04-30

### Changed

- **`fastembed` is now a hard dependency.** The 0.2.0 install advertised
  the semantic encoder as the headline feature but shipped it behind the
  `[embeddings]` extra, so most installs ran in degraded token-only mode
  without realizing it. Reinstall with `pip install --upgrade grafyx-mcp`
  to get the encoder out of the box. The `[embeddings]` extra is kept as
  a no-op alias for backwards compatibility.
- `find_related_code` now reports `"model": "tokens"` (not the configured
  encoder id) when the response was actually scored without the encoder,
  and surfaces a new `degraded_reason` field that distinguishes
  `fastembed_missing` (action: reinstall) from `index_warming_up`
  (action: retry in 30-60 s).

### Fixed

- `get_project_skeleton.directory_stats` undercounted on projects with
  more than 500 files because the underlying `get_all_files()` was
  invoked with the default cap. The skeleton now uses a 10K cap, matching
  `get_module_context`. Reported in the v0.2.0 audit (file count for
  `frontend/` showed 123, but `get_module_context("frontend/src/components")`
  showed 194 just for that subtree).
- TypeScript and JavaScript function signatures are now formatted with
  TS/JS syntax (`function name(arg: T): R` /
  `async function name(arg: T): R`) instead of being incorrectly rendered
  as Python `def name(arg: T) -> R`. Detected from the file extension
  (`.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.cjs`).
- Async factory pattern (`coord = await get_coordinator()`) is now
  resolved by `_augment_index_with_local_var_types`. Previously the
  factory regex required a synchronous call site, so callers of methods
  reached through an awaited factory (e.g. `coord.broadcast_transcript()`)
  were missed entirely. This was the largest single accuracy gap on
  factory-heavy codebases.
- `get_project_skeleton` hints now drill into the largest non-test
  subdirectory of any top-level dir with >100 files, e.g. suggesting
  `get_module_context("backend/app")` instead of the redundant
  `get_module_context("backend")`. Hints are also computed before
  detail-level filtering, so they survive at `detail="signatures"`.

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
