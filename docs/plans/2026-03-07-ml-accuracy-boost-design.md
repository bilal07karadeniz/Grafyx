# Grafyx ML Accuracy Boost — Design Document

**Date**: 2026-03-07
**Goal**: Push Grafyx tool accuracy from 68/100 to 91/100+ using ML models, replacing all brittle heuristics with learned behavior.

## Evaluation Data (Two Independent Tests)

Tested on Eddy (522 files, 994 functions, 321 classes — Python + TypeScript monorepo). Two independent evaluations confirmed the same findings.

### Scores Before

| Tool | Score | Verdict |
|------|-------|---------|
| get_subclasses | 10/10 | Perfect |
| get_unused_symbols (funcs) | 10/10, 92/100 | Perfect (15/15 verified) |
| get_class_context | 9.5/10, 88/100 | Excellent |
| get_project_skeleton | 9/10, 85/100 | Excellent |
| function disambiguation | 95/100 | Excellent |
| get_function_context | 8.5/10, 82-90/100 | Good |
| get_module_context | 8.5/10, 75/100 | Good |
| error handling | 10/10 | Perfect |
| find_related_files | 7.5/10, 75/100 | OK |
| get_conventions | 5.5/10, 82/100 | Mixed |
| get_call_graph | 7/10, 55/100 | Cross-contamination bug |
| get_dependency_graph | 8/10, 30/100 | Reverse deps broken |
| **find_related_code** | **2/10, 15/100** | **BROKEN** |
| circular deps | 25/100 | Broken (depends on reverse deps) |

### Root Causes

1. **Lazy `__getattr__` import pattern**: `__init__.py` lazy imports break search (source token contamination), reverse deps (import index can't trace through), and imported_by tracking
2. **Method name conflation**: `db.refresh()` matched to `refresh()` API endpoint — 55 phantom callers, call graph cross-contamination
3. **No semantic search**: Pure lexical matching — "handle login" can't find `authenticate_user`
4. **Score granularity**: Soft cap at 0.85 compresses all results to same score (0.925 and 0.555 only)

### Design Principle

**Everything that can be learned, MUST be learned.** No hardcoded thresholds, no project-specific heuristics, no magic numbers. The only non-ML changes are data correctness fixes (extracting more information from code, not filtering with arbitrary rules).

---

## Phase 1: Data Quality Fixes (Heuristic Foundation)

Three code-level fixes that improve data quality before ML models see it. Zero thresholds, zero magic numbers.

### Fix 1.1: Import Index — Follow ALL `__init__.py` Re-exports

When resolving `from pkg import X`, if `pkg/__init__.py` exists, follow ALL re-export mechanisms:
- `from .module import X` (explicit re-export)
- `__all__ = [...]` declarations
- `__getattr__` with any lazy import pattern
- Direct assignment `X = module.X`

Resolve to the actual source module, regardless of mechanism.

**Files**: `grafyx/graph/_indexes.py`

### Fix 1.2: Extract Receiver Tokens from Call Sites

Currently `db.refresh()` stored as just `refresh` in caller index. New format:

```
Current:  {function_name: [(caller_file, caller_func, caller_class, trusted)]}
New:      {function_name: [(caller_file, caller_func, caller_class, trusted, receiver_token, has_dot_syntax)]}
```

Pure data extraction — no filtering, no thresholds. ML models use the extra fields.

**Files**: `grafyx/graph/_indexes.py`, `grafyx/graph/_callers.py`

### Fix 1.3: Remove ALL Hardcoded Caps

Replace all `[:20]`, `[:30]`, `MAX_HINTS = 3` with dynamic behavior:
- Return everything, let ML rankers decide what to show
- If truncation needed for output size, let the model rank first, then truncate

**Files**: `grafyx/server/_hints.py`, `grafyx/server/_tools_graph.py`, `grafyx/server/_tools_introspection.py`

---

## Phase 2: ML Models (6 Models, ~28M total params)

### Model Inventory

| # | Model | Architecture | Params | Replaces | Disk |
|---|-------|-------------|--------|----------|------|
| M1 | Relevance Ranker v2 | MLP (42→128→64→1) | 16K | Current 33-feature MLP | 32KB |
| M2 | Caller Disambiguator | MLP (25→64→32→1) | 4K | 4-level heuristic filter | 8KB |
| M3 | Source Token Filter | MLP (15→32→16→1) | 1.5K | Hardcoded 0.6 IDF weight | 3KB |
| M4 | Symbol Importance Ranker | MLP (18→32→16→1) | 1.5K | Caller count heuristic | 3KB |
| M5 | Code Search Encoder | Mamba (8L-384H-16S) | 16M | No semantic search / fastembed | 32MB |
| M6 | Cross-Encoder Reranker | Mamba+Attn (6M+2A-384H) | 12M | MLP scoring for top-K | 24MB |
| | **Total** | | **~28M** | | **~57MB** |

### M1: Relevance Ranker v2 (MLP, 16K params)

Retrain existing model with 9 new features (33 → 42 total).

**New features (on top of existing 33):**
- 33: is_dunder (bool)
- 34: is_init_file (bool)
- 35: is_method (bool)
- 36: is_class (bool)
- 37: receiver_call_ratio (float) — fraction of callers using dot syntax
- 38: source_token_entropy (float) — concentrated vs scattered source matches
- 39: source_unique_token_ratio (float) — tokens unique to this function vs project-wide
- 40: embedding_similarity (float) — cosine sim from M5 (0.0 if M5 not loaded)
- 41: caller_count_normalized (float) — popularity signal

Features 33-39 let the model LEARN that `__getattr__` in `__init__.py` with scattered source tokens is irrelevant. No hardcoded exclusion.

**Training data**: 500K examples (200K existing + 300K new with hard negatives from evaluation results).

### M2: Caller Disambiguator (MLP, 4K params)

Binary classifier: P(this call_site targets this candidate_callee).

**25 features per (call_site, candidate_callee) pair:**
1. receiver_token_overlap_class_name (float)
2. receiver_char_bigram_sim_class_name (float)
3. caller_imports_callee_module (bool)
4. caller_imports_callee_package (bool)
5. file_path_distance (float)
6. same_directory (bool)
7. same_top_package (bool)
8. has_dot_syntax (bool)
9. receiver_is_self (bool)
10. method_uniqueness (float) — 1/count_classes_with_method
11. callee_is_method (bool)
12. callee_is_standalone (bool)
13. same_language (bool)
14. receiver_type_known (bool)
15. receiver_type_matches (bool)
16. callee_param_count (float, normalized)
17. arg_count_matches_params (bool)
18. callee_has_decorator (bool)
19. receiver_name_length (float, normalized)
20. method_name_commonness (float)
21. caller_complexity (float) — LOC normalized
22. callee_is_property (bool)
23. callee_is_classmethod (bool)
24. callee_is_abstractmethod (bool)
25. receiver_is_common_pattern (float) — learned from training

**Training data**: 200K from type-annotated Python codebases where ground truth is known (typed variable → method call → we know the correct class).

Replaces the entire 4-level heuristic filter. For `db.refresh()` the model sees: has_dot_syntax=True, receiver="db", callee is standalone function, receiver_type not matching → P(real) ≈ 0.05.

### M3: Source Token Filter (MLP, 1.5K params)

Predicts P(query_token is semantically relevant to this function).

**15 features per (token, function) pair:**
1. token_in_function_name (bool)
2. token_in_docstring (bool)
3. token_in_parameter_names (bool)
4. token_in_return_type (bool)
5. token_frequency_in_body (float, normalized)
6. token_idf_across_all_functions (float)
7. token_in_import_statements (bool)
8. token_in_string_literals (bool)
9. function_is_dunder (bool)
10. function_in_init_file (bool)
11. function_body_length (float, normalized)
12. token_position_in_body (float) — early=0.0, late=1.0
13. token_in_variable_names (bool)
14. function_unique_token_count (float, normalized)
15. token_char_bigram_sim_to_function_name (float)

Kills `__getattr__` contamination: model learns that token in dict literal inside dunder function in `__init__.py` = not relevant.

**Training data**: 200K from code structure analysis (name/docstring/param tokens = positive, import/string-literal/comment tokens = negative).

### M4: Symbol Importance Ranker (MLP, 1.5K params)

Predicts "how important/interesting is this symbol" for navigation hints and ranking.

**18 features per symbol:**
1. caller_count (normalized by project size)
2. importer_count (normalized)
3. method_count (for classes)
4. subclass_count (for classes)
5. is_entry_point (bool) — has route/task/main/cli decorator
6. has_docstring (bool)
7. docstring_length (normalized)
8. is_abstract (bool)
9. decorator_count (normalized)
10. loc (normalized)
11. parameter_count (normalized)
12. cross_language_reference_count (float)
13. is_in_test_file (bool)
14. module_depth (float)
15. name_specificity (float) — longer names = more specific
16. has_type_annotations (bool)
17. is_class (bool) vs function
18. file_centrality (float) — how connected is this file

**Training data**: 100K from 50+ OSS projects. Labels derived from multi-signal proxy: mentioned in README (0.3), is API entry point (0.25), referenced in >10 files (0.15), is base class with subclasses (0.1), cross-package callers (0.1), exported in `__all__` (0.1). The model learns its own weighting of these structural signals.

### M5: Code Search Encoder (Mamba Bi-Encoder, 16M params)

The semantic brain. Maps queries and code to shared 256-dim embedding space.

```
Architecture:
  Input:      BPE tokens (16K vocab, max 256 tokens)
  Embedding:  16384 × 384 = 6.3M params
  Position:   512 × 384 = 197K (learned absolute)
  Encoder:    8 Mamba blocks
              d_model=384, d_inner=768, d_state=16
              Per block: in_proj(384→1536) + SSM + out_proj(768→384) + LayerNorm
              Per block: ~1.2M params
              8 blocks: 9.6M params
  Pooling:    Mean pool over sequence → 384-dim
  Projection: 384 → 256 = 98K

  Total: ~16.2M params (~32MB float16)
  Inference: ~10-15ms per encoding on CPU, <2ms on GPU
```

**Why Mamba**:
- O(n) inference vs O(n²) for transformer — code sequences are long
- SSMs naturally model sequential dependencies in code
- 16M Mamba params ≈ 30M transformer params in capacity
- Pure numpy inference: selective scan is a simple for loop

**Tokenizer**: BPE with 16K vocab trained on code+English. Merge rules exported as JSON. Encoding implemented in pure Python (~50 lines). No runtime dependency.

**Usage**:
- Index time: Pre-compute embeddings for all functions/classes. Store in numpy array (~500KB per 1000 symbols).
- Query time: Encode query (10ms) → cosine similarity (0.1ms) → top-50 candidates.

**Training data**: 2.5M examples total:
- CodeSearchNet: 1.5M (NL query, code function) pairs
- Synthetic name-based: 500K (split `get_current_user` → "get current user")
- Claude Haiku-generated semantic pairs: 125K (gold-standard conceptual queries)
- Hard negatives: 500K (same-file different function, keyword overlap different intent)

**Training**: Contrastive learning (InfoNCE loss), batch size 512, 10 epochs, ~2-3 hours on RTX 5070.

### M6: Cross-Encoder Reranker (Mamba+Attention Hybrid, 12M params)

Reranks top-15 candidates from M5 with full query×code interaction.

```
Architecture:
  Input:      [query_tokens SEP code_name + code_doc] (max 384 combined)
  Embedding:  Shared with M5 (6.3M) or separate
  Encoder:    6 Mamba blocks (7.2M) + 2 self-attention blocks (2.4M)
              Attention: 384 dim, 6 heads, FFN=1536
              The attention layers capture query↔code interactions
              Mamba layers capture sequential code patterns
  Pooling:    CLS token → 384-dim
  Head:       384 → 128 → 1 (49K)

  Total: ~12M params (~24MB float16)
  Inference: ~15ms per candidate × 15 = ~225ms total
```

**Why hybrid (Jamba-style)**: Mamba alone can't cross-attend between query and code. The 2 attention layers enable bidirectional query↔code matching. Mamba layers efficiently process the long input.

**Training data**: 500K reranking pairs (same sources as M5, formatted as [query SEP code] → relevance_score). Use M5's bi-encoder scores as soft labels for borderline cases.

**Training**: BCE loss, batch size 256, 15 epochs, ~1-2 hours on RTX 5070.

---

## Phase 3: Training Data Strategy

### Data Sources

| Source | Size | Models | How |
|--------|------|--------|-----|
| Existing training data | 200K | M1 | Recompute with 42 features |
| 50+ OSS Python/TS repos | ~500K | M1-M4 | Clone, extract symbols, generate pairs |
| CodeSearchNet (public) | 1.5M | M5, M6 | Download, filter Python+JS subsets |
| Synthetic name-based | 500K | M5, M6 | Split camelCase/snake_case → NL queries |
| Claude Haiku semantic pairs | 125K | M5, M6 | Function body → "what queries find this?" |
| Hard negatives | 500K | M5, M6 | Same-file, keyword-overlap, cross-language |
| Typed codebase analysis | 200K | M2 | Type-annotated code → ground truth callers |
| Code structure analysis | 200K | M3 | Token position analysis in function bodies |
| OSS project analysis | 100K | M4 | README mentions, entry points, connectivity |

**Total**: ~3.8M examples across all models. Zero human labeling — all generated synthetically.

### OSS Repos for Training (50+, all MIT/Apache)

**Web frameworks**: FastAPI, Django, Flask, Starlette, Sanic
**LLM tooling**: LangChain, LlamaIndex, Haystack
**Database**: SQLAlchemy, Tortoise-ORM, SQLModel, Peewee
**Task queues**: Celery, Dramatiq, Huey, RQ
**Data models**: Pydantic, Attrs, Marshmallow
**HTTP**: httpx, requests, aiohttp
**CLI**: Click, Typer, Rich, Textual
**Testing**: pytest, hypothesis
**Frontend**: Next.js examples, React repos, Vue repos
**Typed Python**: mypy, pyright test fixtures
**DevOps**: Ansible modules, Fabric

### Claude Haiku Semantic Pair Generation

For each function in the 50 repos:
```
Prompt: "Given this Python function, generate 5 natural language search
queries a developer would use to find it. Be diverse — include conceptual
queries, not just the function name."

Input: def authenticate_user(email: str, password: str, db: Session) -> User:
       """Verify credentials and return user if valid."""

Output:
1. "verify user login credentials"
2. "check if password matches for email"
3. "handle user authentication"
4. "validate login attempt"
5. "sign in user with email and password"
```

Cost: ~$50-80 for 125K gold-standard pairs. These cover exactly the semantic gap that broke in evaluation.

### Training Pipeline

```
ml/
├── data_common/
│   ├── clone_repos.py            # Clone 50+ repos to local cache
│   ├── extract_symbols.py        # Extract functions/classes/imports
│   ├── repos.json                # Repo list with licenses
│   └── download_codesearchnet.py # Download CodeSearchNet
├── relevance_ranker_v2/
│   ├── generate_data.py          # 500K examples
│   ├── features.py               # 42-feature extraction
│   ├── train.py                  # MLP training (~seconds)
│   └── stress_test.py
├── caller_disambiguator/
│   ├── generate_data.py          # 200K from typed code
│   ├── features.py               # 25-feature extraction
│   ├── train.py
│   └── stress_test.py
├── source_token_filter/
│   ├── generate_data.py          # 200K from code analysis
│   ├── features.py               # 15-feature extraction
│   ├── train.py
│   └── stress_test.py
├── symbol_importance/
│   ├── generate_data.py          # 100K from OSS analysis
│   ├── features.py               # 18-feature extraction
│   ├── train.py
│   └── stress_test.py
├── code_search_encoder/
│   ├── generate_semantic_pairs.py  # Claude Haiku generation
│   ├── generate_synthetic.py       # 500K synthetic + 500K hard negatives
│   ├── train_tokenizer.py          # 16K BPE vocab
│   ├── model.py                    # Mamba bi-encoder (PyTorch)
│   ├── train.py                    # Contrastive learning (~2-3hrs GPU)
│   └── evaluate.py
├── cross_encoder/
│   ├── generate_data.py          # 500K reranking pairs
│   ├── model.py                  # Mamba+Attention hybrid (PyTorch)
│   ├── train.py                  # BCE training (~1-2hrs GPU)
│   └── evaluate.py
└── export_all.py                 # Convert ALL models to numpy .npz
```

**Total training time on RTX 5070**: ~4-6 hours.

---

## Integration Architecture

### New/Modified Files

```
grafyx/
├── search/
│   ├── _relevance.py          # MODIFY: 42 features, load v2 weights
│   ├── _source_filter.py      # NEW: M3 source token quality filter
│   ├── _code_encoder.py       # NEW: M5 Mamba bi-encoder inference
│   ├── _cross_encoder.py      # NEW: M6 Mamba+Attn reranker inference
│   ├── _tokenizer.py          # NEW: BPE tokenizer (pure Python)
│   ├── _mamba.py              # NEW: Mamba forward pass (numpy)
│   ├── searcher.py            # MODIFY: new search pipeline
│   └── model/
│       ├── relevance_weights_v2.npz        # M1
│       ├── caller_disambig_weights.npz     # M2
│       ├── source_filter_weights.npz       # M3
│       ├── symbol_importance_weights.npz   # M4
│       ├── code_encoder_weights.npz        # M5 (~32MB)
│       ├── cross_encoder_weights.npz       # M6 (~24MB)
│       ├── bpe_merges.json                 # Tokenizer
│       └── (existing weights kept)
│
├── graph/
│   ├── _indexes.py            # MODIFY: receiver tokens, __init__.py re-exports
│   ├── _callers.py            # MODIFY: use M2 for disambiguation
│   └── _analysis.py           # MODIFY: use M2 for unused code detection
│
├── server/
│   ├── _hints.py              # MODIFY: use M4 for importance ranking
│   ├── _tools_search.py       # MODIFY: new search pipeline
│   └── _tools_graph.py        # MODIFY: dynamic limits, M2 integration
│
└── ml_inference.py            # NEW: shared model loading, lazy init, GPU detect
```

### Search Pipeline (After)

```
Query arrives
    |
[Existing] Gibberish Detector → block nonsense
    |
[M5] Bi-Encoder: encode query → 256-dim vector (10ms)
    |
Cosine similarity vs pre-computed embeddings → top-50
    |
[M1v2] Relevance Ranker: score each (42 features incl embedding sim)
    |
[M3] Source Token Filter: validate source token relevance per candidate
    |
[M6] Cross-Encoder: rerank top-15 with full query×code attention (225ms)
    |
Return top-10 with calibrated confidence
```

### Caller Resolution Pipeline (After)

```
For each (call_site, candidate_callee) pair:
    |
[M2] Caller Disambiguator: P(real call)
    |
Keep candidates above learned threshold
    |
Used by: get_call_graph, get_dependency_graph, get_function_context, get_unused_symbols
```

### Graceful Degradation

Every model is optional:
- M5/M6 not loaded → keyword-only search (current behavior)
- M2 not loaded → heuristic 4-level disambiguation (current behavior)
- M3 not loaded → all source tokens treated equally (current behavior)
- M4 not loaded → caller count heuristic for hints (current behavior)
- cupy not available → numpy CPU inference (always works)

### GPU Detection

```python
def _get_backend():
    try:
        import cupy as cp
        return cp
    except ImportError:
        return numpy
```

All model code uses `xp = _get_backend()` — same code, CPU or GPU.

---

## Expected Impact

| Tool | Before | After Phase 1 | After Phase 2 | After Phase 3 |
|------|--------|---------------|---------------|---------------|
| find_related_code | 2/10 | 5/10 | 7/10 | **9/10** |
| find_related_files | 7.5/10 | 8/10 | 8.5/10 | **9/10** |
| get_call_graph | 55/100 | 75/100 | **90/100** | 90/100 |
| get_dependency_graph | 30/100 | **80/100** | 85/100 | 85/100 |
| get_function_context | 82/100 | 88/100 | **93/100** | 93/100 |
| navigation hints | 80/100 | 80/100 | **90/100** | 90/100 |
| get_unused_symbols | 92/100 | 92/100 | **95/100** | 95/100 |
| circular deps | 25/100 | **70/100** | 75/100 | 75/100 |
| **Overall** | **68/100** | **78/100** | **86/100** | **91/100** |

---

## Implementation Order

1. **Phase 1**: Data quality fixes (Fix 1.1, 1.2, 1.3)
2. **Phase 2a**: Training infrastructure (clone repos, extract symbols, data generation)
3. **Phase 2b**: Train M1-M4 (MLPs, minutes on GPU)
4. **Phase 2c**: Integrate M1-M4 into codebase
5. **Phase 3a**: Train BPE tokenizer, implement Mamba numpy inference
6. **Phase 3b**: Train M5 (Mamba bi-encoder, 2-3hrs)
7. **Phase 3c**: Train M6 (cross-encoder, 1-2hrs)
8. **Phase 3d**: Integrate M5/M6 into search pipeline
9. **Evaluation**: Re-run accuracy tests, compare against baseline
