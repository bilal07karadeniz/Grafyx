# Grafyx 0.2 Benchmark — Encoder Head-to-Head

**Date:** 2026-04-30
**Grafyx version:** 0.2.0a1
**Scope:** FastAPI + Django (Home Assistant deferred for 0.2.0 GA)
**Eval set:** Per-repo public-function docstrings (first sentence → query, function name → expected). 78 queries on FastAPI, 200 on Django, 278 total.
**Metrics:** nDCG@10 (primary), MRR@10, p50 latency.

## Repos

| Repo | Pinned commit |
|---|---|
| FastAPI | [`4f64b8f6`](https://github.com/tiangolo/fastapi/commit/4f64b8f65191b497255004e896a2bbfd7cfd1886) |
| Django | [`02a7d43d`](https://github.com/django/django/commit/02a7d43d02e2acc9325a5a27eb01ffe7dbba5c7f) |

## Results

| Repo | Encoder | nDCG@10 | MRR@10 | p50 latency | n |
|---|---|---:|---:|---:|---:|
| FastAPI | **jina-v2** | **0.770** | **0.706** | 1058 ms | 78 |
| FastAPI | coderankembed | 0.721 | 0.672 | 885 ms | 78 |
| Django | **jina-v2** | **0.803** | **0.776** | 1865 ms | 200 |
| Django | coderankembed | 0.605 | 0.573 | 1664 ms | 200 |
| **Average** | **jina-v2** | **0.787** | **0.741** | — | 278 |
| **Average** | coderankembed | 0.663 | 0.623 | — | 278 |

## Decision

**`jina-v2` is the default encoder for v0.2.0a1.**

Per the locked selection rule (CodeRankEmbed must lead jina-v2 by ≥0.03 absolute nDCG@10 to ship as default), the picture is unambiguous:

- jina-v2 leads by **12.4 nDCG@10 points** on average across the two repos.
- The Django gap is **19.9 nDCG@10 points** — jina-v2 is far better at retrieving the right symbol from a docstring-style query in a large, mature codebase.
- CodeRankEmbed is ~16% faster on p50 latency, but the accuracy gap dominates.

## CodeRankEmbed remains available

The encoder registry still includes `coderankembed`. Users who want lower latency at the cost of precision can opt in:

```bash
GRAFYX_ENCODER=coderankembed grafyx
```

The model is hosted at [`Bilal7Dev/grafyx-coderankembed-onnx`](https://huggingface.co/Bilal7Dev/grafyx-coderankembed-onnx) (re-host of [`mrsladoje/CodeRankEmbed-onnx-int8`](https://huggingface.co/mrsladoje/CodeRankEmbed-onnx-int8); attribution in the model card).

## Reproduce

```bash
git checkout v0.2.0a1
pip install -e ".[embeddings,bench]"
cd benchmarks
python -m scripts.setup_repos                                 # one-time
python -m scripts.bench_search --encoder jina-v2 --repos fastapi,django
python -m scripts.bench_search --encoder coderankembed --repos fastapi,django
```

Per-query JSONL with raw rankings ships in `docs/benchmarks/0.2.0/per_query/`.

## Caveats

1. **Home Assistant skipped this run.** Home Assistant has ~13K Python files, and the embedding build for that codebase needs ~30 min per encoder; running both encoders against it pushed past the bench harness's wall-clock budget for the alpha. The v0.2.0 GA benchmark will include it.
2. **Bias toward docstring-language queries.** The eval set is the first sentence of each function's docstring, which is a *fair* test (encoder must map natural-language description → function name) but it favors models trained on doc-style retrieval. Real Grafyx queries from MCP clients tend to be terser ("auth middleware", "rate limit"). A second eval pass on hand-written terse queries is on the v0.2.0 GA roadmap.
3. **No "no-encoder" baseline.** Token-only search (the fallback when fastembed isn't installed) wasn't measured — it would require a separate code path. The expected gap is large (encoder ~0.79 vs token ~0.3 historically), but we should publish the number rather than assume it for GA.
