"""Head-to-head search benchmark: M5 baseline (and later: jina-v2, CodeRankEmbed).

Measures nDCG@10 and MRR@10 per repo, per encoder, across the eval_data pairs.

Usage:
    python -m scripts.bench_search --encoder m5
    python -m scripts.bench_search --encoder all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.eval_data import load_eval_pairs

ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = ROOT / "repos"
RESULTS_DIR = ROOT / "results"


def _ndcg_at_k(ranked: list[str], expected: str, k: int = 10) -> float:
    """1.0 if expected is at rank 1; log-discounted otherwise; 0.0 if not in top-k."""
    for i, name in enumerate(ranked[:k]):
        if name == expected:
            return 1.0 / math.log2(i + 2)
    return 0.0


def _mrr_at_k(ranked: list[str], expected: str, k: int = 10) -> float:
    for i, name in enumerate(ranked[:k]):
        if name == expected:
            return 1.0 / (i + 1)
    return 0.0


def _run_grafyx_encoder(repo_path: Path, queries: list[dict], encoder: str) -> list[dict]:
    """Spin up Grafyx against the repo and run each query through find_related_code.

    encoder: "m5" | "jina-v2" | "coderankembed" — controls which encoder backend
    Grafyx uses (set via env var GRAFYX_ENCODER, read by EmbeddingSearcher in Task 7).
    """
    os.environ["GRAFYX_ENCODER"] = encoder
    from grafyx.graph import CodebaseGraph
    from grafyx.search.searcher import CodeSearcher

    graph = CodebaseGraph(str(repo_path), languages=["python"])
    searcher = CodeSearcher(graph)
    # Block on encoder build — for benchmarking we want a warm index.
    searcher.wait_for_index_ready(timeout=600)

    rows: list[dict] = []
    for q in queries:
        t0 = time.perf_counter()
        results = searcher.search(q["query"], max_results=10)
        latency_ms = (time.perf_counter() - t0) * 1000
        ranked = [r.get("name", "") for r in results]
        rows.append({
            "query": q["query"],
            "expected": q["expected_symbol"],
            "ranked_top10": ranked,
            "ndcg10": _ndcg_at_k(ranked, q["expected_symbol"]),
            "mrr10": _mrr_at_k(ranked, q["expected_symbol"]),
            "latency_ms": latency_ms,
        })
    return rows


def run(encoders: list[str]) -> dict:
    pins = json.loads((ROOT / "pinned_commits.json").read_text())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"run_id": run_id, "results": {}}
    for repo_name in pins:
        repo_path = REPOS_DIR / repo_name
        queries = load_eval_pairs(repo_name)
        if not queries:
            print(f"  skip {repo_name} (no eval data)")
            continue
        for enc in encoders:
            print(f"  {repo_name} / {enc}: {len(queries)} queries...")
            rows = _run_grafyx_encoder(repo_path, queries, enc)
            ndcgs = [r["ndcg10"] for r in rows]
            mrrs = [r["mrr10"] for r in rows]
            lats = [r["latency_ms"] for r in rows]
            summary["results"].setdefault(repo_name, {})[enc] = {
                "ndcg10_mean": statistics.mean(ndcgs),
                "mrr10_mean": statistics.mean(mrrs),
                "latency_p50_ms": statistics.median(lats),
                "n": len(rows),
            }
            (out_dir / f"{repo_name}__{enc}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nResults written to {out_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="m5",
                        choices=["m5", "jina-v2", "coderankembed", "all"])
    args = parser.parse_args()
    encs = ["m5", "jina-v2", "coderankembed"] if args.encoder == "all" else [args.encoder]
    run(encs)
