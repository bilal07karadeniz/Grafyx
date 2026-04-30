"""Single-entry-point benchmark runner.

Usage:
    python -m scripts.run_all                     # runs setup + eval data + bench_search (all encoders)
    python -m scripts.run_all --skip-setup        # repos already pinned
"""
from __future__ import annotations

import argparse
import sys

from scripts import setup_repos, eval_data, bench_search


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--encoders", default="all",
                        help="Comma-separated: m5, jina-v2, coderankembed, or 'all'")
    args = parser.parse_args()

    if not args.skip_setup:
        sys.argv = ["setup_repos"]
        setup_repos.main()

    eval_data.build_eval_pairs()

    encs = ["m5", "jina-v2", "coderankembed"] if args.encoders == "all" \
        else [e.strip() for e in args.encoders.split(",")]
    bench_search.run(encs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
