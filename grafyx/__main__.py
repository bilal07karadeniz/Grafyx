"""CLI entry point for Grafyx MCP server."""

# ── Ensure clean stdout for MCP stdio transport ──
# Any stray bytes on stdout corrupt the JSON-RPC stream and kill the
# MCP connection.  graph_sitter creates StreamHandler(sys.stdout) both
# at import time AND lazily during build_graph().
#
# We use TWO layers of protection:
#   Layer 1: Replace sys.stdout with sys.stderr during imports.
#   Layer 2: Monkeypatch logging.StreamHandler.__init__ so that ANY
#            handler targeting stdout is silently redirected to stderr.
#            This catches handlers created at any time, even lazily
#            during background graph initialization.
import os
import sys
import warnings
import logging

_real_stdout = sys.stdout

# Layer 1: hide stdout during imports.
sys.stdout = sys.stderr

# Layer 2: monkeypatch StreamHandler so stdout is NEVER used as a log
# destination.  FastMCP writes JSON-RPC directly to sys.stdout.buffer
# (not via logging), so this doesn't affect the MCP transport.
_orig_sh_init = logging.StreamHandler.__init__


def _safe_sh_init(self, stream=None):  # type: ignore[override]
    if stream is None:
        stream = sys.stderr
    elif stream is _real_stdout or stream is sys.stdout:
        stream = sys.stderr
    _orig_sh_init(self, stream)


logging.StreamHandler.__init__ = _safe_sh_init  # type: ignore[method-assign]

os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.WARNING,
    format="[grafyx] %(levelname)s %(message)s",
    stream=sys.stderr,
    force=True,
)

import argparse

from grafyx import __version__
from grafyx.server import create_server


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="grafyx",
        description="Grafyx MCP server — real-time codebase understanding for AI coding assistants",
    )
    parser.add_argument(
        "--project",
        default=".",
        help="Path to the project to analyze (default: current directory)",
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="Comma-separated list of languages to parse (default: auto-detect)",
    )
    parser.add_argument(
        "--ignore",
        default=None,
        help="Comma-separated additional directory patterns to ignore",
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Disable automatic file watching",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"grafyx {__version__}",
    )

    args = parser.parse_args()

    # Reconfigure logging with user's preferred level
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="[grafyx] %(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )

    languages = args.languages.split(",") if args.languages else None
    ignore_patterns = args.ignore.split(",") if args.ignore else None

    server = create_server(
        project_path=args.project,
        languages=languages,
        ignore_patterns=ignore_patterns,
        watch=not args.no_watch,
        lazy=True,
    )

    # Restore real stdout for FastMCP's stdio transport.
    # The StreamHandler monkeypatch ensures no logger can ever write here.
    sys.stdout = _real_stdout

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
