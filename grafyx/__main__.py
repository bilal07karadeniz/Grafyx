"""CLI entry point for Grafyx MCP server."""

# ── Ensure clean stdout for MCP stdio transport ──
# Must run BEFORE any library imports that might write ANSI or warnings to stdout.
import os
import warnings

os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import argparse
import logging
import sys

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

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[grafyx] %(levelname)s %(message)s",
        stream=sys.stderr,  # Log to stderr so it doesn't interfere with MCP stdio
    )

    languages = args.languages.split(",") if args.languages else None
    ignore_patterns = args.ignore.split(",") if args.ignore else None

    server = create_server(
        project_path=args.project,
        languages=languages,
        ignore_patterns=ignore_patterns,
        watch=not args.no_watch,
    )

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
