"""MCP Tools for fuzzy/semantic code search.

This module provides two search tools that help an AI assistant **discover**
relevant code without knowing exact symbol names:

- **find_related_code** (Tool 5) -- returns individual functions, classes,
  and files ranked by relevance to a natural language query.
- **find_related_files** (Tool 6) -- returns files ranked by how many of
  their symbols match the query, with matching symbols listed per file.

Both tools delegate to ``CodeSearcher`` (``grafyx/search.py``), which
combines fuzzy string matching on symbol names with token-based relevance
scoring.  When confidence is low, the response includes a ``note`` field
suggesting the AI refine its query.

These tools are the primary way an AI goes from "I need to find the
authentication logic" to discovering the specific files and functions
involved.

Registered on the shared ``mcp`` instance from ``_state.py``.
"""

import time as _time

from fastmcp.exceptions import ToolError

from grafyx.server import _state
from grafyx.server._state import _ensure_initialized, mcp
from grafyx.utils import truncate_response


# --- Tool 5: find_related_code ---


@mcp.tool
def find_related_code(description: str, max_results: int = 10) -> dict:
    """Find functions, classes, and files related to a natural language description.

    Use this when you know WHAT you're looking for conceptually but not
    WHERE it lives in the codebase. Prevents duplicating existing functionality.
    """
    _ensure_initialized()
    try:
        if _state._searcher is None:
            raise ToolError("Search not initialized.")

        # CodeSearcher.search() returns a list of dicts, each with:
        #   name, type (function/class/file), file, score, low_confidence
        t0 = _time.perf_counter()
        results = _state._searcher.search(description, max_results=max_results)
        latency_ms = (_time.perf_counter() - t0) * 1000.0

        response: dict = {
            "query": description,
            "results": results,
            "total_results": len(results),
            "model": getattr(_state._searcher, "encoder_meta", {"model": "unknown", "version": ""}),
            "latency_ms": round(latency_ms, 1),
        }

        # When the best match has low confidence, add a hint so the AI
        # knows to try a more specific query rather than trusting the results.
        if results and results[0].get("low_confidence"):
            response["note"] = (
                "Results may be approximate — consider refining your "
                "query with specific function or class names."
            )

        if getattr(_state._searcher, "degraded", False):
            response["degraded"] = True
            # Distinguish "encoder still warming up" from "encoder permanently
            # missing" so the agent knows whether to retry or to act.
            embed = getattr(_state._searcher, "_embedding_searcher", None)
            if embed is None:
                response["degraded_reason"] = "fastembed_missing"
                response["action_hint"] = (
                    "Semantic encoder package is not installed. Reinstall "
                    "with `pip install --upgrade grafyx-mcp` (0.2.1+ "
                    "bundles fastembed as a hard dependency)."
                )
            elif getattr(embed, "_build_error", None):
                response["degraded_reason"] = "build_failed"
                response["action_hint"] = (
                    f"Embedding index build failed: {embed._build_error}. "
                    "Check the server logs for details. Common causes: model "
                    "download failure (network issue), disk space, or ONNX "
                    "runtime incompatibility. Restart the server to retry."
                )
            else:
                response["degraded_reason"] = "index_warming_up"
                response["action_hint"] = (
                    "Embedding index is still being built in the background. "
                    "Retry the same query in 30-60 seconds for full-quality results."
                )
        return truncate_response(response)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in find_related_code: {e}")


# --- Tool 6: find_related_files ---


@mcp.tool
def find_related_files(description: str, max_results: int = 5) -> dict:
    """Find files related to a natural language description by scoring
    symbols inside them.

    Use this when you need to understand which files are relevant to a
    feature or concept. Returns files grouped with their matching symbols.
    """
    _ensure_initialized()
    try:
        if _state._searcher is None:
            raise ToolError("Search not initialized.")

        # CodeSearcher.search_files() aggregates symbol-level scores by
        # file, returning each file with its top-matching symbols.
        results = _state._searcher.search_files(description, max_results=max_results)

        response: dict = {
            "query": description,
            "results": results,
            "total_results": len(results),
        }

        # Same low-confidence hint as find_related_code.
        if results and results[0].get("low_confidence"):
            response["note"] = (
                "Results may be approximate — consider refining your "
                "query with specific file or function names."
            )
        return truncate_response(response)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error in find_related_files: {e}")
