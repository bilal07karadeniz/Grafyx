"""Utility functions for the sample project."""

import json
import os
from typing import Any


def helper_function(value: Any) -> str:
    """Convert any value to a formatted string."""
    if isinstance(value, dict):
        return json.dumps(value, indent=2)
    return str(value)


def format_output(data: dict) -> dict:
    """Format output data with metadata."""
    return {
        "data": data,
        "formatted": True,
        "timestamp": "2025-01-01T00:00:00Z",
    }


def read_config(path: str) -> dict:
    """Read configuration from a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return json.load(f)


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


SUPPORTED_FORMATS = ["json", "csv", "xml"]

MAX_RETRIES = 3
