"""Shared test fixtures for Grafyx tests."""

import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def python_project_path(fixtures_dir) -> Path:
    """Path to the Python test fixture project."""
    return fixtures_dir / "python_project"
