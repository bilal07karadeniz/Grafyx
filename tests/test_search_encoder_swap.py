"""find_related_code is routed through EmbeddingSearcher only (no M5)."""
from __future__ import annotations

import sys

import pytest

from grafyx.graph import CodebaseGraph
from grafyx.search import _embeddings
from grafyx.search.searcher import CodeSearcher


@pytest.fixture
def fixture_graph(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "auth.py").write_text(
        "def login(user, password):\n"
        '    """Validate user credentials and return a session token."""\n'
        "    return None\n"
    )
    (pkg / "rate.py").write_text(
        "def exponential_backoff_delay(attempt):\n"
        '    """Return the wait time for retry with exponential backoff."""\n'
        "    return 2 ** attempt\n"
    )
    return CodebaseGraph(str(tmp_path), languages=["python"])


def test_search_does_not_import_code_encoder(fixture_graph):
    """The M5 module should not be imported anywhere in the search path."""
    sys.modules.pop("grafyx.search._code_encoder", None)
    s = CodeSearcher(fixture_graph)
    s.search("login user", max_results=5)
    assert "grafyx.search._code_encoder" not in sys.modules, (
        "find_related_code is still pulling in M5 module"
    )


@pytest.mark.skipif(
    not _embeddings._HAS_EMBEDDINGS,
    reason="fastembed not installed; embedding-driven retrieval skipped",
)
def test_search_returns_results(fixture_graph):
    s = CodeSearcher(fixture_graph)
    s.wait_for_index_ready(timeout=120)
    results = s.search("validate user credentials", max_results=5)
    assert isinstance(results, list)
    if results:
        assert any("login" in r.get("name", "") for r in results)
