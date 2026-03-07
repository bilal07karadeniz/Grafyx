"""Tests for grafyx.search module."""

from unittest.mock import MagicMock
from grafyx.search import CodeSearcher


def _make_mock_graph():
    graph = MagicMock()
    graph.get_all_functions.return_value = [
        {"name": "process_data", "signature": "def process_data(data: list) -> dict", "file": "main.py", "language": "python", "line": 10, "docstring": "Process raw data"},
        {"name": "validate", "signature": "def validate(data: list) -> list", "file": "main.py", "language": "python", "line": 20, "docstring": "Validate input"},
        {"name": "helper_function", "signature": "def helper_function(value) -> str", "file": "utils.py", "language": "python", "line": 5, "docstring": "Convert value to string"},
    ]
    graph.get_all_classes.return_value = [
        {"name": "DataProcessor", "base_classes": [], "file": "main.py", "language": "python", "line": 30, "docstring": "Processes data in batch", "method_count": 3},
        {"name": "User", "base_classes": [], "file": "models.py", "language": "python", "line": 5, "docstring": "User model"},
    ]
    graph.get_all_files.return_value = [
        {"path": "main.py", "function_count": 3, "class_count": 1, "import_count": 2, "language": "python"},
        {"path": "utils.py", "function_count": 2, "class_count": 0, "import_count": 1, "language": "python"},
        {"path": "models.py", "function_count": 0, "class_count": 2, "import_count": 1, "language": "python"},
    ]
    return graph


class TestCodeSearcher:
    def test_search_exact_name(self):
        searcher = CodeSearcher(_make_mock_graph())
        results = searcher.search("process_data")
        assert len(results) > 0
        assert results[0]["name"] == "process_data"
        assert results[0]["score"] > 0.5

    def test_search_fuzzy_match(self):
        searcher = CodeSearcher(_make_mock_graph())
        results = searcher.search("data processing")
        names = [r["name"] for r in results]
        assert "process_data" in names or "DataProcessor" in names

    def test_search_by_docstring(self):
        searcher = CodeSearcher(_make_mock_graph())
        results = searcher.search("validate input")
        names = [r["name"] for r in results]
        assert "validate" in names

    def test_search_max_results(self):
        searcher = CodeSearcher(_make_mock_graph())
        results = searcher.search("data", max_results=2)
        assert len(results) <= 2

    def test_search_kind_filter(self):
        searcher = CodeSearcher(_make_mock_graph())
        results = searcher.search("data", kind_filter="class")
        for r in results:
            assert r["kind"] == "class"

    def test_search_returns_scores(self):
        searcher = CodeSearcher(_make_mock_graph())
        results = searcher.search("process")
        for r in results:
            assert 0 < r["score"] <= 1.0

    def test_search_files(self):
        searcher = CodeSearcher(_make_mock_graph())
        results = searcher.search_files("data processing")
        assert len(results) > 0
        assert "file" in results[0]
        assert "score" in results[0]
        assert "relevant_symbols" in results[0]
