"""Tests for gibberish query scoring penalty (P2 fix).

Nonsense queries should score significantly lower than meaningful queries.
"""

from unittest.mock import MagicMock
from grafyx.search import CodeSearcher


def _make_mock_graph():
    graph = MagicMock()
    graph.get_all_functions.return_value = [
        {"name": "process_data", "signature": "def process_data(data)", "file": "main.py",
         "language": "python", "line": 10, "docstring": "Process raw data into records"},
        {"name": "authenticate_user", "signature": "def authenticate_user(username, password)",
         "file": "auth.py", "language": "python", "line": 5,
         "docstring": "Authenticate a user with credentials"},
        {"name": "send_email", "signature": "def send_email(to, subject, body)",
         "file": "services/email.py", "language": "python", "line": 1,
         "docstring": "Send an email notification"},
    ]
    graph.get_all_classes.return_value = [
        {"name": "DataProcessor", "base_classes": [], "file": "main.py",
         "language": "python", "line": 30, "docstring": "Batch data processor",
         "method_count": 3},
        {"name": "UserService", "base_classes": [], "file": "services/user.py",
         "language": "python", "line": 1, "docstring": "User management service"},
    ]
    graph.get_all_files.return_value = [
        {"path": "main.py", "function_count": 2, "class_count": 1,
         "import_count": 3, "language": "python"},
        {"path": "auth.py", "function_count": 1, "class_count": 0,
         "import_count": 2, "language": "python"},
        {"path": "services/email.py", "function_count": 1, "class_count": 0,
         "import_count": 1, "language": "python"},
        {"path": "services/user.py", "function_count": 0, "class_count": 1,
         "import_count": 1, "language": "python"},
    ]
    graph.get_importers.return_value = []
    graph.get_forward_imports.return_value = []
    graph.get_callers.return_value = []
    return graph


class TestGibberishScoring:
    def test_gibberish_query_scores_below_threshold(self):
        """Nonsense queries should have max score well below 0.5."""
        searcher = CodeSearcher(_make_mock_graph())
        results = searcher.search("xyzzy foobar qlrmph zbnkwt")
        if results:
            assert results[0]["score"] < 0.40, (
                f"Gibberish query scored {results[0]['score']:.2f}, expected < 0.40"
            )

    def test_meaningful_query_scores_higher_than_gibberish(self):
        """A real query should score significantly higher than gibberish."""
        searcher = CodeSearcher(_make_mock_graph())
        real_results = searcher.search("process data")
        gibberish_results = searcher.search("xyzzy foobar qlrmph zbnkwt")
        real_best = real_results[0]["score"] if real_results else 0
        gibberish_best = gibberish_results[0]["score"] if gibberish_results else 0
        assert real_best > gibberish_best + 0.15, (
            f"Real query ({real_best:.2f}) should score >0.15 higher than "
            f"gibberish ({gibberish_best:.2f})"
        )

    def test_partial_gibberish_penalized(self):
        """Queries where >50% of tokens are gibberish should be penalized."""
        searcher = CodeSearcher(_make_mock_graph())
        # 1 real token + 3 gibberish = 75% gibberish
        results = searcher.search("data xyzzy qlrmph zbnkwt")
        if results:
            # Should still find data-related results but at lower scores
            assert results[0]["score"] < 0.60, (
                f"Mostly-gibberish query scored {results[0]['score']:.2f}, expected < 0.60"
            )
