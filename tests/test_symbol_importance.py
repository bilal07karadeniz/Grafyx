"""Tests for Symbol Importance (M4) integration in hints module."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from grafyx.server._hints import (
    _extract_importance_features,
    _score_symbol_importance,
    IMPORTANCE_FEATURE_COUNT,
)


def _make_mock_graph(
    caller_index=None,
    import_index=None,
    class_method_names=None,
    class_defined_in=None,
):
    """Create a mock graph with configurable indexes."""
    graph = MagicMock()
    graph._caller_index = caller_index or {}
    graph._import_index = import_index or {}
    graph._class_method_names = class_method_names or {}
    graph._class_defined_in = class_defined_in or {}
    return graph


class TestExtractImportanceFeatures:
    """Test _extract_importance_features returns correct shape and values."""

    def test_returns_correct_shape(self):
        """Feature vector should have exactly IMPORTANCE_FEATURE_COUNT elements."""
        graph = _make_mock_graph()
        vec = _extract_importance_features("process_order", "orders.py", "function", graph)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (IMPORTANCE_FEATURE_COUNT,)
        assert vec.dtype == np.float32

    def test_caller_count_feature(self):
        """Feature 0 should reflect caller count, normalized."""
        graph = _make_mock_graph(
            caller_index={
                "process_order": [
                    {"name": f"caller_{i}", "file": f"file_{i}.py"}
                    for i in range(10)
                ],
            }
        )
        vec = _extract_importance_features("process_order", "orders.py", "function", graph)
        # 10 callers / 20.0 = 0.5
        assert abs(vec[0] - 0.5) < 1e-6

    def test_cross_file_caller_count(self):
        """Feature 1 should count unique files of callers."""
        graph = _make_mock_graph(
            caller_index={
                "process_order": [
                    {"name": "caller_a", "file": "routes.py"},
                    {"name": "caller_b", "file": "jobs.py"},
                    {"name": "caller_c", "file": "routes.py"},  # duplicate file
                ],
            }
        )
        vec = _extract_importance_features("process_order", "orders.py", "function", graph)
        # 2 unique files / 10.0 = 0.2
        assert abs(vec[1] - 0.2) < 1e-6

    def test_api_indicator_feature(self):
        """Feature 3 should detect API-related file paths."""
        graph = _make_mock_graph()
        vec = _extract_importance_features(
            "handle_request", "api/routes.py", "function", graph,
        )
        assert vec[3] == 1.0

    def test_non_api_file(self):
        """Feature 3 should be 0.0 for non-API files."""
        graph = _make_mock_graph()
        vec = _extract_importance_features(
            "process_order", "services/orders.py", "function", graph,
        )
        assert vec[3] == 0.0

    def test_entry_point_detection(self):
        """Feature 4 should detect main/entry point functions."""
        graph = _make_mock_graph()
        vec_main = _extract_importance_features("main", "app.py", "function", graph)
        assert vec_main[4] == 1.0

        vec_other = _extract_importance_features("process", "app.py", "function", graph)
        assert vec_other[4] == 0.0

    def test_import_count_feature(self):
        """Feature 9 should reflect how many files import this file."""
        graph = _make_mock_graph(
            import_index={
                "orders.py": ["routes.py", "jobs.py", "tests/test_orders.py"],
            }
        )
        vec = _extract_importance_features("process_order", "orders.py", "function", graph)
        # 3 importers / 20.0 = 0.15
        assert abs(vec[9] - 0.15) < 1e-6

    def test_class_method_count_feature(self):
        """Feature 12 should reflect method count for classes."""
        graph = _make_mock_graph(
            class_method_names={
                "OrderService": {"process_order", "cancel_order", "get_order"},
            }
        )
        vec = _extract_importance_features(
            "OrderService", "orders.py", "class", graph,
        )
        # 3 methods / 20.0 = 0.15
        assert abs(vec[12] - 0.15) < 1e-6

    def test_test_function_detection(self):
        """Feature 15 should detect test files and test functions."""
        graph = _make_mock_graph()

        # Test file path
        vec_test_file = _extract_importance_features(
            "process_order", "tests/test_orders.py", "function", graph,
        )
        assert vec_test_file[15] == 1.0

        # Test function name
        vec_test_func = _extract_importance_features(
            "test_process_order", "orders.py", "function", graph,
        )
        assert vec_test_func[15] == 1.0

        # Non-test
        vec_normal = _extract_importance_features(
            "process_order", "services/orders.py", "function", graph,
        )
        assert vec_normal[15] == 0.0

    def test_file_depth_feature(self):
        """Feature 16 should reflect path depth."""
        graph = _make_mock_graph()
        vec = _extract_importance_features(
            "foo", "a/b/c/d.py", "function", graph,
        )
        # 4 parts / 8.0 = 0.5
        assert abs(vec[16] - 0.5) < 1e-6

    def test_name_length_feature(self):
        """Feature 17 should reflect symbol name length."""
        graph = _make_mock_graph()
        vec = _extract_importance_features(
            "process_order", "orders.py", "function", graph,
        )
        # 13 chars / 30.0 ≈ 0.4333
        assert abs(vec[17] - 13.0 / 30.0) < 1e-5

    def test_all_features_within_range(self):
        """All feature values should be in [0.0, 1.0]."""
        graph = _make_mock_graph(
            caller_index={
                "process_order": [
                    {"name": f"caller_{i}", "file": f"file_{i}.py"}
                    for i in range(5)
                ],
            },
            import_index={
                "orders.py": ["routes.py", "jobs.py"],
            },
            class_method_names={
                "OrderService": {"method_a", "method_b"},
            },
        )
        vec = _extract_importance_features("process_order", "orders.py", "function", graph)
        for i, val in enumerate(vec):
            assert 0.0 <= val <= 1.0, f"Feature {i} out of range: {val}"

    def test_high_importance_vs_low_importance(self):
        """Symbols with many callers should have higher feature values
        than isolated symbols."""
        graph = _make_mock_graph(
            caller_index={
                "core_function": [
                    {"name": f"caller_{i}", "file": f"file_{i}.py"}
                    for i in range(15)
                ],
            },
            import_index={
                "core.py": [f"mod_{i}.py" for i in range(10)],
            },
        )

        vec_high = _extract_importance_features("core_function", "core.py", "function", graph)
        vec_low = _extract_importance_features("helper", "utils.py", "function", graph)

        # High-importance symbol should have higher caller and import features
        assert vec_high[0] > vec_low[0]  # caller_count
        assert vec_high[9] > vec_low[9]  # import_count

    def test_empty_symbol_file(self):
        """Should handle empty file path gracefully."""
        graph = _make_mock_graph()
        vec = _extract_importance_features("foo", "", "function", graph)
        assert vec.shape == (IMPORTANCE_FEATURE_COUNT,)
        assert vec[16] == 0.0  # file depth should be 0


class TestScoreSymbolImportance:
    """Test _score_symbol_importance with heuristic fallback."""

    def test_fallback_uses_counts(self):
        """Without M4 model, should use caller + import count heuristic."""
        graph = _make_mock_graph(
            caller_index={
                "process_order": [
                    {"name": "caller_a", "file": "routes.py"},
                    {"name": "caller_b", "file": "jobs.py"},
                ],
            },
            import_index={
                "orders.py": ["routes.py", "jobs.py", "tests.py"],
            },
        )
        score = _score_symbol_importance("process_order", "orders.py", "function", graph)
        # (2 callers + 3 importers) / 50.0 = 0.10
        assert abs(score - 0.10) < 1e-6

    def test_fallback_capped_at_one(self):
        """Heuristic score should be capped at 1.0."""
        graph = _make_mock_graph(
            caller_index={
                "core": [{"name": f"c{i}", "file": f"f{i}.py"} for i in range(40)],
            },
            import_index={
                "core.py": [f"mod_{i}.py" for i in range(20)],
            },
        )
        score = _score_symbol_importance("core", "core.py", "function", graph)
        # (40 + 20) / 50.0 = 1.2, should be capped at 1.0
        assert score == 1.0

    def test_fallback_zero_for_isolated_symbol(self):
        """Isolated symbols with no callers or importers should score 0."""
        graph = _make_mock_graph()
        score = _score_symbol_importance("orphan_func", "orphan.py", "function", graph)
        assert score == 0.0

    def test_returns_float(self):
        """Score should always be a float."""
        graph = _make_mock_graph()
        score = _score_symbol_importance("foo", "bar.py", "function", graph)
        assert isinstance(score, float)

    def test_score_within_range(self):
        """Score should be in [0.0, 1.0]."""
        graph = _make_mock_graph(
            caller_index={
                "func": [{"name": "a", "file": "b.py"}],
            },
        )
        score = _score_symbol_importance("func", "file.py", "function", graph)
        assert 0.0 <= score <= 1.0
