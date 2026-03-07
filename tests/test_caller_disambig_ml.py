"""Tests for ML-based caller disambiguation (M2 integration).

Tests cover:
    1. Feature extraction (shape, dtype, individual feature correctness)
    2. Fallback to heuristic when M2 weights aren't available
    3. ML path activation when model is available (mocked)
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from grafyx.graph._caller_features import (
    FEATURE_COUNT,
    _char_bigrams,
    _split_tokens,
    extract_caller_features,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal mock graph with the indexes get_callers needs
# ---------------------------------------------------------------------------

def _make_mock_graph(**overrides):
    graph = MagicMock()
    graph._class_defined_in = overrides.get("class_defined_in", {})
    graph._forward_import_index = overrides.get("forward_import_index", {})
    graph._class_method_names = overrides.get("class_method_names", {})
    graph._file_class_methods = overrides.get("file_class_methods", {})
    graph._import_index = overrides.get("import_index", {})
    graph._caller_index = overrides.get("caller_index", {})
    return graph


# ===========================================================================
# Feature extraction unit tests
# ===========================================================================


class TestHelpers:
    """Test the private helper functions."""

    def test_char_bigrams_basic(self):
        assert _char_bigrams("ab") == {"ab"}
        assert _char_bigrams("abc") == {"ab", "bc"}

    def test_char_bigrams_strips_non_alpha(self):
        # "My_Class" -> "myclass" -> bigrams of "myclass"
        bg = _char_bigrams("My_Class")
        assert "my" in bg
        assert "cl" in bg

    def test_char_bigrams_short(self):
        assert _char_bigrams("a") == set()
        assert _char_bigrams("") == set()

    def test_split_tokens_camel_case(self):
        tokens = _split_tokens("MyDatabaseClass")
        assert "my" in tokens
        assert "database" in tokens
        assert "class" in tokens

    def test_split_tokens_snake_case(self):
        tokens = _split_tokens("my_database_class")
        assert "my" in tokens
        assert "database" in tokens

    def test_split_tokens_empty(self):
        assert _split_tokens("") == set()
        assert _split_tokens(None) == set()

    def test_split_tokens_short_parts_filtered(self):
        # Single char parts like "a" should be filtered (len < 2)
        tokens = _split_tokens("a_b_cd")
        assert "cd" in tokens
        assert "a" not in tokens
        assert "b" not in tokens


class TestCallerFeatureExtraction:
    """Test extract_caller_features produces correct feature vectors."""

    def test_feature_count_constant(self):
        assert FEATURE_COUNT == 25

    def test_feature_vector_shape_and_dtype(self):
        graph = _make_mock_graph(
            class_defined_in={"MyClass": {"/project/my.py"}},
            class_method_names={"MyClass": {"do_thing"}},
        )
        caller = {
            "name": "handler",
            "file": "/project/app.py",
            "_receivers": {"db"},
            "_has_dot_syntax": True,
        }
        vec = extract_caller_features(caller, "MyClass", "do_thing", graph)
        assert vec.shape == (FEATURE_COUNT,)
        assert vec.dtype == np.float32

    def test_all_zeros_for_minimal_caller(self):
        """A caller with no file, no receivers -> mostly zeros."""
        graph = _make_mock_graph()
        caller = {"name": "f"}
        vec = extract_caller_features(caller, "", "", graph)
        # Most features should be 0 (except feature 9 = 1.0 / 1 = 1.0 for uniqueness,
        # and feature 11 = 1.0 for callee_is_standalone since target_class_name is "")
        assert vec[7] == 0.0  # no dot syntax
        assert vec[8] == 0.0  # no self receiver
        assert vec[9] == 1.0  # 1/max(0,1) = 1.0 (no classes have this method)

    def test_dot_syntax_feature(self):
        graph = _make_mock_graph()
        caller_dot = {"name": "f", "file": "/a.py", "_has_dot_syntax": True, "_receivers": {"db"}}
        caller_no_dot = {"name": "f", "file": "/a.py", "_has_dot_syntax": False}

        vec_dot = extract_caller_features(caller_dot, "Cls", "m", graph)
        vec_no = extract_caller_features(caller_no_dot, "Cls", "m", graph)

        assert vec_dot[7] == 1.0  # has_dot_syntax
        assert vec_no[7] == 0.0

    def test_receiver_is_self(self):
        graph = _make_mock_graph()

        caller_self = {"name": "f", "file": "/a.py", "_receivers": {"self.db"}}
        caller_bare_self = {"name": "f", "file": "/a.py", "_receivers": {"self"}}
        caller_no_self = {"name": "f", "file": "/a.py", "_receivers": {"db"}}

        vec_self = extract_caller_features(caller_self, "Cls", "m", graph)
        vec_bare = extract_caller_features(caller_bare_self, "Cls", "m", graph)
        vec_no = extract_caller_features(caller_no_self, "Cls", "m", graph)

        assert vec_self[8] == 1.0
        assert vec_bare[8] == 1.0
        assert vec_no[8] == 0.0

    def test_method_uniqueness(self):
        graph = _make_mock_graph(
            class_method_names={
                "ClassA": {"process"},
                "ClassB": {"process"},
                "ClassC": {"process"},
            }
        )
        caller = {"name": "f", "file": "/a.py"}
        vec = extract_caller_features(caller, "ClassA", "process", graph)
        assert abs(vec[9] - 1.0 / 3.0) < 0.01  # 1 / 3 classes

    def test_method_uniqueness_single_class(self):
        graph = _make_mock_graph(
            class_method_names={"OnlyClass": {"unique_method"}}
        )
        caller = {"name": "f", "file": "/a.py"}
        vec = extract_caller_features(caller, "OnlyClass", "unique_method", graph)
        assert vec[9] == 1.0  # 1 / 1

    def test_callee_is_method_vs_standalone(self):
        graph = _make_mock_graph()

        vec_method = extract_caller_features({"name": "f"}, "MyClass", "do", graph)
        vec_standalone = extract_caller_features({"name": "f"}, "", "do", graph)

        assert vec_method[10] == 1.0  # is_method
        assert vec_method[11] == 0.0  # not standalone
        assert vec_standalone[10] == 0.0
        assert vec_standalone[11] == 1.0

    def test_caller_imports_callee_module(self):
        graph = _make_mock_graph(
            class_defined_in={"Database": {"/project/db.py"}},
            forward_import_index={"/project/api.py": ["/project/db.py"]},
        )
        caller = {"name": "handler", "file": "/project/api.py"}
        vec = extract_caller_features(caller, "Database", "execute", graph)
        assert vec[2] == 1.0  # imports the module

    def test_caller_does_not_import_callee(self):
        graph = _make_mock_graph(
            class_defined_in={"Database": {"/project/db.py"}},
            forward_import_index={"/project/api.py": ["/project/utils.py"]},
        )
        caller = {"name": "handler", "file": "/project/api.py"}
        vec = extract_caller_features(caller, "Database", "execute", graph)
        assert vec[2] == 0.0

    def test_same_directory(self):
        graph = _make_mock_graph(
            class_defined_in={"Cls": {"/project/src/cls.py"}},
        )
        caller_same = {"name": "f", "file": "/project/src/other.py"}
        caller_diff = {"name": "f", "file": "/project/tests/test.py"}

        vec_same = extract_caller_features(caller_same, "Cls", "m", graph)
        vec_diff = extract_caller_features(caller_diff, "Cls", "m", graph)

        assert vec_same[5] == 1.0
        assert vec_diff[5] == 0.0

    def test_file_path_distance(self):
        graph = _make_mock_graph(
            class_defined_in={"Cls": {"/project/src/models/cls.py"}},
        )
        # Same project, same src dir
        caller_close = {"name": "f", "file": "/project/src/views/view.py"}
        # Totally different path
        caller_far = {"name": "f", "file": "/other/lib/thing.py"}

        vec_close = extract_caller_features(caller_close, "Cls", "m", graph)
        vec_far = extract_caller_features(caller_far, "Cls", "m", graph)

        assert vec_close[4] > vec_far[4]  # closer path = higher score

    def test_receiver_token_overlap(self):
        graph = _make_mock_graph()

        # "user_service" receiver vs "UserService" class
        # tokens: {"user", "service"} overlap completely
        caller_match = {
            "name": "f", "file": "/a.py",
            "_receivers": {"user_service"}, "_has_dot_syntax": True,
        }
        vec_match = extract_caller_features(caller_match, "UserService", "do", graph)

        # "cache" receiver vs "UserService" class -> no overlap
        caller_no = {
            "name": "f", "file": "/a.py",
            "_receivers": {"cache"}, "_has_dot_syntax": True,
        }
        vec_no = extract_caller_features(caller_no, "UserService", "do", graph)

        assert vec_match[0] > vec_no[0]

    def test_receiver_char_bigram_similarity(self):
        graph = _make_mock_graph()

        # "db" receiver vs "Database" -> some bigram overlap ("da","at","ab"... vs "db")
        caller = {
            "name": "f", "file": "/a.py",
            "_receivers": {"database_conn"}, "_has_dot_syntax": True,
        }
        vec = extract_caller_features(caller, "Database", "execute", graph)
        assert vec[1] > 0.0  # some bigram overlap

    def test_trusted_entry_features(self):
        graph = _make_mock_graph()
        caller = {"name": "f", "file": "/a.py", "_trusted": True}
        vec = extract_caller_features(caller, "Cls", "m", graph)
        assert vec[13] == 1.0  # receiver_type_known
        assert vec[14] == 1.0  # receiver_type_matches

    def test_method_name_commonness(self):
        graph = _make_mock_graph(
            class_method_names={
                f"Class{i}": {"save"} for i in range(8)
            }
        )
        caller = {"name": "f", "file": "/a.py"}
        vec = extract_caller_features(caller, "Class0", "save", graph)
        assert abs(vec[19] - 0.8) < 0.01  # 8 / 10 = 0.8

    def test_receiver_is_common_pattern(self):
        graph = _make_mock_graph()

        caller_common = {"name": "f", "file": "/a.py", "_receivers": {"self.db"}}
        caller_uncommon = {"name": "f", "file": "/a.py", "_receivers": {"self.frobnicator"}}

        vec_common = extract_caller_features(caller_common, "Cls", "m", graph)
        vec_uncommon = extract_caller_features(caller_uncommon, "Cls", "m", graph)

        assert vec_common[24] == 1.0  # "db" is common
        assert vec_uncommon[24] == 0.0  # "frobnicator" not common

    def test_receiver_name_length(self):
        graph = _make_mock_graph()

        caller_short = {"name": "f", "file": "/a.py", "_receivers": {"db"}}
        caller_long = {"name": "f", "file": "/a.py", "_receivers": {"database_connection_pool"}}

        vec_short = extract_caller_features(caller_short, "Cls", "m", graph)
        vec_long = extract_caller_features(caller_long, "Cls", "m", graph)

        assert vec_short[18] < vec_long[18]

    def test_same_language_feature(self):
        graph = _make_mock_graph(
            class_defined_in={"Cls": {"/project/cls.py"}},
        )
        caller_py = {"name": "f", "file": "/project/app.py"}
        vec = extract_caller_features(caller_py, "Cls", "m", graph)
        assert vec[12] == 1.0  # both .py

    def test_empty_receivers_handled(self):
        """Callers with None or empty _receivers shouldn't crash."""
        graph = _make_mock_graph()

        caller_none = {"name": "f", "file": "/a.py", "_receivers": None}
        caller_empty = {"name": "f", "file": "/a.py", "_receivers": set()}
        caller_missing = {"name": "f", "file": "/a.py"}

        # None of these should raise
        vec1 = extract_caller_features(caller_none, "Cls", "m", graph)
        vec2 = extract_caller_features(caller_empty, "Cls", "m", graph)
        vec3 = extract_caller_features(caller_missing, "Cls", "m", graph)

        assert vec1.shape == (FEATURE_COUNT,)
        assert vec2.shape == (FEATURE_COUNT,)
        assert vec3.shape == (FEATURE_COUNT,)


# ===========================================================================
# Integration with get_callers: fallback and ML path tests
# ===========================================================================


class TestCallerDisambiguationFallback:
    """Test that get_callers falls back to heuristic when M2 not available."""

    def _make_graph_for_callers(self):
        """Build a mock graph suitable for calling get_callers()."""
        from grafyx.graph._callers import CallerQueryMixin

        graph = MagicMock(spec=CallerQueryMixin)
        graph._lock = MagicMock()
        graph._lock.__enter__ = MagicMock(return_value=None)
        graph._lock.__exit__ = MagicMock(return_value=False)
        graph._caller_index = {
            "refresh": [
                {
                    "name": "handler",
                    "file": "/api.py",
                    "_has_dot_syntax": True,
                    "_receivers": {"db"},
                },
            ]
        }
        graph._class_method_names = {
            "Database": {"refresh"},
            "Router": {"refresh"},
        }
        graph._file_class_methods = {}
        graph._class_defined_in = {"Database": {"/db.py"}}
        graph._import_index = {}
        graph._forward_import_index = {}
        graph._file_symbol_imports = {}

        # Bind the mixin methods
        graph.get_callers = CallerQueryMixin.get_callers.__get__(graph)
        graph._get_class_importer_files = CallerQueryMixin._get_class_importer_files.__get__(graph)
        return graph

    def test_fallback_to_heuristic_without_model(self):
        """When M2 weights don't exist, existing 4-level heuristic should work."""
        graph = self._make_graph_for_callers()

        with patch("grafyx.graph._callers.get_model", return_value=None):
            result = graph.get_callers("refresh", "Database")

        assert isinstance(result, list)

    def test_fallback_returns_filtered_results(self):
        """Heuristic path should still filter correctly."""
        graph = self._make_graph_for_callers()

        # Add a caller from a file that imports /db.py (should pass Level 3)
        graph._caller_index["refresh"].append({
            "name": "importer_func",
            "file": "/consumer.py",
            "_has_dot_syntax": True,
            "_receivers": {"database"},
        })
        graph._import_index = {"/db.py": ["/consumer.py"]}
        graph._forward_import_index = {"/consumer.py": ["/db.py"]}

        with patch("grafyx.graph._callers.get_model", return_value=None):
            result = graph.get_callers("refresh", "Database")

        # importer_func from /consumer.py should be kept (imports the module)
        names = [r["name"] for r in result]
        assert "importer_func" in names

    def test_no_class_name_returns_all(self):
        """Without class_name, all callers returned unfiltered."""
        graph = self._make_graph_for_callers()

        with patch("grafyx.graph._callers.get_model", return_value=None):
            result = graph.get_callers("refresh")

        assert len(result) == 1
        assert result[0]["name"] == "handler"


class TestCallerDisambiguationML:
    """Test the ML path when model IS available."""

    def _make_graph_for_callers(self):
        from grafyx.graph._callers import CallerQueryMixin

        graph = MagicMock(spec=CallerQueryMixin)
        graph._lock = MagicMock()
        graph._lock.__enter__ = MagicMock(return_value=None)
        graph._lock.__exit__ = MagicMock(return_value=False)
        graph._caller_index = {
            "execute": [
                {
                    "name": "real_caller",
                    "file": "/api.py",
                    "_has_dot_syntax": True,
                    "_receivers": {"db"},
                },
                {
                    "name": "false_positive",
                    "file": "/tool.py",
                    "_has_dot_syntax": True,
                    "_receivers": {"executor"},
                },
            ]
        }
        graph._class_method_names = {
            "Database": {"execute"},
            "ToolExecutor": {"execute"},
        }
        graph._file_class_methods = {}
        graph._class_defined_in = {"Database": {"/db.py"}}
        graph._import_index = {}
        graph._forward_import_index = {}
        graph._file_symbol_imports = {}

        graph.get_callers = CallerQueryMixin.get_callers.__get__(graph)
        graph._get_class_importer_files = CallerQueryMixin._get_class_importer_files.__get__(graph)
        return graph

    def test_ml_path_filters_by_score(self):
        """ML model should filter out low-scoring callers."""
        graph = self._make_graph_for_callers()

        mock_model = MagicMock()
        # First caller scores high, second scores low
        mock_model.predict = MagicMock(side_effect=[0.85, 0.15])

        with patch("grafyx.graph._callers.get_model", return_value=mock_model):
            result = graph.get_callers("execute", "Database")

        names = [r["name"] for r in result]
        assert "real_caller" in names
        assert "false_positive" not in names

    def test_ml_path_keeps_trusted_entries(self):
        """Trusted entries should bypass ML scoring."""
        graph = self._make_graph_for_callers()

        # Make second caller trusted
        graph._caller_index["execute"][1]["_trusted"] = True

        mock_model = MagicMock()
        # Only called for the non-trusted caller
        mock_model.predict = MagicMock(return_value=0.1)  # low score

        with patch("grafyx.graph._callers.get_model", return_value=mock_model):
            result = graph.get_callers("execute", "Database")

        names = [r["name"] for r in result]
        # trusted entry should be kept regardless of score
        assert "false_positive" in names
        # non-trusted with score 0.1 should be filtered
        assert "real_caller" not in names

    def test_ml_path_keeps_same_class_callers(self):
        """Callers from the same class should be kept without ML scoring."""
        graph = self._make_graph_for_callers()

        # Make first caller part of Database class
        graph._caller_index["execute"][0]["class"] = "Database"

        mock_model = MagicMock()
        # Only called for the second caller
        mock_model.predict = MagicMock(return_value=0.1)

        with patch("grafyx.graph._callers.get_model", return_value=mock_model):
            result = graph.get_callers("execute", "Database")

        names = [r["name"] for r in result]
        assert "real_caller" in names  # same class -> kept
        assert "false_positive" not in names  # low score -> filtered

    def test_ml_model_predict_called_with_features(self):
        """Verify that model.predict receives a numpy array of correct shape."""
        graph = self._make_graph_for_callers()

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=0.5)

        with patch("grafyx.graph._callers.get_model", return_value=mock_model):
            graph.get_callers("execute", "Database")

        # predict should have been called for each non-same-class caller
        assert mock_model.predict.call_count == 2
        # Check that the argument is a numpy array with correct shape
        for call_args in mock_model.predict.call_args_list:
            features = call_args[0][0]
            assert isinstance(features, np.ndarray)
            assert features.shape == (FEATURE_COUNT,)
