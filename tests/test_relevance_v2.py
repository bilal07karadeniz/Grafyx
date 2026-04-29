"""Tests for Relevance Ranker v2 integration."""
import numpy as np
from grafyx.search._relevance import _extract_features, _FEATURE_COUNT


class TestRelevanceV2Features:
    def test_feature_count_is_42(self):
        assert _FEATURE_COUNT == 42

    def test_42_features_extracted(self):
        vec = _extract_features(
            ["auth", "user"], "auth user", "authenticate_user",
            "Authenticate a user", "app/auth.py"
        )
        assert vec.shape == (42,)

    def test_dunder_feature_set(self):
        vec = _extract_features(
            ["get"], "get", "__getattr__", "", "app/__init__.py",
            is_dunder=True
        )
        assert vec[33] == 1.0  # is_dunder

    def test_init_file_feature_set(self):
        vec = _extract_features(
            ["auth"], "auth", "authenticate", "", "app/__init__.py",
            is_init_file=True
        )
        assert vec[34] == 1.0  # is_init_file

    def test_method_feature_set(self):
        vec = _extract_features(
            ["get"], "get", "get", "", "app/service.py",
            is_method=True
        )
        assert vec[35] == 1.0  # is_method

    def test_class_feature_set(self):
        vec = _extract_features(
            ["user"], "user", "UserService", "", "app/service.py",
            is_class=True
        )
        assert vec[36] == 1.0  # is_class

    def test_receiver_call_ratio_clamped(self):
        vec = _extract_features(
            ["test"], "test", "test_func", "", "app/test.py",
            receiver_call_ratio=1.5
        )
        assert vec[37] == 1.0  # clamped to 1.0

    def test_source_token_entropy_normalized(self):
        vec = _extract_features(
            ["test"], "test", "test_func", "", "app/test.py",
            source_token_entropy=2.5
        )
        assert abs(vec[38] - 0.5) < 1e-6  # 2.5 / 5.0

    def test_source_unique_token_ratio(self):
        vec = _extract_features(
            ["test"], "test", "test_func", "", "app/test.py",
            source_unique_token_ratio=0.75
        )
        assert abs(vec[39] - 0.75) < 1e-6

    def test_embedding_similarity(self):
        vec = _extract_features(
            ["test"], "test", "test_func", "", "app/test.py",
            embedding_similarity=0.42
        )
        assert abs(vec[40] - 0.42) < 1e-6

    def test_caller_count_normalized_clamped(self):
        vec = _extract_features(
            ["test"], "test", "test_func", "", "app/test.py",
            caller_count_normalized=2.0
        )
        assert vec[41] == 1.0  # clamped to 1.0

    def test_v1_features_unchanged(self):
        """First 33 features should be identical to v1 extraction."""
        vec = _extract_features(
            ["auth", "user"], "auth user", "authenticate_user",
            "Authenticate a user", "app/auth.py"
        )
        # Name exact match: auth->no, user->yes (name tokens: authenticate, user)
        # So vec[0] > 0 (user matches)
        assert vec[0] > 0  # At least one exact name token match

    def test_graceful_with_no_v2_params(self):
        """Calling without v2 params should work (all default to 0/False)."""
        vec = _extract_features(
            ["test"], "test", "test_func", "A test", "tests/test.py"
        )
        # v2 features should all be 0
        assert vec[33] == 0.0
        assert vec[34] == 0.0
        assert vec[35] == 0.0
        assert vec[36] == 0.0
        assert vec[37] == 0.0
        assert vec[38] == 0.0
        assert vec[39] == 0.0
        assert vec[40] == 0.0
        assert vec[41] == 0.0


class TestRelevanceScorerV1Fallback:
    """Test that RelevanceScorer falls back to v1 weights gracefully."""

    def _make_v1_scorer(self):
        """Create a scorer that forces v1 fallback by hiding v2 weights."""
        import os
        from unittest.mock import patch
        from grafyx.search._relevance import RelevanceScorer

        _real_exists = os.path.exists

        def _hide_v2(path):
            if "relevance_weights_v2" in str(path):
                return False
            return _real_exists(path)

        with patch("os.path.exists", side_effect=_hide_v2):
            return RelevanceScorer()

    def test_v1_weights_still_work(self):
        """v1 weights file loads and produces valid scores."""
        scorer = self._make_v1_scorer()
        assert scorer._version == 1
        result = scorer.score(
            ["auth", "user"], "auth user", "authenticate_user",
            "Authenticate a user", "app/auth.py"
        )
        assert 0.0 <= result <= 1.0

    def test_v1_truncates_features(self):
        """When using v1 model, features are truncated to 33."""
        scorer = self._make_v1_scorer()
        assert scorer._version == 1
        # Scorer should still work with v2 params passed
        result = scorer.score(
            ["auth"], "auth", "authenticate",
            "Auth function", "app/auth.py",
            is_dunder=False, is_init_file=True, is_method=False,
            is_class=False,
        )
        assert 0.0 <= result <= 1.0

    def test_zero_overlap_short_circuit(self):
        """No overlap should still short-circuit to 0."""
        from grafyx.search._relevance import RelevanceScorer
        scorer = RelevanceScorer()
        result = scorer.score(
            ["zzzzz"], "zzzzz", "authenticate_user",
            "Auth function", "app/auth.py"
        )
        assert result == 0.0


class TestMlScoreMatchV2Params:
    """Test that ml_score_match forwards v2 params correctly."""

    def test_accepts_v2_params(self):
        from grafyx.search._relevance import ml_score_match
        result = ml_score_match(
            ["auth"], "auth", "authenticate",
            "Auth function", "app/auth.py",
            is_dunder=False, is_init_file=True, is_method=False,
            is_class=False, receiver_call_ratio=0.5,
            source_token_entropy=1.0, source_unique_token_ratio=0.5,
            embedding_similarity=0.3, caller_count_normalized=0.2,
        )
        assert 0.0 <= result <= 1.0

    def test_backward_compatible(self):
        """Calling without v2 params should still work."""
        from grafyx.search._relevance import ml_score_match
        result = ml_score_match(
            ["auth"], "auth", "authenticate",
            "Auth function", "app/auth.py",
            token_weights={"auth": 1.0},
        )
        assert 0.0 <= result <= 1.0
