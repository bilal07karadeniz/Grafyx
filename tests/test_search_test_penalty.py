"""Tests for test file scoring penalty in search.

Production files should score higher than test files with equivalent
name/docstring matches, because users searching for functionality
typically want the implementation, not the test.
"""

from grafyx.search._scoring import ScoringMixin


class TestTestFilePenalty:
    """Test files should receive a scoring penalty vs production files."""

    def setup_method(self):
        self.scorer = ScoringMixin()

    def test_production_file_scores_higher(self):
        """Same function name in test vs production file: production wins."""
        prod_score = self.scorer._score_match(
            query_tokens=["webrtc", "client"],
            query_lower="webrtc client",
            name="WebRTCClient",
            docstring="WebRTC client for voice communication",
            file_path="backend/app/voice/webrtc_client.py",
        )
        test_score = self.scorer._score_match(
            query_tokens=["webrtc", "client"],
            query_lower="webrtc client",
            name="WebRTCClient",
            docstring="WebRTC client for voice communication",
            file_path="load_tests/webrtc_client.py",
        )
        assert prod_score > test_score, (
            f"Production score ({prod_score:.3f}) should be > test score ({test_score:.3f})"
        )

    def test_test_directory_penalized(self):
        """Files in tests/ directory should get penalized."""
        prod_score = self.scorer._score_match(
            query_tokens=["auth", "handler"],
            query_lower="auth handler",
            name="authenticate",
            docstring="Handle authentication",
            file_path="src/auth/handler.py",
        )
        test_score = self.scorer._score_match(
            query_tokens=["auth", "handler"],
            query_lower="auth handler",
            name="authenticate",
            docstring="Handle authentication",
            file_path="tests/test_auth/test_handler.py",
        )
        assert prod_score > test_score

    def test_non_test_files_unpenalized(self):
        """Non-test files in various directories should not be penalized."""
        score1 = self.scorer._score_match(
            query_tokens=["process", "data"],
            query_lower="process data",
            name="process_data",
            docstring="Process incoming data",
            file_path="src/processing/data.py",
        )
        score2 = self.scorer._score_match(
            query_tokens=["process", "data"],
            query_lower="process data",
            name="process_data",
            docstring="Process incoming data",
            file_path="lib/processing/data.py",
        )
        # Both non-test, should score identically
        assert abs(score1 - score2) < 0.01
