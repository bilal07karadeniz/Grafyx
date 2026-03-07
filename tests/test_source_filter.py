"""Tests for the Source Token Filter (M3) module."""

import numpy as np
import pytest

from grafyx.search._source_filter import (
    FEATURE_COUNT,
    _extract_token_features,
    filter_source_tokens,
)


# --- Sample function sources for testing ---

SAMPLE_SOURCE_SIMPLE = """\
def process_order(order_id, customer):
    \"\"\"Process a customer order.\"\"\"
    validated = validate(order_id)
    if validated:
        db.save(order_id, customer)
        return True
    return False
"""

SAMPLE_SOURCE_WITH_IMPORTS = """\
def fetch_data(url):
    import requests
    from urllib.parse import urlparse
    response = requests.get(url)
    return response.json()
"""

SAMPLE_SOURCE_WITH_GETATTR = """\
def __getattr__(name):
    \"\"\"Dynamic attribute access.\"\"\"
    if name == 'jwt':
        return get_jwt_handler()
    if name == 'auth':
        return get_auth_handler()
    raise AttributeError(name)
"""

SAMPLE_SOURCE_WITH_STRINGS = """\
def log_event(event_type):
    message = "Processing event for analytics pipeline"
    logger.info(message)
    db.insert(event_type)
"""

SAMPLE_SOURCE_WITH_DECORATOR = """\
@router.post("/orders")
@require_auth
def create_order(request):
    order = Order(request.data)
    order.save()
    return order
"""


class TestExtractTokenFeatures:
    """Test _extract_token_features returns correct shape and values."""

    def test_returns_correct_shape(self):
        """Feature vector should have exactly FEATURE_COUNT elements."""
        vec = _extract_token_features(
            "order", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (FEATURE_COUNT,)
        assert vec.dtype == np.float32

    def test_token_in_function_name(self):
        """Feature 0 should be 1.0 when token appears in function name."""
        vec = _extract_token_features(
            "process", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        assert vec[0] == 1.0

    def test_token_not_in_function_name(self):
        """Feature 0 should be 0.0 when token is NOT in function name."""
        vec = _extract_token_features(
            "validate", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        assert vec[0] == 0.0

    def test_token_in_docstring(self):
        """Feature 1 should be 1.0 when token appears in docstring."""
        vec = _extract_token_features(
            "customer", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
            function_docstring="Process a customer order.",
        )
        assert vec[1] == 1.0

    def test_token_not_in_docstring(self):
        """Feature 1 should be 0.0 when token is not in docstring."""
        vec = _extract_token_features(
            "validate", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
            function_docstring="Process a customer order.",
        )
        assert vec[1] == 0.0

    def test_token_in_param_names(self):
        """Feature 2 should be 1.0 when token appears in parameter names."""
        vec = _extract_token_features(
            "order_id", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        assert vec[2] == 1.0

    def test_token_in_import_statement(self):
        """Feature 4 should be 1.0 when token appears in import statement."""
        vec = _extract_token_features(
            "requests", "fetch_data", SAMPLE_SOURCE_WITH_IMPORTS, "api.py",
        )
        assert vec[4] == 1.0

    def test_token_not_in_import(self):
        """Feature 4 should be 0.0 when token is not in any import."""
        vec = _extract_token_features(
            "order", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        assert vec[4] == 0.0

    def test_token_in_string_literal(self):
        """Feature 5 should be 1.0 when token appears in a string literal."""
        vec = _extract_token_features(
            "analytics", "log_event", SAMPLE_SOURCE_WITH_STRINGS, "logging.py",
        )
        assert vec[5] == 1.0

    def test_token_in_getattr_body(self):
        """Feature 7 should be 1.0 when token is in a __getattr__ function."""
        vec = _extract_token_features(
            "jwt", "__getattr__", SAMPLE_SOURCE_WITH_GETATTR, "module.py",
        )
        assert vec[7] == 1.0

    def test_token_not_in_getattr(self):
        """Feature 7 should be 0.0 for normal functions."""
        vec = _extract_token_features(
            "order", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        assert vec[7] == 0.0

    def test_token_in_decorator(self):
        """Feature 3 should be 1.0 when token appears in a decorator."""
        vec = _extract_token_features(
            "router", "create_order", SAMPLE_SOURCE_WITH_DECORATOR, "routes.py",
        )
        assert vec[3] == 1.0

    def test_token_in_comment(self):
        """Feature 6 should be 1.0 when token appears in a comment."""
        source = "def foo():\n    x = 1  # process the order here\n    return x"
        vec = _extract_token_features("order", "foo", source, "foo.py")
        assert vec[6] == 1.0

    def test_stop_word_detection(self):
        """Feature 10 should be 1.0 for stop words."""
        vec = _extract_token_features(
            "the", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        assert vec[10] == 1.0

    def test_keyword_detection(self):
        """Feature 11 should be 1.0 for Python keywords."""
        vec = _extract_token_features(
            "return", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        assert vec[11] == 1.0

    def test_identifier_detection(self):
        """Feature 14 should be 1.0 for valid identifiers, 0.0 otherwise."""
        vec_ident = _extract_token_features(
            "order_id", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        assert vec_ident[14] == 1.0

        vec_non_ident = _extract_token_features(
            "123abc", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        assert vec_non_ident[14] == 0.0

    def test_source_line_count_normalized(self):
        """Feature 12 should scale with line count, capped at 1.0."""
        vec = _extract_token_features(
            "order", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        line_count = len(SAMPLE_SOURCE_SIMPLE.split('\n'))
        expected = min(line_count / 100.0, 1.0)
        assert abs(vec[12] - expected) < 1e-6

    def test_token_length_normalized(self):
        """Feature 9 should scale with token length, capped at 1.0."""
        vec = _extract_token_features(
            "order", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
        )
        assert abs(vec[9] - 5.0 / 20.0) < 1e-6  # "order" is 5 chars

    def test_empty_docstring_gives_zero(self):
        """Feature 1 should be 0.0 when no docstring is provided."""
        vec = _extract_token_features(
            "order", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
            function_docstring="",
        )
        assert vec[1] == 0.0

    def test_all_features_within_range(self):
        """All feature values should be in [0.0, 1.0]."""
        vec = _extract_token_features(
            "order", "process_order", SAMPLE_SOURCE_SIMPLE, "orders.py",
            function_docstring="Process a customer order.",
        )
        for i, val in enumerate(vec):
            assert 0.0 <= val <= 1.0, f"Feature {i} out of range: {val}"

    def test_token_in_name_vs_only_in_import(self):
        """Token in function name should have very different feature profile
        from token only in import statement."""
        # Token in name
        vec_name = _extract_token_features(
            "fetch", "fetch_data", SAMPLE_SOURCE_WITH_IMPORTS, "api.py",
        )
        # Token only in import
        vec_import = _extract_token_features(
            "urlparse", "fetch_data", SAMPLE_SOURCE_WITH_IMPORTS, "api.py",
        )
        # Name token has feature 0 set, import-only does not
        assert vec_name[0] == 1.0
        assert vec_import[0] == 0.0
        # Import-only token has feature 4 set
        assert vec_import[4] == 1.0


class TestFilterSourceTokens:
    """Test filter_source_tokens() public API."""

    def test_fallback_all_ones_when_no_model(self):
        """Without M3 model weights, all tokens should get weight 1.0."""
        # Since source_filter_weights.npz does not exist yet, model is None
        result = filter_source_tokens(
            ["order", "process", "validate"],
            "process_order",
            SAMPLE_SOURCE_SIMPLE,
            "orders.py",
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"order", "process", "validate"}
        for token, weight in result.items():
            assert weight == 1.0, f"Expected 1.0 for {token}, got {weight}"

    def test_returns_dict_with_all_tokens(self):
        """Result dict should contain an entry for every input token."""
        result = filter_source_tokens(
            ["order", "customer", "missing_token"],
            "process_order",
            SAMPLE_SOURCE_SIMPLE,
            "orders.py",
        )
        assert "order" in result
        assert "customer" in result
        assert "missing_token" in result

    def test_empty_query_tokens(self):
        """Empty token list should return empty dict."""
        result = filter_source_tokens(
            [],
            "process_order",
            SAMPLE_SOURCE_SIMPLE,
            "orders.py",
        )
        assert result == {}

    def test_with_docstring_parameter(self):
        """Should accept and pass through function_docstring parameter."""
        result = filter_source_tokens(
            ["order"],
            "process_order",
            SAMPLE_SOURCE_SIMPLE,
            "orders.py",
            function_docstring="Process a customer order.",
        )
        assert "order" in result
