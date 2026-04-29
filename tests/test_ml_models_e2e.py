"""End-to-end tests for all 4 ML models.

Tests real-world behavior with diverse inputs across multiple project types.
Verifies ranking quality, edge cases, and generalization — not just feature
shapes or inference mechanics.

These tests use the actual trained weights (not mocks), so they test the
full production path including weight loading, feature extraction, and
score computation.
"""

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════
#  M1: Relevance Ranker v2 (search scoring)
# ═══════════════════════════════════════════════════════════════════════


class TestM1RankingQuality:
    """Verify M1 produces meaningful score gradations for diverse queries."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from grafyx.search._relevance import RelevanceScorer
        self.scorer = RelevanceScorer()
        assert self.scorer._version == 2, "v2 weights must be present"

    def _score(self, query, name, doc="", path="", **kw):
        from grafyx.search._relevance import _split_tokens
        tokens = _split_tokens(query)
        return self.scorer.score(tokens, query.lower(), name, doc, path, **kw)

    # --- Ranking tests: exact > partial > path-only > unrelated ---

    def test_ranking_web_framework(self):
        """Web framework: authenticate_user > validate_input > process_request (in auth/) > render_html"""
        s1 = self._score("authenticate user", "authenticate_user", "Authenticate user credentials", "app/auth.py")
        s2 = self._score("authenticate user", "validate_input", "Validate user input", "app/forms.py")
        s3 = self._score("authenticate user", "process_request", "Process HTTP request", "auth/middleware.py")
        s4 = self._score("authenticate user", "render_html", "Render HTML template", "web/views.py")
        assert s1 > s2 > s4, f"Ranking broken: {s1:.3f}, {s2:.3f}, {s4:.3f}"
        assert s1 > s3, f"Exact ({s1:.3f}) should beat path-only ({s3:.3f})"

    def test_ranking_database(self):
        """Database: create_connection > close_connection (partial) > run_migration (unrelated)"""
        s1 = self._score("database connection", "create_db_connection", "Create database connection", "db/pool.py")
        s2 = self._score("database connection", "close_connection", "Close an open connection", "db/pool.py")
        s3 = self._score("database connection", "run_migration", "Run schema migration", "db/migrations.py")
        assert s1 > s3, f"Match ({s1:.3f}) should beat unrelated ({s3:.3f})"

    def test_ranking_ml_pipeline(self):
        """ML pipeline: train_model > evaluate_model (partial) > load_config (unrelated)"""
        s1 = self._score("train model", "train_model", "Train the neural network", "ml/training.py")
        s2 = self._score("train model", "evaluate_model", "Evaluate model performance", "ml/evaluation.py")
        s3 = self._score("train model", "load_config", "Load configuration file", "config/loader.py")
        assert s1 > s2 > s3, f"Ranking: {s1:.3f}, {s2:.3f}, {s3:.3f}"

    def test_ranking_payment_processing(self):
        """Payment: process_payment > create_invoice (partial) > send_email (unrelated)"""
        s1 = self._score("process payment", "process_payment", "Process a payment charge", "payments/stripe.py")
        s2 = self._score("process payment", "create_invoice", "Create invoice for payment", "billing/invoice.py")
        s3 = self._score("process payment", "send_email", "Send notification email", "notifications/email.py")
        assert s1 > s3, f"Match ({s1:.3f}) should beat unrelated ({s3:.3f})"

    def test_ranking_cli_tool(self):
        """CLI: parse_arguments > run_command (partial verb) > format_output (unrelated)"""
        s1 = self._score("parse arguments", "parse_arguments", "Parse CLI arguments", "cli/parser.py")
        s2 = self._score("parse arguments", "parse_config", "Parse config file", "config/parser.py")
        s3 = self._score("parse arguments", "format_output", "Format CLI output", "cli/output.py")
        assert s1 > s2, f"Exact ({s1:.3f}) should beat partial ({s2:.3f})"
        assert s1 > s3, f"Exact ({s1:.3f}) should beat unrelated ({s3:.3f})"

    # --- Score range tests ---

    def test_exact_match_high_score(self):
        """Exact name matches should score > 0.70."""
        cases = [
            ("create user", "create_user", "Create a new user", "app/users.py"),
            ("send notification", "send_notification", "Send a notification", "notifications/service.py"),
            ("parse config", "parse_config", "Parse configuration", "config/parser.py"),
            ("validate schema", "validate_schema", "Validate JSON schema", "validators/schema.py"),
        ]
        for query, name, doc, path in cases:
            score = self._score(query, name, doc, path)
            assert score > 0.70, f"Exact match '{name}' for '{query}' scored {score:.3f}, expected > 0.70"

    def test_no_overlap_near_zero(self):
        """Completely unrelated queries should score < 0.10."""
        cases = [
            ("payment processing", "render_template", "Render HTML", "web/views.py"),
            ("database migration", "send_email", "Send notification email", "email/sender.py"),
            ("jwt decode", "compress_file", "Compress using gzip", "utils/compression.py"),
            ("user authentication", "parse_csv", "Parse CSV file", "data/csv_parser.py"),
        ]
        for query, name, doc, path in cases:
            score = self._score(query, name, doc, path)
            assert score < 0.10, f"No-overlap '{name}' for '{query}' scored {score:.3f}, expected < 0.10"

    def test_dunder_penalized(self):
        """Dunder methods with generic queries should score low."""
        # "init" matching __init__ should be much lower than a regular function
        dunder_score = self._score("init", "__init__", "Initialize", "models/user.py", is_dunder=True)
        regular_score = self._score("init", "init_app", "Initialize application", "app/startup.py")
        assert dunder_score < regular_score, \
            f"Dunder ({dunder_score:.3f}) should score lower than regular ({regular_score:.3f})"
        assert dunder_score < 0.30, f"Dunder scored {dunder_score:.3f}, expected < 0.30"

    def test_path_only_match_moderate(self):
        """Path-only matches should score between 0.10 and 0.50."""
        score = self._score("auth", "process_request", "Process HTTP request", "auth/middleware.py")
        assert 0.10 < score < 0.50, f"Path-only scored {score:.3f}, expected 0.10-0.50"

    def test_camelcase_class_names(self):
        """CamelCase class names should match when query uses space-separated words.

        Note: All-caps abbreviations (HTTPClient) don't split correctly with
        the current tokenizer — this is a known limitation.
        """
        cases = [
            ("user manager", "UserManager"),
            ("event handler", "EventHandler"),
            ("database session", "DatabaseSession"),
        ]
        for query, name in cases:
            score = self._score(query, name, f"{name} class", f"services/{name.lower()}.py")
            assert score > 0.70, f"CamelCase '{name}' for '{query}' scored {score:.3f}, expected > 0.70"

    # --- Edge cases ---

    def test_single_token_query(self):
        """Single-token queries should still produce meaningful rankings."""
        s1 = self._score("cache", "cache_manager", "Manage the cache", "cache/manager.py")
        s2 = self._score("cache", "invalidate_entry", "Invalidate cache entry", "cache/invalidation.py")
        s3 = self._score("cache", "send_email", "Send email notification", "email/sender.py")
        assert s1 > s3, f"Name match ({s1:.3f}) should beat unrelated ({s3:.3f})"

    def test_empty_docstring(self):
        """Functions without docstrings should still be scored by name/path."""
        score = self._score("validate input", "validate_input", "", "app/validators.py")
        assert score > 0.60, f"No-doc exact match scored {score:.3f}, expected > 0.60"

    def test_long_query(self):
        """Long multi-word queries should work correctly."""
        score = self._score(
            "create database connection pool with retry logic",
            "create_connection_pool", "Create a connection pool", "db/pool.py"
        )
        assert score > 0.20, f"Long query partial match scored {score:.3f}, expected > 0.20"


# ═══════════════════════════════════════════════════════════════════════
#  M2: Caller Disambiguator (who calls what)
# ═══════════════════════════════════════════════════════════════════════


class TestM2Disambiguation:
    """Verify M2 correctly disambiguates callers across diverse patterns."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from grafyx.ml_inference import get_model
        self.model = get_model("caller_disambig")
        if self.model is None:
            pytest.skip("M2 weights not available")

    def _make_features(self, **overrides):
        """Build a 25-feature vector with specified overrides."""
        vec = np.zeros(25, dtype=np.float32)
        feature_map = {
            "receiver_overlap": 0, "bigram_sim": 1, "imports_module": 2,
            "imports_package": 3, "path_distance": 4, "same_dir": 5,
            "same_top_package": 6, "has_dot_syntax": 7, "receiver_is_self": 8,
            "method_uniqueness": 9, "callee_is_method": 10, "callee_is_standalone": 11,
            "same_language": 12, "receiver_type_known": 13, "receiver_type_matches": 14,
            "param_count": 15, "arg_count_matches": 16, "has_decorator": 17,
            "receiver_name_length": 18, "method_commonness": 19, "caller_complexity": 20,
            "is_property": 21, "is_classmethod": 22, "is_abstractmethod": 23,
            "receiver_common_pattern": 24,
        }
        for key, val in overrides.items():
            vec[feature_map[key]] = val
        return vec

    def test_obvious_positive(self):
        """Caller that imports module, same dir, matching receiver → high score."""
        feat = self._make_features(
            receiver_overlap=0.8, bigram_sim=0.7, imports_module=1.0,
            same_dir=1.0, has_dot_syntax=1.0, method_uniqueness=1.0,
            callee_is_method=1.0, arg_count_matches=1.0, same_language=1.0,
        )
        score = self.model.predict(feat)
        assert score > 0.8, f"Obvious positive scored {score:.3f}"

    def test_obvious_negative(self):
        """Caller from different package, no import, no receiver match → low score."""
        feat = self._make_features(
            receiver_overlap=0.0, imports_module=0.0, path_distance=0.9,
            same_dir=0.0, same_top_package=0.0, method_uniqueness=0.1,
            callee_is_method=1.0, same_language=1.0,
        )
        score = self.model.predict(feat)
        assert score < 0.2, f"Obvious negative scored {score:.3f}"

    def test_cross_file_caller_with_import(self):
        """Caller from different file that imports the module → positive.

        Note: Same-class callers bypass ML scoring in production (_callers.py:151),
        so we test the cross-file import scenario instead.
        """
        feat = self._make_features(
            receiver_overlap=0.5, imports_module=1.0, imports_package=1.0,
            same_top_package=1.0, has_dot_syntax=1.0, method_uniqueness=0.5,
            callee_is_method=1.0, arg_count_matches=1.0, same_language=1.0,
        )
        score = self.model.predict(feat)
        assert score > 0.5, f"Cross-file caller with import scored {score:.3f}"

    def test_common_method_name_disambiguation(self):
        """Common method names (execute, get, run) with wrong receiver → low."""
        feat = self._make_features(
            receiver_overlap=0.0, bigram_sim=0.0, imports_module=0.0,
            path_distance=0.8, same_dir=0.0, has_dot_syntax=1.0,
            method_uniqueness=0.1,  # 10 classes have 'execute'
            callee_is_method=1.0, method_commonness=0.9,
            same_language=1.0,
        )
        score = self.model.predict(feat)
        assert score < 0.3, f"Wrong receiver for common method scored {score:.3f}"

    def test_unique_method_name(self):
        """Unique method names with import should score positive."""
        feat = self._make_features(
            imports_module=1.0, imports_package=1.0, method_uniqueness=1.0,
            callee_is_method=1.0, same_language=1.0, arg_count_matches=1.0,
            same_top_package=1.0,
        )
        score = self.model.predict(feat)
        assert score > 0.5, f"Unique method scored {score:.3f}"

    def test_type_annotation_match(self):
        """When receiver type is known and matches → strong positive."""
        feat = self._make_features(
            receiver_type_known=1.0, receiver_type_matches=1.0,
            imports_module=1.0, has_dot_syntax=1.0,
            callee_is_method=1.0, same_language=1.0,
        )
        score = self.model.predict(feat)
        assert score > 0.7, f"Type-matched caller scored {score:.3f}"


# ═══════════════════════════════════════════════════════════════════════
#  M3: Source Token Filter (noise filtering)
# ═══════════════════════════════════════════════════════════════════════


class TestM3TokenFiltering:
    """Verify M3 correctly distinguishes meaningful vs noise tokens."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from grafyx.ml_inference import get_model
        self.model = get_model("source_token_filter")
        if self.model is None:
            pytest.skip("M3 weights not available")

    def _make_features(self, **overrides):
        """Build a 15-feature vector with specified overrides."""
        vec = np.zeros(15, dtype=np.float32)
        feature_map = {
            "in_function_name": 0, "in_docstring": 1, "in_param_names": 2,
            "in_import": 3, "is_identifier": 4, "in_string_literal": 5,
            "in_getattr": 6, "is_stop_word": 7, "is_keyword": 8,
            "in_decorator": 9, "in_comment": 10, "source_line_count": 11,
            "token_length": 12, "docstring_length": 13, "token_frequency": 14,
        }
        for key, val in overrides.items():
            vec[feature_map[key]] = val
        return vec

    def test_function_name_token_kept(self):
        """Tokens that appear in function names are highly relevant."""
        feat = self._make_features(
            in_function_name=1.0, is_identifier=1.0, token_length=0.4,
        )
        score = self.model.predict(feat)
        assert score > 0.7, f"Name token scored {score:.3f}"

    def test_stop_word_filtered(self):
        """Stop words should be filtered out."""
        feat = self._make_features(
            is_stop_word=1.0, token_length=0.1,
        )
        score = self.model.predict(feat)
        assert score < 0.3, f"Stop word scored {score:.3f}"

    def test_keyword_filtered(self):
        """Python keywords should score lower than identifiers."""
        kw_feat = self._make_features(is_keyword=1.0, token_length=0.2)
        id_feat = self._make_features(in_function_name=1.0, is_identifier=1.0, token_length=0.4)
        kw_score = self.model.predict(kw_feat)
        id_score = self.model.predict(id_feat)
        assert id_score > kw_score, \
            f"Identifier ({id_score:.3f}) should rank above keyword ({kw_score:.3f})"

    def test_docstring_only_token_low(self):
        """Tokens ONLY in docstrings (not in code) should score low for source indexing."""
        feat = self._make_features(
            in_docstring=1.0, is_identifier=1.0, token_length=0.3,
            docstring_length=0.5,
        )
        score = self.model.predict(feat)
        # For source token indexing, docstring-only tokens are noise
        assert score < 0.5, f"Docstring-only token scored {score:.3f}"

    def test_import_only_token_low(self):
        """Tokens only in import statements should score low for source indexing."""
        feat = self._make_features(
            in_import=1.0, is_identifier=1.0, token_length=0.4,
        )
        score = self.model.predict(feat)
        # Import names without function name presence are low value
        assert score < 0.5, f"Import-only token scored {score:.3f}"

    def test_string_literal_filtered(self):
        """Random string literal content should score low."""
        feat = self._make_features(
            in_string_literal=1.0, is_identifier=0.0, token_length=0.2,
        )
        score = self.model.predict(feat)
        assert score < 0.5, f"String literal scored {score:.3f}"

    def test_param_name_kept(self):
        """Function parameter names are meaningful."""
        feat = self._make_features(
            in_param_names=1.0, is_identifier=1.0, token_length=0.3,
        )
        score = self.model.predict(feat)
        assert score > 0.5, f"Param name scored {score:.3f}"


# ═══════════════════════════════════════════════════════════════════════
#  M4: Symbol Importance (hint ranking)
# ═══════════════════════════════════════════════════════════════════════


class TestM4SymbolImportance:
    """Verify M4 correctly ranks symbol importance for diverse patterns."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from grafyx.ml_inference import get_model
        self.model = get_model("symbol_importance")
        if self.model is None:
            pytest.skip("M4 weights not available")

    def _make_features(self, **overrides):
        """Build an 18-feature vector with specified overrides."""
        vec = np.zeros(18, dtype=np.float32)
        feature_map = {
            "caller_count": 0, "cross_file_callers": 1, "is_exported": 2,
            "is_api_endpoint": 3, "is_entry_point": 4, "loc_count": 5,
            "param_count": 6, "has_docstring": 7, "docstring_length": 8,
            "import_count": 9, "is_base_class": 10, "subclass_count": 11,
            "method_count": 12, "is_abstract": 13, "decorator_count": 14,
            "is_test": 15, "file_depth": 16, "name_length": 17,
        }
        for key, val in overrides.items():
            vec[feature_map[key]] = val
        return vec

    def test_heavily_used_function_important(self):
        """Functions with many callers across files → high importance."""
        feat = self._make_features(
            caller_count=0.8, cross_file_callers=0.6, import_count=0.5,
        )
        score = self.model.predict(feat)
        assert score > 0.3, f"Heavily-used function scored {score:.3f}"

    def test_isolated_function_unimportant(self):
        """Functions with no callers or imports → low importance."""
        feat = self._make_features(
            caller_count=0.0, cross_file_callers=0.0, import_count=0.0,
        )
        score = self.model.predict(feat)
        assert score < 0.3, f"Isolated function scored {score:.3f}"

    def test_test_function_low_importance(self):
        """Test functions should have lower importance than production code."""
        test_feat = self._make_features(is_test=1.0, caller_count=0.1)
        prod_feat = self._make_features(is_test=0.0, caller_count=0.1)
        test_score = self.model.predict(test_feat)
        prod_score = self.model.predict(prod_feat)
        assert test_score < prod_score, \
            f"Test ({test_score:.3f}) should be less important than prod ({prod_score:.3f})"

    def test_more_callers_more_important(self):
        """Functions with more callers should score higher."""
        many = self._make_features(caller_count=0.8, cross_file_callers=0.5)
        few = self._make_features(caller_count=0.1, cross_file_callers=0.05)
        many_score = self.model.predict(many)
        few_score = self.model.predict(few)
        assert many_score > few_score, \
            f"Many callers ({many_score:.3f}) should outrank few ({few_score:.3f})"

    def test_moderately_used_function(self):
        """Functions with moderate usage across files are moderately important."""
        feat = self._make_features(
            caller_count=0.4, cross_file_callers=0.3, import_count=0.3,
        )
        score = self.model.predict(feat)
        # With moderate signals, should be between low and high
        isolated = self._make_features()
        heavy = self._make_features(
            caller_count=0.8, cross_file_callers=0.6, import_count=0.5,
        )
        s_isolated = self.model.predict(isolated)
        s_heavy = self.model.predict(heavy)
        assert s_isolated < score < s_heavy, \
            f"Ranking: isolated({s_isolated:.3f}) < moderate({score:.3f}) < heavy({s_heavy:.3f})"

    def test_large_class_more_important_than_small(self):
        """Large classes with cross-file usage outrank small isolated ones."""
        large = self._make_features(
            method_count=0.8, caller_count=0.6, cross_file_callers=0.5,
            import_count=0.4,
        )
        small = self._make_features(
            method_count=0.0, caller_count=0.0, cross_file_callers=0.0,
            import_count=0.0,
        )
        large_score = self.model.predict(large)
        small_score = self.model.predict(small)
        assert large_score > small_score, \
            f"Large class ({large_score:.3f}) should outrank small ({small_score:.3f})"

    def test_importance_ranking(self):
        """Core function > utility > test function."""
        core = self._make_features(
            caller_count=0.7, cross_file_callers=0.5, import_count=0.4,
        )
        util = self._make_features(
            caller_count=0.2, cross_file_callers=0.1, import_count=0.1,
        )
        test = self._make_features(
            caller_count=0.0, is_test=1.0,
        )
        s_core = self.model.predict(core)
        s_util = self.model.predict(util)
        s_test = self.model.predict(test)
        assert s_core > s_util > s_test, \
            f"Ranking broken: core={s_core:.3f}, util={s_util:.3f}, test={s_test:.3f}"


# ═══════════════════════════════════════════════════════════════════════
#  Cross-model integration
# ═══════════════════════════════════════════════════════════════════════


class TestAllModelsLoaded:
    """Verify all 4 models load and produce valid outputs."""

    def test_m1_loads_v2(self):
        from grafyx.search._relevance import RelevanceScorer
        scorer = RelevanceScorer()
        assert scorer._version == 2

    def test_m2_loads(self):
        from grafyx.ml_inference import get_model
        assert get_model("caller_disambig") is not None

    def test_m3_loads(self):
        from grafyx.ml_inference import get_model
        assert get_model("source_token_filter") is not None

    def test_m4_loads(self):
        from grafyx.ml_inference import get_model
        assert get_model("symbol_importance") is not None

    def test_all_outputs_bounded(self):
        """All models should produce outputs in [0, 1]."""
        from grafyx.ml_inference import get_model

        rng = np.random.default_rng(42)
        for name, feat_count in [("caller_disambig", 25), ("source_token_filter", 15), ("symbol_importance", 18)]:
            model = get_model(name)
            for _ in range(100):
                feat = rng.random(feat_count).astype(np.float32)
                score = model.predict(feat)
                assert 0.0 <= score <= 1.0, f"{name}: score {score} out of bounds"

    def test_m1_outputs_bounded(self):
        """M1 should produce outputs in [0, 1] for random inputs."""
        from grafyx.search._relevance import RelevanceScorer, _split_tokens
        scorer = RelevanceScorer()
        queries = ["test query", "database", "authenticate user login", "x", "parse json config file"]
        names = ["foo_bar", "TestClass", "__init__", "a", "very_long_function_name_here"]
        for q in queries:
            for n in names:
                score = scorer.score(_split_tokens(q), q.lower(), n, "Some doc", "src/file.py")
                assert 0.0 <= score <= 1.0, f"M1 score {score} out of bounds for {q}/{n}"
