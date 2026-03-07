"""Stress test for Relevance Ranker v2.

Loads the trained model and runs a suite of known-answer test cases.
Must achieve >= 95% accuracy across all categories.

Usage:
    python stress_test.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features, _split_tokens, FEATURE_COUNT

MODEL_DIR = Path(__file__).parent / "model"


class RelevanceRankerV2Inference:
    """Numpy-only inference for the trained v2 model."""

    def __init__(self, model_dir: str | None = None):
        model_dir = model_dir or str(MODEL_DIR)
        weights_path = Path(model_dir) / "relevance_weights_v2.npz"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model not found at {weights_path}. Run train.py first."
            )
        weights = np.load(weights_path)
        # Pre-transpose for fast matmul: x @ wT + b
        self.w1T = np.ascontiguousarray(weights["w1"].T)
        self.b1 = weights["b1"]
        self.w2T = np.ascontiguousarray(weights["w2"].T)
        self.b2 = weights["b2"]
        self.w3T = np.ascontiguousarray(weights["w3"].T)
        self.b3 = weights["b3"]

    def score(self, features: np.ndarray) -> float:
        """Run forward pass and return sigmoid probability."""
        x = features.astype(np.float32)
        x = np.maximum(0, x @ self.w1T + self.b1)
        x = np.maximum(0, x @ self.w2T + self.b2)
        logit = float((x @ self.w3T + self.b3).item())
        logit = np.clip(logit, -500, 500)
        return float(1.0 / (1.0 + np.exp(-logit)))


# ── Test cases ────────────────────────────────────────────────────
# Each tuple: (query, name, docstring, file_path, expected_relevant,
#              category, extra_kwargs)
# expected_relevant: True if score should be >= 0.5, False if < 0.5


TEST_CASES = [
    # ── Exact name matches (should be positive) ──────────────────
    ("get_user", "get_user", "Fetch a user by ID.", "app/models.py",
     True, "exact_match", {}),
    ("process_payment", "process_payment", "Process a payment.",
     "payments/handler.py", True, "exact_match", {}),
    ("validate_input", "validate_input", "Validate user input.",
     "utils/validation.py", True, "exact_match", {}),
    ("send_email", "send_email", "Send an email notification.",
     "notifications/email.py", True, "exact_match", {}),
    ("create_session", "create_session", "Create a new session.",
     "auth/session.py", True, "exact_match", {}),
    ("parse_config", "parse_config", "Parse config file.",
     "config/parser.py", True, "exact_match", {}),
    ("build_query", "build_query", "Build a SQL query.",
     "db/query.py", True, "exact_match", {}),
    ("handle_request", "handle_request", "Handle incoming request.",
     "server/handler.py", True, "exact_match", {}),

    # ── Partial / stem matches (should be positive) ───────────────
    ("user", "get_user_by_id", "Fetch a user from database.",
     "models/user.py", True, "partial_match", {}),
    ("validate", "validate_email_address", "Validate email.",
     "utils/validation.py", True, "partial_match", {}),
    ("cache", "CacheManager", "Manages the caching layer.",
     "cache/manager.py", True, "partial_match", {}),
    ("auth middleware", "AuthenticationMiddleware", "Auth middleware.",
     "middleware/auth.py", True, "partial_match", {}),
    ("database connection", "DatabaseConnectionPool", "Connection pool.",
     "db/pool.py", True, "partial_match", {}),

    # ── Docstring-only matches (should be positive) ───────────────
    ("rate limit", "throttle_request",
     "Apply rate limiting to incoming API requests.",
     "middleware/throttle.py", True, "doc_match", {}),
    ("retry logic", "execute_with_backoff",
     "Execute a function with exponential backoff retry logic.",
     "utils/retry.py", True, "doc_match", {}),

    # ── No overlap at all (should be negative) ────────────────────
    ("database migration", "render_template", "Render a Jinja template.",
     "views/render.py", False, "no_overlap", {}),
    ("websocket handler", "parse_csv_file", "Parse a CSV file.",
     "utils/csv.py", False, "no_overlap", {}),
    ("encryption key", "format_date", "Format a datetime object.",
     "utils/format.py", False, "no_overlap", {}),
    ("queue worker", "calculate_tax", "Calculate sales tax.",
     "billing/tax.py", False, "no_overlap", {}),
    ("health check", "compress_image", "Compress an image file.",
     "media/compress.py", False, "no_overlap", {}),
    ("jwt token", "merge_sort", "Sort using merge sort algorithm.",
     "algorithms/sort.py", False, "no_overlap", {}),
    ("pagination", "encrypt_password", "Encrypt a user password.",
     "auth/crypto.py", False, "no_overlap", {}),

    # ── Dunder negatives (should be negative) ─────────────────────
    ("user login", "__init__", "", "models/user.py",
     False, "dunder_negative", {"is_dunder": True}),
    ("database query", "__repr__", "Return string representation.",
     "db/models.py", False, "dunder_negative", {"is_dunder": True}),
    ("process data", "__str__", "String representation.",
     "data/processor.py", False, "dunder_negative", {"is_dunder": True}),
    ("send message", "__eq__", "Equality check.",
     "messaging/message.py", False, "dunder_negative", {"is_dunder": True}),
    ("file upload", "__hash__", "Hash function.",
     "storage/file.py", False, "dunder_negative", {"is_dunder": True}),
    ("cache invalidation", "__len__", "Return length.",
     "cache/store.py", False, "dunder_negative", {"is_dunder": True}),

    # ── __init__.py negatives ─────────────────────────────────────
    ("config loader", "register_plugins", "",
     "plugins/__init__.py", False, "init_file_negative",
     {"is_init_file": True}),
    ("database models", "setup_logging", "",
     "logging/__init__.py", False, "init_file_negative",
     {"is_init_file": True}),

    # ── Method vs standalone disambiguation ───────────────────────
    ("database refresh", "refresh", "Refresh the database session.",
     "db/session.py", True, "method_match", {"is_method": True}),
    ("cache get", "get", "Get value from cache.",
     "cache/backend.py", True, "method_match", {"is_method": True}),

    # ── Class matches ─────────────────────────────────────────────
    ("user model", "UserModel", "Represents a user in the database.",
     "models/user.py", True, "class_match", {"is_class": True}),
    ("http client", "HTTPClient", "HTTP client with connection pooling.",
     "http/client.py", True, "class_match", {"is_class": True}),

    # ── Path-only matches ─────────────────────────────────────────
    ("auth", "check_permissions", "Check user permissions.",
     "auth/permissions.py", True, "path_match", {}),
    ("middleware", "apply_cors", "Apply CORS headers.",
     "middleware/cors.py", True, "path_match", {}),

    # ── Verb-only matches (hard negatives — verbs too generic) ────
    ("get", "set_timeout", "Set a timeout value.",
     "utils/timeout.py", False, "verb_hard_negative", {}),
    ("run", "stop_server", "Stop the HTTP server.",
     "server/lifecycle.py", False, "verb_hard_negative", {}),
    ("check", "delete_record", "Delete a database record.",
     "db/records.py", False, "verb_hard_negative", {}),

    # ── CamelCase query matching ──────────────────────────────────
    ("UserService", "UserService", "Service for user operations.",
     "services/user.py", True, "camel_case", {}),
    ("HttpClient", "HTTPClient", "HTTP client.",
     "http/client.py", True, "camel_case", {}),

    # ── Substring matches ─────────────────────────────────────────
    ("auth", "authenticate_user", "Authenticate a user.",
     "auth/handler.py", True, "substring", {}),
    ("serial", "serialize_response", "Serialize API response.",
     "api/serializers.py", True, "substring", {}),
]


def run_stress_test():
    """Run all test cases and report results."""
    try:
        scorer = RelevanceRankerV2Inference()
    except FileNotFoundError as e:
        print(f"SKIP: {e}")
        return

    print(f"Running {len(TEST_CASES)} stress test cases...\n")

    category_results: dict[str, list[bool]] = {}
    failures: list[tuple[str, str, float, bool]] = []

    for (query, name, docstring, file_path, expected,
         category, extra_kwargs) in TEST_CASES:
        tokens = _split_tokens(query)
        query_lower = query.lower()

        features = extract_features(
            query_tokens=tokens,
            query_lower=query_lower,
            name=name,
            docstring=docstring,
            file_path=file_path,
            **extra_kwargs,
        )

        score = scorer.score(features)
        predicted = score >= 0.5
        correct = predicted == expected

        category_results.setdefault(category, []).append(correct)

        if not correct:
            failures.append((query, name, score, expected))

    # ── Report ────────────────────────────────────────────────────
    total_correct = 0
    total_count = 0

    print(f"{'Category':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    print("-" * 52)

    for cat, results in sorted(category_results.items()):
        n_correct = sum(results)
        n_total = len(results)
        acc = n_correct / n_total if n_total else 0
        total_correct += n_correct
        total_count += n_total
        status = "PASS" if acc >= 0.90 else "FAIL"
        print(f"  {cat:<23} {n_correct:>6}/{n_total:<6} {acc:>8.1%}  {status}")

    overall_acc = total_correct / total_count if total_count else 0
    print("-" * 52)
    print(f"  {'OVERALL':<23} {total_correct:>6}/{total_count:<6} "
          f"{overall_acc:>8.1%}")

    if failures:
        print(f"\n--- Failures ({len(failures)}) ---")
        for query, name, score, expected in failures:
            exp_label = "RELEVANT" if expected else "IRRELEVANT"
            print(f"  query={query!r:30s} name={name!r:30s} "
                  f"score={score:.3f} expected={exp_label}")

    print()
    if overall_acc >= 0.95:
        print(f"PASS: {overall_acc:.1%} >= 95% threshold")
    else:
        print(f"FAIL: {overall_acc:.1%} < 95% threshold")
        sys.exit(1)


if __name__ == "__main__":
    run_stress_test()
