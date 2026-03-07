"""Stress test for Caller Disambiguator.

Tests disambiguation of common ambiguous patterns:
  - db.refresh() != standalone refresh()
  - self.cache.get() != standalone get()
  - session.commit() resolves to SQLAlchemy Session
  - Multiple classes with same method name

Must achieve >= 95% accuracy.

Usage:
    python stress_test.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features, FEATURE_COUNT

MODEL_DIR = Path(__file__).parent / "model"


class CallerDisambiguatorInference:
    """Numpy-only inference for the trained caller disambiguator."""

    def __init__(self, model_dir: str | None = None):
        model_dir = model_dir or str(MODEL_DIR)
        weights_path = Path(model_dir) / "caller_disambiguator_weights.npz"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model not found at {weights_path}. Run train.py first."
            )
        weights = np.load(weights_path)
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
# Each test case: dict with call_site info, callee info, expected label,
# and a category string.


TEST_CASES = [
    # ── db.refresh() should match Session.refresh, not standalone refresh ──
    {
        "call_site": {
            "receiver": "db",
            "method": "refresh",
            "caller_file": "app/services/user.py",
            "caller_imports": ["sqlalchemy.orm", "app.models"],
            "arg_count": 1,
            "has_dot_syntax": True,
            "caller_loc": 45,
        },
        "callee": {
            "name": "refresh",
            "class_name": "Session",
            "file": "sqlalchemy/orm/session.py",
            "params": ["self", "instance"],
            "decorators": [],
        },
        "expected": True,
        "category": "db_method_positive",
    },
    {
        "call_site": {
            "receiver": "db",
            "method": "refresh",
            "caller_file": "app/services/user.py",
            "caller_imports": ["sqlalchemy.orm", "app.models"],
            "arg_count": 1,
            "has_dot_syntax": True,
            "caller_loc": 45,
        },
        "callee": {
            "name": "refresh",
            "class_name": "",
            "file": "app/utils/cache.py",
            "params": ["key"],
            "decorators": [],
        },
        "expected": False,
        "category": "db_method_negative",
    },

    # ── self.cache.get() should match CacheBackend.get, not dict.get ──
    {
        "call_site": {
            "receiver": "self.cache",
            "method": "get",
            "caller_file": "app/services/data.py",
            "caller_imports": ["app.cache.backend"],
            "arg_count": 1,
            "has_dot_syntax": True,
            "caller_loc": 30,
        },
        "callee": {
            "name": "get",
            "class_name": "CacheBackend",
            "file": "app/cache/backend.py",
            "params": ["self", "key"],
            "decorators": [],
        },
        "expected": True,
        "category": "chained_receiver_positive",
    },
    {
        "call_site": {
            "receiver": "self.cache",
            "method": "get",
            "caller_file": "app/services/data.py",
            "caller_imports": ["app.cache.backend"],
            "arg_count": 1,
            "has_dot_syntax": True,
            "caller_loc": 30,
        },
        "callee": {
            "name": "get",
            "class_name": "",
            "file": "builtins.py",
            "params": ["key", "default"],
            "decorators": [],
        },
        "expected": False,
        "category": "chained_receiver_negative",
    },

    # ── session.commit() resolves to SQLAlchemy Session ──
    {
        "call_site": {
            "receiver": "session",
            "method": "commit",
            "caller_file": "app/db/operations.py",
            "caller_imports": ["sqlalchemy.orm.Session"],
            "arg_count": 0,
            "has_dot_syntax": True,
            "caller_loc": 20,
        },
        "callee": {
            "name": "commit",
            "class_name": "Session",
            "file": "sqlalchemy/orm/session.py",
            "params": ["self"],
            "decorators": [],
        },
        "expected": True,
        "category": "import_match_positive",
    },
    {
        "call_site": {
            "receiver": "session",
            "method": "commit",
            "caller_file": "app/db/operations.py",
            "caller_imports": ["sqlalchemy.orm.Session"],
            "arg_count": 0,
            "has_dot_syntax": True,
            "caller_loc": 20,
        },
        "callee": {
            "name": "commit",
            "class_name": "GitRepo",
            "file": "git/repo.py",
            "params": ["self", "message"],
            "decorators": [],
        },
        "expected": False,
        "category": "import_match_negative",
    },

    # ── self.method() within same class ──
    {
        "call_site": {
            "receiver": "self",
            "method": "validate",
            "caller_file": "app/models/user.py",
            "caller_imports": [],
            "arg_count": 0,
            "has_dot_syntax": True,
            "caller_loc": 15,
        },
        "callee": {
            "name": "validate",
            "class_name": "UserModel",
            "file": "app/models/user.py",
            "params": ["self"],
            "decorators": [],
        },
        "expected": True,
        "category": "self_method_positive",
    },
    {
        "call_site": {
            "receiver": "self",
            "method": "validate",
            "caller_file": "app/models/user.py",
            "caller_imports": [],
            "arg_count": 0,
            "has_dot_syntax": True,
            "caller_loc": 15,
        },
        "callee": {
            "name": "validate",
            "class_name": "PaymentModel",
            "file": "app/models/payment.py",
            "params": ["self"],
            "decorators": [],
        },
        "expected": False,
        "category": "self_method_negative",
    },

    # ── Standalone function call (no dot syntax) ──
    {
        "call_site": {
            "receiver": "",
            "method": "parse_config",
            "caller_file": "app/main.py",
            "caller_imports": ["app.config.parse_config"],
            "arg_count": 1,
            "has_dot_syntax": False,
            "caller_loc": 10,
        },
        "callee": {
            "name": "parse_config",
            "class_name": "",
            "file": "app/config.py",
            "params": ["path"],
            "decorators": [],
        },
        "expected": True,
        "category": "standalone_positive",
    },
    {
        "call_site": {
            "receiver": "",
            "method": "parse_config",
            "caller_file": "app/main.py",
            "caller_imports": ["app.config.parse_config"],
            "arg_count": 1,
            "has_dot_syntax": False,
            "caller_loc": 10,
        },
        "callee": {
            "name": "parse_config",
            "class_name": "ConfigParser",
            "file": "lib/parser.py",
            "params": ["self", "raw_data"],
            "decorators": [],
        },
        "expected": False,
        "category": "standalone_negative",
    },

    # ── Same directory proximity ──
    {
        "call_site": {
            "receiver": "handler",
            "method": "process",
            "caller_file": "app/handlers/main.py",
            "caller_imports": ["app.handlers.request"],
            "arg_count": 1,
            "has_dot_syntax": True,
            "caller_loc": 25,
        },
        "callee": {
            "name": "process",
            "class_name": "RequestHandler",
            "file": "app/handlers/request.py",
            "params": ["self", "data"],
            "decorators": [],
        },
        "expected": True,
        "category": "proximity_positive",
    },
    {
        "call_site": {
            "receiver": "handler",
            "method": "process",
            "caller_file": "app/handlers/main.py",
            "caller_imports": ["app.handlers.request"],
            "arg_count": 1,
            "has_dot_syntax": True,
            "caller_loc": 25,
        },
        "callee": {
            "name": "process",
            "class_name": "BatchProcessor",
            "file": "lib/batch/processor.py",
            "params": ["self", "items", "callback"],
            "decorators": [],
        },
        "expected": False,
        "category": "proximity_negative",
    },

    # ── Property access should NOT be a call ──
    {
        "call_site": {
            "receiver": "user",
            "method": "name",
            "caller_file": "app/views/profile.py",
            "caller_imports": ["app.models.User"],
            "arg_count": 0,
            "has_dot_syntax": True,
            "caller_loc": 10,
        },
        "callee": {
            "name": "name",
            "class_name": "User",
            "file": "app/models/user.py",
            "params": ["self"],
            "decorators": ["@property"],
        },
        "expected": True,
        "category": "property_positive",
    },

    # ── classmethod vs regular method ──
    {
        "call_site": {
            "receiver": "User",
            "method": "create",
            "caller_file": "app/services/auth.py",
            "caller_imports": ["app.models.User"],
            "arg_count": 2,
            "has_dot_syntax": True,
            "caller_loc": 35,
        },
        "callee": {
            "name": "create",
            "class_name": "User",
            "file": "app/models/user.py",
            "params": ["cls", "name", "email"],
            "decorators": ["@classmethod"],
        },
        "expected": True,
        "category": "classmethod_positive",
    },

    # ── Type annotation match ──
    {
        "call_site": {
            "receiver": "client",
            "method": "send",
            "caller_file": "app/api/gateway.py",
            "caller_imports": ["httpx"],
            "arg_count": 2,
            "has_dot_syntax": True,
            "caller_loc": 50,
            "type_annotation": "HTTPClient",
        },
        "callee": {
            "name": "send",
            "class_name": "HTTPClient",
            "file": "httpx/client.py",
            "params": ["self", "request", "timeout"],
            "decorators": [],
        },
        "expected": True,
        "category": "type_annotation_positive",
    },
    {
        "call_site": {
            "receiver": "client",
            "method": "send",
            "caller_file": "app/api/gateway.py",
            "caller_imports": ["httpx"],
            "arg_count": 2,
            "has_dot_syntax": True,
            "caller_loc": 50,
            "type_annotation": "HTTPClient",
        },
        "callee": {
            "name": "send",
            "class_name": "WebSocketClient",
            "file": "ws/client.py",
            "params": ["self", "message"],
            "decorators": [],
        },
        "expected": False,
        "category": "type_annotation_negative",
    },
]


def _extract_module(file_path: str) -> str:
    parts = file_path.replace("\\", "/").split("/")
    name = parts[-1] if parts else ""
    return name.replace(".py", "")


def _extract_package(file_path: str) -> str:
    parts = file_path.replace("\\", "/").split("/")
    for p in parts:
        if p not in ("src", "lib", "app", ""):
            return p
    return parts[0] if parts else ""


def run_stress_test():
    """Run all test cases and report results."""
    try:
        scorer = CallerDisambiguatorInference()
    except FileNotFoundError as e:
        print(f"SKIP: {e}")
        return

    print(f"Running {len(TEST_CASES)} stress test cases...\n")

    category_results: dict[str, list[bool]] = {}
    failures: list[tuple[str, str, float, bool]] = []

    for tc in TEST_CASES:
        cs = tc["call_site"]
        callee = tc["callee"]
        expected = tc["expected"]
        category = tc["category"]

        features = extract_features(
            receiver_text=cs.get("receiver", ""),
            method_name=cs.get("method", ""),
            caller_file=cs.get("caller_file", ""),
            caller_imports=cs.get("caller_imports", []),
            arg_count=cs.get("arg_count", 0),
            has_dot_syntax=cs.get("has_dot_syntax", False),
            caller_loc=cs.get("caller_loc", 0),
            callee_name=callee.get("name", ""),
            callee_class_name=callee.get("class_name", ""),
            callee_file=callee.get("file", ""),
            callee_module=_extract_module(callee.get("file", "")),
            callee_package=_extract_package(callee.get("file", "")),
            callee_param_count=len(callee.get("params", [])),
            callee_is_method=bool(callee.get("class_name")),
            callee_decorators=callee.get("decorators", []),
            receiver_type_annotation=cs.get("type_annotation", ""),
        )

        score = scorer.score(features)
        predicted = score >= 0.5
        correct = predicted == expected

        category_results.setdefault(category, []).append(correct)

        if not correct:
            desc = (f"{cs.get('receiver', '')}.{cs.get('method', '')}() "
                    f"-> {callee.get('class_name', '')}.{callee.get('name', '')}")
            failures.append((desc, category, score, expected))

    # ── Report ────────────────────────────────────────────────────
    total_correct = 0
    total_count = 0

    print(f"{'Category':<30} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    print("-" * 57)

    for cat, results in sorted(category_results.items()):
        n_correct = sum(results)
        n_total = len(results)
        acc = n_correct / n_total if n_total else 0
        total_correct += n_correct
        total_count += n_total
        status = "PASS" if acc >= 0.90 else "FAIL"
        print(f"  {cat:<28} {n_correct:>6}/{n_total:<6} {acc:>8.1%}  {status}")

    overall_acc = total_correct / total_count if total_count else 0
    print("-" * 57)
    print(f"  {'OVERALL':<28} {total_correct:>6}/{total_count:<6} "
          f"{overall_acc:>8.1%}")

    if failures:
        print(f"\n--- Failures ({len(failures)}) ---")
        for desc, cat, score, expected in failures:
            exp_label = "MATCH" if expected else "NO_MATCH"
            print(f"  {desc:50s} score={score:.3f} expected={exp_label} ({cat})")

    print()
    if overall_acc >= 0.95:
        print(f"PASS: {overall_acc:.1%} >= 95% threshold")
    else:
        print(f"FAIL: {overall_acc:.1%} < 95% threshold")
        sys.exit(1)


if __name__ == "__main__":
    run_stress_test()
