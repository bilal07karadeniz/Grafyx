"""Evaluate the trained code search encoder."""
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"

# Test queries from evaluation results
TEST_QUERIES = [
    {"query": "JWT authentication token verification", "expected": ["create_access_token", "verify_token", "authenticate"]},
    {"query": "file upload S3 storage", "expected": ["upload_file", "S3StorageService", "save_file"]},
    {"query": "password hash bcrypt", "expected": ["verify_password", "hash_password", "get_password_hash"]},
    {"query": "WebSocket message handler", "expected": ["websocket_endpoint", "handle_message", "on_message"]},
    {"query": "database connection pool", "expected": ["get_db", "create_engine", "SessionLocal"]},
    {"query": "send email notification", "expected": ["send_email", "send_notification", "notify_user"]},
    {"query": "rate limit middleware", "expected": ["rate_limit", "check_rate_limit", "RateLimiter"]},
    {"query": "user session management", "expected": ["create_session", "get_session", "SessionManager"]},
]


def evaluate():
    """Evaluate code search encoder on test queries."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "grafyx" / "search"))

    from _tokenizer import CodeTokenizer
    from _mamba import MambaBlock

    weights_path = MODEL_DIR / "code_encoder_weights.npz"
    if not weights_path.exists():
        print("ERROR: Model not trained yet. Run train.py first.")
        return

    tokenizer = CodeTokenizer()
    if not tokenizer.is_available:
        print("ERROR: Tokenizer not available.")
        return

    # Load code embeddings from test data
    test_file = DATA_DIR / "synthetic_test.jsonl"
    if not test_file.exists():
        print("ERROR: Test data not found.")
        return

    print("Loading test data and computing embeddings...")
    # (Would load model and compute embeddings here)

    print("\nEvaluation Results:")
    print(f"{'Query':<45} {'Top-5 Hit?':<10} {'Rank':<6}")
    print("-" * 65)

    for test in TEST_QUERIES:
        query = test["query"]
        expected = test["expected"]
        # Placeholder -- real evaluation would use the model
        print(f"{query:<45} {'TBD':<10} {'TBD':<6}")

    print("\nFull evaluation requires trained model. Run train.py first.")


if __name__ == "__main__":
    evaluate()
