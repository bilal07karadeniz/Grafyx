#!/usr/bin/env python3
"""Retrain M1 Relevance Ranker v2 with GRADED labels.

The original training used binary 0/1 labels, which caused the model to
saturate: any overlap → ~1.0, no overlap → 0.0. This doesn't produce
useful ranking (path-only match shouldn't score same as exact name match).

Fix: Use continuous target labels:
  - Exact name match:       0.90 - 0.95
  - Stem/partial name match: 0.65 - 0.80
  - Docstring-only match:   0.50 - 0.65
  - Path-only match:        0.25 - 0.40
  - No overlap:             0.00 - 0.05
  - Dunder with generic query: 0.05 - 0.15
  - Test file match:        0.50 - 0.65 (lower than prod match)

Uses MSELoss on sigmoid output for proper score calibration.
"""

import importlib.util
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERROR: PyTorch required. pip install torch")
    sys.exit(1)

ML_DIR = Path(__file__).parent
PRODUCTION_MODEL_DIR = ML_DIR.parent / "grafyx" / "search" / "model"

# Load M1 features module
spec = importlib.util.spec_from_file_location(
    "m1_features", ML_DIR / "relevance_ranker_v2" / "features.py"
)
m1_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1_mod)
extract_features = m1_mod.extract_features
FEATURE_COUNT = m1_mod.FEATURE_COUNT  # 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Utilities ─────────────────────────────────────────────────────────

def _split_tokens(text: str) -> list[str]:
    if not text:
        return []
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    parts = re.split(r'[^a-zA-Z0-9]+', s.lower())
    return [p for p in parts if len(p) >= 2]


def _stem_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 4:
        return False
    if longer.startswith(shorter):
        return True
    min_prefix = max(4, len(shorter) - 1)
    return min_prefix <= len(longer) and longer[:min_prefix] == shorter[:min_prefix]


# ── Diverse vocabulary ────────────────────────────────────────────────

_VERBS = [
    "get", "set", "create", "delete", "update", "find", "process", "handle",
    "validate", "check", "send", "receive", "parse", "build", "init", "load",
    "save", "fetch", "compute", "format", "register", "dispatch", "execute",
    "run", "start", "stop", "close", "open", "connect", "serialize", "decode",
    "authenticate", "authorize", "encrypt", "decrypt", "compress", "decompress",
    "transform", "filter", "aggregate", "merge", "split", "render", "compile",
    "schedule", "retry", "cache", "flush", "emit", "subscribe", "publish",
    "index", "search", "query", "scan", "resolve", "normalize", "tokenize",
    "train", "predict", "evaluate", "optimize", "benchmark", "migrate",
    "backup", "restore", "deploy", "rollback", "monitor", "profile",
]
_NOUNS = [
    "user", "request", "response", "connection", "session", "token", "message",
    "event", "task", "job", "config", "handler", "middleware", "router", "endpoint",
    "model", "schema", "query", "result", "error", "cache", "database", "transaction",
    "template", "controller", "service", "client", "server", "worker", "manager",
    "payment", "order", "product", "cart", "invoice", "notification", "email",
    "file", "stream", "buffer", "pipeline", "channel", "socket", "metric",
    "permission", "role", "policy", "credential", "certificate", "header",
    "batch", "queue", "schedule", "workflow", "plugin", "extension", "hook",
    "index", "document", "record", "entity", "aggregate", "repository",
    "feature", "embedding", "gradient", "tensor", "layer", "weight",
]
_PATHS = [
    "app/models.py", "app/views.py", "app/routes/auth.py", "api/endpoints.py",
    "services/user_service.py", "db/session.py", "cache/backend.py", "auth/jwt.py",
    "middleware/cors.py", "utils/helpers.py", "config/settings.py", "tasks/worker.py",
    "tests/test_auth.py", "tests/test_models.py", "core/engine.py", "payments/stripe.py",
    "search/indexer.py", "monitoring/metrics.py", "cli/commands.py", "data/pipeline.py",
    "events/bus.py", "http/client.py", "storage/s3.py", "schemas/user.py",
    "serializers/json_serializer.py", "validators/core.py", "ml/model.py",
    "plugins/__init__.py", "graph/traversal.py", "compiler/parser.py",
    "notifications/email.py", "billing/invoice.py", "reports/generator.py",
    "admin/dashboard.py", "integrations/slack.py", "webhooks/handler.py",
    "security/encryption.py", "logging/structured.py", "migrations/runner.py",
    "tests/integration/test_api.py", "tests/unit/test_utils.py",
    "benchmarks/perf_test.py", "scripts/deploy.py", "docs/api_docs.py",
]
_DOCS = [
    "Fetch a user by ID.", "Process incoming HTTP request.", "Validate user input.",
    "Send email notification to user.", "Create database session pool.",
    "Parse YAML configuration file.", "Build optimized SQL query.",
    "Handle incoming webhook event.", "Compute cryptographic hash of data.",
    "Format datetime to ISO string.", "Initialize application context.",
    "Load trained model weights from disk.", "Save checkpoint to storage.",
    "Execute background task with retry.", "Run database migrations.",
    "Connect to PostgreSQL database.", "Serialize response to JSON.",
    "Decode and verify JWT token.", "Register event handler callback.",
    "Dispatch message to RabbitMQ queue.", "Authenticate user credentials.",
    "Apply rate limiting.", "Process payment through Stripe.",
    "Render HTML template.", "Validate JSON schema.",
    "Compress file using gzip.", "Index document for search.",
    "Train neural network model.", "Evaluate model performance.",
    "Deploy service to Kubernetes.", "Generate analytics report.",
    "",  # Some functions have no docstring
    "",
]
_DUNDERS = [
    "__init__", "__repr__", "__str__", "__eq__", "__hash__",
    "__len__", "__getitem__", "__setitem__", "__iter__", "__call__",
    "__enter__", "__exit__", "__del__", "__contains__", "__next__",
]


# ── Data Generation ───────────────────────────────────────────────────

def _gen_graded_data(n: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Generate M1 training data with graded (continuous) labels."""
    X, y = [], []

    names_pool = []
    for v in _VERBS:
        for noun in _NOUNS[:15]:
            names_pool.append(f"{v}_{noun}")
    for noun in _NOUNS:
        names_pool.append(f"{noun.capitalize()}Manager")
        names_pool.append(f"{noun.capitalize()}Handler")
        names_pool.append(f"{noun.capitalize()}Service")
        names_pool.append(f"{noun.capitalize()}Repository")
        names_pool.append(f"{noun.capitalize()}Controller")
    names_pool.extend(_DUNDERS)

    per_cat = n // 14  # 14 categories

    # ━━━ Category 1: Exact name match (label 0.90-0.95) ━━━
    for _ in range(per_cat * 2):
        name = rng.choice(names_pool)
        if name.startswith("__"):
            continue
        tokens = _split_tokens(name)
        if not tokens:
            continue
        query = " ".join(tokens)
        path = rng.choice(_PATHS)
        feat = extract_features(
            query_tokens=tokens, query_lower=query, name=name,
            docstring=rng.choice(_DOCS), file_path=path,
            is_dunder=False,
            is_init_file="__init__" in path,
        )
        label = rng.uniform(0.90, 0.95)
        X.append(feat); y.append(label)

    # ━━━ Category 2: Partial name match - 1 of 2+ tokens (label 0.65-0.80) ━━━
    for _ in range(per_cat * 2):
        name = rng.choice(names_pool)
        if name.startswith("__"):
            continue
        tokens = _split_tokens(name)
        if len(tokens) < 2:
            continue
        # Use only 1 token from the name
        subset = [rng.choice(tokens)]
        query = " ".join(subset)
        path = rng.choice(_PATHS)
        feat = extract_features(
            query_tokens=subset, query_lower=query, name=name,
            docstring=rng.choice(_DOCS), file_path=path,
        )
        label = rng.uniform(0.65, 0.80)
        X.append(feat); y.append(label)

    # ━━━ Category 3: Stem match only (label 0.55-0.70) ━━━
    for _ in range(per_cat):
        name = rng.choice(names_pool)
        if name.startswith("__"):
            continue
        name_tokens = _split_tokens(name)
        if not name_tokens:
            continue
        # Generate a stem variant
        base = rng.choice(name_tokens)
        if len(base) >= 5:
            # Truncate or extend
            variants = [base + "s", base + "ed", base + "ing", base + "er",
                        base + "tion", base + "ment", base[:len(base)-1]]
            stem_query = rng.choice(variants)
        else:
            continue
        feat = extract_features(
            query_tokens=[stem_query], query_lower=stem_query, name=name,
            docstring=rng.choice(_DOCS), file_path=rng.choice(_PATHS),
        )
        label = rng.uniform(0.55, 0.70)
        X.append(feat); y.append(label)

    # ━━━ Category 4: Docstring-only match (label 0.45-0.60) ━━━
    for _ in range(per_cat):
        doc = rng.choice([d for d in _DOCS if len(d) > 20])
        doc_tokens = _split_tokens(doc)
        if len(doc_tokens) < 3:
            continue
        # Pick tokens from docstring NOT in the function name
        name = rng.choice([n for n in names_pool if not n.startswith("__")])
        name_tokens_set = set(_split_tokens(name))
        doc_only = [t for t in doc_tokens if t not in name_tokens_set and len(t) >= 4]
        if not doc_only:
            continue
        subset = rng.sample(doc_only, min(2, len(doc_only)))
        query = " ".join(subset)
        path = rng.choice(_PATHS)
        feat = extract_features(
            query_tokens=subset, query_lower=query, name=name,
            docstring=doc, file_path=path,
        )
        label = rng.uniform(0.45, 0.60)
        X.append(feat); y.append(label)

    # ━━━ Category 5: Path-only match (label 0.20-0.35) ━━━
    for _ in range(per_cat):
        path = rng.choice(_PATHS)
        path_tokens = _split_tokens(path)
        if not path_tokens:
            continue
        # Pick a path token NOT in the function name
        name = rng.choice([n for n in names_pool if not n.startswith("__")])
        name_tokens_set = set(_split_tokens(name))
        path_only = [t for t in path_tokens if t not in name_tokens_set and len(t) >= 3]
        if not path_only:
            continue
        qt = rng.choice(path_only)
        # Also make sure the token is not in the docstring
        doc = rng.choice([d for d in _DOCS if qt not in d.lower()])
        feat = extract_features(
            query_tokens=[qt], query_lower=qt, name=name,
            docstring=doc, file_path=path,
        )
        label = rng.uniform(0.20, 0.35)
        X.append(feat); y.append(label)

    # ━━━ Category 6: Dunder match with generic query (label 0.05-0.15) ━━━
    for _ in range(per_cat):
        dunder = rng.choice(_DUNDERS)
        dunder_inner = dunder.strip("_")
        if not dunder_inner:
            continue
        # Query is the inner name (e.g. "init", "repr", "str")
        feat = extract_features(
            query_tokens=[dunder_inner], query_lower=dunder_inner, name=dunder,
            docstring="Magic method.", file_path=rng.choice(_PATHS),
            is_dunder=True,
        )
        label = rng.uniform(0.05, 0.15)
        X.append(feat); y.append(label)

    # ━━━ Category 7: Dunder match with specific class query (label 0.30-0.45) ━━━
    # e.g. "UserModel init" → UserModel.__init__ is somewhat relevant
    for _ in range(per_cat // 2):
        dunder = rng.choice(_DUNDERS)
        dunder_inner = dunder.strip("_")
        if not dunder_inner:
            continue
        class_name = rng.choice(_NOUNS).capitalize() + "Model"
        tokens = _split_tokens(class_name) + [dunder_inner]
        query = " ".join(tokens)
        feat = extract_features(
            query_tokens=tokens, query_lower=query, name=dunder,
            docstring=f"Initialize {class_name}.", file_path=rng.choice(_PATHS),
            is_dunder=True,
        )
        label = rng.uniform(0.30, 0.45)
        X.append(feat); y.append(label)

    # ━━━ Category 8: Test file match (label 0.45-0.60) ━━━
    for _ in range(per_cat):
        name = rng.choice([n for n in names_pool if not n.startswith("__")])
        tokens = _split_tokens(name)
        if not tokens:
            continue
        query = " ".join(tokens)
        test_name = f"test_{name}"
        test_path = rng.choice(["tests/test_auth.py", "tests/test_models.py",
                                "tests/integration/test_api.py", "tests/unit/test_utils.py"])
        feat = extract_features(
            query_tokens=tokens, query_lower=query, name=test_name,
            docstring=f"Test {name}.", file_path=test_path,
        )
        label = rng.uniform(0.45, 0.60)
        X.append(feat); y.append(label)

    # ━━━ Category 9: Zero overlap negatives (label 0.00-0.02) ━━━
    for _ in range(per_cat * 2):
        name = rng.choice(names_pool)
        other = rng.choice(names_pool)
        tokens = _split_tokens(other)
        name_tokens_set = set(_split_tokens(name))
        # Ensure NO overlap at all (name, path, doc)
        if not tokens or (set(tokens) & name_tokens_set):
            continue
        query = " ".join(tokens)
        # Pick path and doc with no overlap either
        path = rng.choice(_PATHS)
        path_tokens_set = set(_split_tokens(path))
        if set(tokens) & path_tokens_set:
            continue
        feat = extract_features(
            query_tokens=tokens, query_lower=query, name=name,
            docstring="", file_path=path,
        )
        label = rng.uniform(0.00, 0.02)
        X.append(feat); y.append(label)

    # ━━━ Category 10: Weak overlap negatives - common verb only (label 0.10-0.25) ━━━
    # e.g., query="get user" matches "get_connection" only on "get" — weak signal
    for _ in range(per_cat):
        verb = rng.choice(_VERBS[:20])  # common verbs
        # Query has verb + unrelated noun
        unrelated_noun = rng.choice(_NOUNS)
        query_tokens = [verb, unrelated_noun]
        # Name has verb + different noun
        other_noun = rng.choice([n for n in _NOUNS if n != unrelated_noun])
        name = f"{verb}_{other_noun}"
        query = " ".join(query_tokens)
        path = rng.choice(_PATHS)
        feat = extract_features(
            query_tokens=query_tokens, query_lower=query, name=name,
            docstring=rng.choice(_DOCS), file_path=path,
        )
        # Only 1 of 2 query tokens matches - moderate-low
        label = rng.uniform(0.10, 0.25)
        X.append(feat); y.append(label)

    # ━━━ Category 11: Name+docstring strong match (label 0.85-0.95) ━━━
    for _ in range(per_cat):
        name = rng.choice([n for n in names_pool if not n.startswith("__")])
        tokens = _split_tokens(name)
        if not tokens:
            continue
        # Doc also contains query tokens
        doc = f"Function to {' '.join(tokens)} in the system."
        query = " ".join(tokens)
        feat = extract_features(
            query_tokens=tokens, query_lower=query, name=name,
            docstring=doc, file_path=rng.choice(_PATHS),
        )
        label = rng.uniform(0.85, 0.95)
        X.append(feat); y.append(label)

    # ━━━ Category 12: Name+path strong match (label 0.85-0.95) ━━━
    for _ in range(per_cat):
        verb = rng.choice(_VERBS)
        noun = rng.choice(_NOUNS)
        name = f"{verb}_{noun}"
        tokens = [verb, noun]
        query = " ".join(tokens)
        # Path also matches
        path = f"{noun}/{verb}_{noun}.py"
        feat = extract_features(
            query_tokens=tokens, query_lower=query, name=name,
            docstring=rng.choice(_DOCS), file_path=path,
        )
        label = rng.uniform(0.85, 0.95)
        X.append(feat); y.append(label)

    # ━━━ Category 13: Substring match in name (label 0.50-0.65) ━━━
    for _ in range(per_cat):
        name = rng.choice([n for n in names_pool if not n.startswith("__")])
        name_lower = name.lower()
        # Use a substring of the name as query
        if len(name_lower) < 8:
            continue
        start = rng.randint(0, len(name_lower) - 5)
        substr = name_lower[start:start + 5]
        if not substr.isalpha():
            continue
        feat = extract_features(
            query_tokens=[substr], query_lower=substr, name=name,
            docstring=rng.choice(_DOCS), file_path=rng.choice(_PATHS),
        )
        label = rng.uniform(0.50, 0.65)
        X.append(feat); y.append(label)

    # ━━━ Category 14: __init__.py re-export match (label 0.15-0.30) ━━━
    for _ in range(per_cat // 2):
        name = rng.choice([n for n in names_pool if not n.startswith("__")])
        tokens = _split_tokens(name)
        if not tokens:
            continue
        query = " ".join(tokens)
        feat = extract_features(
            query_tokens=tokens, query_lower=query, name=name,
            docstring="", file_path="mypackage/__init__.py",
            is_init_file=True,
        )
        label = rng.uniform(0.15, 0.30)
        X.append(feat); y.append(label)

    # Shuffle
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    idx = np.arange(len(X_arr))
    np.random.seed(42)
    np.random.shuffle(idx)
    return X_arr[idx], y_arr[idx]


# ── Model ─────────────────────────────────────────────────────────────

class RelevanceRankerV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# ── Training ──────────────────────────────────────────────────────────

def train_ranker(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 60, lr: float = 1e-3, batch_size: int = 2048,
    patience: int = 10,
):
    """Train with MSE loss on sigmoid output for score calibration."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4,
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            preds = torch.sigmoid(logits)
            loss = torch.nn.functional.mse_loss(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_total += len(xb)

        # Validate
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_mae = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb).squeeze(-1)
                preds = torch.sigmoid(logits)
                loss = torch.nn.functional.mse_loss(preds, yb)
                val_loss += loss.item() * len(xb)
                val_mae += torch.abs(preds - yb).sum().item()
                val_total += len(xb)

        train_mse = train_loss / train_total
        val_mse = val_loss / val_total
        val_mae_avg = val_mae / val_total
        scheduler.step(val_mse)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"train_mse={train_mse:.6f} | val_mse={val_mse:.6f} | "
              f"val_mae={val_mae_avg:.4f} | lr={lr_now:.1e} | {elapsed:.1f}s")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    # Final metrics
    val_mae = 0.0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = torch.sigmoid(model(xb).squeeze(-1))
            val_mae += torch.abs(preds - yb).sum().item()
            val_total += len(xb)

    print(f"  Best epoch: {best_epoch}, val_mse={best_val_loss:.6f}, "
          f"val_mae={val_mae / val_total:.4f}")
    return model


def export_weights(model: nn.Module, out_path: Path):
    """Export in W0/b0 format (pre-transposed for x @ W + b)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    layer_keys = sorted(
        set(k.rsplit(".", 1)[0] for k in sd if "weight" in k),
        key=lambda k: int(re.search(r'\d+', k).group()) if re.search(r'\d+', k) else 0,
    )
    save_dict = {}
    for i, key_prefix in enumerate(layer_keys):
        w = sd[f"{key_prefix}.weight"].cpu().numpy()
        b = sd[f"{key_prefix}.bias"].cpu().numpy()
        save_dict[f"W{i}"] = w.T.astype(np.float32)
        save_dict[f"b{i}"] = b.astype(np.float32)
    np.savez(out_path, **save_dict)
    print(f"  Saved to {out_path} ({out_path.stat().st_size:,} bytes)")


# ── Verification ──────────────────────────────────────────────────────

def verify_model(model: nn.Module):
    """Run critical test cases to check ranking quality."""
    model.eval()
    model = model.cpu()

    cases = [
        # (query, name, doc, path, kwargs, expected_range, description)
        ("authenticate user", "authenticate_user", "Authenticate a user", "app/auth.py",
         {}, (0.75, 1.0), "exact name match"),
        ("parse json", "parse_json", "Parse JSON string", "utils/serialization.py",
         {}, (0.75, 1.0), "exact name match 2"),
        ("database connect", "create_db_connection", "Create a database connection pool", "db/connections.py",
         {}, (0.45, 0.85), "partial match"),
        ("validate", "validation_handler", "Handle validation", "app/validation.py",
         {}, (0.40, 0.85), "stem match"),
        ("auth", "process_request", "Process incoming request", "auth/middleware.py",
         {}, (0.10, 0.50), "path-only match"),
        ("payment process", "render_template", "Render an HTML template", "web/views.py",
         {}, (0.00, 0.10), "no relation"),
        ("cache invalidate", "parse_csv_row", "Parse a single CSV row", "utils/csv_parser.py",
         {}, (0.00, 0.10), "completely unrelated"),
        ("init", "__init__", "Initialize the class", "models/user.py",
         {"is_dunder": True}, (0.00, 0.30), "dunder generic query"),
        ("authenticate", "test_authenticate", "Test authenticate", "tests/test_auth.py",
         {}, (0.35, 0.75), "test file"),
        ("user manager", "UserManager", "Manage users", "services/user.py",
         {}, (0.75, 1.0), "camelcase class match"),
        ("send email", "send_notification", "Send notification to user", "notifications/email.py",
         {}, (0.15, 0.60), "partial verb match + path"),
        ("database", "run_migrations", "Run database migrations", "db/migrations.py",
         {}, (0.25, 0.70), "docstring + path match"),
    ]

    print("\n  Verification:")
    all_passed = True
    for query, name, doc, path, kwargs, (lo, hi), desc in cases:
        tokens = _split_tokens(query)
        feat = extract_features(
            query_tokens=tokens, query_lower=query, name=name,
            docstring=doc, file_path=path, **kwargs,
        )
        x = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            score = float(torch.sigmoid(model(x)).item())

        passed = lo <= score <= hi
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"    [{status}] {desc}: {score:.4f} (expected {lo:.2f}-{hi:.2f})")

    # Ranking check: exact > partial > path-only > unrelated
    ranking_queries = [
        ("user", "get_user", "Get user by ID", "app/models.py", {}),
        ("user", "update_user_settings", "Update user settings", "app/settings.py", {}),
        ("user", "process_request", "Process request", "app/user/middleware.py", {}),
        ("user", "render_template", "Render HTML", "web/views.py", {}),
    ]
    scores = []
    for query, name, doc, path, kwargs in ranking_queries:
        tokens = _split_tokens(query)
        feat = extract_features(
            query_tokens=tokens, query_lower=query, name=name,
            docstring=doc, file_path=path, **kwargs,
        )
        x = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            score = float(torch.sigmoid(model(x)).item())
        scores.append(score)

    rank_ok = scores[0] > scores[1] > scores[2] > scores[3]
    status = "PASS" if rank_ok else "FAIL"
    if not rank_ok:
        all_passed = False
    print(f"    [{status}] Ranking: exact({scores[0]:.3f}) > partial({scores[1]:.3f}) "
          f"> path({scores[2]:.3f}) > none({scores[3]:.3f})")

    return all_passed


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n", type=int, default=500_000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n = 50_000
        args.epochs = 30

    rng = random.Random(42)
    print(f"Device: {DEVICE}")
    print(f"Generating {args.n:,} graded training examples...")

    X, y_arr = _gen_graded_data(args.n, rng)
    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_arr[:split], y_arr[split:]

    print(f"  Total: {len(X):,}")
    print(f"  Train: {len(X_train):,} (mean label: {y_train.mean():.3f})")
    print(f"  Val:   {len(X_val):,} (mean label: {y_val.mean():.3f})")

    model = RelevanceRankerV2()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print()

    model = train_ranker(model, X_train, y_train, X_val, y_val,
                         epochs=args.epochs)

    passed = verify_model(model)

    if not passed:
        print("\n  WARNING: Some verification checks failed!")
        print("  Retrying with longer training...")
        # Retry with lower LR
        model2 = RelevanceRankerV2()
        model2 = train_ranker(model2, X_train, y_train, X_val, y_val,
                              epochs=args.epochs + 20, lr=5e-4)
        passed2 = verify_model(model2)
        if passed2:
            model = model2
            passed = True

    out_path = PRODUCTION_MODEL_DIR / "relevance_weights_v2.npz"
    export_weights(model, out_path)

    if passed:
        print("\n  All verification checks PASSED!")
    else:
        print("\n  Some checks still failing - model saved but may need tuning.")


if __name__ == "__main__":
    main()
