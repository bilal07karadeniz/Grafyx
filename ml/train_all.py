#!/usr/bin/env python3
"""Master training script for all 4 MLPs.

Generates diverse synthetic training data, trains each model with PyTorch,
exports weights in the correct format for production inference, and copies
to grafyx/search/model/.

Models trained:
  M1: Relevance Ranker v2      (42 -> 128 -> 64 -> 1)
  M2: Caller Disambiguator     (25 -> 64 -> 32 -> 1)
  M3: Source Token Filter       (15 -> 32 -> 16 -> 1)
  M4: Symbol Importance         (18 -> 32 -> 16 -> 1)

Weight format:
  W0, b0, W1, b1, W2, b2  (pre-transposed for x @ W + b)

Usage:
    python ml/train_all.py [--epochs 40] [--quick]
"""

import argparse
import json
import os
import random
import re
import shutil
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

# ── Import feature modules without name clashes ──────────────────────
import importlib.util

def _load_features_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m1_features_mod = _load_features_module("m1_features", ML_DIR / "relevance_ranker_v2" / "features.py")
m2_features_mod = _load_features_module("m2_features", ML_DIR / "caller_disambiguator" / "features.py")
m3_features_mod = _load_features_module("m3_features", ML_DIR / "source_token_filter" / "features.py")
m4_features_mod = _load_features_module("m4_features", ML_DIR / "symbol_importance" / "features.py")

m1_extract_features = m1_features_mod.extract_features
M1_FEATURES = m1_features_mod.FEATURE_COUNT
m2_extract_features = m2_features_mod.extract_features
M2_FEATURES = m2_features_mod.FEATURE_COUNT
m3_extract_features = m3_features_mod.extract_features
M3_FEATURES = m3_features_mod.FEATURE_COUNT
m4_extract_features = m4_features_mod.extract_features
M4_FEATURES = m4_features_mod.FEATURE_COUNT

# ═══════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════

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
    if min_prefix <= len(longer) and longer[:min_prefix] == shorter[:min_prefix]:
        return True
    return False


def _char_bigrams(text: str) -> set[str]:
    t = re.sub(r'[^a-z]', '', text.lower())
    if len(t) < 2:
        return set()
    return {t[i:i+2] for i in range(len(t) - 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 2048,
    patience: int = 8,
) -> dict:
    """Train a binary classifier with early stopping. Returns best metrics."""
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
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
        train_loss = train_correct = train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_correct += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
            train_total += len(xb)

        # Validate
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(xb)
                val_correct += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
                val_total += len(xb)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        val_loss_avg = val_loss / val_total
        scheduler.step(val_loss_avg)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
              f"val_loss={val_loss_avg:.4f} | {elapsed:.1f}s")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
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

    # Final test on val set with best weights
    val_correct = val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb).squeeze(-1)
            val_correct += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
            val_total += len(xb)

    best_val_acc = val_correct / val_total
    print(f"  Best epoch: {best_epoch}, val_acc={best_val_acc:.4f}")
    return {"best_epoch": best_epoch, "val_acc": best_val_acc}


def export_weights(model: nn.Module, out_path: Path):
    """Export weights in W0/b0 format (pre-transposed for x @ W + b)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()

    # Collect Linear layers in order
    layer_keys = sorted(
        set(k.rsplit(".", 1)[0] for k in sd if "weight" in k),
        key=lambda k: int(re.search(r'\d+', k).group()) if re.search(r'\d+', k) else 0,
    )

    save_dict = {}
    for i, key_prefix in enumerate(layer_keys):
        w = sd[f"{key_prefix}.weight"].cpu().numpy()  # (out, in)
        b = sd[f"{key_prefix}.bias"].cpu().numpy()      # (out,)
        # Transpose to (in, out) for x @ W + b
        save_dict[f"W{i}"] = w.T.astype(np.float32)
        save_dict[f"b{i}"] = b.astype(np.float32)

    np.savez(out_path, **save_dict)
    size = out_path.stat().st_size
    print(f"  Saved {out_path.name} ({size:,} bytes)")


# ═══════════════════════════════════════════════════════════════════════
#  M1: RELEVANCE RANKER V2 (42 features)
# ═══════════════════════════════════════════════════════════════════════

class RelevanceRankerV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M1_FEATURES, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)


# Domains for diverse queries
_VERBS = ["get", "set", "create", "delete", "update", "find", "process", "handle",
          "validate", "check", "send", "receive", "parse", "build", "init", "load",
          "save", "fetch", "compute", "format", "register", "dispatch", "execute",
          "run", "start", "stop", "close", "open", "connect", "serialize", "decode"]
_NOUNS = ["user", "request", "response", "connection", "session", "token", "message",
          "event", "task", "job", "config", "handler", "middleware", "router", "endpoint",
          "model", "schema", "query", "result", "error", "cache", "database", "transaction",
          "template", "controller", "service", "client", "server", "worker", "manager"]
_PATHS = [
    "app/models.py", "app/views.py", "app/routes/auth.py", "api/endpoints.py",
    "services/user.py", "db/session.py", "cache/backend.py", "auth/jwt.py",
    "middleware/cors.py", "utils/helpers.py", "config/settings.py", "tasks/worker.py",
    "tests/test_auth.py", "tests/test_models.py", "core/engine.py", "payments/stripe.py",
    "search/indexer.py", "monitoring/metrics.py", "cli/commands.py", "data/pipeline.py",
    "events/bus.py", "http/client.py", "storage/s3.py", "schemas/user.py",
    "serializers/json.py", "validators/core.py", "ml/model.py", "plugins/__init__.py",
]
_DOCS = [
    "Fetch a user by ID.", "Process incoming request.", "Validate user input.",
    "Send email notification.", "Create database session.", "Parse configuration file.",
    "Build SQL query.", "Handle incoming request.", "Compute hash of data.",
    "Format date string.", "Initialize application.", "Load model weights.",
    "Save checkpoint.", "Execute background task.", "Run migrations.",
    "Connect to database.", "Serialize response.", "Decode JWT token.",
    "Register event handler.", "Dispatch message to queue.",
    "Apply rate limiting to incoming API requests.",
    "Execute a function with exponential backoff retry logic.",
    "Manage the caching layer for frequently accessed data.",
    "Process a payment charge through the payment gateway.",
    "Render a Jinja2 template to HTML string.",
    "Authenticate user credentials against the database.",
    "Validate JSON schema against the specification.",
    "",  # Some functions have no docstring
]


def _gen_m1_data(n: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Generate M1 relevance training data."""
    X, y = [], []

    # Function/class names pool
    names_pool = []
    for v in _VERBS:
        for n_ in _NOUNS[:10]:
            names_pool.append(f"{v}_{n_}")
    for n_ in _NOUNS:
        names_pool.append(f"{n_.capitalize()}Manager")
        names_pool.append(f"{n_.capitalize()}Handler")
        names_pool.append(f"{n_.capitalize()}Service")
    dunders = ["__init__", "__repr__", "__str__", "__eq__", "__hash__",
               "__len__", "__getitem__", "__setitem__", "__iter__", "__call__"]
    names_pool.extend(dunders)

    target_per_category = n // 10

    # 1. Exact match positives (name tokens = query)
    for _ in range(target_per_category * 2):
        name = rng.choice(names_pool)
        tokens = _split_tokens(name)
        if not tokens:
            continue
        query = " ".join(tokens)
        doc = rng.choice(_DOCS)
        path = rng.choice(_PATHS)
        feat = m1_extract_features(
            query_tokens=tokens, query_lower=query, name=name,
            docstring=doc, file_path=path,
            is_dunder=name.startswith("__") and name.endswith("__"),
            is_init_file="__init__" in path,
        )
        X.append(feat); y.append(1.0)

    # 2. Partial match positives (subset of name tokens)
    for _ in range(target_per_category):
        name = rng.choice(names_pool)
        tokens = _split_tokens(name)
        if len(tokens) < 2:
            continue
        k = rng.randint(1, len(tokens))
        subset = tokens[:k]
        query = " ".join(subset)
        feat = m1_extract_features(
            query_tokens=subset, query_lower=query, name=name,
            docstring=rng.choice(_DOCS), file_path=rng.choice(_PATHS),
        )
        X.append(feat); y.append(1.0)

    # 3. Docstring match positives
    for _ in range(target_per_category):
        doc = rng.choice([d for d in _DOCS if d])
        doc_tokens = _split_tokens(doc)
        if len(doc_tokens) < 3:
            continue
        k = min(4, len(doc_tokens))
        subset = rng.sample(doc_tokens, k)
        query = " ".join(subset)
        name = rng.choice(names_pool)
        feat = m1_extract_features(
            query_tokens=subset, query_lower=query, name=name,
            docstring=doc, file_path=rng.choice(_PATHS),
        )
        X.append(feat); y.append(1.0)

    # 4. Path match positives
    for _ in range(target_per_category):
        path = rng.choice(_PATHS)
        path_tokens = _split_tokens(path)
        if not path_tokens:
            continue
        qt = rng.choice(path_tokens)
        name = rng.choice(names_pool)
        feat = m1_extract_features(
            query_tokens=[qt], query_lower=qt, name=name,
            docstring=rng.choice(_DOCS), file_path=path,
        )
        X.append(feat); y.append(1.0)

    # 5. Verb+noun semantic positives
    for _ in range(target_per_category):
        verb = rng.choice(_VERBS)
        noun = rng.choice(_NOUNS)
        query = f"{verb} {noun}"
        tokens = [verb, noun]
        # Find a name that contains at least one token
        matching_names = [n for n in names_pool if any(t in _split_tokens(n) for t in tokens)]
        if matching_names:
            name = rng.choice(matching_names)
        else:
            name = f"{verb}_{noun}"
        feat = m1_extract_features(
            query_tokens=tokens, query_lower=query, name=name,
            docstring=rng.choice(_DOCS), file_path=rng.choice(_PATHS),
        )
        X.append(feat); y.append(1.0)

    # 6. Random mismatch negatives (no token overlap)
    for _ in range(target_per_category * 2):
        name = rng.choice(names_pool)
        other_name = rng.choice(names_pool)
        tokens = _split_tokens(other_name)
        name_tokens = set(_split_tokens(name))
        if not tokens or (set(tokens) & name_tokens):
            continue
        query = " ".join(tokens)
        feat = m1_extract_features(
            query_tokens=tokens, query_lower=query, name=name,
            docstring="", file_path=rng.choice(_PATHS),
        )
        X.append(feat); y.append(0.0)

    # 7. Dunder hard negatives
    for _ in range(target_per_category):
        dunder = rng.choice(dunders)
        other_name = rng.choice([n for n in names_pool if not n.startswith("__")])
        tokens = _split_tokens(other_name)
        if not tokens:
            continue
        query = " ".join(tokens)
        feat = m1_extract_features(
            query_tokens=tokens, query_lower=query, name=dunder,
            docstring="Magic method.", file_path=rng.choice(_PATHS),
            is_dunder=True,
        )
        X.append(feat); y.append(0.0)

    # 8. __init__.py negatives
    for _ in range(target_per_category):
        name = rng.choice(names_pool)
        tokens = _split_tokens(rng.choice(names_pool))
        if not tokens:
            continue
        query = " ".join(tokens)
        feat = m1_extract_features(
            query_tokens=tokens, query_lower=query, name=name,
            docstring="", file_path="plugins/__init__.py",
            is_init_file=True,
        )
        X.append(feat); y.append(0.0)

    # 9. Verb-only hard negatives (too generic)
    for _ in range(target_per_category // 2):
        verb = rng.choice(_VERBS)
        # Pick a name that doesn't contain the verb
        candidates = [n for n in names_pool if verb not in _split_tokens(n)]
        if not candidates:
            continue
        name = rng.choice(candidates)
        feat = m1_extract_features(
            query_tokens=[verb], query_lower=verb, name=name,
            docstring=rng.choice(_DOCS), file_path=rng.choice(_PATHS),
        )
        X.append(feat); y.append(0.0)

    # Shuffle before returning
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    idx = np.arange(len(X_arr))
    np.random.seed(42)
    np.random.shuffle(idx)
    return X_arr[idx], y_arr[idx]


# ═══════════════════════════════════════════════════════════════════════
#  M2: CALLER DISAMBIGUATOR (25 features)
# ═══════════════════════════════════════════════════════════════════════

class CallerDisambiguator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M2_FEATURES, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x)


_CLASSES = [
    ("DatabaseSession", "db/session.py", "db"), ("CacheManager", "cache/manager.py", "cache"),
    ("UserRepository", "models/user.py", "models"), ("HTTPClient", "http/client.py", "http"),
    ("TaskWorker", "tasks/worker.py", "tasks"), ("EventBus", "events/bus.py", "events"),
    ("AuthProvider", "auth/provider.py", "auth"), ("PaymentProcessor", "payments/stripe.py", "payments"),
    ("SearchEngine", "search/engine.py", "search"), ("Router", "routes/router.py", "routes"),
    ("Middleware", "middleware/base.py", "middleware"), ("Serializer", "serializers/base.py", "serializers"),
    ("Validator", "validators/core.py", "validators"), ("Logger", "logging/logger.py", "logging"),
    ("FileStorage", "storage/local.py", "storage"), ("MessageQueue", "queue/broker.py", "queue"),
]
_COMMON_METHODS = ["execute", "get", "set", "run", "process", "handle", "validate",
                   "create", "update", "delete", "save", "load", "close", "send", "connect"]
_RECEIVERS = ["self", "cls", "db", "session", "cache", "client", "app", "request",
              "response", "handler", "manager", "service", "worker", "engine"]


def _gen_m2_data(n: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Generate M2 caller disambiguation training data."""
    X, y = [], []
    target = n // 8

    for _ in range(target):
        # Pick correct class and method
        cls_name, cls_file, cls_pkg = rng.choice(_CLASSES)
        method = rng.choice(_COMMON_METHODS)
        receiver = rng.choice([t.lower() for t in _split_tokens(cls_name)] + ["self"])

        # POSITIVE: correct class
        feat = m2_extract_features(
            receiver_text=receiver, method_name=method,
            caller_file=rng.choice(_PATHS), caller_imports=[cls_pkg, cls_file.replace("/", ".")],
            arg_count=rng.randint(0, 4), has_dot_syntax=True,
            caller_loc=rng.randint(10, 150),
            callee_name=method, callee_class_name=cls_name,
            callee_file=cls_file, callee_module=cls_file.split("/")[-1].replace(".py", ""),
            callee_package=cls_pkg, callee_param_count=rng.randint(1, 5),
            callee_is_method=True, callee_decorators=[],
            method_count_with_same_name=rng.randint(2, 8),
            method_name_frequency=rng.uniform(0.001, 0.05),
            receiver_type_annotation=cls_name,
        )
        X.append(feat); y.append(1.0)

        # NEGATIVE: wrong class
        wrong_cls = rng.choice([c for c in _CLASSES if c[0] != cls_name])
        feat = m2_extract_features(
            receiver_text=receiver, method_name=method,
            caller_file=rng.choice(_PATHS), caller_imports=[cls_pkg],
            arg_count=rng.randint(0, 4), has_dot_syntax=True,
            caller_loc=rng.randint(10, 150),
            callee_name=method, callee_class_name=wrong_cls[0],
            callee_file=wrong_cls[1], callee_module=wrong_cls[1].split("/")[-1].replace(".py", ""),
            callee_package=wrong_cls[2], callee_param_count=rng.randint(1, 5),
            callee_is_method=True, callee_decorators=[],
            method_count_with_same_name=rng.randint(2, 8),
            method_name_frequency=rng.uniform(0.001, 0.05),
            receiver_type_annotation=cls_name,
        )
        X.append(feat); y.append(0.0)

    # Same-class self.method() -> positive
    for _ in range(target):
        cls_name, cls_file, cls_pkg = rng.choice(_CLASSES)
        method = rng.choice(_COMMON_METHODS)
        feat = m2_extract_features(
            receiver_text="self", method_name=method,
            caller_file=cls_file, caller_imports=[],
            arg_count=rng.randint(0, 3), has_dot_syntax=True,
            caller_loc=rng.randint(10, 100),
            callee_name=method, callee_class_name=cls_name,
            callee_file=cls_file, callee_module=cls_file.split("/")[-1].replace(".py", ""),
            callee_package=cls_pkg, callee_param_count=rng.randint(1, 4),
            callee_is_method=True, callee_decorators=[],
            method_count_with_same_name=1,
            method_name_frequency=rng.uniform(0.001, 0.02),
            receiver_type_annotation=cls_name,
        )
        X.append(feat); y.append(1.0)

    # No import -> negative
    for _ in range(target):
        cls_name, cls_file, cls_pkg = rng.choice(_CLASSES)
        method = rng.choice(_COMMON_METHODS)
        feat = m2_extract_features(
            receiver_text=rng.choice(_RECEIVERS), method_name=method,
            caller_file=rng.choice(_PATHS), caller_imports=[],
            arg_count=rng.randint(0, 4), has_dot_syntax=True,
            caller_loc=rng.randint(10, 150),
            callee_name=method, callee_class_name=cls_name,
            callee_file=cls_file, callee_module=cls_file.split("/")[-1].replace(".py", ""),
            callee_package=cls_pkg, callee_param_count=rng.randint(1, 5),
            callee_is_method=True, callee_decorators=[],
            method_count_with_same_name=rng.randint(3, 10),
            method_name_frequency=rng.uniform(0.01, 0.1),
            receiver_type_annotation="",
        )
        X.append(feat); y.append(0.0)

    # Receiver matches class name tokens -> positive
    for _ in range(target):
        cls_name, cls_file, cls_pkg = rng.choice(_CLASSES)
        method = rng.choice(_COMMON_METHODS)
        cls_tokens = _split_tokens(cls_name)
        receiver = "_".join(cls_tokens[-2:]) if len(cls_tokens) > 1 else cls_tokens[0] if cls_tokens else "obj"
        feat = m2_extract_features(
            receiver_text=receiver, method_name=method,
            caller_file=rng.choice(_PATHS), caller_imports=[cls_pkg],
            arg_count=rng.randint(0, 4), has_dot_syntax=True,
            caller_loc=rng.randint(10, 150),
            callee_name=method, callee_class_name=cls_name,
            callee_file=cls_file, callee_module=cls_file.split("/")[-1].replace(".py", ""),
            callee_package=cls_pkg, callee_param_count=rng.randint(1, 5),
            callee_is_method=True, callee_decorators=[],
            method_count_with_same_name=rng.randint(2, 6),
            method_name_frequency=rng.uniform(0.001, 0.05),
            receiver_type_annotation=cls_name,
        )
        X.append(feat); y.append(1.0)

    # Different directory, no type annotation -> negative
    for _ in range(target):
        cls_name, cls_file, cls_pkg = rng.choice(_CLASSES)
        method = rng.choice(_COMMON_METHODS)
        different_paths = [p for p in _PATHS if not p.startswith(cls_pkg)]
        caller_file = rng.choice(different_paths) if different_paths else rng.choice(_PATHS)
        feat = m2_extract_features(
            receiver_text=rng.choice(["obj", "item", "thing", "other"]),
            method_name=method,
            caller_file=caller_file, caller_imports=[],
            arg_count=rng.randint(0, 4), has_dot_syntax=True,
            caller_loc=rng.randint(10, 150),
            callee_name=method, callee_class_name=cls_name,
            callee_file=cls_file, callee_module=cls_file.split("/")[-1].replace(".py", ""),
            callee_package=cls_pkg, callee_param_count=rng.randint(1, 5),
            callee_is_method=True, callee_decorators=[],
            method_count_with_same_name=rng.randint(3, 10),
            method_name_frequency=rng.uniform(0.01, 0.1),
            receiver_type_annotation="",
        )
        X.append(feat); y.append(0.0)

    # Standalone function (no dot syntax) -> negative for method
    for _ in range(target):
        cls_name, cls_file, cls_pkg = rng.choice(_CLASSES)
        method = rng.choice(_COMMON_METHODS)
        feat = m2_extract_features(
            receiver_text="", method_name=method,
            caller_file=rng.choice(_PATHS), caller_imports=[],
            arg_count=rng.randint(0, 4), has_dot_syntax=False,
            caller_loc=rng.randint(10, 150),
            callee_name=method, callee_class_name=cls_name,
            callee_file=cls_file, callee_module=cls_file.split("/")[-1].replace(".py", ""),
            callee_package=cls_pkg, callee_param_count=rng.randint(1, 5),
            callee_is_method=True, callee_decorators=[],
            method_count_with_same_name=rng.randint(2, 8),
            method_name_frequency=rng.uniform(0.01, 0.1),
            receiver_type_annotation="",
        )
        X.append(feat); y.append(0.0)

    # Property/classmethod callee distinctions
    for _ in range(target):
        cls_name, cls_file, cls_pkg = rng.choice(_CLASSES)
        method = rng.choice(_COMMON_METHODS)
        is_prop = rng.random() > 0.5
        feat = m2_extract_features(
            receiver_text="self" if not is_prop else cls_name,
            method_name=method,
            caller_file=cls_file, caller_imports=[cls_pkg],
            arg_count=0 if is_prop else rng.randint(0, 3),
            has_dot_syntax=True,
            caller_loc=rng.randint(10, 100),
            callee_name=method, callee_class_name=cls_name,
            callee_file=cls_file, callee_module=cls_file.split("/")[-1].replace(".py", ""),
            callee_package=cls_pkg, callee_param_count=0 if is_prop else rng.randint(1, 4),
            callee_is_method=True,
            callee_decorators=["property"] if is_prop else ["classmethod"] if rng.random() > 0.5 else [],
            method_count_with_same_name=1,
            method_name_frequency=rng.uniform(0.001, 0.02),
            receiver_type_annotation=cls_name,
        )
        X.append(feat); y.append(1.0)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    idx = np.arange(len(X_arr))
    np.random.seed(43)
    np.random.shuffle(idx)
    return X_arr[idx], y_arr[idx]


# ═══════════════════════════════════════════════════════════════════════
#  M3: SOURCE TOKEN FILTER (15 features)
# ═══════════════════════════════════════════════════════════════════════

class SourceTokenFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M3_FEATURES, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
    def forward(self, x): return self.net(x)


def _gen_m3_data(n: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Generate M3 source token filter training data."""
    X, y = [], []

    # Realistic function templates
    funcs = [
        {"name": "get_user_by_id", "doc": "Fetch a user record by their unique identifier.",
         "params": ["user_id", "include_deleted"], "decorators": [],
         "source": 'def get_user_by_id(user_id: int, include_deleted: bool = False):\n    """Fetch a user record by their unique identifier."""\n    import logging\n    # Check cache first\n    cached = self.cache.get(f"user:{user_id}")\n    if cached:\n        return cached\n    query = self.session.query(User).filter(User.id == user_id)\n    if not include_deleted:\n        query = query.filter(User.deleted_at.is_(None))\n    result = query.first()\n    return result\n'},
        {"name": "process_payment", "doc": "Process a payment charge through Stripe.",
         "params": ["amount", "currency", "source_token"], "decorators": ["router.post"],
         "source": 'def process_payment(amount: float, currency: str, source_token: str):\n    """Process a payment charge through Stripe."""\n    import stripe\n    # Validate amount\n    if amount <= 0:\n        raise ValueError("Amount must be positive")\n    intent = stripe.PaymentIntent.create(\n        amount=int(amount * 100),\n        currency=currency,\n        payment_method=source_token,\n    )\n    return {"status": intent.status, "id": intent.id}\n'},
        {"name": "validate_email", "doc": "Validate email address format.",
         "params": ["email"], "decorators": [],
         "source": 'def validate_email(email: str) -> bool:\n    """Validate email address format."""\n    import re\n    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"\n    return bool(re.match(pattern, email))\n'},
        {"name": "build_query", "doc": "Build an SQL query with dynamic filters.",
         "params": ["table", "filters", "order_by"], "decorators": [],
         "source": 'def build_query(table: str, filters: dict, order_by: str = None):\n    """Build an SQL query with dynamic filters."""\n    sql = f"SELECT * FROM {table}"\n    conditions = []\n    for key, value in filters.items():\n        conditions.append(f"{key} = %s")\n    if conditions:\n        sql += " WHERE " + " AND ".join(conditions)\n    if order_by:\n        sql += f" ORDER BY {order_by}"\n    return sql\n'},
        {"name": "send_notification", "doc": "Send push notification to user devices.",
         "params": ["user_id", "title", "body"], "decorators": ["task"],
         "source": 'def send_notification(user_id: int, title: str, body: str):\n    """Send push notification to user devices."""\n    from firebase_admin import messaging\n    tokens = get_device_tokens(user_id)\n    message = messaging.MulticastMessage(\n        notification=messaging.Notification(title=title, body=body),\n        tokens=tokens,\n    )\n    response = messaging.send_multicast(message)\n    logger.info(f"Sent {response.success_count}/{len(tokens)} notifications")\n'},
        {"name": "parse_config", "doc": "Parse YAML configuration file.",
         "params": ["path"], "decorators": [],
         "source": 'def parse_config(path: str) -> dict:\n    """Parse YAML configuration file."""\n    import yaml\n    with open(path, "r") as f:\n        config = yaml.safe_load(f)\n    # Validate required keys\n    required = ["database", "cache", "secret_key"]\n    for key in required:\n        if key not in config:\n            raise KeyError(f"Missing required config key: {key}")\n    return config\n'},
        {"name": "hash_password", "doc": "Hash a password using bcrypt.",
         "params": ["password"], "decorators": [],
         "source": 'def hash_password(password: str) -> str:\n    """Hash a password using bcrypt."""\n    import bcrypt\n    salt = bcrypt.gensalt(rounds=12)\n    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)\n    return hashed.decode("utf-8")\n'},
        {"name": "create_session", "doc": "Create a new database session.",
         "params": ["engine"], "decorators": ["contextmanager"],
         "source": 'def create_session(engine):\n    """Create a new database session."""\n    from sqlalchemy.orm import Session\n    session = Session(bind=engine)\n    try:\n        yield session\n        session.commit()\n    except Exception:\n        session.rollback()\n        raise\n    finally:\n        session.close()\n'},
    ]

    target_per_func = n // len(funcs)

    for func in funcs:
        name = func["name"]
        doc = func["doc"]
        params = func["params"]
        decorators = func["decorators"]
        source = func["source"]
        source_tokens = set(_split_tokens(source[:3000]))

        for token in source_tokens:
            name_tokens = set(_split_tokens(name))
            doc_tokens = set(_split_tokens(doc))
            param_tokens = set()
            for p in params:
                param_tokens.update(_split_tokens(p))

            # Classify
            if token in name_tokens or token in doc_tokens or token in param_tokens:
                label = 1.0
            elif any(token in line.lower() for line in source.split("\n")
                     if line.strip().startswith("import ") or line.strip().startswith("from ")):
                label = 0.0
            elif any(token in line.lower() for line in source.split("\n")
                     if line.strip().startswith("#")):
                label = 0.0
            else:
                # Body logic tokens - weak negative
                label = 0.0 if rng.random() > 0.3 else 1.0

            feat = m3_extract_features(
                token=token, function_name=name, docstring=doc,
                param_names=params, decorator_names=decorators,
                source_code=source,
            )
            X.append(feat); y.append(label)

            if len(X) >= n:
                break
        if len(X) >= n:
            break

    # Pad with more varied examples if needed
    while len(X) < n:
        func = rng.choice(funcs)
        name = func["name"]
        source = func["source"]
        name_tokens = list(set(_split_tokens(name)))
        if name_tokens:
            token = rng.choice(name_tokens)
            feat = m3_extract_features(
                token=token, function_name=name, docstring=func["doc"],
                param_names=func["params"], decorator_names=func["decorators"],
                source_code=source,
            )
            X.append(feat); y.append(1.0)

        # Random noise token
        noise = rng.choice(["numpy", "logging", "typing", "self", "None", "True",
                            "return", "import", "class", "lambda", "async", "await",
                            "json", "yaml", "os", "sys", "re", "math"])
        feat = m3_extract_features(
            token=noise, function_name=name, docstring=func["doc"],
            param_names=func["params"], decorator_names=func["decorators"],
            source_code=source,
        )
        X.append(feat); y.append(0.0)

    X_arr = np.array(X[:n], dtype=np.float32)
    y_arr = np.array(y[:n], dtype=np.float32)
    idx = np.arange(len(X_arr))
    np.random.seed(44)
    np.random.shuffle(idx)
    return X_arr[idx], y_arr[idx]


# ═══════════════════════════════════════════════════════════════════════
#  M4: SYMBOL IMPORTANCE (18 features)
# ═══════════════════════════════════════════════════════════════════════

class SymbolImportanceRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M4_FEATURES, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
    def forward(self, x): return self.net(x)


def _gen_m4_data(n: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Generate M4 symbol importance training data."""
    X, y = [], []
    target = n // 6

    # 1. API endpoints (important)
    for _ in range(target):
        verb = rng.choice(_VERBS)
        noun = rng.choice(_NOUNS)
        name = f"{verb}_{noun}"
        path = rng.choice(["app/routes/auth.py", "api/endpoints.py", "routes/users.py",
                          "views/products.py", "controllers/orders.py"])
        feat = m4_extract_features(
            name=name, file_path=path, source=f"def {name}():\n    pass\n",
            docstring=f"{verb.capitalize()} {noun}.",
            param_names=["request"], decorators=["router.get", "router.post"],
            caller_count=rng.randint(5, 50),
            cross_file_caller_count=rng.randint(3, 20),
            is_exported_in_all=rng.random() > 0.5,
            import_count=rng.randint(5, 30),
        )
        X.append(feat); y.append(1.0)

    # 2. Base classes with subclasses (important)
    for _ in range(target):
        name = rng.choice(["BaseModel", "AbstractHandler", "BaseService", "BaseRepository",
                           "AbstractValidator", "BaseSerializer", "AbstractProvider"])
        feat = m4_extract_features(
            name=name, file_path="core/base.py",
            source=f"class {name}(ABC):\n    pass\n",
            docstring=f"Abstract base {name}.",
            base_classes=["ABC"], methods=["process", "validate", "execute"],
            caller_count=rng.randint(10, 60),
            cross_file_caller_count=rng.randint(5, 30),
            subclass_count=rng.randint(3, 15),
            import_count=rng.randint(10, 40),
        )
        X.append(feat); y.append(1.0)

    # 3. Widely imported utilities (important)
    for _ in range(target):
        name = rng.choice(["get_settings", "get_db", "get_logger", "create_app",
                           "init_database", "setup_logging", "get_cache", "format_response"])
        feat = m4_extract_features(
            name=name, file_path="core/utils.py",
            source=f"def {name}():\n    pass\n",
            docstring=f"Get or create {name.replace('get_', '')}.",
            param_names=[],
            caller_count=rng.randint(20, 100),
            cross_file_caller_count=rng.randint(10, 50),
            is_exported_in_all=True,
            import_count=rng.randint(15, 50),
        )
        X.append(feat); y.append(1.0)

    # 4. Test functions (not important)
    for _ in range(target):
        name = f"test_{rng.choice(_VERBS)}_{rng.choice(_NOUNS)}"
        feat = m4_extract_features(
            name=name, file_path=f"tests/test_{rng.choice(_NOUNS)}.py",
            source=f"def {name}():\n    assert True\n",
            docstring=f"Test {name.replace('test_', '')}.",
            param_names=["client"], decorators=["pytest.mark.asyncio"],
            caller_count=0, cross_file_caller_count=0,
        )
        X.append(feat); y.append(0.0)

    # 5. Private helpers (not important)
    for _ in range(target):
        name = f"_{rng.choice(_VERBS)}_{rng.choice(_NOUNS)}"
        feat = m4_extract_features(
            name=name, file_path=rng.choice(_PATHS),
            source=f"def {name}():\n    pass\n",
            docstring="",
            param_names=rng.sample(["x", "y", "data", "value"], rng.randint(0, 2)),
            caller_count=rng.randint(0, 3),
            cross_file_caller_count=0,
        )
        X.append(feat); y.append(0.0)

    # 6. Dunder methods (not important)
    for _ in range(target):
        name = rng.choice(["__init__", "__repr__", "__str__", "__eq__", "__hash__",
                          "__len__", "__iter__", "__getitem__"])
        feat = m4_extract_features(
            name=name, file_path=rng.choice(_PATHS),
            source=f"def {name}(self):\n    pass\n",
            docstring=f"Magic method {name}.",
            param_names=["self"],
            caller_count=rng.randint(0, 5),
            cross_file_caller_count=0,
        )
        X.append(feat); y.append(0.0)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    idx = np.arange(len(X_arr))
    np.random.seed(45)
    np.random.shuffle(idx)
    return X_arr[idx], y_arr[idx]


# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Train all 4 MLPs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--quick", action="store_true", help="Quick mode: less data, fewer epochs")
    args = parser.parse_args()

    epochs = 15 if args.quick else args.epochs
    rng = random.Random(42)

    if args.quick:
        data_sizes = {"m1": 50_000, "m2": 30_000, "m3": 20_000, "m4": 20_000}
    else:
        data_sizes = {"m1": 500_000, "m2": 200_000, "m3": 200_000, "m4": 100_000}

    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print()

    PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── M1: Relevance Ranker v2 ──────────────────────────────────────
    print("=" * 60)
    print("M1: Relevance Ranker v2 (42 features)")
    print("=" * 60)
    print(f"Generating {data_sizes['m1']:,} training examples...")
    X, y = _gen_m1_data(data_sizes["m1"], rng)
    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"  Train: {len(X_train):,} (pos={int(y_train.sum()):,}, neg={int(len(y_train)-y_train.sum()):,})")
    print(f"  Val:   {len(X_val):,} (pos={int(y_val.sum()):,}, neg={int(len(y_val)-y_val.sum()):,})")

    model = RelevanceRankerV2()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    metrics = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)
    results["M1"] = metrics

    out_path = PRODUCTION_MODEL_DIR / "relevance_weights_v2.npz"
    export_weights(model, out_path)
    print()

    # ── M2: Caller Disambiguator ─────────────────────────────────────
    print("=" * 60)
    print("M2: Caller Disambiguator (25 features)")
    print("=" * 60)
    print(f"Generating {data_sizes['m2']:,} training examples...")
    X, y = _gen_m2_data(data_sizes["m2"], rng)
    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"  Train: {len(X_train):,} (pos={int(y_train.sum()):,}, neg={int(len(y_train)-y_train.sum()):,})")
    print(f"  Val:   {len(X_val):,} (pos={int(y_val.sum()):,}, neg={int(len(y_val)-y_val.sum()):,})")

    model = CallerDisambiguator()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    metrics = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)
    results["M2"] = metrics

    out_path = PRODUCTION_MODEL_DIR / "caller_disambig_weights.npz"
    export_weights(model, out_path)
    print()

    # ── M3: Source Token Filter ──────────────────────────────────────
    print("=" * 60)
    print("M3: Source Token Filter (15 features)")
    print("=" * 60)
    print(f"Generating {data_sizes['m3']:,} training examples...")
    X, y = _gen_m3_data(data_sizes["m3"], rng)
    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"  Train: {len(X_train):,} (pos={int(y_train.sum()):,}, neg={int(len(y_train)-y_train.sum()):,})")
    print(f"  Val:   {len(X_val):,} (pos={int(y_val.sum()):,}, neg={int(len(y_val)-y_val.sum()):,})")

    model = SourceTokenFilter()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    metrics = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)
    results["M3"] = metrics

    out_path = PRODUCTION_MODEL_DIR / "source_token_filter_weights.npz"
    export_weights(model, out_path)
    print()

    # ── M4: Symbol Importance ────────────────────────────────────────
    print("=" * 60)
    print("M4: Symbol Importance (18 features)")
    print("=" * 60)
    print(f"Generating {data_sizes['m4']:,} training examples...")
    X, y = _gen_m4_data(data_sizes["m4"], rng)
    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"  Train: {len(X_train):,} (pos={int(y_train.sum()):,}, neg={int(len(y_train)-y_train.sum()):,})")
    print(f"  Val:   {len(X_val):,} (pos={int(y_val.sum()):,}, neg={int(len(y_val)-y_val.sum()):,})")

    model = SymbolImportanceRanker()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    metrics = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)
    results["M4"] = metrics

    out_path = PRODUCTION_MODEL_DIR / "symbol_importance_weights.npz"
    export_weights(model, out_path)
    print()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    all_good = True
    for name, metrics in results.items():
        status = "PASS" if metrics["val_acc"] >= 0.88 else "WARN"
        if metrics["val_acc"] < 0.85:
            status = "FAIL"
            all_good = False
        print(f"  {name}: val_acc={metrics['val_acc']:.4f} (epoch {metrics['best_epoch']}) [{status}]")

    print()
    print("Weight files in production:")
    for f in sorted(PRODUCTION_MODEL_DIR.glob("*.npz")):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")

    if all_good:
        print("\nAll models trained successfully!")
    else:
        print("\nWARNING: Some models have low accuracy. Consider adding more training data.")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
