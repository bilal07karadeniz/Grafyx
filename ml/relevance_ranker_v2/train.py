"""Train Relevance Ranker v2 MLP (42 -> 128 -> 64 -> 1).

Reads JSONL data produced by generate_data.py.
Exports pre-transposed weight matrices as relevance_weights_v2.npz
for numpy-only inference.

Usage:
    python train.py [--epochs 40] [--lr 1e-3] [--patience 8] [--batch-size 2048]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERROR: PyTorch is required for training.")
    print("  pip install torch")
    sys.exit(1)

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"

FEATURE_COUNT = 42


# ── Model ─────────────────────────────────────────────────────────


class RelevanceRankerV2(nn.Module):
    """Binary relevance classifier: 42 -> 128 -> 64 -> 1."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Data loading ──────────────────────────────────────────────────


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a JSONL split into numpy arrays."""
    path = DATA_DIR / f"{name}.jsonl"
    features_list = []
    labels_list = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            features_list.append(obj["features"])
            labels_list.append(obj["label"])
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.float32)
    print(f"  {name}: {len(X)} examples, "
          f"pos={int(y.sum())}, neg={int(len(y) - y.sum())}, "
          f"feature_dim={X.shape[1]}")
    return X, y


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap numpy arrays into a PyTorch DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ── Training loop ─────────────────────────────────────────────────


def train(
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 2048,
    patience: int = 8,
):
    """Train the model with early stopping on validation loss."""
    print("Loading data...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    assert X_train.shape[1] == FEATURE_COUNT, (
        f"Expected {FEATURE_COUNT} features, got {X_train.shape[1]}"
    )

    train_loader = make_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RelevanceRankerV2().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += len(X_batch)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * len(X_batch)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += len(X_batch)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # ── Early stopping ────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            # Save best checkpoint
            torch.save(model.state_dict(), MODEL_DIR / "best_checkpoint.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"\nBest epoch: {best_epoch}, "
          f"val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")

    # ── Export weights ────────────────────────────────────────────
    print("\nExporting weights...")
    model.load_state_dict(torch.load(MODEL_DIR / "best_checkpoint.pt"))
    model.eval()

    export_weights(model)

    # ── Final test evaluation ─────────────────────────────────────
    print("\nEvaluating on test set...")
    X_test, y_test = load_split("test")
    test_loader = make_dataloader(X_test, y_test, batch_size, shuffle=False)

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            test_correct += (preds == y_batch).sum().item()
            test_total += len(X_batch)

    test_acc = test_correct / test_total
    print(f"Test accuracy: {test_acc:.4f} ({test_correct}/{test_total})")


def export_weights(model: RelevanceRankerV2):
    """Export pre-transposed weights for numpy-only inference.

    The runtime scorer computes:
        x = relu(x @ w1T + b1)
        x = relu(x @ w2T + b2)
        logit = x @ w3T + b3

    We store w1T, w2T, w3T (already transposed) so inference
    is a simple matmul with no transpose at runtime.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    sd = model.state_dict()
    # net.0 = Linear(42, 128), net.3 = Linear(128, 64), net.5 = Linear(64, 1)
    w1 = sd["net.0.weight"].cpu().numpy()  # (128, 42)
    b1 = sd["net.0.bias"].cpu().numpy()    # (128,)
    w2 = sd["net.3.weight"].cpu().numpy()  # (64, 128)
    b2 = sd["net.3.bias"].cpu().numpy()    # (64,)
    w3 = sd["net.5.weight"].cpu().numpy()  # (1, 64)
    b3 = sd["net.5.bias"].cpu().numpy()    # (1,)

    out_path = MODEL_DIR / "relevance_weights_v2.npz"
    np.savez(
        out_path,
        w1=w1,  # (128, 42) — will be transposed at load time: (42, 128)
        b1=b1,
        w2=w2,  # (64, 128) -> (128, 64)
        b2=b2,
        w3=w3,  # (1, 64)  -> (64, 1)
        b3=b3,
    )
    print(f"  Saved {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"  Shapes: w1={w1.shape}, w2={w2.shape}, w3={w3.shape}")


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train Relevance Ranker v2 MLP"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
