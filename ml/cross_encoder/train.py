"""Train the Mamba+Attention cross-encoder for reranking.

Loss: BCEWithLogitsLoss
Batch size: 256
LR: 3e-4 with cosine decay
Epochs: 15
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"


def train(
    batch_size: int = 256,
    lr: float = 3e-4,
    epochs: int = 15,
    device: str = "auto",
):
    """Train the cross-encoder reranker."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "code_search_encoder"))
    from dataset import CrossEncoderDataset
    from model import CrossEncoderReranker, export_to_numpy

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "grafyx" / "search"))
    from _tokenizer import CodeTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = CodeTokenizer()
    if not tokenizer.is_available:
        print("ERROR: BPE tokenizer not available.")
        return

    # Load data
    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "val.jsonl"
    if not train_file.exists():
        print("ERROR: Training data not found. Run generate_data.py first.")
        return

    train_ds = CrossEncoderDataset(train_file, tokenizer, max_length=512)
    val_ds = CrossEncoderDataset(val_file, tokenizer, max_length=512) if val_file.exists() else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    model = CrossEncoderReranker(vocab_size=tokenizer.vocab_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids).squeeze(-1)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = (logits > 0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 100 == 0:
                acc = correct / max(total, 1)
                print(f"  Epoch {epoch+1}/{epochs} | Step {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

        avg_loss = total_loss / len(train_loader)
        acc = correct / max(total, 1)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.3f}")

        # Validation
        if val_ds:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch["input_ids"].to(device)).squeeze(-1)
                    labels = batch["label"].to(device)
                    val_loss += criterion(logits, labels).item()
                    val_correct += ((logits > 0).float() == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(val_loader)
            val_acc = val_correct / max(val_total, 1)
            print(f"  Val: loss={val_loss:.4f}, acc={val_acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
                print(f"  Saved best model")

    # Export
    print("Exporting to numpy...")
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", weights_only=True))
    export_to_numpy(model, str(MODEL_DIR / "cross_encoder_weights.npz"))
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    train(**vars(args))
