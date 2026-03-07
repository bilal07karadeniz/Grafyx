"""Train the Mamba bi-encoder for code search.

Loss: InfoNCE with in-batch negatives
Batch size: 512 (with gradient accumulation if needed)
LR: 5e-4 with cosine decay + warmup
Epochs: 10
"""
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"


def train(
    batch_size: int = 512,
    lr: float = 5e-4,
    epochs: int = 10,
    warmup_steps: int = 500,
    grad_accum: int = 1,
    device: str = "auto",
):
    """Train the code search encoder."""
    from model import CodeSearchEncoder, info_nce_loss, export_to_numpy
    from dataset import CodeSearchDataset

    # Import tokenizer for encoding
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "grafyx" / "search"))
    from _tokenizer import CodeTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (must have BPE files trained first)
    tokenizer = CodeTokenizer()
    if not tokenizer.is_available:
        print("ERROR: BPE tokenizer not trained yet. Run train_tokenizer.py first.")
        return

    # Load datasets
    train_files = sorted(DATA_DIR.glob("*train*.jsonl"))
    if not train_files:
        print("ERROR: No training data found. Run download/generate scripts first.")
        return

    print(f"Loading training data from {len(train_files)} files...")
    train_ds = CodeSearchDataset(train_files[0], tokenizer)
    # TODO: Concatenate multiple data sources

    val_files = sorted(DATA_DIR.glob("*val*.jsonl"))
    val_ds = CodeSearchDataset(val_files[0], tokenizer) if val_files else None

    train_loader = DataLoader(train_ds, batch_size=batch_size // grad_accum,
                              shuffle=True, num_workers=4, pin_memory=True)

    # Model
    model = CodeSearchEncoder(vocab_size=tokenizer.vocab_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader))

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        steps = 0

        for batch_idx, batch in enumerate(train_loader):
            query_ids = batch["query_ids"].to(device)
            code_ids = batch["code_ids"].to(device)

            query_emb = model(query_ids)
            code_emb = model(code_ids)

            loss = info_nce_loss(query_emb, code_emb) / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps += 1

            total_loss += loss.item() * grad_accum

            if batch_idx % 100 == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}/{epochs} | Step {batch_idx}/{len(train_loader)} | Loss: {avg:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: avg train loss = {avg_loss:.4f}")

        # Validation
        if val_ds:
            model.eval()
            val_loss = 0
            val_loader = DataLoader(val_ds, batch_size=batch_size // grad_accum, num_workers=2)
            with torch.no_grad():
                for batch in val_loader:
                    q = model(batch["query_ids"].to(device))
                    c = model(batch["code_ids"].to(device))
                    val_loss += info_nce_loss(q, c).item()
            val_loss /= len(val_loader)
            print(f"  Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
                print(f"  Saved best model (val_loss={val_loss:.4f})")

    # Export to numpy
    print("Exporting to numpy...")
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", weights_only=True))
    export_to_numpy(model, str(MODEL_DIR / "code_encoder_weights.npz"))
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    train(**vars(args))
