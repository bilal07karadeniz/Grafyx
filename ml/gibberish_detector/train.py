"""Train the gibberish detector model.

Usage: python train.py

Uses PyTorch for training, exports weights as numpy (.npz) for inference.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from features import (
    build_vocabulary, build_bigram_stats, text_to_features,
    save_vocabulary, save_bigram_stats, get_feature_size,
    BIGRAM_VOCAB_SIZE,
)

# --- Hyperparameters ---
TRIGRAM_VOCAB_SIZE = 0  # Not used — bigrams provide universal coverage
HIDDEN_1 = 128
HIDDEN_2 = 32
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 30
PATIENCE = 7


class GibberishDetector(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_data(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def prepare_dataset(examples, trigram_vocab, bigram_logprobs):
    features = np.stack([
        text_to_features(ex["query"], trigram_vocab, bigram_logprobs)
        for ex in examples
    ])
    labels = np.array([ex["label"] for ex in examples], dtype=np.float32)
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))


def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def export_weights(model, path):
    state = model.state_dict()
    np.savez(path,
        w1=state["net.0.weight"].cpu().numpy(),
        b1=state["net.0.bias"].cpu().numpy(),
        w2=state["net.3.weight"].cpu().numpy(),
        b2=state["net.3.bias"].cpu().numpy(),
        w3=state["net.6.weight"].cpu().numpy(),
        b3=state["net.6.bias"].cpu().numpy(),
    )
    file_size = os.path.getsize(path) / (1024 * 1024)
    print(f"Exported weights to {path} ({file_size:.1f} MB)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_data = load_data("data/train.jsonl")
    val_data = load_data("data/val.jsonl")
    test_data = load_data("data/test.jsonl")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    print("Building trigram vocabulary...")
    train_texts = [ex["query"] for ex in train_data]
    trigram_vocab = build_vocabulary(train_texts, max_vocab=TRIGRAM_VOCAB_SIZE)
    print(f"Trigram vocabulary: {len(trigram_vocab)}")

    print("Building character bigram statistics...")
    real_texts = [ex["query"] for ex in train_data if ex["label"] == 1]
    bigram_logprobs = build_bigram_stats(real_texts)
    print(f"Bigram stats: {len(bigram_logprobs)} entries")

    os.makedirs("model", exist_ok=True)
    save_vocabulary(trigram_vocab, "model/vocab.json")
    save_bigram_stats(bigram_logprobs, "model/bigram_stats.json")

    input_size = get_feature_size(len(trigram_vocab))
    print(f"Input size: {input_size} "
          f"({BIGRAM_VOCAB_SIZE} bigrams + {len(trigram_vocab)} trigrams + 8 meta)")

    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_data, trigram_vocab, bigram_logprobs)
    val_dataset = prepare_dataset(val_data, trigram_vocab, bigram_logprobs)
    test_dataset = prepare_dataset(test_data, trigram_vocab, bigram_logprobs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = GibberishDetector(input_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    patience_counter = 0

    print("\nTraining...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        start = time.time()

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        train_acc = train_correct / train_total
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - start

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "model/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load("model/best_model.pt", weights_only=True))
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, device, criterion
    )
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    tp = ((test_preds == 1) & (test_labels == 1)).sum()
    tn = ((test_preds == 0) & (test_labels == 0)).sum()
    fp = ((test_preds == 1) & (test_labels == 0)).sum()
    fn = ((test_preds == 0) & (test_labels == 1)).sum()
    print(f"\nConfusion Matrix:")
    print(f"  TP (real->real):      {tp}")
    print(f"  TN (gib->gib):       {tn}")
    print(f"  FP (gib->real):      {fp}")
    print(f"  FN (real->gib):      {fn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    export_weights(model, "model/gibberish_weights.npz")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
