"""M5/M6 Prototype Training — All-in-one script for RTX 5070.

Trains small versions of:
  M5: Bi-encoder for semantic code search (FeedForward, 4 layers, 128 hidden)
  M6: Cross-encoder for result reranking (3 FF + 1 Attention, 128 hidden)

Also trains a BPE tokenizer from real source code.

Usage:
    python ml/train_m5m6_prototype.py              # Train everything
    python ml/train_m5m6_prototype.py --only bpe   # Just tokenizer
    python ml/train_m5m6_prototype.py --only m5    # Just M5
    python ml/train_m5m6_prototype.py --only m6    # Just M6
    python ml/train_m5m6_prototype.py --only test  # Just quality test
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ML_DIR = Path(__file__).parent
PROJECT_DIR = ML_DIR.parent
MODEL_DIR = PROJECT_DIR / "grafyx" / "search" / "model"
REAL_SYMBOLS = ML_DIR / "real_symbols.json"
ALL_SYMBOLS = ML_DIR / "all_symbols.json"
LLM_PAIRS = ML_DIR / "llm_training_pairs.json"


# ═══════════════════════════════════════════════════════════════════════
#  PART 1: BPE Tokenizer Training (pure Python, no sentencepiece)
# ═══════════════════════════════════════════════════════════════════════

def _collect_source_texts(symbols_data, max_texts: int = 10000) -> list[str]:
    """Collect source code + docstrings for tokenizer training.

    Accepts either:
    - Legacy format: list of repo dicts with "sources", "functions", "classes"
    - Flat format: list of symbol dicts with "source", "docstring" (from extract_symbols.py)
    """
    texts = []

    # Detect format: flat list of symbols vs list of repos
    if symbols_data and isinstance(symbols_data[0], dict) and "type" in symbols_data[0]:
        # Flat format (all_symbols.json)
        for sym in symbols_data:
            source = sym.get("source", "")
            if source:
                texts.append(source[:2000])
                if len(texts) >= max_texts:
                    return texts
            doc = sym.get("docstring", "")
            if doc:
                texts.append(doc[:500])
                if len(texts) >= max_texts:
                    return texts
    else:
        # Legacy format (real_symbols.json)
        for repo in symbols_data:
            for src in repo.get("sources", {}).values():
                texts.append(src[:2000])
                if len(texts) >= max_texts:
                    return texts
            for func in repo.get("functions", []):
                if func.get("docstring"):
                    texts.append(func["docstring"][:500])
                    if len(texts) >= max_texts:
                        return texts
    return texts


def _pre_tokenize(text: str) -> list[str]:
    """Split into words on whitespace/punctuation, lowercase."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def train_bpe(vocab_size: int = 8192, min_frequency: int = 3, symbols_file: str | None = None):
    """Train BPE tokenizer from real source code. Pure Python."""
    print("=" * 60)
    print("TRAINING BPE TOKENIZER")
    print("=" * 60)

    # Prefer all_symbols.json (broader coverage) if available
    sf = Path(symbols_file) if symbols_file else None
    if sf and sf.exists():
        data_path = sf
    elif ALL_SYMBOLS.exists():
        data_path = ALL_SYMBOLS
    else:
        data_path = REAL_SYMBOLS

    print(f"Loading from {data_path.name}...")
    with open(data_path, encoding="utf-8") as f:
        symbols_data = json.load(f)

    texts = _collect_source_texts(symbols_data)
    print(f"Collected {len(texts)} text samples")

    # Pre-tokenize into words
    all_words = []
    for text in texts:
        all_words.extend(_pre_tokenize(text))
    print(f"Total words: {len(all_words)}")

    # Count word frequencies
    word_freq = Counter(all_words)
    print(f"Unique words: {len(word_freq)}")

    # Initialize vocab with characters + special tokens
    special = {"<pad>": 0, "<unk>": 1, "<sep>": 2, "<cls>": 3}
    char_freq = Counter()
    for word, freq in word_freq.items():
        for ch in word:
            char_freq[ch] += freq

    # Build initial vocab: special + all chars
    vocab = dict(special)
    for ch in sorted(char_freq.keys()):
        if ch not in vocab:
            vocab[ch] = len(vocab)

    print(f"Initial vocab (chars): {len(vocab)}")

    # Represent each word as list of character tokens
    # Only keep words above min_frequency
    word_splits = {}
    for word, freq in word_freq.items():
        if freq >= min_frequency:
            word_splits[word] = (list(word), freq)

    merges = []
    target = vocab_size - len(vocab)

    t0 = time.time()
    for step in range(target):
        # Count adjacent pairs
        pair_freq = Counter()
        for chars, freq in word_splits.values():
            for i in range(len(chars) - 1):
                pair_freq[(chars[i], chars[i + 1])] += freq

        if not pair_freq:
            break

        # Find most frequent pair
        best_pair = pair_freq.most_common(1)[0]
        (a, b), best_count = best_pair

        if best_count < min_frequency:
            break

        # Merge this pair in all words
        merged = a + b
        for word in word_splits:
            chars, freq = word_splits[word]
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            word_splits[word] = (new_chars, freq)

        # Add to vocab and merges
        if merged not in vocab:
            vocab[merged] = len(vocab)
        merges.append([a, b])

        if (step + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Step {step+1}/{target} | Vocab: {len(vocab)} | "
                  f"Last merge: '{a}'+''{b}'->'{merged}' (freq={best_count}) | "
                  f"{elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nBPE training done in {elapsed:.1f}s")
    print(f"Final vocab size: {len(vocab)}")
    print(f"Merges: {len(merges)}")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    merges_path = MODEL_DIR / "bpe_merges.json"
    vocab_path = MODEL_DIR / "bpe_vocab.json"

    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges, f)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    print(f"Saved: {merges_path} ({merges_path.stat().st_size // 1024}KB)")
    print(f"Saved: {vocab_path} ({vocab_path.stat().st_size // 1024}KB)")

    # Quick test
    sys.path.insert(0, str(PROJECT_DIR / "grafyx" / "search"))
    from _tokenizer import CodeTokenizer
    tok = CodeTokenizer()
    tok._loaded = False  # Force reload
    test_texts = [
        "get user authentication",
        "def parse_config(path: str):",
        "class HttpClient(BaseClient):",
    ]
    for t in test_texts:
        ids = tok.encode(t, max_length=32)
        decoded = tok.decode(ids)
        non_pad = sum(1 for i in ids if i != 0)
        print(f"  '{t}' -> {non_pad} tokens -> '{decoded}'")

    return vocab, merges


# ═══════════════════════════════════════════════════════════════════════
#  PART 2: Data Preparation
# ═══════════════════════════════════════════════════════════════════════

def _build_symbol_text(sym: dict) -> str:
    """Build text representation for a symbol (for code encoder)."""
    name = sym.get("name", "")
    doc = (sym.get("docstring") or "")[:200]
    file_path = sym.get("file", "")
    class_name = sym.get("class_name", "")

    parts = [name]
    if class_name:
        parts.append(class_name)
    if doc:
        parts.append(doc)
    if file_path:
        # Extract module path tokens
        mod = file_path.replace("/", " ").replace("\\", " ").replace(".py", "")
        parts.append(mod)
    return " ".join(parts)


def prepare_bi_encoder_data(repos=None, llm_pairs=None, train_data_path: str | None = None):
    """Prepare (query, code_text, label) triples for bi-encoder training.

    If train_data_path is provided, loads pre-prepared data (from generate_training_data_m5.py).
    Otherwise falls back to legacy inline logic using repos + llm_pairs.
    """
    if train_data_path and Path(train_data_path).exists():
        # New path: load pre-prepared training data
        with open(train_data_path, encoding="utf-8") as f:
            pairs = json.load(f)

        positives = []
        negatives = []
        all_code_texts_set = set()

        for p in pairs:
            code_text = _build_symbol_text(p)
            all_code_texts_set.add(code_text)
            if p.get("relevance", 0.5) > 0.5:
                positives.append((p["query"], code_text))
            elif p.get("relevance", 0.5) < 0.35:
                negatives.append((p["query"], code_text))

        print(f"Bi-encoder data (from {Path(train_data_path).name}): "
              f"{len(positives)} positives, {len(negatives)} explicit negatives")
        print(f"  (in-batch negatives will provide ~{len(positives)} more per batch)")
        return positives, negatives, list(all_code_texts_set)

    # Legacy path: inline generation from repos + llm_pairs
    all_symbols = []
    for repo in (repos or []):
        for func in repo["functions"]:
            if func.get("docstring") and not func["name"].startswith("_"):
                all_symbols.append(func)
        for cls in repo["classes"]:
            if cls.get("docstring") and not cls["name"].startswith("_"):
                all_symbols.append(cls)

    symbol_texts = {
        (s["name"], s.get("file", "")): _build_symbol_text(s)
        for s in all_symbols
    }

    rng = random.Random(42)

    # From LLM pairs: HIGH = positive, MED = soft positive, LOW = negative
    positives = []  # (query, code_text) pairs
    negatives = []

    for pair in (llm_pairs or []):
        key = (pair["name"], pair.get("file", ""))
        code_text = symbol_texts.get(key)
        if not code_text:
            code_text = _build_symbol_text(pair)
        query = pair["query"]

        if pair["relevance"] > 0.7:
            positives.append((query, code_text))
        elif pair["relevance"] < 0.35:
            negatives.append((query, code_text))

    # Generate extra positives from docstrings (zero-shot)
    extra_positives = []
    for sym in rng.sample(all_symbols, min(3000, len(all_symbols))):
        doc = (sym.get("docstring") or "").strip()
        if not doc or len(doc) < 20:
            continue
        # Use first sentence of docstring as pseudo-query
        first_sent = doc.split(".")[0].split("\n")[0].strip().lower()
        if len(first_sent) > 10:
            code_text = _build_symbol_text(sym)
            extra_positives.append((first_sent, code_text))

    positives.extend(extra_positives[:2000])
    print(f"Bi-encoder data: {len(positives)} positives, {len(negatives)} explicit negatives")
    print(f"  (in-batch negatives will provide ~{len(positives)} more negatives per batch)")

    return positives, negatives, list(symbol_texts.values())


def prepare_cross_encoder_data(repos, llm_pairs):
    """Prepare (query, code_text, label) triples for cross-encoder training."""
    all_symbols = []
    for repo in repos:
        for func in repo["functions"]:
            if func.get("docstring") and not func["name"].startswith("_"):
                all_symbols.append(func)
        for cls in repo["classes"]:
            if cls.get("docstring") and not cls["name"].startswith("_"):
                all_symbols.append(cls)

    symbol_texts = {
        (s["name"], s.get("file", "")): _build_symbol_text(s)
        for s in all_symbols
    }
    all_code_texts = list(symbol_texts.values())

    rng = random.Random(42)
    triples = []  # (query, code_text, label)

    for pair in llm_pairs:
        key = (pair["name"], pair.get("file", ""))
        code_text = symbol_texts.get(key, _build_symbol_text(pair))
        query = pair["query"]

        if pair["relevance"] > 0.7:
            triples.append((query, code_text, 1.0))
            # Add a hard negative: random code for same query
            neg_text = rng.choice(all_code_texts)
            triples.append((query, neg_text, 0.0))
        elif pair["relevance"] < 0.35:
            triples.append((query, code_text, 0.0))

    # Extra from docstrings
    for sym in rng.sample(all_symbols, min(2000, len(all_symbols))):
        doc = (sym.get("docstring") or "").strip()
        if not doc or len(doc) < 20:
            continue
        first_sent = doc.split(".")[0].split("\n")[0].strip().lower()
        if len(first_sent) > 10:
            code_text = _build_symbol_text(sym)
            triples.append((first_sent, code_text, 1.0))
            neg_text = rng.choice(all_code_texts)
            triples.append((first_sent, neg_text, 0.0))

    rng.shuffle(triples)
    pos = sum(1 for _, _, l in triples if l > 0.5)
    neg = len(triples) - pos
    print(f"Cross-encoder data: {len(triples)} total ({pos} pos, {neg} neg)")
    return triples


# ═══════════════════════════════════════════════════════════════════════
#  PART 3: M5 Bi-Encoder Training
# ═══════════════════════════════════════════════════════════════════════

def train_m5(
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 5e-4,
    train_data: str | None = None,
    val_data: str | None = None,
    temperature: float = 0.05,
    weight_decay: float = 0.02,
    dropout: float = 0.15,
    patience: int = 5,
):
    """Train M5 bi-encoder with validation, early stopping, and LR scheduling."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    print("\n" + "=" * 60)
    print("TRAINING M5 BI-ENCODER")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Hyperparams: epochs={epochs}, batch_size={batch_size}, lr={lr}, "
          f"temp={temperature}, wd={weight_decay}, dropout={dropout}, patience={patience}")

    # Load tokenizer
    sys.path.insert(0, str(PROJECT_DIR / "grafyx" / "search"))
    from _tokenizer import CodeTokenizer
    tokenizer = CodeTokenizer()
    if not tokenizer.is_available:
        print("ERROR: BPE tokenizer not trained. Run with --only bpe first.")
        return
    actual_vocab = tokenizer.vocab_size
    print(f"Tokenizer vocab: {actual_vocab}")

    # Load training data
    if train_data:
        positives, negatives, all_code_texts = prepare_bi_encoder_data(
            train_data_path=train_data
        )
    else:
        with open(REAL_SYMBOLS, encoding="utf-8") as f:
            repos = json.load(f)
        with open(LLM_PAIRS, encoding="utf-8") as f:
            llm_pairs = json.load(f)
        positives, negatives, all_code_texts = prepare_bi_encoder_data(repos, llm_pairs)

    # Load validation data
    val_positives = None
    val_code_texts = None
    if val_data and Path(val_data).exists():
        val_pos, _, val_codes = prepare_bi_encoder_data(train_data_path=val_data)
        val_positives = val_pos
        val_code_texts = val_codes
        print(f"Validation: {len(val_positives)} positive pairs, {len(val_code_texts)} code texts")

    # ── Model ──
    D_MODEL = 128
    N_LAYERS = 4
    EMBED_DIM = 64
    MAX_SEQ = 128
    EXPAND = 2

    class FeedForwardLayer(nn.Module):
        def __init__(self, d_model, expand=2):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * expand),
                nn.SiLU(),
                nn.Linear(d_model * expand, d_model),
            )

        def forward(self, x):
            return x + self.ff(self.norm(x))

    class BiEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(actual_vocab, D_MODEL, padding_idx=0)
            self.pos_embedding = nn.Embedding(MAX_SEQ, D_MODEL)
            self.dropout = nn.Dropout(dropout)
            self.layers = nn.ModuleList([
                FeedForwardLayer(D_MODEL, EXPAND) for _ in range(N_LAYERS)
            ])
            self.final_norm = nn.LayerNorm(D_MODEL)
            self.projection = nn.Linear(D_MODEL, EMBED_DIM)

        def forward(self, input_ids):
            B, S = input_ids.shape
            pos = torch.arange(S, device=input_ids.device)
            x = self.embedding(input_ids) + self.pos_embedding(pos)
            x = self.dropout(x)
            for layer in self.layers:
                x = layer(x)
            x = self.final_norm(x)
            mask = (input_ids != 0).unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            emb = self.projection(x)
            return F.normalize(emb, p=2, dim=-1)

    model = BiEncoder().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"M5: {param_count:,} params ({param_count/1e6:.2f}M)")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Encode all texts to token IDs
    def encode_batch(texts, max_len=MAX_SEQ):
        return torch.tensor(
            [tokenizer.encode(t, max_length=max_len) for t in texts],
            dtype=torch.long,
        )

    # Validation function
    def compute_val_metrics():
        if not val_positives:
            return None
        model.eval()
        with torch.no_grad():
            # Compute val loss (InfoNCE on val positives)
            val_loss_total = 0
            val_batches = 0
            for i in range(0, min(len(val_positives), 2000), batch_size):
                batch = val_positives[i:i + batch_size]
                if len(batch) < 4:
                    continue
                queries, codes = zip(*batch)
                q_ids = encode_batch(queries).to(device)
                c_ids = encode_batch(codes).to(device)
                q_emb = model(q_ids)
                c_emb = model(c_ids)
                logits = q_emb @ c_emb.T / temperature
                labels = torch.arange(len(batch), device=device)
                loss = (F.cross_entropy(logits, labels) +
                        F.cross_entropy(logits.T, labels)) / 2
                val_loss_total += loss.item()
                val_batches += 1

            val_loss = val_loss_total / max(val_batches, 1)

            # Compute recall@10 on a subset
            recall_10 = 0
            n_eval = 0
            # Encode a pool of code texts for recall computation
            pool_size = min(len(val_code_texts), 500)
            pool_texts = val_code_texts[:pool_size]
            pool_ids = encode_batch(pool_texts).to(device)
            pool_embs = []
            for j in range(0, len(pool_ids), batch_size):
                pool_embs.append(model(pool_ids[j:j + batch_size]))
            pool_embs = torch.cat(pool_embs, dim=0)

            for query, code_text in val_positives[:200]:
                q_emb = model(encode_batch([query]).to(device))
                # Add the true positive to pool if not there
                c_emb = model(encode_batch([code_text]).to(device))
                all_embs = torch.cat([c_emb, pool_embs], dim=0)  # true pos is index 0
                sims = (q_emb @ all_embs.T).squeeze(0)
                top_10 = torch.argsort(-sims)[:10]
                if 0 in top_10:
                    recall_10 += 1
                n_eval += 1

            recall_10 = recall_10 / max(n_eval, 1)

            # MRR
            mrr_total = 0
            for query, code_text in val_positives[:200]:
                q_emb = model(encode_batch([query]).to(device))
                c_emb = model(encode_batch([code_text]).to(device))
                all_embs = torch.cat([c_emb, pool_embs], dim=0)
                sims = (q_emb @ all_embs.T).squeeze(0)
                ranked = torch.argsort(-sims)
                rank = (ranked == 0).nonzero(as_tuple=True)[0].item() + 1
                mrr_total += 1.0 / rank
            mrr = mrr_total / max(n_eval, 1)

        return {"val_loss": val_loss, "recall@10": recall_10, "mrr": mrr}

    # Training loop with early stopping
    rng = random.Random(42)
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    patience_counter = 0
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        rng.shuffle(positives)
        total_loss = 0
        n_batches = 0

        for i in range(0, len(positives), batch_size):
            batch = positives[i:i + batch_size]
            if len(batch) < 4:
                continue

            queries, codes = zip(*batch)
            q_ids = encode_batch(queries).to(device)
            c_ids = encode_batch(codes).to(device)

            q_emb = model(q_ids)
            c_emb = model(c_ids)

            # InfoNCE loss
            logits = q_emb @ c_emb.T / temperature
            labels = torch.arange(len(batch), device=device)
            loss = (F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)) / 2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t_start
        current_lr = scheduler.get_last_lr()[0]

        # Validation
        val_metrics = compute_val_metrics()
        if val_metrics:
            val_loss = val_metrics["val_loss"]
            print(f"  Epoch {epoch+1}/{epochs} | Train: {avg_loss:.4f} | "
                  f"Val: {val_loss:.4f} | R@10: {val_metrics['recall@10']:.3f} | "
                  f"MRR: {val_metrics['mrr']:.3f} | LR: {current_lr:.2e} | {elapsed:.0f}s")

            # Early stopping on val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ML_DIR / "m5_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                    break
        else:
            if (epoch + 1) % 3 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.2e} | {elapsed:.0f}s")
            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                torch.save(model.state_dict(), ML_DIR / "m5_best.pt")

    model.load_state_dict(torch.load(ML_DIR / "m5_best.pt", weights_only=True))
    if val_metrics:
        print(f"\nM5 training done. Best val_loss: {best_val_loss:.4f}")
    else:
        print(f"\nM5 training done. Best train_loss: {best_train_loss:.4f}")

    # Export to numpy
    _export_m5(model, actual_vocab, D_MODEL, N_LAYERS, EMBED_DIM, MAX_SEQ, EXPAND)

    return model, tokenizer


def _export_m5(model, vocab_size, d_model, n_layers, embed_dim, max_seq, expand):
    """Export M5 weights to production-compatible numpy format."""
    import numpy as np

    state = model.state_dict()
    weights = {}

    # Embeddings
    weights["token_embedding"] = state["embedding.weight"].cpu().numpy()
    weights["pos_embedding"] = state["pos_embedding.weight"].cpu().numpy()

    # FeedForward layers — export with keys that production code can detect
    for i in range(n_layers):
        prefix = f"layer_{i}_"
        # LayerNorm
        weights[f"{prefix}norm.weight"] = state[f"layers.{i}.norm.weight"].cpu().numpy()
        weights[f"{prefix}norm.bias"] = state[f"layers.{i}.norm.bias"].cpu().numpy()
        # FeedForward: ff.0 = Linear(d_model, d_inner), ff.2 = Linear(d_inner, d_model)
        # Export as ff_w1, ff_b1, ff_w2, ff_b2 (pre-transposed for x @ W + b)
        weights[f"{prefix}ff.w1"] = state[f"layers.{i}.ff.0.weight"].cpu().numpy().T
        weights[f"{prefix}ff.b1"] = state[f"layers.{i}.ff.0.bias"].cpu().numpy()
        weights[f"{prefix}ff.w2"] = state[f"layers.{i}.ff.2.weight"].cpu().numpy().T
        weights[f"{prefix}ff.b2"] = state[f"layers.{i}.ff.2.bias"].cpu().numpy()

    # Final norm + projection
    weights["final_norm_w"] = state["final_norm.weight"].cpu().numpy()
    weights["final_norm_b"] = state["final_norm.bias"].cpu().numpy()
    weights["projection_w"] = state["projection.weight"].cpu().numpy().T  # Pre-transpose
    weights["projection_b"] = state["projection.bias"].cpu().numpy()

    # Save
    out_path = MODEL_DIR / "code_encoder_weights.npz"
    np.savez(out_path, **weights)
    size = out_path.stat().st_size
    print(f"M5 exported: {out_path} ({size // 1024}KB, {len(weights)} arrays)")


# ═══════════════════════════════════════════════════════════════════════
#  PART 4: M6 Cross-Encoder Training
# ═══════════════════════════════════════════════════════════════════════

def train_m6(epochs: int = 15, batch_size: int = 128, lr: float = 3e-4):
    """Train small M6 cross-encoder prototype."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW

    print("\n" + "=" * 60)
    print("TRAINING M6 CROSS-ENCODER (prototype)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sys.path.insert(0, str(PROJECT_DIR / "grafyx" / "search"))
    from _tokenizer import CodeTokenizer
    tokenizer = CodeTokenizer()
    if not tokenizer.is_available:
        print("ERROR: BPE tokenizer not trained.")
        return
    actual_vocab = tokenizer.vocab_size

    with open(REAL_SYMBOLS, encoding="utf-8") as f:
        repos = json.load(f)
    with open(LLM_PAIRS, encoding="utf-8") as f:
        llm_pairs = json.load(f)

    triples = prepare_cross_encoder_data(repos, llm_pairs)

    # ── Prototype model ──
    D_MODEL = 128
    N_FF = 3
    N_ATTN = 1
    N_HEADS = 4
    MAX_SEQ = 192  # Longer for combined query+code
    EXPAND = 2

    class FeedForwardLayer(nn.Module):
        def __init__(self, d_model, expand=2):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * expand),
                nn.SiLU(),
                nn.Linear(d_model * expand, d_model),
            )

        def forward(self, x):
            return x + self.ff(self.norm(x))

    class CrossEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(actual_vocab, D_MODEL, padding_idx=0)
            self.pos_embedding = nn.Embedding(MAX_SEQ, D_MODEL)
            self.dropout = nn.Dropout(0.1)
            self.ff_layers = nn.ModuleList([
                FeedForwardLayer(D_MODEL, EXPAND) for _ in range(N_FF)
            ])
            self.attn_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=D_MODEL, nhead=N_HEADS,
                    dim_feedforward=D_MODEL * 4, batch_first=True,
                    dropout=0.1,
                )
                for _ in range(N_ATTN)
            ])
            self.final_norm = nn.LayerNorm(D_MODEL)
            self.head = nn.Sequential(
                nn.Linear(D_MODEL, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )

        def forward(self, input_ids):
            B, S = input_ids.shape
            pos = torch.arange(S, device=input_ids.device)
            x = self.embedding(input_ids) + self.pos_embedding(pos)
            x = self.dropout(x)
            for layer in self.ff_layers:
                x = layer(x)
            for layer in self.attn_layers:
                x = layer(x)
            x = self.final_norm(x)
            cls = x[:, 0, :]  # CLS token pooling
            return self.head(cls)

    model = CrossEncoder().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"M6 prototype: {param_count:,} params ({param_count/1e6:.2f}M)")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    def encode_combined(query, code_text, max_len=MAX_SEQ):
        combined = f"{query} [SEP] {code_text}"
        return tokenizer.encode(combined, max_length=max_len)

    # Training
    rng = random.Random(42)
    best_loss = float("inf")
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        rng.shuffle(triples)
        total_loss = 0
        correct = 0
        total = 0
        n_batches = 0

        for i in range(0, len(triples), batch_size):
            batch = triples[i:i + batch_size]
            if len(batch) < 4:
                continue

            input_ids = torch.tensor(
                [encode_combined(q, c) for q, c, _ in batch],
                dtype=torch.long,
            ).to(device)
            labels = torch.tensor(
                [l for _, _, l in batch], dtype=torch.float32,
            ).to(device)

            logits = model(input_ids).squeeze(-1)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = (logits > 0).float()
            correct += (preds == labels).sum().item()
            total += len(labels)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        acc = correct / max(total, 1)
        elapsed = time.time() - t_start

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.3f} | {elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ML_DIR / "m6_best.pt")

    model.load_state_dict(torch.load(ML_DIR / "m6_best.pt", weights_only=True))
    print(f"\nM6 training done. Best loss: {best_loss:.4f}")

    # Export
    _export_m6(model, actual_vocab, D_MODEL, N_FF, N_ATTN, N_HEADS, MAX_SEQ, EXPAND)

    return model, tokenizer


def _export_m6(model, vocab_size, d_model, n_ff, n_attn, n_heads, max_seq, expand):
    """Export M6 weights to production-compatible numpy format."""
    import numpy as np

    state = model.state_dict()
    weights = {}

    # Embeddings
    weights["token_embedding"] = state["embedding.weight"].cpu().numpy()
    weights["pos_embedding"] = state["pos_embedding.weight"].cpu().numpy()

    # FeedForward layers (same naming as Mamba layers but with ff.* keys)
    for i in range(n_ff):
        prefix = f"mamba_{i}_"
        weights[f"{prefix}norm.weight"] = state[f"ff_layers.{i}.norm.weight"].cpu().numpy()
        weights[f"{prefix}norm.bias"] = state[f"ff_layers.{i}.norm.bias"].cpu().numpy()
        weights[f"{prefix}ff.w1"] = state[f"ff_layers.{i}.ff.0.weight"].cpu().numpy().T
        weights[f"{prefix}ff.b1"] = state[f"ff_layers.{i}.ff.0.bias"].cpu().numpy()
        weights[f"{prefix}ff.w2"] = state[f"ff_layers.{i}.ff.2.weight"].cpu().numpy().T
        weights[f"{prefix}ff.b2"] = state[f"ff_layers.{i}.ff.2.bias"].cpu().numpy()

    # Attention layers — DON'T pre-transpose: loading code already does .T
    for i in range(n_attn):
        prefix = f"attn_{i}_"
        attn_state = model.attn_layers[i].state_dict()
        for k, v in attn_state.items():
            weights[f"{prefix}{k}"] = v.cpu().numpy()
        # Store n_heads so loading code can use correct value
        weights[f"{prefix}n_heads"] = np.array([n_heads])

    # Final norm
    weights["final_norm_w"] = state["final_norm.weight"].cpu().numpy()
    weights["final_norm_b"] = state["final_norm.bias"].cpu().numpy()

    # Head layers — export with sequential indices (0, 1, 2...)
    # The Sequential may have gaps (0, 3) due to ReLU/Dropout, but loading expects (0, 1, ...)
    head_idx = 0
    head_state = model.head.state_dict()
    # Group by layer index
    layer_indices = sorted(set(int(k.split(".")[0]) for k in head_state.keys()))
    for orig_idx in layer_indices:
        w_key = f"{orig_idx}.weight"
        b_key = f"{orig_idx}.bias"
        if w_key in head_state:
            w = head_state[w_key].cpu().numpy().T  # Pre-transpose
            b = head_state[b_key].cpu().numpy()
            weights[f"head_{head_idx}.weight"] = w
            weights[f"head_{head_idx}.bias"] = b
            head_idx += 1

    out_path = MODEL_DIR / "cross_encoder_weights.npz"
    np.savez(out_path, **weights)
    size = out_path.stat().st_size
    print(f"M6 exported: {out_path} ({size // 1024}KB, {len(weights)} arrays)")


# ═══════════════════════════════════════════════════════════════════════
#  PART 5: Quality Test
# ═══════════════════════════════════════════════════════════════════════

def quality_test():
    """Test M5 quality: can it find functions with zero keyword overlap?"""
    import numpy as np

    print("\n" + "=" * 60)
    print("QUALITY TEST")
    print("=" * 60)

    # Ensure grafyx package is importable
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from grafyx.search._code_encoder import CodeEncoder

    print("\n--- M5 Bi-Encoder ---")
    encoder = CodeEncoder()
    if not encoder.is_available:
        print("M5 not available (weights missing)")
        return

    # Build index from available symbols
    symbols = []
    if ALL_SYMBOLS.exists():
        with open(ALL_SYMBOLS, encoding="utf-8") as f:
            all_syms = json.load(f)
        for sym in all_syms:
            if sym.get("docstring") and not sym["name"].startswith("_"):
                symbols.append({
                    "name": sym["name"],
                    "docstring": sym.get("docstring", ""),
                    "file": sym.get("file", ""),
                    "kind": sym.get("type", "function"),
                })
                if len(symbols) >= 5000:
                    break
    elif REAL_SYMBOLS.exists():
        with open(REAL_SYMBOLS, encoding="utf-8") as f:
            repos = json.load(f)
        for repo in repos:
            for func in repo["functions"]:
                if func.get("docstring") and not func["name"].startswith("_"):
                    symbols.append({
                        "name": func["name"],
                        "docstring": func.get("docstring", ""),
                        "file": func.get("file", ""),
                        "kind": "function",
                    })
                    if len(symbols) >= 2000:
                        break
            if len(symbols) >= 2000:
                break
    else:
        print("No symbol data found")
        return

    print(f"Building index with {len(symbols)} symbols...")
    t0 = time.time()
    encoder.build_index(symbols)
    print(f"Index built in {time.time() - t0:.1f}s")

    # Test queries — including ones with no keyword overlap
    test_queries = [
        # Direct keyword matches (should be easy)
        ("parse config", "keyword match"),
        ("http client", "keyword match"),
        # Semantic queries (harder — tests if encoder learned meaning)
        ("read data from file", "semantic"),
        ("handle user authentication", "semantic"),
        ("convert json to object", "semantic"),
        ("send network request", "semantic"),
        ("validate input parameters", "semantic"),
        # Conceptual queries (hardest — no keywords in common)
        ("login functionality", "conceptual"),
        ("database connection pool", "conceptual"),
        ("retry with exponential backoff", "conceptual"),
    ]

    for query, qtype in test_queries:
        t0 = time.time()
        results = encoder.search(query, top_k=5)
        elapsed = (time.time() - t0) * 1000
        if results:
            top = results[0]
            print(f"  [{qtype:15s}] '{query}' -> {top[0]:30s} ({top[2]:.3f}) [{elapsed:.1f}ms]")
        else:
            print(f"  [{qtype:15s}] '{query}' -> NO RESULTS [{elapsed:.1f}ms]")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train M5/M6 prototype")
    parser.add_argument("--only", choices=["bpe", "m5", "m6", "test"],
                        help="Train only one component")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--bpe-vocab", type=int, default=8192)
    parser.add_argument("--symbols-file", type=str, default=None,
                        help="Symbols file for BPE training (default: all_symbols.json or real_symbols.json)")
    parser.add_argument("--train-data", type=str, default=None,
                        help="Pre-prepared training data JSON for M5 (from generate_training_data_m5.py)")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Validation data JSON for M5")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()

    if args.only == "bpe" or args.only is None:
        train_bpe(vocab_size=args.bpe_vocab, symbols_file=args.symbols_file)

    if args.only == "m5" or args.only is None:
        train_m5(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_data=args.train_data,
            val_data=args.val_data,
            temperature=args.temperature,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=args.patience,
        )

    if args.only == "m6" or args.only is None:
        train_m6(epochs=args.epochs, batch_size=args.batch_size)

    if args.only == "test" or args.only is None:
        quality_test()

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total time: {total/60:.1f} minutes")


if __name__ == "__main__":
    main()
