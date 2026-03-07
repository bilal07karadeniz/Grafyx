"""Mamba+Attention cross-encoder for search result reranking.

Architecture: 6 Mamba layers + 2 Transformer attention layers (~12M params).
Input: [CLS] query [SEP] code [SEP] [PAD...]
Output: relevance score (sigmoid)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA:
            self.mamba = Mamba(d_model=d_model, d_state=d_state,
                              d_conv=d_conv, expand=expand)
        else:
            d_inner = d_model * expand
            self.mamba = nn.Sequential(
                nn.Linear(d_model, d_inner), nn.SiLU(),
                nn.Linear(d_inner, d_model),
            )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class CrossEncoderReranker(nn.Module):
    """Mamba+Attention hybrid. 6 Mamba + 2 Attention layers.

    Processes combined [query SEP code] input for full interaction.
    Outputs a single relevance score.
    """

    def __init__(self, vocab_size=16384, d_model=384, n_mamba=6,
                 n_attn=2, n_heads=6, d_state=16, d_conv=4, expand=2,
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Mamba layers (efficient sequence processing)
        self.mamba_layers = nn.ModuleList([
            MambaLayer(d_model, d_state, d_conv, expand)
            for _ in range(n_mamba)
        ])

        # Attention layers (query-code cross-interaction)
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, batch_first=True,
                dropout=dropout,
            )
            for _ in range(n_attn)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, seq_len) -> scores: (batch, 1)."""
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        x = self.dropout(x)

        for layer in self.mamba_layers:
            x = layer(x)

        for layer in self.attn_layers:
            x = layer(x)

        x = self.final_norm(x)
        # CLS pooling (first token)
        cls = x[:, 0, :]
        return self.head(cls)


def export_to_numpy(model: CrossEncoderReranker, output_path: str):
    """Export cross-encoder weights to numpy for production inference."""
    import numpy as np

    state = model.state_dict()
    weights = {}

    weights["token_embedding"] = state["embedding.weight"].cpu().numpy()
    weights["pos_embedding"] = state["pos_embedding.weight"].cpu().numpy()

    for i in range(len(model.mamba_layers)):
        prefix = f"mamba_{i}_"
        for k, v in model.mamba_layers[i].state_dict().items():
            weights[prefix + k] = v.cpu().numpy()

    for i in range(len(model.attn_layers)):
        prefix = f"attn_{i}_"
        for k, v in model.attn_layers[i].state_dict().items():
            weights[prefix + k] = v.cpu().numpy()

    weights["final_norm_w"] = state["final_norm.weight"].cpu().numpy()
    weights["final_norm_b"] = state["final_norm.bias"].cpu().numpy()

    for k, v in model.head.state_dict().items():
        w = v.cpu().numpy()
        if "weight" in k:
            w = w.T  # Pre-transpose for numpy inference
        weights[f"head_{k}"] = w

    np.savez(output_path, **weights)
    print(f"Exported {len(weights)} weight arrays to {output_path}")
