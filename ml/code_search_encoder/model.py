"""Mamba bi-encoder for code search. PyTorch training model.

Architecture: 8 Mamba layers, 384 hidden dim, 256 embedding dim (~16M params).
Uses the mamba-ssm library for GPU-accelerated selective scan during training.
The trained weights are exported to numpy for production inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("WARNING: mamba-ssm not installed. Install with: pip install mamba-ssm")


class MambaLayer(nn.Module):
    """Single Mamba layer with pre-norm and residual."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA:
            self.mamba = Mamba(d_model=d_model, d_state=d_state,
                              d_conv=d_conv, expand=expand)
        else:
            # Fallback: simple linear layer for testing without mamba-ssm
            d_inner = d_model * expand
            self.mamba = nn.Sequential(
                nn.Linear(d_model, d_inner),
                nn.SiLU(),
                nn.Linear(d_inner, d_model),
            )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class CodeSearchEncoder(nn.Module):
    """Mamba bi-encoder: maps text to 256-dim embedding.

    Input: token IDs (batch, seq_len)
    Output: L2-normalized embeddings (batch, embed_dim)
    """

    def __init__(self, vocab_size=16384, d_model=384, n_layers=8,
                 d_state=16, d_conv=4, expand=2, embed_dim=256,
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, seq_len) -> embeddings: (batch, embed_dim)."""
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # Mean pooling (exclude padding tokens)
        mask = (input_ids != 0).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        emb = self.projection(x)
        # L2 normalize
        return F.normalize(emb, p=2, dim=-1)


def info_nce_loss(query_emb, code_emb, temperature=0.07):
    """InfoNCE contrastive loss with in-batch negatives.

    Args:
        query_emb: (batch, embed_dim) L2-normalized query embeddings
        code_emb: (batch, embed_dim) L2-normalized code embeddings
        temperature: scaling factor

    Returns:
        loss: scalar
    """
    # Similarity matrix: (batch, batch)
    logits = query_emb @ code_emb.T / temperature
    # Labels: diagonal elements are positives
    labels = torch.arange(logits.shape[0], device=logits.device)
    # Cross-entropy in both directions
    loss_q2c = F.cross_entropy(logits, labels)
    loss_c2q = F.cross_entropy(logits.T, labels)
    return (loss_q2c + loss_c2q) / 2


def export_to_numpy(model: CodeSearchEncoder, output_path: str):
    """Export trained model weights to numpy .npz for production inference."""
    import numpy as np

    state = model.state_dict()
    weights = {}

    # Embedding layers
    weights["token_embedding"] = state["embedding.weight"].cpu().numpy()
    weights["pos_embedding"] = state["pos_embedding.weight"].cpu().numpy()

    # Mamba layers
    for i, layer in enumerate(model.layers):
        prefix = f"layer_{i}_"
        layer_state = {k: v.cpu().numpy() for k, v in layer.state_dict().items()}
        for k, v in layer_state.items():
            weights[prefix + k] = v

    # Final norm and projection
    weights["final_norm_w"] = state["final_norm.weight"].cpu().numpy()
    weights["final_norm_b"] = state["final_norm.bias"].cpu().numpy()
    weights["projection_w"] = state["projection.weight"].cpu().numpy().T  # Pre-transpose
    weights["projection_b"] = state["projection.bias"].cpu().numpy()

    np.savez(output_path, **weights)
    print(f"Exported {len(weights)} weight arrays to {output_path}")
