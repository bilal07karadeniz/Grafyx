"""Dataset classes for code search encoder training."""
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset


class CodeSearchDataset(Dataset):
    """Dataset for contrastive code search training.

    Each example has a query and code string. The training loop
    uses in-batch negatives for InfoNCE loss.
    """

    def __init__(self, data_path: Path, tokenizer, max_length: int = 256):
        self.examples = []
        with open(data_path) as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(ex)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query_ids = self.tokenizer.encode(ex["query"], max_length=self.max_length)
        code_text = f"{ex.get('name', '')} {ex.get('code', '')}"
        code_ids = self.tokenizer.encode(code_text, max_length=self.max_length)
        return {
            "query_ids": torch.tensor(query_ids, dtype=torch.long),
            "code_ids": torch.tensor(code_ids, dtype=torch.long),
            "label": ex.get("label", 1),
        }


class CrossEncoderDataset(Dataset):
    """Dataset for cross-encoder reranking training.

    Each example is a (query, code) pair with a binary relevance label.
    The tokenizer combines them as: [CLS] query [SEP] code [SEP] [PAD...]
    """

    def __init__(self, data_path: Path, tokenizer, max_length: int = 512):
        self.examples = []
        with open(data_path) as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(ex)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Combine query and code with separator
        combined = f"{ex['query']} [SEP] {ex.get('name', '')} {ex.get('code', '')}"
        ids = self.tokenizer.encode(combined, max_length=self.max_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(float(ex.get("label", 1)), dtype=torch.float32),
        }
