"""Upload the converted ONNX model to HuggingFace Hub.

Target repo: bilal07karadeniz/grafyx-coderankembed-onnx (public, MIT).

    pip install huggingface_hub
    huggingface-cli login    # one-time; paste a WRITE token from
                             # https://huggingface.co/settings/tokens
    python ml/upload_coderankembed_onnx.py

This is a one-shot script. The Grafyx runtime fetches the uploaded files
via fastembed.add_custom_model — see grafyx/search/_embeddings.py.
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "bilal07karadeniz/grafyx-coderankembed-onnx"
OUT = Path(__file__).parent / "coderankembed_onnx_out"


_README = """\
# grafyx-coderankembed-onnx

ONNX-int8 export of [`nomic-ai/CodeRankEmbed`](https://huggingface.co/nomic-ai/CodeRankEmbed)
for use with [Grafyx](https://github.com/bilal07karadeniz/Grafyx) via
[fastembed](https://github.com/qdrant/fastembed).

- **Base model:** nomic-ai/CodeRankEmbed (137M params)
- **License:** MIT (inherits the base model's license)
- **Format:** ONNX, dynamic int8 quantized (AVX-512-VNNI)
- **Pooling:** CLS
- **Normalization:** L2 (cosine-ready)
- **Embedding dim:** 768
- **Query prefix (required):** `Represent this query for searching relevant code: `

Used by Grafyx's `find_related_code` MCP tool when `GRAFYX_ENCODER=coderankembed`.
"""


def main() -> None:
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

    # Bundle a minimal README so the HF page is not empty.
    readme = OUT / "README.md"
    if not readme.exists():
        readme.write_text(_README, encoding="utf-8")

    api.upload_folder(
        folder_path=str(OUT),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Initial CodeRankEmbed ONNX-int8 export for Grafyx 0.2.0",
    )
    print(f"Uploaded to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
