"""One-shot: convert nomic-ai/CodeRankEmbed to ONNX-int8 for Grafyx.

Run once locally with a Python venv that has torch + transformers + optimum.
Outputs go to ml/coderankembed_onnx_out/ — uploaded separately by
upload_coderankembed_onnx.py. The Grafyx runtime never invokes this code:
fastembed reads the uploaded ONNX from HF and caches it locally.

    python -m venv /tmp/grafyx-convert
    source /tmp/grafyx-convert/bin/activate    # WSL
    # source /tmp/grafyx-convert/Scripts/activate    # Windows
    pip install "torch>=2.7" "transformers>=4.53" "optimum[onnxruntime]>=1.22" "huggingface_hub"
    python ml/convert_coderankembed_to_onnx.py

Why a separate venv:
    Grafyx's runtime install is fastembed-only (~150 MB of ONNX runtime).
    torch + transformers + optimum is ~5 GB and is *only* needed to produce
    the ONNX export. Keep them out of the project venv so Grafyx stays
    light.
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent / "coderankembed_onnx_out"


def main() -> None:
    from huggingface_hub import snapshot_download
    from optimum.onnxruntime import (
        ORTModelForFeatureExtraction,
        ORTQuantizer,
    )
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    model_id = "nomic-ai/CodeRankEmbed"
    OUT.mkdir(parents=True, exist_ok=True)

    # CodeRankEmbed ships custom modeling code whose state_dict loader has
    # a bug in the HF-fallback branch: it defaults to safe_serialization=False
    # and probes for pytorch_model.bin first, but the repo only ships
    # model.safetensors. The local-file branch of the same loader handles
    # safetensors correctly, so we snapshot-download the repo first and
    # then point optimum at the local directory.
    print(f"Snapshot-downloading {model_id}...")
    local_model_path = snapshot_download(repo_id=model_id)
    print(f"  -> {local_model_path}")

    print(f"Exporting {model_id} to ONNX...")
    model = ORTModelForFeatureExtraction.from_pretrained(
        local_model_path, export=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)

    print("Quantizing to int8 (dynamic, AVX-512-VNNI)...")
    quantizer = ORTQuantizer.from_pretrained(OUT)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=OUT, quantization_config=qconfig)

    # Replace the fp32 model with the quantized one to keep the download
    # small (otherwise users get both files and waste disk).
    fp32 = OUT / "model.onnx"
    int8 = OUT / "model_quantized.onnx"
    if int8.exists():
        if fp32.exists():
            fp32.unlink()
        int8.rename(fp32)
        print(f"Quantized model -> {fp32}")

    size_mb = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"\nDone. Output: {OUT} ({size_mb:.1f} MB total)")
    print("\nSanity check:")
    _smoke_test()


def _smoke_test() -> None:
    """Verify the converted model produces sane 768-dim normalized embeddings."""
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    import numpy as np

    m = ORTModelForFeatureExtraction.from_pretrained(str(OUT))
    t = AutoTokenizer.from_pretrained(str(OUT))
    inp = t(
        ["Represent this query for searching relevant code: parse json"],
        return_tensors="np",
    )
    out = m(**inp)
    emb = out.last_hidden_state[:, 0, :]
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    assert emb.shape[1] == 768, f"unexpected embedding dim: {emb.shape}"
    norm = float(np.linalg.norm(emb))
    print(f"  shape: {emb.shape}, norm: {norm:.4f}")
    assert abs(norm - 1.0) < 1e-3, f"embedding not normalized (norm={norm})"
    print("  ok")


if __name__ == "__main__":
    main()
