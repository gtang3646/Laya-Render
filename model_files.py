"""Resolve the quantized model file, downloading from the HuggingFace Hub if absent.

Keeps the 300-370 MB weights out of the git repo (Render clones it) while letting local runs use
a file that is already on disk.
"""
import os

from huggingface_hub import hf_hub_download

VARIANTS = {
    "wo8": {"file": "laya_wo8_emb.onnx", "data": "laya_wo8_emb.onnx.data",
            "note": "int8 weight-only, lossless (0.9100 vs fp32 0.9092)"},
    "wo4": {"file": "laya_wo4_emb.onnx", "data": "laya_wo4_emb.onnx.data",
            "note": "int4 weight-only, -0.61 pt, smaller memory footprint"},
}


def resolve_model(variant: str, repo_id: str, token=None) -> str:
    """Return a local path to the .onnx, with its sibling .data next to it.

    LAYA_MODEL_PATH overrides the hub entirely (point it at the .onnx for local testing).
    Otherwise both files are fetched; hf_hub_download lands every file of a revision in one
    snapshot directory, so the .onnx and its .data end up side by side, which is how
    onnxruntime resolves external data.
    """
    if variant not in VARIANTS:
        raise ValueError("unknown variant %r (choose from %s)" % (variant, sorted(VARIANTS)))

    local = os.environ.get("LAYA_MODEL_PATH")
    if local and os.path.exists(local):
        return local

    files = VARIANTS[variant]
    hf_hub_download(repo_id, files["data"], token=token)
    return hf_hub_download(repo_id, files["file"], token=token)
