"""Resolve the quantized model file, downloading from the HuggingFace Hub if absent.

Keeps the 300-370 MB weights out of the git repo (Render clones it) while letting local runs use
a file that is already on disk.
"""
import os
import shutil

from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
STAGING = os.path.join(HERE, "models")

VARIANTS = {
    "wo8": {"file": "laya_wo8_emb.onnx", "data": "laya_wo8_emb.onnx.data",
            "note": "int8 weight-only, lossless (0.9100 vs fp32 0.9092)"},
    "wo4": {"file": "laya_wo4_emb.onnx", "data": "laya_wo4_emb.onnx.data",
            "note": "int4 weight-only, -0.61 pt, smaller memory footprint"},
}


def resolve_model(variant: str, repo_id: str, token=None) -> str:
    """Return a local path to the .onnx with its sibling .data next to it.

    LAYA_MODEL_PATH overrides the hub entirely (point it at the .onnx for local testing).
    Otherwise both files are fetched from the hub. hf_hub_download returns paths inside the
    blob cache, where each file lands under a hash-named directory, so the .onnx and its .data
    end up in different directories. onnxruntime requires external data next to the model and
    fails with "External data path escapes model directory" when they are not, so both files
    are copied into a single staging directory first.
    """
    if variant not in VARIANTS:
        raise ValueError("unknown variant %r (choose from %s)" % (variant, sorted(VARIANTS)))

    local = os.environ.get("LAYA_MODEL_PATH")
    if local and os.path.exists(local):
        return local

    files = VARIANTS[variant]
    staged_dir = os.path.join(STAGING, variant)
    staged = os.path.join(staged_dir, files["file"])
    if os.path.exists(staged):
        return staged

    os.makedirs(staged_dir, exist_ok=True)
    for key in ("data", "file"):
        src = hf_hub_download(repo_id, files[key], token=token)
        shutil.copyfile(src, os.path.join(staged_dir, files[key]))
    return staged
