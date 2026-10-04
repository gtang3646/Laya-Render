"""Simulate the Render failure and confirm the fix.

Reproduces the deployed environment: huggingface_hub's blob cache puts each file in a
hash-named directory, so the .onnx and its .data land in different directories and
onnxruntime rejects the external-data path. Monkeypatches hf_hub_download to return exactly
that layout, then checks that resolve_model stages the files side by side and that ORT can
open the session.

Set LAYA_QUANT_DIR to the folder holding the built .onnx files if it is not ../quant.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QUANT = os.environ.get("LAYA_QUANT_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "quant")


def main():
    import model_files

    variant = model_files.VARIANTS["wo8"]
    src_onnx = os.path.join(QUANT, variant["file"])
    src_data = os.path.join(QUANT, variant["data"])
    missing = [p for p in (src_onnx, src_data) if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing local model files (set LAYA_QUANT_DIR): %s" % missing)

    # Two unrelated directories, as the blob cache would produce them.
    fake_blobs = tempfile.mkdtemp(prefix="blobA_")
    other_blobs = tempfile.mkdtemp(prefix="blobB_")
    onnx_in_cache = os.path.join(fake_blobs, "somehash.onnx")
    data_in_cache = os.path.join(other_blobs, "anotherhash.data")
    shutil.copyfile(src_onnx, onnx_in_cache)
    shutil.copyfile(src_data, data_in_cache)

    calls = []

    def fake_download(repo_id, filename, token=None):
        calls.append((repo_id, filename))
        return onnx_in_cache if filename == variant["file"] else data_in_cache

    model_files.hf_hub_download = fake_download

    # Fresh staging dir so the copy path actually runs.
    staging = os.path.join(model_files.STAGING, "wo8")
    if os.path.isdir(staging):
        shutil.rmtree(staging)

    got = model_files.resolve_model("wo8", "gtang0115/laya-quantized")

    assert got == os.path.join(staging, variant["file"]), got
    sibling = os.path.join(os.path.dirname(got), variant["data"])
    assert os.path.exists(sibling), "staged .data missing"
    assert len(calls) == 2, calls
    print("staged: %s" % got)
    print("calls : %s" % [c[1] for c in calls])

    # The original failure was here: ORT opening the session.
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(got, so, providers=["CPUExecutionProvider"])
    print("ORT session opened, inputs: %s" % [i.name for i in sess.get_inputs()])

    shutil.rmtree(staging)
    shutil.rmtree(fake_blobs)
    shutil.rmtree(other_blobs)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
