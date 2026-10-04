"""Upload the quantized models + tokenizer to the HuggingFace Hub (run once, locally).

    set HF_TOKEN=hf_xxx
    python upload_model.py --repo your-username/laya-quantized --variant wo8

The repo is created if it does not exist. Render then downloads from this repo at startup, so the
git repository you deploy from only carries the ~33 MB tokenizer, not the model weights.
"""
import argparse
import os

from huggingface_hub import HfApi, create_repo

from model_files import VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))
QUANT = os.path.join(HERE, "..", "quant")
TOKENIZER = os.path.join(HERE, "tokenizer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. your-username/laya-quantized")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="wo8")
    ap.add_argument("--quant-dir", default=QUANT)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN first (https://huggingface.co/settings/tokens, write access)")

    create_repo(args.repo, token=token, exist_ok=True, repo_type="model")
    api = HfApi(token=token)

    files = VARIANTS[args.variant]
    for name in (files["file"], files["data"]):
        src = os.path.join(args.quant_dir, name)
        if not os.path.exists(src):
            raise SystemExit("missing %s -- build it first in D:\\Laya\\quant" % src)
        api.upload_file(path_or_fileobj=src, path_in_repo=name, repo_id=args.repo, repo_type="model")
        print("uploaded %s" % name)

    for name in ("tokenizer.json", "tokenizer_config.json"):
        api.upload_file(path_or_fileobj=os.path.join(TOKENIZER, name), path_in_repo=name,
                        repo_id=args.repo, repo_type="model")
        print("uploaded %s" % name)

    print("done -> https://huggingface.co/%s" % args.repo)


if __name__ == "__main__":
    main()
