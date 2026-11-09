"""Download pinned BioLinkBERT-base (Apache-2.0) for extension E5 (CPU only; never imports torch).

Revision b71f5d70f063d1c8f1124070ce86f1ee463ca1fe; pytorch_model.bin size and SHA-256 must equal
the Hugging Face LFS record. Files go to unified/models/biolinkbert_base/ with a manifest.
The checkpoint is loaded later with weights_only=True and no remote code.
"""
import hashlib
import json
import os
import shutil
import urllib.request

import env
import runlib

REPO, REV = "michiyasunaga/BioLinkBERT-base", "b71f5d70f063d1c8f1124070ce86f1ee463ca1fe"
FILES = {"config.json": None, "vocab.txt": None, "tokenizer_config.json": None, "special_tokens_map.json": None,
         "README.md": None, "pytorch_model.bin": (433019313, "acc5ae5f16206b893adfcf10772ee4472e24d7847145ac961097bd06129e2ece")}
OUT = os.path.join(env.ROOT, "models", "biolinkbert_base")


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "download_manifest.json")):
        raise SystemExit("already prepared")
    if shutil.disk_usage(OUT).free < 433019313 + 2 * 1024**3:
        raise SystemExit("insufficient disk for download plus 2 GiB reserve")
    manifest = {"repo": REPO, "revision": REV, "license": "Apache-2.0", "utc": runlib.utc(), "files": {}}
    for name, expect in FILES.items():
        target = os.path.join(OUT, name)
        url = f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"
        req = urllib.request.Request(url, headers={"User-Agent": "unified-study/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(target + ".partial", "xb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        size, digest = os.path.getsize(target + ".partial"), runlib.sha256_file(target + ".partial")
        if expect and (size, digest) != expect:
            raise SystemExit(f"{name} failed verification: {size} {digest}")
        os.replace(target + ".partial", target)
        manifest["files"][name] = {"bytes": size, "sha256": digest}
        print(name, size, digest[:12], flush=True)
    with open(os.path.join(OUT, "download_manifest.json"), "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
