"""Download the pinned PubMedQA PQA-A (artificial) and PQA-U (unlabeled) subsets.

CPU-only data preparation (reads Parquet, so arrow is allowed; never imports torch).
Source: huggingface.co/datasets/qiaojin/PubMedQA (MIT) at a fixed revision. Each file's
size and SHA-256 must equal the Hugging Face LFS record. Existing verified files are
reused; partial downloads are never renamed into place.

Output (unified/data/):
  raw/pqa_artificial.parquet, raw/pqa_unlabeled.parquet     verified upstream files
  pqa_artificial.jsonl, pqa_unlabeled.jsonl                 one record per PMID with
      id, question, contexts (list), long_answer, label (PQA-A only: "yes"/"no")
  extra_manifest.json                                        hashes, counts, label mix
The long_answer (conclusion) is stored for protocol-specified privileged-information use
only; every model-input builder in this study must exclude it (tested separately).
"""
import datetime
import hashlib
import json
import os
import shutil
import sys
import urllib.request

import env

env.allow_arrow()
import pyarrow.parquet as pq  # noqa: E402

REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"
BASE = f"https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/{REVISION}"
FILES = {
    "pqa_artificial": ("pqa_artificial/train-00000-of-00001.parquet", 233411194,
                       "cdee2d3c01369f47d2ea7b89cd32ee64b142cc0debcb863bf39b3a81a3cece05"),
    "pqa_unlabeled": ("pqa_unlabeled/train-00000-of-00001.parquet", 66010017,
                      "2f390ea5e6e3dc53b2736a52be2f594e05f5d8a599c3de754ad72f1068251e6e"),
}
DATA = os.path.join(env.ROOT, "data")
RAW = os.path.join(DATA, "raw")
RESERVE = 2 * 1024**3


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name):
    remote, size, digest = FILES[name]
    target = os.path.join(RAW, f"{name}.parquet")
    if os.path.exists(target):
        if os.path.getsize(target) == size and sha256(target) == digest:
            return target
        raise RuntimeError(f"Existing {target} does not match the pinned record; inspect it manually")
    if shutil.disk_usage(RAW).free < size + RESERVE:
        raise RuntimeError("Insufficient disk space for download plus the 2 GiB reserve")
    partial = target + ".partial"
    if os.path.exists(partial):
        raise RuntimeError(f"Interrupted download {partial} exists; inspect and remove it manually")
    request = urllib.request.Request(f"{BASE}/{remote}", headers={"User-Agent": "unified-study/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, open(partial, "xb") as out:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != size:
            raise RuntimeError(f"Unexpected Content-Length {declared} for {name}")
        received = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            received += len(block)
            if received > size:
                raise RuntimeError("Download exceeds the pinned size")
            out.write(block)
    if os.path.getsize(partial) != size or sha256(partial) != digest:
        raise RuntimeError(f"Downloaded {name} failed size/SHA-256 verification; kept {partial}")
    os.replace(partial, target)
    return target


def convert(name, path):
    table = pq.read_table(path)
    columns = table.column_names
    out_path = os.path.join(DATA, f"{name}.jsonl")
    if os.path.exists(out_path):
        raise RuntimeError(f"{out_path} exists; conversions are write-once")
    counts, seen, n = {}, set(), 0
    with open(out_path + ".partial", "x", encoding="utf-8") as out:
        for batch in table.to_batches(max_chunksize=4096):
            for row in batch.to_pylist():
                pid = str(row["pubid"])
                if pid in seen:
                    raise ValueError(f"Duplicate PMID {pid} in {name}")
                seen.add(pid)
                ctx = row["context"]
                record = {"id": pid, "question": row["question"], "contexts": list(ctx["contexts"]),
                          "long_answer": row.get("long_answer")}
                if name == "pqa_artificial":
                    label = row["final_decision"]
                    if label not in ("yes", "no"):
                        raise ValueError(f"Unexpected PQA-A label {label!r}")
                    record["label"] = label
                    counts[label] = counts.get(label, 0) + 1
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    os.replace(out_path + ".partial", out_path)
    return {"rows": n, "columns": columns, "label_counts": counts, "jsonl": os.path.basename(out_path),
            "jsonl_sha256": sha256(out_path), "unique_ids": len(seen)}


def main():
    os.makedirs(RAW, exist_ok=True)
    manifest_path = os.path.join(DATA, "extra_manifest.json")
    if os.path.exists(manifest_path):
        raise RuntimeError("extra_manifest.json exists; preparation already completed")
    record = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source": "https://huggingface.co/datasets/qiaojin/PubMedQA", "revision": REVISION,
              "license": "MIT (dataset repository)", "python": sys.version, "files": {}}
    for name in FILES:
        path = fetch(name)
        info = convert(name, path)
        info.update({"parquet": os.path.relpath(path, env.ROOT), "parquet_bytes": os.path.getsize(path),
                     "parquet_sha256": FILES[name][2]})
        record["files"][name] = info
        print(name, json.dumps({k: v for k, v in info.items() if k != "columns"}), flush=True)
    with open(manifest_path, "x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
