"""Build the 500-example non-test PQA-L pool with conclusions (CPU only).

The local domain_distill/data/{train,dev}.json files (400 + 100 = the 500 non-test PQA-L
records) omit `long_answer`. The privileged (reasoning-free) teacher needs it, so this
script downloads the pinned PQA-L parquet (1,075,513 bytes, MIT) and keeps ONLY rows whose
PMID is in the local non-test pool. Test rows are removed by a PMID filter at the Arrow
level before any text is converted to Python objects, and nothing about them is written.
Every kept record's question, contexts and label must equal the local copy exactly.

Output: data/pqal_pool.jsonl (500 records: id, question, contexts, long_answer, label)
        data/pqal_pool_manifest.json
"""
import datetime
import hashlib
import json
import os
import urllib.request

import env

env.allow_arrow()
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"
URL = (f"https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/{REVISION}/"
       "pqa_labeled/train-00000-of-00001.parquet")
SIZE, SHA = 1075513, "3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c"
DATA = os.path.join(env.ROOT, "data")
LOCAL = os.path.join(env.WORKSPACE, "domain_distill", "data")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    env.seal()  # the historical test files stay sealed; this script never needs them
    out_path = os.path.join(DATA, "pqal_pool.jsonl")
    if os.path.exists(out_path):
        raise RuntimeError("pqal_pool.jsonl exists; write-once")
    raw_path = os.path.join(DATA, "raw", "pqa_labeled.parquet")
    if not os.path.exists(raw_path):
        with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "unified-study/1.0"}),
                                    timeout=60) as r:
            blob = r.read(SIZE + 1)
        if len(blob) != SIZE or sha256_bytes(blob) != SHA:
            raise RuntimeError("PQA-L parquet failed size/SHA-256 verification")
        with open(raw_path, "xb") as f:
            f.write(blob)
    with open(raw_path, "rb") as f:
        if sha256_bytes(f.read()) != SHA:
            raise RuntimeError("Stored PQA-L parquet hash mismatch")
    local = {}
    for split in ("train", "dev"):
        with open(os.path.join(LOCAL, f"{split}.json"), encoding="utf-8") as f:
            for row in json.load(f):
                local[str(row["id"])] = row
    if len(local) != 500:
        raise ValueError("Expected 500 local non-test records")
    table = pq.read_table(raw_path, columns=["pubid", "question", "context", "long_answer", "final_decision"])
    keep = pc.is_in(pc.cast(table["pubid"], pa.string()), value_set=pa.array(sorted(local)))
    table = table.filter(keep)
    if table.num_rows != 500:
        raise ValueError(f"Expected 500 matching rows, found {table.num_rows}")
    records = []
    for row in table.to_pylist():
        pid = str(row["pubid"])
        ref = local[pid]
        if row["question"] != ref["question"] or list(row["context"]["contexts"]) != ref["contexts"] \
                or row["final_decision"] != ref["label"]:
            raise ValueError(f"PMID {pid} differs from the local prepared record")
        if not row["long_answer"]:
            raise ValueError(f"PMID {pid} has no long_answer")
        records.append({"id": pid, "question": ref["question"], "contexts": ref["contexts"],
                        "long_answer": row["long_answer"], "label": ref["label"]})
    records.sort(key=lambda r: int(r["id"]))
    with open(out_path, "x", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_path, "rb") as f:
        digest = sha256_bytes(f.read())
    manifest = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "source": URL,
                "parquet_sha256": SHA, "records": len(records), "pool_sha256": digest,
                "label_counts": {k: sum(r["label"] == k for r in records) for k in ("yes", "no", "maybe")},
                "test_rows_processed": False,
                "note": "Test rows removed by PMID filter before conversion; local question/context/label verified"}
    with open(os.path.join(DATA, "pqal_pool_manifest.json"), "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
