"""Label-free evaluation inputs for the official PQA-L test (logged sealed-access exception).

This is one of the three scripts allowed to use env.sealed_access (PROTOCOL Part B §B12). It
reads the historical test file and the PQA-L parquet ONLY to write:
  data/sealed_inputs/pqal_test_inputs.jsonl   id, question, contexts, section_labels
                                              (NO label, NO long_answer)
  data/sealed_inputs/test_donors.json         3 label-stratified, evidence-length-quintile
                                              derangement donor lists (donor ids only; the
                                              different-label constraint is asserted here)
  data/pqal_pool_sections.json                section labels for the 500 non-test records
Labels are held in memory only for the donor assertion and are never written or printed.
CPU only; never imports torch.
"""
import hashlib
import json
import os

import env

env.allow_arrow()
import pyarrow.parquet as pq  # noqa: E402

import pubmedqa_inputs as pi  # noqa: E402

DATA = os.path.join(env.ROOT, "data")
OUT = os.path.join(DATA, "sealed_inputs")
TEST_JSON = os.path.join(env.WORKSPACE, "domain_distill", "data", "test.json")
PARQUET = os.path.join(DATA, "raw", "pqa_labeled.parquet")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    env.seal()
    os.makedirs(OUT, exist_ok=True)
    targets = [os.path.join(OUT, "pqal_test_inputs.jsonl"), os.path.join(OUT, "test_donors.json"),
               os.path.join(DATA, "pqal_pool_sections.json")]
    if any(os.path.exists(t) for t in targets):
        raise RuntimeError("evaluation inputs already prepared (write-once)")
    with open(os.path.join(DATA, "pqal_pool.jsonl"), encoding="utf-8") as f:
        pool = {json.loads(line)["id"]: json.loads(line) for line in f}
    with env.sealed_access(TEST_JSON, PARQUET):
        with open(TEST_JSON, encoding="utf-8") as f:
            test = json.load(f)
        table = pq.read_table(PARQUET, columns=["pubid", "question", "context"]).to_pylist()
    sections = {str(r["pubid"]): list(r["context"]["labels"]) for r in table}
    contexts = {str(r["pubid"]): list(r["context"]["contexts"]) for r in table}
    test_ids = [str(r["id"]) for r in test]
    if len(test_ids) != 500 or len(set(test_ids)) != 500 or set(test_ids) & set(pool):
        raise ValueError("unexpected test id set")
    rows, labels, lengths = [], [], []
    for r in sorted(test, key=lambda r: int(r["id"])):
        pid = str(r["id"])
        if contexts[pid] != r["contexts"]:
            raise ValueError(f"context mismatch for {pid}")
        rows.append({"id": pid, "question": r["question"], "contexts": r["contexts"], "section_labels": sections[pid]})
        labels.append(pi.label_index(r["label"]))
        lengths.append(len(" ".join(r["contexts"])))
    donors = []
    for d in range(3):
        idx = pi.stratified_derangement(labels, lengths, seed=d + 1)
        assert all(labels[j] != labels[i] for i, j in enumerate(idx))
        donors.append([rows[j]["id"] for j in idx])
    with open(targets[0], "x", encoding="utf-8") as f:
        for r in rows:
            assert set(r) == {"id", "question", "contexts", "section_labels"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(targets[1], "x", encoding="utf-8") as f:
        json.dump({"order": [r["id"] for r in rows], "donors": donors,
                   "rule": "stratified_derangement(labels, evidence_char_lengths, seed=1..3)"}, f)
    with open(targets[2], "x", encoding="utf-8") as f:
        json.dump({pid: sections[pid] for pid in sorted(pool, key=int)}, f)
    print(json.dumps({t: sha(t) for t in targets}, indent=1))


if __name__ == "__main__":
    main()
