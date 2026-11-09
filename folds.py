"""Seeded stratified 10-fold split of the 500 non-test PQA-L records (PROTOCOL Part A §3.0).

Per label (yes, no, maybe): PMIDs sorted numerically, shuffled with random.Random(20260925)
(one generator, labels in that order); the three lists are concatenated and position j is
assigned to fold j mod 10. No test data is read (the seal is active).
"""
import collections
import json
import os
import random

import env

SEED = 20260925


def make_folds(rows, k=10, seed=SEED):
    rng = random.Random(seed)
    ordered = []
    for label in ("yes", "no", "maybe"):
        ids = sorted((r["id"] for r in rows if r["label"] == label), key=int)
        rng.shuffle(ids)
        ordered += ids
    folds = [[] for _ in range(k)]
    for j, pid in enumerate(ordered):
        folds[j % k].append(pid)
    return [sorted(f, key=int) for f in folds]


def main():
    env.seal()
    out = os.path.join(env.ROOT, "data", "pqal_folds.json")
    if os.path.exists(out):
        raise RuntimeError("pqal_folds.json exists (write-once)")
    with open(os.path.join(env.ROOT, "data", "pqal_pool.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    folds = make_folds(rows)
    label = {r["id"]: r["label"] for r in rows}
    summary = [dict(collections.Counter(label[i] for i in fold)) for fold in folds]
    with open(out, "x", encoding="utf-8") as f:
        json.dump({"seed": SEED, "rule": __doc__.strip(), "folds": folds, "label_counts": summary}, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
