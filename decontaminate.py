"""Near-duplicate audit of PQA-A/PQA-U against all 1,000 PQA-L records (CPU only).

Integrity note: this is the ONLY unified-study step that opens the historical PQA-L test
file. It reads question and context text solely to find overlapping PQA-A/PQA-U records;
it never reads, stores or reports test labels, and it writes only overlap statistics and
the PQA-A/PQA-U PMIDs to exclude. Every later process keeps the test file sealed.

Criteria (fixed before inspecting any overlap):
  exclude a PQA-A/PQA-U record if
    (1) its normalized question equals that of any PQA-L record, or
    (2) the word 5-gram Jaccard similarity of its contexts with any PQA-L record is >= 0.5.
  Counts at Jaccard thresholds 0.2/0.3/0.5/0.8 are also reported for transparency.
"""
import collections
import datetime
import hashlib
import json
import os
import re

import env

DATA = os.path.join(env.ROOT, "data")
PQAL = os.path.join(env.WORKSPACE, "domain_distill", "data")
THRESHOLDS = (0.2, 0.3, 0.5, 0.8)
EXCLUDE_AT = 0.5
TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text):
    return TOKEN.findall(text.lower())


def shingles(words, n=5):
    if len(words) < n:
        return {hash(" ".join(words))} if words else set()
    return {hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


def normalized_question(text):
    return " ".join(tokens(text))


def load_pqal():
    records = []
    for split in ("train", "dev", "test"):
        with open(os.path.join(PQAL, f"{split}.json"), encoding="utf-8") as f:
            for row in json.load(f):
                records.append({"id": str(row["id"]), "split": split, "question": row["question"],
                                "contexts": row["contexts"]})
    ids = [r["id"] for r in records]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError("Expected 1,000 unique PQA-L records")
    return records


def main():
    out_path = os.path.join(DATA, "decontamination.json")
    if os.path.exists(out_path):
        raise RuntimeError("decontamination.json exists; the audit is write-once")
    targets = load_pqal()
    target_q = collections.defaultdict(list)
    index = collections.defaultdict(set)
    sizes = []
    for t, record in enumerate(targets):
        target_q[normalized_question(record["question"])].append(t)
        sh = shingles(tokens(" ".join(record["contexts"])))
        sizes.append(len(sh))
        for s in sh:
            index[s].add(t)
    result = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "criteria": {"question_exact_normalized": True, "context_word5gram_jaccard_exclude_at": EXCLUDE_AT},
              "test_labels_read": False, "subsets": {}}
    for subset in ("pqa_artificial", "pqa_unlabeled"):
        counts = {str(th): 0 for th in THRESHOLDS}
        by_split = {str(th): {"train": 0, "dev": 0, "test": 0} for th in THRESHOLDS}
        exact_q = {"train": 0, "dev": 0, "test": 0}
        excluded, max_j = [], []
        n = 0
        with open(os.path.join(DATA, f"{subset}.jsonl"), encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                n += 1
                q_hits = target_q.get(normalized_question(row["question"]), [])
                sh = shingles(tokens(" ".join(row["contexts"])))
                hits = collections.Counter()
                for s in sh:
                    for t in index.get(s, ()):
                        hits[t] += 1
                best, best_t = 0.0, None
                for t, inter in hits.items():
                    j = inter / (len(sh) + sizes[t] - inter)
                    if j > best:
                        best, best_t = j, t
                for th in THRESHOLDS:
                    if best >= th:
                        counts[str(th)] += 1
                        by_split[str(th)][targets[best_t]["split"]] += 1
                for t in q_hits:
                    exact_q[targets[t]["split"]] += 1
                if q_hits or best >= EXCLUDE_AT:
                    excluded.append(row["id"])
                if best >= 0.2:
                    max_j.append(round(best, 4))
        result["subsets"][subset] = {"records": n, "exact_question_matches_by_pqal_split": exact_q,
                                     "context_jaccard_at_least": counts,
                                     "context_jaccard_at_least_by_pqal_split": by_split,
                                     "excluded_ids": sorted(excluded, key=int), "excluded_count": len(excluded),
                                     "jaccard_values_at_least_0.2": sorted(max_j, reverse=True)[:200]}
        print(subset, json.dumps({k: v for k, v in result["subsets"][subset].items()
                                  if k not in ("excluded_ids", "jaccard_values_at_least_0.2")}), flush=True)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    print("decontamination.json", digest)


if __name__ == "__main__":
    main()
