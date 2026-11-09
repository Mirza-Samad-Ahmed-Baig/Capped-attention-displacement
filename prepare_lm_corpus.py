"""Fresh biomedical LM corpus from PQA-A abstracts (CPU only; never imports torch).

Documents: PQA-A records (all PMIDs are disjoint from PQA-L; the decontamination audit
found no PQA-A near-duplicates of PQA-L), text = contexts joined by newlines, then the
conclusion. No PQA-L text is used.
Split by salted hash: h = int(sha256(f"{SALT}:{pmid}")[:16], 16) % 100;
  h == 0 -> test (sealed), h == 1 -> validation, otherwise train.
Test and validation documents that near-duplicate any train document (word 5-gram Jaccard
>= 0.5) are removed and counted. A byte-level BPE tokenizer (vocab 16,384) is fitted on
train documents only. Token streams are uint16 memmaps, documents separated by <|endoftext|>,
in increasing hash order (a fixed, label-free document order).

Outputs in data/lm/: tokenizer.json, {train,val,test}.bin, {split}_docs.json (PMIDs and token
offsets), lm_manifest.json. The test stream is written but never read before the single
prespecified unseal.
"""
import collections
import datetime
import hashlib
import json
import os
import re

import numpy as np

import env

SALT = "unified-lm-v1"
VOCAB = 16384
DATA = os.path.join(env.ROOT, "data")
OUT = os.path.join(DATA, "lm")
TOKEN = re.compile(r"[a-z0-9]+")
EOT = "<|endoftext|>"


def split_of(pmid):
    h = int(hashlib.sha256(f"{SALT}:{pmid}".encode()).hexdigest()[:16], 16) % 100
    return "test" if h == 0 else "val" if h == 1 else "train", h


def shingles(text, n=5):
    w = TOKEN.findall(text.lower())
    return {hash(" ".join(w[i:i + n])) for i in range(max(1, len(w) - n + 1))} if w else set()


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main():
    if os.path.exists(os.path.join(OUT, "lm_manifest.json")):
        raise RuntimeError("LM corpus already prepared (write-once)")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(DATA, "decontamination.json"), encoding="utf-8") as f:
        excluded = set(json.load(f)["subsets"]["pqa_artificial"]["excluded_ids"])
    docs = {"train": [], "val": [], "test": []}
    with open(os.path.join(DATA, "pqa_artificial.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in excluded:
                continue
            split, _ = split_of(r["id"])
            key = hashlib.sha256(f"{SALT}:{r['id']}".encode()).hexdigest()
            docs[split].append((key, r["id"], "\n".join(r["contexts"]) + "\n" + (r["long_answer"] or "")))
    for split in docs:
        docs[split].sort()
    # near-duplicate filter of held-out documents against train
    held = [(s, i) for s in ("val", "test") for i in range(len(docs[s]))]
    index = collections.defaultdict(set)
    sizes = {}
    for s, i in held:
        sh = shingles(docs[s][i][2])
        sizes[(s, i)] = len(sh)
        for x in sh:
            index[x].add((s, i))
    best = collections.defaultdict(float)
    for _, _, text in docs["train"]:
        sh = shingles(text)
        hits = collections.Counter()
        for x in sh:
            for h in index.get(x, ()):
                hits[h] += 1
        for h, inter in hits.items():
            j = inter / (len(sh) + sizes[h] - inter)
            if j > best[h]:
                best[h] = j
    removed = {s: sorted(docs[s][i][1] for (ss, i), j in best.items() if ss == s and j >= 0.5) for s in ("val", "test")}
    for s in ("val", "test"):
        drop = set(removed[s])
        docs[s] = [d for d in docs[s] if d[1] not in drop]
    # tokenizer fitted on train only
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator((d[2] for d in docs["train"]), vocab_size=VOCAB, min_frequency=2,
                            special_tokens=[EOT], show_progress=False)
    tok_path = os.path.join(OUT, "tokenizer.json")
    tok.save(tok_path)
    eot = tok.token_to_id(EOT)
    manifest = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "salt": SALT, "vocab": tok.get_vocab_size(),
                "eot_id": eot, "source": "data/pqa_artificial.jsonl", "tokenizer_sha256": sha256_file(tok_path),
                "heldout_near_duplicates_removed": {s: len(v) for s, v in removed.items()},
                "removed_ids": removed, "splits": {}}
    for split in ("train", "val", "test"):
        path = os.path.join(OUT, f"{split}.bin")
        offsets, total = [], 0
        with open(path, "xb") as out:
            for start in range(0, len(docs[split]), 2048):
                chunk = docs[split][start:start + 2048]
                encs = tok.encode_batch([d[2] for d in chunk])
                for (_, pid, _), e in zip(chunk, encs):
                    ids = np.asarray(e.ids + [eot], dtype=np.uint16)
                    out.write(ids.tobytes())
                    offsets.append([pid, total, int(ids.size)])
                    total += int(ids.size)
        with open(os.path.join(OUT, f"{split}_docs.json"), "x", encoding="utf-8") as f:
            json.dump(offsets, f)
        manifest["splits"][split] = {"documents": len(docs[split]), "tokens": total, "sha256": sha256_file(path)}
        print(split, manifest["splits"][split], flush=True)
    with open(os.path.join(OUT, "lm_manifest.json"), "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "removed_ids"}, indent=2))


if __name__ == "__main__":
    main()
