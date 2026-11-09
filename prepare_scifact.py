"""SciFact oracle claim-abstract pairs for extension E4 (CPU only; never imports torch).

Source: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz (claims CC BY 4.0,
abstracts ODC-By 1.0); size 3,115,079 B and ETag (= MD5) cb7da4d8609e30f2c7483b61aa447f7e.
Pairs (Wadden et al., 2020, oracle setting): claims with evidence are paired with each evidence
document (label SUPPORT or CONTRADICT); claims without evidence are paired with each cited document
(label NOT ENOUGH INFO). Labels reuse the 3-class head: SUPPORT->"yes", CONTRADICT->"no", NEI->"maybe".
Gold rationale sentences are marked section label "RESULTS" (all others "OTHER"), so the existing
`results_deleted` input mode is the rationale-deletion probe.

Outputs (unified/data/scifact/):
  train_pairs.jsonl           labelled training pairs (id, question=claim, contexts, section_labels, label)
  sealed_inputs/dev_inputs.jsonl, sealed_inputs/dev_donors.json   label-free one-shot test inputs
  sealed_labels/dev_labels.json   (sealed: added to env.SEALED; read only by unseal)
  manifest.json
"""
import hashlib
import io
import json
import os
import tarfile
import urllib.request

import env
import pubmedqa_inputs as pi
import runlib

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SIZE, ETAG = 3115079, "cb7da4d8609e30f2c7483b61aa447f7e"
OUT = os.path.join(env.ROOT, "data", "scifact")
MAP = {"SUPPORT": "yes", "CONTRADICT": "no", "NEI": "maybe"}


def pairs(claims, corpus):
    out = []
    for c in claims:
        ev = {int(k): v for k, v in (c.get("evidence") or {}).items()}
        doc_ids = list(dict.fromkeys([int(d) for d in c["cited_doc_ids"]] + list(ev)))
        for doc_id in doc_ids:
            doc = corpus[doc_id]
            if doc_id in ev:
                sets = ev[doc_id]
                rationale = {s for e in sets for s in e["sentences"]}
                label = MAP[sets[0]["label"]]
                sections = ["RESULTS" if i in rationale else "OTHER" for i in range(len(doc["abstract"]))]
            else:  # cited but no evidence: NOT ENOUGH INFO
                label, sections = "maybe", ["OTHER"] * len(doc["abstract"])
            out.append({"id": f"{c['id']}_{doc_id}", "question": c["claim"], "contexts": doc["abstract"],
                        "section_labels": sections, "label": label})
    return out


def main():
    os.makedirs(os.path.join(OUT, "sealed_inputs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "sealed_labels"), exist_ok=True)
    if os.path.exists(os.path.join(OUT, "manifest.json")):
        raise SystemExit("already prepared")
    with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "unified-study/1.0"}), timeout=120) as r:
        blob = r.read(SIZE + 1)
    if len(blob) != SIZE or hashlib.md5(blob).hexdigest() != ETAG:
        raise SystemExit("SciFact archive failed size/ETag verification")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".jsonl") and "cross_validation" not in m.name:
                files[os.path.basename(m.name)] = [json.loads(l) for l in tar.extractfile(m).read().decode("utf-8").splitlines() if l.strip()]
    corpus = {d["doc_id"]: d for d in files["corpus.jsonl"]}
    train, dev = pairs(files["claims_train.jsonl"], corpus), pairs(files["claims_dev.jsonl"], corpus)
    counts = lambda rows: {k: sum(r["label"] == k for r in rows) for k in ("yes", "no", "maybe")}
    with open(os.path.join(OUT, "train_pairs.jsonl"), "x", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    labels = [pi.label_index(r["label"]) for r in dev]
    lengths = [len(" ".join(r["contexts"])) for r in dev]
    donors = [[dev[j]["id"] for j in pi.stratified_derangement(labels, lengths, seed=d + 1)] for d in range(3)]
    with open(os.path.join(OUT, "sealed_inputs", "dev_inputs.jsonl"), "x", encoding="utf-8") as f:
        for r in dev:
            f.write(json.dumps({k: r[k] for k in ("id", "question", "contexts", "section_labels")}, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "sealed_inputs", "dev_donors.json"), "x", encoding="utf-8") as f:
        json.dump({"order": [r["id"] for r in dev], "donors": donors}, f)
    with open(os.path.join(OUT, "sealed_labels", "dev_labels.json"), "x", encoding="utf-8") as f:
        json.dump({r["id"]: r["label"] for r in dev}, f)
    manifest = {"utc": runlib.utc(), "source": URL, "etag_md5": ETAG, "bytes": SIZE,
                "license": "claims CC BY 4.0; abstracts ODC-By 1.0 (SciFact LICENSE.md)",
                "train_pairs": len(train), "train_label_counts": counts(train), "dev_pairs": len(dev),
                "dev_labels_written_only_to": "sealed_labels/dev_labels.json",
                "sha256": {p: runlib.sha256_file(os.path.join(OUT, p)) for p in
                           ("train_pairs.jsonl", "sealed_inputs/dev_inputs.jsonl", "sealed_inputs/dev_donors.json",
                            "sealed_labels/dev_labels.json")}}
    with open(os.path.join(OUT, "manifest.json"), "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
