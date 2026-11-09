"""The single unseal (PROTOCOL A §3.0, B12). Reads the historical PQA-L test labels ONCE and scores
every sealed test file; writes results/unsealed.json (write-once). Refuses to run unless:
  (1) results/cv_report.json exists and its SHA-256 is recorded in FREEZE_ADDENDUM.jsonl;
  (2) every declared run in every study registry is complete or listed in results/cuts.json;
  (3) results/unsealed.json does not exist.
Every sealed file's SHA-256 is checked against the hash its run recorded.
"""
import json
import os
import sys

import numpy as np

import analysis_io as aio
import env
import runlib
import stats_unified as su

LABELS = ("yes", "no", "maybe")
TEST_JSON = os.path.join(env.WORKSPACE, "domain_distill", "data", "test.json")
EXCLUDED_STUDIES = ("S2_base",)  # superseded by the G1 fallback; its sealed files were deleted unopened


def preconditions():
    out = os.path.join(aio.ROOT, "results", "unsealed.json")
    if os.path.exists(out):
        raise SystemExit("unsealed.json exists: the test sets have already been unsealed")
    cv = os.path.join(aio.ROOT, "results", "cv_report.json")
    if not os.path.exists(cv):
        raise SystemExit("cv_report.json missing")
    digest = runlib.sha256_file(cv)
    with open(os.path.join(aio.ROOT, "FREEZE_ADDENDUM.jsonl"), encoding="utf-8") as f:
        recorded = any(json.loads(l).get("cv_report_sha256") == digest for l in f if l.strip())
    if not recorded:
        raise SystemExit("cv_report.json hash is not recorded in FREEZE_ADDENDUM.jsonl")
    cuts_path = os.path.join(aio.ROOT, "results", "cuts.json")
    cuts = set(aio.load(cuts_path)["cut_run_ids"]) if os.path.exists(cuts_path) else set()
    incomplete = []
    for base, dirs, files in os.walk(aio.STUDIES):
        if "registry.json" in files and not any(os.path.relpath(base, aio.STUDIES).startswith(e + os.sep) or
                                                os.path.relpath(base, aio.STUDIES) == e for e in EXCLUDED_STUDIES):
            for spec in runlib.Registry(base).pending():
                if spec["run_id"] not in cuts:
                    incomplete.append(spec["run_id"])
    if incomplete:
        raise SystemExit(f"{len(incomplete)} declared runs are neither complete nor cut, e.g. {incomplete[:5]}")
    return out, aio.load(cv)


RUN_RE = __import__("re").compile(r"^(?P<study>[^_]+)_(?P<tag>.+)_f(?P<fold>\d+)(?:_[sr](?P<rep>\d+))?$")
SCIFACT_LABELS = os.path.join(aio.ROOT, "data", "scifact", "sealed_labels", "dev_labels.json")
DONOR_FILES = [os.path.join(aio.ROOT, "data", "sealed_inputs", "test_donors.json"),
               os.path.join(aio.ROOT, "data", "scifact", "sealed_inputs", "dev_donors.json")]


def encoder_run_paths():
    """(study_prefix, run_id, path) for every encoder-type result that wrote a sealed test file."""
    out = []
    for rid, path in aio.index_runs("").items():
        study = os.path.relpath(path, aio.STUDIES).split(os.sep)[0]
        if study == "audit":
            continue  # audit replays (audit.py) are checks, never analysed
        out.append((study, rid, path))
    return sorted(out)


def score_s1(test_labels, cv):
    donor_sets = [aio.load(p) for p in DONOR_FILES if os.path.exists(p)]
    out = {}
    for study, rid, path in encoder_run_paths():
        r = aio.load(path)
        if "test_sealed_sha256" not in r:
            continue  # LM run
        sealed = aio.sealed_path(path)
        if r.get("test_sealed_sha256") is None or not os.path.exists(sealed):
            continue  # unselected tuning level (deleted unopened) or no test evaluation
        if runlib.sha256_file(sealed) != r["test_sealed_sha256"]:
            raise RuntimeError(f"sealed hash mismatch for {rid}")
        s = aio.load(sealed)
        ids = s["ids"]
        donors = next(d for d in donor_sets if d["order"] == ids)
        y = np.array([test_labels[i] for i in ids])
        lab = {i: test_labels[i] for i in ids}
        m = RUN_RE.match(rid)
        tag, fold, rep = m.group("tag"), int(m.group("fold")), int(m.group("rep") or 1)
        majority = float(np.bincount(y, minlength=3).max() / len(y))
        t = su.fit_temperature(np.array(r["dev"]["original"]), np.array(r["eval_labels"]))
        if True:
            full = np.array(s["logits"]["original"])
            p_raw, p_ts = su.softmax(full), su.softmax(full, t)
            eye = np.eye(3)[y]
            mm_keys = sorted(k for k in s["logits"] if k.startswith("mismatched_"))
            donor_labels = [[lab[i] for i in donors["donors"][int(k.split("_")[1])]] for k in mm_keys]
            out.setdefault("S1" if study == "S1" else "EXT", {})[rid] = {
                "study": study, "tag": tag, "fold": fold, "rep": rep, "test_majority": majority,
                "test": su.classification_metrics(full, y), "test_ts": su.classification_metrics(full, y, t),
                "per_example_brier": ((p_raw - eye) ** 2).sum(1).round(6).tolist(),
                "per_example_brier_ts": ((p_ts - eye) ** 2).sum(1).round(6).tolist(),
                "correct": (p_raw.argmax(1) == y).astype(int).tolist(),
                **su.failure_and_collapse(full, y, majority),
                "evidence_use_test": su.evidence_use(full, np.array(s["logits"]["question_only"]),
                                                     [np.array(s["logits"][k]) for k in mm_keys], y, donor_labels),
                "results_deleted_test_accuracy": su.accuracy(su.softmax(np.array(s["logits"]["results_deleted"])), y),
                "pred_share": {c: (np.bincount(np.array(s["logits"][c]).argmax(1), minlength=3) / len(y)).round(6).tolist()
                               for c in ("original", "question_only", "results_deleted")},
                "question_only_test": su.classification_metrics(np.array(s["logits"]["question_only"]), y)}
    return out


def score_s2():
    out = {}
    for tag, units in sorted(aio.s2_units().items()):
        for seed, path in sorted(units.items()):
            r = aio.load(path)
            rec = {"tag": tag, "seed": seed}
            for which in ("final", "best"):
                sealed = aio.sealed_path(path, f"_{which}")
                digest = (r.get("sealed_sha256") or {}).get(which)
                if digest is None or not os.path.exists(sealed):
                    continue
                if runlib.sha256_file(sealed) != digest:
                    raise RuntimeError(f"sealed hash mismatch for {r['spec']['run_id']} {which}")
                s = aio.load(sealed)
                sums = np.array([d[1] for d in s["per_doc"]])
                counts = np.array([d[2] for d in s["per_doc"]])
                rec[which] = {"nll_per_token": float(sums.sum() / counts.sum()), "doc_sum": sums.round(6).tolist(),
                              "doc_count": counts.tolist(),
                              "bucket_nll": [bs / bc if bc else None for bs, bc in zip(s["bucket_sum"], s["bucket_count"])],
                              "bucket_count": s["bucket_count"]}
            if "final" in rec:
                out[r["spec"]["run_id"]] = rec
    return out


def main():
    out, cv = preconditions()
    with env.sealed_access(TEST_JSON, SCIFACT_LABELS):
        with open(TEST_JSON, encoding="utf-8") as f:
            test_labels = {str(r["id"]): LABELS.index(r["label"]) for r in json.load(f)}
        if os.path.exists(SCIFACT_LABELS):
            with open(SCIFACT_LABELS, encoding="utf-8") as f:
                test_labels.update({k: LABELS.index(v) for k, v in json.load(f).items()})
    enc = score_s1(test_labels, cv)
    record = {"utc": runlib.utc(), "cv_report_sha256": runlib.sha256_file(os.path.join(aio.ROOT, "results", "cv_report.json")),
              "S1": enc.get("S1", {}), "EXT": enc.get("EXT", {}), "S2": score_s2()}
    digest = runlib.write_json_once(out, record)
    print(f"unsealed.json {digest}: S1 runs {len(record['S1'])}, extension encoder runs {len(record['EXT'])}, "
          f"S2 runs {len(record['S2'])}")


if __name__ == "__main__":
    env.seal()
    main()
