"""Development / validation report (no sealed access). Its SHA-256 must be recorded in
FREEZE_ADDENDUM.jsonl before unseal.py may run (PROTOCOL A §3.0).

S1: per run dev-fold metrics (original evidence), fitted temperature (used later for TS-Brier),
    failure/collapse, dev evidence-use diagnostics, realized budget statistics, cost.
S2: per run final and best validation NLL, best step, realized budget statistics, cost.
"""
import json
import os
import sys

import numpy as np

import analysis_io as aio
import runlib
import stats_unified as su

LABELS = ("yes", "no", "maybe")


def pool_labels():
    with open(os.path.join(aio.ROOT, "data", "pqal_pool.jsonl"), encoding="utf-8") as f:
        return {json.loads(l)["id"]: LABELS.index(json.loads(l)["label"]) for l in f}


def budget_summary(trace):
    if not trace:
        return None
    last = trace[-1]["layers"]
    rows = [l for l in last if l.get("rows")]
    if not rows:
        return None
    return {k: float(np.mean([l[k] for l in rows])) for k in ("candidate_tv", "realized_tv", "active_rate", "realized_tv_sq")
            if all(k in l for l in rows)}


def s1_run(path, labels_by_id):
    r = aio.load(path)
    y = np.array(r["eval_labels"])
    dev = r["dev"]
    full = np.array(dev["original"])
    majority = np.bincount(y, minlength=3).max() / len(y)
    t = su.fit_temperature(full, y)
    donors = [[labels_by_id[i] for i in d] for d in r.get("dev_donors", [])]
    mm = [np.array(dev[f"mismatched_{d}"]) for d in range(len(donors))]
    out = {"run_id": r["spec"]["run_id"], "fold": r["spec"]["fold"], "seed": r["spec"]["seed"],
           "dev": su.classification_metrics(full, y), "dev_ts": su.classification_metrics(full, y, t), "temperature": t,
           **su.failure_and_collapse(full, y, majority), "dev_majority": float(majority),
           "evidence_use_dev": su.evidence_use(full, np.array(dev["question_only"]), mm, y, donors) if donors else None,
           "results_deleted_dev_accuracy": su.accuracy(su.softmax(np.array(dev["results_deleted"])), y),
           "budget": budget_summary(r.get("budget_trace")), "train_seconds": r["train_seconds"],
           "peak_allocated_bytes": r["peak_allocated_bytes"], "trainable_parameters": r["trainable_parameters"],
           "final_train_loss": r["trace"][-1][1], "test_sealed_sha256": r.get("test_sealed_sha256")}
    return out


def s2_run(path):
    r = aio.load(path)
    vals = [e["val_nll"] for e in r["validation"]]
    return {"run_id": r["spec"]["run_id"], "seed": r["spec"]["seed"], "val_final": vals[-1], "val_best": min(vals),
            "best_step": r["best_validation_step"], "budget": budget_summary(r.get("budget_trace")),
            "train_seconds": r["train_seconds"], "tokens_seen": r["tokens_seen"], "sealed_sha256": r.get("sealed_sha256")}


def main():
    out_path = os.environ.get("CV_REPORT_OUT") or os.path.join(aio.ROOT, "results", "cv_report.json")
    labels = pool_labels()
    report = {"utc": runlib.utc(), "S1": {}, "S2": {}}
    for tag, units in sorted(aio.s1_units().items()):
        report["S1"][tag] = {f"{fold}_{rep}": s1_run(p, labels) for (fold, rep), p in sorted(units.items())}
    for tag, units in sorted(aio.s2_units().items()):
        report["S2"][tag] = {str(seed): s2_run(p) for seed, p in sorted(units.items())}
    # pooled out-of-fold dev accuracy per S1 arm (primary replicate), for literature comparison
    pooled = {}
    for tag, units in report["S1"].items():
        prim = [u for k, u in units.items() if k.endswith("_1")]
        if len(prim) == 10:
            pooled[tag] = {"oof_accuracy": float(np.mean([u["dev"]["accuracy"] for u in prim])),
                           "oof_brier": float(np.mean([u["dev"]["brier"] for u in prim]))}
    report["S1_pooled_oof"] = pooled
    digest = runlib.write_json_once(out_path, report)
    print("cv_report.json", digest, "S1 arms", len(report["S1"]), "S2 arms", len(report["S2"]))
    return digest


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
