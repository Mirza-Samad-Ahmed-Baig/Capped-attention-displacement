"""Prespecified selection rules and gates (PROTOCOL A §3.2-3.3, B6). Prints decisions only.

  g1 FOLDER                 S2 base LR: lowest final validation NLL; gate G1 requires the
                            winner's final val NLL >= its minimum over evaluations + 0.02.
  ge FOLDER                 S1 gate G_E on N and AD@0.10, folds 0-1: switch recipe once if >= 3
                            of 4 runs fail (dev acc <= fold majority) or collapse (>= 90% one class).
  levels FOLDER REGIME      per-family level choice on tuning units (lowest mean dev Brier for the
                            encoder / final val NLL for the LM; ties within 0.005 -> lower dose).
Decisions are written write-once to FOLDER/decisions_<name>.json.
"""
import json
import math
import os
import sys

import runlib


def softmax(row):
    m = max(row)
    e = [math.exp(v - m) for v in row]
    s = sum(e)
    return [v / s for v in e]


def brier(logits, labels):
    total = 0.0
    for row, y in zip(logits, labels):
        p = softmax(row)
        total += sum((pc - (1.0 if c == y else 0.0)) ** 2 for c, pc in enumerate(p))
    return total / len(labels)


def accuracy(logits, labels):
    return sum(max(range(3), key=lambda c: row[c]) == y for row, y in zip(logits, labels)) / len(labels)


def load_runs(folder):
    out = {}
    for name in os.listdir(os.path.join(folder, "runs")):
        with open(os.path.join(folder, "runs", name), encoding="utf-8") as f:
            r = json.load(f)
        out[r["spec"]["run_id"]] = r
    return out


def g1(folder):
    runs = load_runs(folder)
    final = {rid: r["validation"][-1]["val_nll"] for rid, r in runs.items()}
    winner = min(final, key=final.get)
    vals = [e["val_nll"] for e in runs[winner]["validation"]]
    overfit = vals[-1] >= min(vals) + 0.02
    decision = {"rule": "G1", "lr_winner": winner, "lr": runs[winner]["spec"]["recipe"]["lr"],
                "overfits": overfit, "action": "proceed" if overfit else "halve corpus to >=500k tokens, 16 epochs, redo base tuning once"}
    return decision


def ge(folder, ids):
    runs = load_runs(folder)
    bad = 0
    for rid in ids:
        r = runs[rid]
        logits, labels = r["dev"]["original"], r["eval_labels"]
        majority = max(labels.count(c) for c in range(3)) / len(labels)
        preds = [max(range(3), key=lambda c: row[c]) for row in logits]
        collapse = max(preds.count(c) for c in range(3)) / len(preds) >= 0.9
        if accuracy(logits, labels) <= majority or collapse:
            bad += 1
    return {"rule": "G_E", "runs": ids, "failed_or_collapsed": bad,
            "action": "switch to fallback recipe (reinit top-2 layers, lr 3e-5)" if bad >= 3 else "proceed"}


def choose_level(scores):
    """scores: {level(float): mean score (lower better)}; ties within 0.005 -> lower dose."""
    best = min(scores.values())
    return min(l for l, s in scores.items() if s <= best + 0.005)


def parse_tag(run_id):
    """'S1_AD@0.05_f0' -> ('AD', 0.05); 'S1_N_f0' -> ('N', None)."""
    body = run_id.split("_", 1)[1].rsplit("_", 1)[0]
    if "@" in body:
        fam, level = body.split("@")
        return fam, float(level)
    return body, None


def levels_s1(study):
    runs = {}
    for sub in ("n", "gate", "tuning"):
        runs.update(load_runs(os.path.join(study, sub)))
    score = {}
    for rid, r in runs.items():
        if r["spec"]["fold"] not in (0, 1):
            continue
        fam, level = parse_tag(rid)
        score.setdefault((fam, level), []).append(brier(r["dev"]["original"], r["eval_labels"]))
    mean = {k: sum(v) / len(v) for k, v in score.items() if len(v) == 2}
    selected = {}
    for fam in ("AD", "DK", "RA", "HU", "MX", "LS"):
        selected[fam] = choose_level({lv: s for (f, lv), s in mean.items() if f == fam})
    pool = {"N": mean[("N", None)], **{f: mean[(f, selected[f])] for f in selected}}
    c_star = min(pool, key=pool.get)
    return runs, {"rule": "levels", "regime": "encoder", "selected": selected, "C_star": c_star,
                  "C_star_level": selected.get(c_star), "RA_star": selected["RA"],
                  "criterion": "mean dev Brier over folds 0-1 (100 examples); ties within 0.005 -> lower dose"}


def levels_s2(study):
    runs = load_runs(os.path.join(study, "tuning"))
    base = os.path.join(os.path.dirname(study), "S2_base2")
    winner = json.load(open(os.path.join(base, "decisions_G1.json"), encoding="utf-8"))["lr_winner"]
    n_run = load_runs(base)[winner]
    final = {("N", None): n_run["validation"][-1]["val_nll"]}
    for rid, r in runs.items():
        final[parse_tag(rid)] = r["validation"][-1]["val_nll"]
    selected = {}
    for fam in ("AD", "DK", "RA", "HU", "RD"):
        selected[fam] = choose_level({lv: s for (f, lv), s in final.items() if f == fam})
    pool = {"N": final[("N", None)], **{f: final[(f, selected[f])] for f in selected}}
    c_star = min(pool, key=pool.get)
    return runs, {"rule": "levels", "regime": "lm", "selected": selected, "C_star": c_star,
                  "C_star_level": selected.get(c_star), "RA_star": selected["RA"],
                  "criterion": "final validation NLL at seed 1; ties within 0.005 -> lower dose"}


def prune_unselected(study, runs, decision, keep_always):
    """Delete, unopened, sealed test files of tuning runs whose level was not selected."""
    deleted = []
    for rid in runs:
        fam, level = parse_tag(rid)
        if fam in ("N",) or (fam, level) in keep_always:
            continue
        if decision["selected"].get(fam) == level:
            continue
        for sub in ("tuning", "gate"):
            sealed = os.path.join(study, sub, "sealed")
            for name in os.listdir(sealed) if os.path.isdir(sealed) else []:
                if name.startswith(rid + ".") or name.startswith(rid + "_"):
                    os.remove(os.path.join(sealed, name))
                    deleted.append(name)
    return deleted


def main():
    cmd, folder = sys.argv[1], os.path.abspath(sys.argv[2])
    if cmd == "g1":
        decision = g1(folder)
    elif cmd == "ge":
        decision = ge(folder, sys.argv[3].split(","))
    elif cmd == "g3":
        runs = load_runs(folder)
        acc = []
        for f in range(5):
            r = runs[f"ES3_T_f{f}"]
            logits, labels = r["exported_logits"]["dev_priv"]["logits"], r["eval_labels"]
            acc += [int(max(range(3), key=lambda c: row[c]) == y) for row, y in zip(logits, labels)]
        pooled = sum(acc) / len(acc)
        decision = {"rule": "G3", "pooled_privileged_oof_accuracy_T0_T4": pooled, "examples": len(acc),
                    "action": "proceed with students" if pooled >= 0.65 else "cut students (E-S3 descriptive teachers only)"}
    elif cmd == "levels":
        import manipulation_check  # PROTOCOL B11, before any finals are declared
        mc_path = os.path.join(folder, "manipulation_B11.json")
        mc = (json.load(open(mc_path, encoding="utf-8")) if os.path.exists(mc_path) else
              manipulation_check.check(os.path.join(folder, "tuning"), "S1" if sys.argv[3] == "encoder" else "S2", mc_path))
        if mc["recalibrate"]:
            raise SystemExit("B11: realized-displacement ratio outside [0.8, 1.25]; recalibrate once before finals")
        runs, decision = (levels_s1 if sys.argv[3] == "encoder" else levels_s2)(folder)
        core = {("HB", 0.10), ("HF", 0.10), ("RB", 0.10), ("RA", 0.10), ("HU", 0.10)}
        decision["deleted_unopened"] = prune_unselected(folder, runs, decision, core)
    else:
        raise SystemExit("unknown command")
    path = os.path.join(folder, f"decisions_{decision['rule']}.json")
    runlib.write_json_once(path, decision)
    print(json.dumps({k: v for k, v in decision.items()}, indent=1))


if __name__ == "__main__":
    main()
