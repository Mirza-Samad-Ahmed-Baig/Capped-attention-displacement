"""Confirmatory, secondary and descriptive analysis (PROTOCOL A §2, B1-B6, B10). No torch, no sealed
access: reads results/unsealed.json, results/cv_report.json and the selection decisions.
Writes results/analysis.json and paper/tables/*.tex.
"""
import json
import os

import numpy as np

import analysis_io as aio
import analyze_ext
import runlib
import stats_unified as su

ENC_MARGIN, LM_MARGIN = 0.015, 0.010


def enc_tag(fam, level):
    if fam == "N":
        return "N"
    return f"{fam}@{level}" if fam in ("MX", "LS") else f"{fam}@{level:.2f}"


def lm_tag(fam, level):
    if fam == "N":
        return "N"
    return f"RD@{level}" if fam == "RD" else f"{fam}@{level:.2f}"


def enc_values(uns, tag, rep=1, key="per_example_brier"):
    runs = {v["fold"]: v for v in uns["S1"].values() if v["tag"] == tag and v["rep"] == rep}
    if sorted(runs) != list(range(10)):
        return None
    return np.array([runs[f][key] for f in range(10)])  # [folds, examples]


def enc_fold_means(uns, tag, rep=1, key="per_example_brier"):
    m = enc_values(uns, tag, rep, key)
    return None if m is None else m.mean(1)


def lm_values(uns, tag, which="final"):
    runs = {v["seed"]: v for v in uns["S2"].values() if v["tag"] == tag and which in v}
    if sorted(runs) != list(range(1, 11)):
        return None
    return np.array([runs[s][which]["nll_per_token"] for s in range(1, 11)])


def encoder_contrast(uns, a, b, use_p1=False):
    """HB-type contrast a - b on test Brier (raw primary; TS co-criterion; two-level bootstrap)."""
    va, vb = enc_values(uns, a), enc_values(uns, b)
    ta, tb = enc_values(uns, a, key="per_example_brier_ts"), enc_values(uns, b, key="per_example_brier_ts")
    if va is None or vb is None:
        return {"available": False}
    da, db = va.mean(1), vb.mean(1)
    tsa, tsb = ta.mean(1), tb.mean(1)
    if use_p1:
        a2, b2 = enc_fold_means(uns, a, 2), enc_fold_means(uns, b, 2)
        if a2 is not None and b2 is not None:
            da, db = (da + a2) / 2, (db + b2) / 2
            ta2, tb2 = enc_fold_means(uns, a, 2, "per_example_brier_ts"), enc_fold_means(uns, b, 2, "per_example_brier_ts")
            tsa, tsb = (tsa + ta2) / 2, (tsb + tb2) / 2
    diff, diff_ts = da - db, tsa - tsb
    return {"available": True, "a": a, "b": b, "units": 10, "p1_averaged": use_p1,
            "mean_diff": float(diff.mean()), "sd_diff": float(diff.std(ddof=1)), "diffs": diff.round(6).tolist(),
            "p_raw": su.sign_flip_test(diff), "p_ts": su.sign_flip_test(diff_ts), "mean_diff_ts": float(diff_ts.mean()),
            "ci95_signflip": su.sign_flip_ci(diff, 0.05, 1e-4),
            "ci95_two_level": su.two_level_bootstrap(va, vb, reps=10000, level=0.95),
            "ci90_two_level": su.two_level_bootstrap(va, vb, reps=10000, level=0.90),
            "acc_diff": float((enc_fold_means(uns, a, key="correct") - enc_fold_means(uns, b, key="correct")).mean())}


def lm_contrast(uns, a, b):
    fa, fb = lm_values(uns, a, "final"), lm_values(uns, b, "final")
    ba, bb = lm_values(uns, a, "best"), lm_values(uns, b, "best")
    if fa is None or fb is None:
        return {"available": False}
    d, dbest = fa - fb, ba - bb
    return {"available": True, "a": a, "b": b, "units": 10, "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "diffs": d.round(6).tolist(), "p_final": su.sign_flip_test(d), "p_best": su.sign_flip_test(dbest),
            "mean_diff_best": float(dbest.mean()), "ci90_t": su.t_ci(d, 0.90), "ci95_signflip": su.sign_flip_ci(d, 0.05, 1e-5)}


def confirmatory(uns):
    s1 = aio.load(os.path.join(aio.STUDIES, "S1", "decisions_levels.json"))
    s2 = aio.load(os.path.join(aio.STUDIES, "S2", "decisions_levels.json"))
    ra_e, c_e = enc_tag("RA", s1["RA_star"]), enc_tag(s1["C_star"], s1["C_star_level"])
    ra_l, c_l = lm_tag("RA", s2["RA_star"]), lm_tag(s2["C_star"], s2["C_star_level"])
    tests = {"H1-E": encoder_contrast(uns, "HB@0.10", "HF@0.10", use_p1=True),
             "H2-E": encoder_contrast(uns, "HB@0.10", ra_e), "H3-E": encoder_contrast(uns, "HB@0.10", c_e),
             "H1-L": lm_contrast(uns, "HB@0.10", "HF@0.10"), "H2-L": lm_contrast(uns, "HB@0.10", ra_l),
             "H3-L": lm_contrast(uns, "HB@0.10", c_l)}
    keys = [k for k, v in tests.items() if v.get("available")]
    primary = [tests[k]["p_raw"] if k.endswith("E") else tests[k]["p_final"] for k in keys]
    adj = su.holm(primary)
    for k, a in zip(keys, adj):
        t = tests[k]
        t["p_holm"] = a
        factor = a / max(1e-300, (t["p_raw"] if k.endswith("E") else t["p_final"]))
        if k.endswith("E"):
            co = min(1.0, t["p_ts"] * factor)
            lo95, hi95 = t["ci95_two_level"]
            sig = a < 0.05 and (hi95 < 0 or lo95 > 0)
            if sig and co < 0.05 and np.sign(t["mean_diff_ts"]) == np.sign(t["mean_diff"]):
                t["verdict"] = "Better" if t["mean_diff"] < 0 else "Worse"
            elif sig:
                t["verdict"] = "calibration only"
            elif -ENC_MARGIN < t["ci90_two_level"][0] and t["ci90_two_level"][1] < ENC_MARGIN:
                t["verdict"] = "Equivalent"
            else:
                t["verdict"] = "Inconclusive"
        else:
            co = min(1.0, t["p_best"] * factor)
            if a < 0.05 and co < 0.05 and np.sign(t["mean_diff_best"]) == np.sign(t["mean_diff"]):
                t["verdict"] = "Better" if t["mean_diff"] < 0 else "Worse"
            elif -LM_MARGIN < t["ci90_t"][0] and t["ci90_t"][1] < LM_MARGIN:
                t["verdict"] = "Equivalent"
            else:
                t["verdict"] = "Inconclusive"
        t["co_criterion_p_holm_scaled"] = co
    return {"tests": tests, "comparators": {"RA*_E": ra_e, "C*_E": c_e, "RA*_L": ra_l, "C*_L": c_l},
            "holm_family": keys}


def secondary(uns, extra=None):
    out = {"S-HBvsHU-E": encoder_contrast(uns, "HB@0.10", "HU@0.10"), "S-RBvsRA-E": encoder_contrast(uns, "RB@0.10", "RA@0.10"),
           "S-HBvsHU-L": lm_contrast(uns, "HB@0.10", "HU@0.10"), "S-RBvsRA-L": lm_contrast(uns, "RB@0.10", "RA@0.10")}
    # 2x2 interaction (HB - HF) - (RB - RA) on per-fold test Brier
    cells = {t: enc_fold_means(uns, t) for t in ("HB@0.10", "HF@0.10", "RB@0.10", "RA@0.10")}
    if all(v is not None for v in cells.values()):
        inter = (cells["HB@0.10"] - cells["HF@0.10"]) - (cells["RB@0.10"] - cells["RA@0.10"])
        out["S-interaction-E"] = {"mean": float(inter.mean()), "p": su.sign_flip_test(inter), "diffs": inter.round(6).tolist()}
    # evidence use HB vs N (per fold)
    def ev(tag, key):
        runs = {v["fold"]: v for v in uns["S1"].values() if v["tag"] == tag and v["rep"] == 1}
        return np.array([runs[f]["evidence_use_test"][key] for f in range(10)]) if sorted(runs) == list(range(10)) else None
    for key, name in (("evidence_use_gap", "S-EUG"), ("flip_rate", "S-flip")):
        a, b = ev("HB@0.10", key), ev("N", key)
        if a is not None and b is not None:
            d = a - b
            out[name] = {"mean_diff": float(d.mean()), "p": su.sign_flip_test(d), "hb": float(a.mean()), "n": float(b.mean())}
    out.update(extra or {})  # E-S3 S-a / S-b when gates G3 and G4 pass (PROTOCOL B1, B7)
    names = [k for k, v in out.items() if isinstance(v, dict) and ("p_raw" in v or "p_final" in v or "p" in v)]
    ps = [out[k].get("p_raw", out[k].get("p_final", out[k].get("p"))) for k in names]
    for k, a in zip(names, su.holm(ps)):
        t = out[k]
        t["p_holm_F2"] = a
        if "ci95_two_level" in t:  # encoder contrast: the F1 encoder verdict rules with the F2 Holm p
            t["verdict_F2"] = analyze_ext.enc_verdict(t, a, a / max(1e-300, t["p_raw"]))
        elif "p_final" in t:
            co = min(1.0, t["p_best"] * a / max(1e-300, t["p_final"]))
            if a < 0.05 and co < 0.05 and np.sign(t["mean_diff_best"]) == np.sign(t["mean_diff"]):
                t["verdict_F2"] = "Better" if t["mean_diff"] < 0 else "Worse"
            elif -LM_MARGIN < t["ci90_t"][0] and t["ci90_t"][1] < LM_MARGIN:
                t["verdict_F2"] = "Equivalent"
            else:
                t["verdict_F2"] = "Inconclusive"
    return out


def descriptive(uns, cv):
    enc = {}
    for tag in sorted({v["tag"] for v in uns["S1"].values()}):
        runs = [v for v in uns["S1"].values() if v["tag"] == tag and v["rep"] == 1]
        if not runs:
            continue
        def ms(key, sub="test"):
            x = np.array([r[sub][key] for r in runs])
            return [float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0]
        enc[tag] = {"n_runs": len(runs), "brier": ms("brier"), "brier_ts": ms("brier", "test_ts"), "accuracy": ms("accuracy"),
                    "macro_f1": ms("macro_f1"), "auroc": ms("auroc"), "nll": ms("nll"), "ece15": ms("ece15"),
                    "failures": int(sum(r["failed"] for r in runs)), "collapses": int(sum(r["collapsed"] for r in runs)),
                    "evidence_use_gap": float(np.mean([r["evidence_use_test"]["evidence_use_gap"] for r in runs])),
                    "flip_rate": float(np.mean([r["evidence_use_test"]["flip_rate"] for r in runs])),
                    "donor_following": float(np.mean([r["evidence_use_test"]["donor_label_following"] for r in runs])),
                    "results_deleted_acc": float(np.mean([r["results_deleted_test_accuracy"] for r in runs])),
                    "question_only_acc": float(np.mean([r["question_only_test"]["accuracy"] for r in runs])),
                    "ensemble_accuracy": None}
        cvr = [u for k, u in cv["S1"].get(tag, {}).items() if k.endswith("_1")]
        if cvr:
            enc[tag]["realized_tv"] = float(np.mean([u["budget"]["realized_tv"] for u in cvr if u.get("budget")])) if any(u.get("budget") for u in cvr) else None
            enc[tag]["train_minutes"] = float(np.mean([u["train_seconds"] for u in cvr]) / 60)
            enc[tag]["dev_brier"] = float(np.mean([u["dev"]["brier"] for u in cvr]))
    lm = {}
    for tag in sorted({v["tag"] for v in uns["S2"].values()}):
        runs = [v for v in uns["S2"].values() if v["tag"] == tag]
        f = np.array([r["final"]["nll_per_token"] for r in runs])
        b = np.array([r["best"]["nll_per_token"] for r in runs if "best" in r])
        lm[tag] = {"n_runs": len(runs), "final": [float(f.mean()), float(f.std(ddof=1)) if len(f) > 1 else 0.0],
                   "best": [float(b.mean()), float(b.std(ddof=1)) if len(b) > 1 else 0.0]}
    return {"S1": enc, "S2": lm}


def main():
    uns = aio.load(os.path.join(aio.ROOT, "results", "unsealed.json"))
    cv = aio.load(os.path.join(aio.ROOT, "results", "cv_report.json"))
    conf = confirmatory(uns)
    ext = analyze_ext.run(uns, cv, conf)
    f2_extra = {k: ext["ES3"][k] for k in ("S-a", "S-b")
                if ext["ES3"].get("confirmatory_in_F2") and ext["ES3"].get(k, {}).get("available")}
    result = {"utc": runlib.utc(), "confirmatory": conf, "secondary": secondary(uns, f2_extra),
              "descriptive": descriptive(uns, cv), "extensions": ext}
    path = os.path.join(aio.ROOT, "results", "analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for k, t in result["confirmatory"]["tests"].items():
        if t.get("available"):
            print(k, t["a"], "vs", t["b"], "mean diff", round(t["mean_diff"], 5), "p_holm", round(t["p_holm"], 4), t["verdict"])
    print("analysis.json", runlib.sha256_file(path))


if __name__ == "__main__":
    main()
