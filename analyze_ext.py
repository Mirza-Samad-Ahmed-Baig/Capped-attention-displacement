"""Extension, venue and exploratory analyses (PROTOCOL A §2 exploratory, §3.5; B1 F2, B5, B6, B7, B10).
No torch, no sealed access. Inputs: results/unsealed.json, results/cv_report.json, run records (dev and
training traces only), selection decisions and census predictions. Called by analyze.py.

Interpretations fixed before the unseal (FREEZE_ADDENDUM.jsonl):
  * B5 venue rule: E4 and E5 have no HF arm, so "reproduces the H1-E verdict" is assessed on their only cap
    contrast, HB@0.10 - HU@0.10, judged by the H1-E rules (unit sign-flip test, two-level bootstrap CI, TS
    co-criterion, margin 0.015), unadjusted. It reproduces if both verdicts are the same one of
    Better / Worse / Equivalent.
  * E2 "upper 90% bound of the reduction": the upper limit of a two-sided 90% fold-cluster bootstrap interval
    (10^4 resamples of folds, both seeds kept) for rate(F-AD) - rate(arm).
  * E-S3 S-a and S-b join the F2 Holm family only if G3 and G4 pass; otherwise E-S3 is descriptive.
  * Extension contrasts other than S-a, S-b are reported with unadjusted p and labelled descriptive.
  * E6 was not run (budget); E8 (HBsg@0.10 - HB@0.10 in the LM) follows the H1-L rules, descriptive.
"""
import json
import os

import numpy as np

import analysis_io as aio
import stats_unified as su

ENC_MARGIN, LM_MARGIN = 0.015, 0.010
FAIL_ACC = 0.5520  # PROTOCOL A §2 / §3.5: failure = test accuracy <= 55.20%


def by_unit(bucket, study, tag):
    return {(v["fold"], v["rep"]): v for v in bucket.values() if v.get("study") == study and v["tag"] == tag}


def enc_pair(a, b, n_expected, units=None):
    """Paired encoder contrast a - b on per-example test Brier (raw) with the TS co-criterion."""
    keys = sorted(set(a) & set(b))
    if units is not None:
        keys = [k for k in keys if k in units]
    if len(keys) != n_expected:
        return {"available": False, "units_found": len(keys)}
    va = np.array([a[k]["per_example_brier"] for k in keys])
    vb = np.array([b[k]["per_example_brier"] for k in keys])
    ta = np.array([a[k]["per_example_brier_ts"] for k in keys])
    tb = np.array([b[k]["per_example_brier_ts"] for k in keys])
    diff, diff_ts = va.mean(1) - vb.mean(1), ta.mean(1) - tb.mean(1)
    acc = np.array([np.mean(a[k]["correct"]) - np.mean(b[k]["correct"]) for k in keys])
    return {"available": True, "units": len(keys), "mean_diff": float(diff.mean()), "sd_diff": float(diff.std(ddof=1)),
            "diffs": diff.round(6).tolist(), "p_raw": su.sign_flip_test(diff), "p_ts": su.sign_flip_test(diff_ts),
            "mean_diff_ts": float(diff_ts.mean()), "acc_diff": float(acc.mean()),
            "ci95_two_level": su.two_level_bootstrap(va, vb, reps=10000, level=0.95),
            "ci90_two_level": su.two_level_bootstrap(va, vb, reps=10000, level=0.90)}


def enc_verdict(t, p_adj=None, factor=1.0):
    if not t.get("available"):
        return "unavailable"
    p = t["p_raw"] if p_adj is None else p_adj
    co = min(1.0, t["p_ts"] * factor)
    lo95, hi95 = t["ci95_two_level"]
    sig = p < 0.05 and (hi95 < 0 or lo95 > 0)
    if sig and co < 0.05 and np.sign(t["mean_diff_ts"]) == np.sign(t["mean_diff"]):
        return "Better" if t["mean_diff"] < 0 else "Worse"
    if sig:
        return "calibration only"
    if -ENC_MARGIN < t["ci90_two_level"][0] and t["ci90_two_level"][1] < ENC_MARGIN:
        return "Equivalent"
    return "Inconclusive"


def lm_pair(uns, a, b, seeds=range(1, 11)):
    ra = {v["seed"]: v for v in uns["S2"].values() if v["tag"] == a}
    rb = {v["seed"]: v for v in uns["S2"].values() if v["tag"] == b}
    if not all(s in ra and s in rb and "best" in ra[s] and "best" in rb[s] for s in seeds):
        return {"available": False}
    d = np.array([ra[s]["final"]["nll_per_token"] - rb[s]["final"]["nll_per_token"] for s in seeds])
    db = np.array([ra[s]["best"]["nll_per_token"] - rb[s]["best"]["nll_per_token"] for s in seeds])
    t = {"available": True, "a": a, "b": b, "units": len(d), "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
         "diffs": d.round(6).tolist(), "p_final": su.sign_flip_test(d), "p_best": su.sign_flip_test(db),
         "mean_diff_best": float(db.mean()), "ci90_t": su.t_ci(d, 0.90)}
    if t["p_final"] < 0.05 and t["p_best"] < 0.05 and np.sign(t["mean_diff"]) == np.sign(t["mean_diff_best"]):
        t["verdict"] = "Better" if t["mean_diff"] < 0 else "Worse"
    elif -LM_MARGIN < t["ci90_t"][0] and t["ci90_t"][1] < LM_MARGIN:
        t["verdict"] = "Equivalent"
    else:
        t["verdict"] = "Inconclusive"
    return t


def contrast(bucket, study, a, b, n, units=None):
    t = enc_pair(by_unit(bucket, study, a), by_unit(bucket, study, b), n, units)
    t.update({"a": a, "b": b, "study": study})
    t["verdict_unadjusted"] = enc_verdict(t)
    return t


# ------------------------------------------------------------------------------------ E2
def e2(ext):
    arms = ("F-AD", "F-HB@0.10", "F-MX")
    fails = {a: {k: bool(v["test"]["accuracy"] <= FAIL_ACC + 1e-12) for k, v in by_unit(ext, "E2", a).items()} for a in arms}
    if any(len(fails[a]) != 20 for a in arms):
        return {"available": False, "runs": {a: len(fails[a]) for a in arms}}
    out = {"available": True, "failure_rate": {a: float(np.mean(list(fails[a].values()))) for a in arms}}
    out["inconclusive_by_rule"] = bool(out["failure_rate"]["F-AD"] < 0.30)
    keys = sorted(fails["F-AD"])
    rng = np.random.default_rng(0)
    folds = sorted({k[0] for k in keys})
    tests, ps = {}, []
    for arm in ("F-HB@0.10", "F-MX"):
        fa = [fails[arm][k] for k in keys]
        fb = [fails["F-AD"][k] for k in keys]
        obs, p = su.stratified_permutation_failure(fa, fb, [k[0] for k in keys], reps=100000, alternative="less")
        boot = []
        for _ in range(10000):
            pick = rng.choice(folds, len(folds))
            ks = [k for f in pick for k in keys if k[0] == f]
            boot.append(np.mean([fails["F-AD"][k] for k in ks]) - np.mean([fails[arm][k] for k in ks]))
        lo, hi = np.quantile(boot, [0.05, 0.95])
        tests[arm] = {"reduction": float(-obs), "p_one_sided": p, "reduction_ci90": [float(lo), float(hi)]}
        ps.append(p)
    for arm, adj in zip(tests, su.holm(ps)):
        t = tests[arm]
        t["p_holm"] = adj
        if out["inconclusive_by_rule"]:
            t["verdict"] = "inconclusive (F-AD failed in fewer than 30% of runs)"
        elif adj < 0.05:
            t["verdict"] = "reduces failures"
        elif t["reduction_ci90"][1] < 0.25:
            t["verdict"] = "does not reduce failures"
        else:
            t["verdict"] = "inconclusive"
    out["tests"] = tests
    return out


# --------------------------------------------------------------------------- E4 / E5
def replication(ext, study, n, extra_arms):
    out = {"HB-HU": contrast(ext, study, "HB@0.10", "HU@0.10", n)}
    for a in ("HB@0.10", "HU@0.10") + tuple(extra_arms):
        out[f"{a}-N"] = contrast(ext, study, a, "N", n)
    diag = {}
    for tag in {v["tag"] for v in ext.values() if v.get("study") == study}:
        runs = list(by_unit(ext, study, tag).values())
        diag[tag] = {"runs": len(runs),
                     "claim_or_question_only_accuracy": float(np.mean([r["question_only_test"]["accuracy"] for r in runs])),
                     "evidence_deleted_accuracy": float(np.mean([r["results_deleted_test_accuracy"] for r in runs])),
                     "maybe_or_NEI_share_shift_on_deletion": float(np.mean([r["pred_share"]["results_deleted"][2] -
                                                                            r["pred_share"]["original"][2] for r in runs]))}
    out["diagnostics"] = diag
    return out


# ------------------------------------------------------------------------------ E-S3
def load_study_runs(prefix):
    return {rid: aio.load(p) for rid, p in aio.index_runs(prefix).items()}


def argmax_acc(logits, labels):
    return float(np.mean(np.asarray(logits).argmax(1) == np.asarray(labels)))


def es3(uns):
    g3_path = os.path.join(aio.STUDIES, "ES3", "teachers", "decisions_G3.json")
    if not os.path.exists(g3_path):
        return {"available": False}
    g3 = aio.load(g3_path)
    out = {"available": True, "G3": g3}
    if "proceed" not in g3["action"]:
        out["students"] = False
        return out
    runs = load_study_runs("ES3")
    s_n = [runs[f"ES3_S-N_f{f}"] for f in range(5) if f"ES3_S-N_f{f}" in runs]
    correct = [c for r in s_n for c in (np.asarray(r["dev"]["original"]).argmax(1) == np.asarray(r["eval_labels"]))]
    g4 = float(np.mean(correct)) if len(s_n) == 5 else None
    out["G4"] = {"pooled_S-N_oof_accuracy_folds_0_4": g4, "threshold": 0.57, "pass": g4 is not None and g4 >= 0.57}
    out["confirmatory_in_F2"] = out["G4"]["pass"]
    ext = uns["EXT"]
    out["S-a"] = contrast(ext, "ES3", "S-HB@0.10", "S-HU@0.10", 10)
    out["S-b"] = contrast(ext, "ES3", "S-HB@0.10", "S-N", 10)
    out["S-H1"] = contrast(ext, "ES3", "S-HB@0.10", "S-HF@0.10", 10)
    out["S-KD"] = contrast(ext, "ES3", "S-N", "S-noKD", 10)
    n1 = {v["fold"]: v for v in uns["S1"].values() if v["tag"] == "N" and v["rep"] == 1}
    retention, agreement = {}, {}
    for tag in ("S-N", "S-HU@0.10", "S-HB@0.10", "S-HF@0.10", "S-noKD"):
        u = {k[0]: v for k, v in by_unit(ext, "ES3", tag).items()}
        if len(u) == 10 and len(n1) == 10:
            retention[tag] = {"accuracy_ratio": float(np.mean([u[f]["test"]["accuracy"] / n1[f]["test"]["accuracy"] for f in range(10)])),
                              "brier_diff": float(np.mean([u[f]["test"]["brier"] - n1[f]["test"]["brier"] for f in range(10)]))}
        agree = []
        for f in range(10):
            s, t = runs.get(f"ES3_{tag}_f{f}"), runs.get(f"ES3_T_f{f}")
            if s and t:
                sp = np.asarray(s["dev"]["original"]).argmax(1)
                tp = np.asarray(t["exported_logits"]["dev_priv"]["logits"]).argmax(1)
                agree.append(float(np.mean(sp == tp)))
        agreement[tag] = float(np.mean(agree)) if agree else None
    out["retention_vs_S1_N"] = retention
    out["dev_agreement_with_privileged_teacher"] = agreement
    cost = os.path.join(aio.ROOT, "results", "cost.json")
    out["cost"] = aio.load(cost) if os.path.exists(cost) else None
    return out


# ------------------------------------------------------------------------------ E1 / E3
def e1(uns):
    path = os.path.join(aio.STUDIES, "S1", "decisions_E1.json")
    if not os.path.exists(path):
        return {"available": False}
    alpha = aio.load(path)["selected_alpha"]
    t = contrast(uns["S1"], "S1", "HB@0.10", f"RDROP@{alpha}", 10)
    return {"available": t.get("available", False), "selected_alpha": alpha, "HB-RDROP*": t}


def e3(uns, lm_pair_fn=lm_pair):
    return {"S1 DB-HB": contrast(uns["S1"], "S1", "DB@0.10", "HB@0.10", 10), "S2 DB-HB": lm_pair_fn(uns, "DB@0.10", "HB@0.10")}


# ------------------------------------------------------------------------------ E6 / E8
# B8 rule "each extension starts only if its full cost fits under the 90-h cap", applied per extension
# (FREEZE_ADDENDUM, 2026-11-08, recorded before E8 was declared and before the unseal).
E6_STATUS = {"run": False, "reason": "not run (budget): after E3 the compute used was 85.11 GPU-h, and E6's "
                                     "expected 5.6 GPU-h would exceed the 90 GPU-h cap (B8, B13)"}


def e8(uns, lm_pair_fn=lm_pair):
    """E8 (PROTOCOL B10): paired LM contrast HBsg@0.10 - HB@0.10 (alpha-detached minus standard HB) over seeds
    1-10, by the H1-L rules in lm_pair (final-checkpoint test NLL per token; sign-flip p on final and best;
    90% t-interval against the 0.010 margin). Unadjusted, outside the Holm family, labelled descriptive."""
    t = lm_pair_fn(uns, "HBsg@0.10", "HB@0.10")
    t["label"] = "descriptive (unadjusted)"
    return t


# ------------------------------------------------------------------ exploratory (Part A §2)
def exploratory(uns, cv):
    pred = aio.load(os.path.join(aio.ROOT, "results", "predictions.json"))
    out = {}
    s1 = uns["S1"]
    brier = lambda tag: [v["test"]["brier"] for v in s1.values() if v["tag"] == tag and v["rep"] == 1]
    hb = brier("HB@0.10")
    out["brown_forsythe_test_brier_HB@0.10_vs"] = {
        t: dict(zip(("statistic", "p"), su.brown_forsythe(hb, brier(t)))) for t in ("HF@0.10", "HU@0.10", "N", "RA@0.10")
        if len(brier(t)) >= 2 and len(hb) >= 2}
    # dose-response on dev metrics only: (HB@l - HU@l) against the census clipped excess at l
    pts_x, pts_y = [], []
    for lvl in ("0.05", "0.10", "0.20"):
        hbu, huu = cv["S1"].get(f"HB@{lvl}", {}), cv["S1"].get(f"HU@{lvl}", {})
        for fold in (0, 1):
            k = f"{fold}_1"
            if k in hbu and k in huu:
                pts_x.append(pred["encoder"]["levels"][lvl]["HB"]["excess"])
                pts_y.append(hbu[k]["dev"]["brier"] - huu[k]["dev"]["brier"])
    if len(pts_x) >= 4:
        rho, p = su.spearman_perm(pts_x, pts_y)
        out["dose_response_encoder_dev"] = {"points": len(pts_x), "spearman": rho, "p": p}
    # DAM: per-run test Brier ~ realized D + realized S2 + census B + fold effects, fitted on N/AD/DK/RA
    rows = []
    for v in s1.values():
        if v["rep"] != 1 or "@" not in v["tag"] and v["tag"] != "N":
            continue
        fam, lvl = (v["tag"].split("@") + [None])[:2]
        if fam not in ("N", "AD", "DK", "RA", "HU", "HB", "HF", "RB"):
            continue
        unit = cv["S1"].get(v["tag"], {}).get(f"{v['fold']}_1")
        b = (unit or {}).get("budget") or {}
        census_b = 0.0 if fam in ("N", "AD") else pred["encoder"]["levels"].get(lvl, {}).get(fam, {}).get("B")
        if census_b is None:
            continue
        rows.append((fam, v["fold"], b.get("realized_tv", 0.0), b.get("realized_tv_sq", 0.0), census_b, v["test"]["brier"]))
    fit = [r for r in rows if r[0] in ("N", "AD", "DK", "RA")]
    if len(fit) > 15:
        def design(rs):
            x = np.array([[1.0, r[2], r[3], r[4]] + [1.0 if r[1] == f else 0.0 for f in range(1, 10)] for r in rs])
            return x, np.array([r[5] for r in rs])
        X, y = design(fit)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        dam = {"coef": dict(zip(["intercept", "D", "S2", "B"], coef[:4].round(6).tolist())),
               "r2": float(1 - res.var() / y.var()), "prediction_error": {}}
        for fam in ("HU", "HB", "HF", "RB"):
            rs = [r for r in rows if r[0] == fam]
            if rs:
                Xp, yp = design(rs)
                dam["prediction_error"][fam] = float(np.mean(yp - Xp @ coef))
        out["DAM_descriptive"] = dam
    # LM test NLL by training-frequency bucket (descriptive)
    buckets = {}
    for v in uns["S2"].values():
        if "final" in v:
            buckets.setdefault(v["tag"], []).append(v["final"]["bucket_nll"])
    out["lm_bucket_nll_final"] = {t: np.nanmean(np.array(b, dtype=float), 0).round(5).tolist() for t, b in buckets.items()}
    return out


# ----------------------------------------------------------------------------- venue
def venue(conf, e4r, e5r):
    tests = conf["tests"]
    v = {k: tests[k].get("verdict") for k in tests}
    h1e = v.get("H1-E")
    rep = {}
    for name, r in (("E4", e4r), ("E5", e5r)):
        c = (r or {}).get("HB-HU", {})
        rep[name] = {"verdict": c.get("verdict_unadjusted"),
                     "reproduces_H1-E": h1e in ("Better", "Worse", "Equivalent") and c.get("verdict_unadjusted") == h1e}
    enc = [v.get(k) for k in ("H1-E", "H2-E", "H3-E")]
    return {"verdicts": v, "replication": rep,
            "all_encoder_F1_inconclusive": all(x == "Inconclusive" for x in enc),
            "H1_signs_disagree": bool(tests.get("H1-E", {}).get("available") and tests.get("H1-L", {}).get("available") and
                                       np.sign(tests["H1-E"]["mean_diff"]) != np.sign(tests["H1-L"]["mean_diff"])),
            "neural_networks_eligible_B5": any(r["reproduces_H1-E"] for r in rep.values()),
            "note": "the outcome->claim table (PROTOCOL A §2, B5) maps these facts to the claim; applied by the authors"}


def run(uns, cv, conf):
    ext = uns.get("EXT", {})
    e4r = replication(ext, "E4", 10, (next((t for t in {v["tag"] for v in ext.values() if v.get("study") == "E4"}
                                             if t.startswith("RA@")), "RA@0.05"),))
    e5r = replication(ext, "E5", 10, ("AD@0.20",))
    s1 = uns["S1"]
    oof = {k: contrast(s1, "S1", "HB@0.10", conf["comparators"][c], 8, units={(f, 1) for f in range(2, 10)})
           for k, c in (("H2-E folds 2-9", "RA*_E"), ("H3-E folds 2-9", "C*_E"))}
    p1_seed2 = contrast({k: v for k, v in s1.items() if v["rep"] == 2},
                        "S1", "HB@0.10", "HF@0.10", 10, units={(f, 2) for f in range(10)})
    return {"E2": e2(ext), "E4": e4r, "E5": e5r, "ES3": es3(uns), "E1": e1(uns), "E3": e3(uns),
            "E6": E6_STATUS, "E8": e8(uns), "B6_oof_replication": oof, "P1_seed2_H1-E": p1_seed2, "exploratory": exploratory(uns, cv),
            "venue": venue(conf, e4r, e5r)}
