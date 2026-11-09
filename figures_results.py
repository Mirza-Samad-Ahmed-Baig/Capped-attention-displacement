"""Result figures F2-F6 (PROTOCOL A §6), written before the unseal and run once afterwards.

F2 forest plot of the F1 and F2 contrasts with equivalence margins; F3 test loss against realized mean
displacement; F4 LM validation curves (training records only); F5 per-run PubMedQA test accuracy with the
majority line; F6 evidence-use diagnostics. Reads results/analysis.json, results/unsealed.json,
results/cv_report.json and LM run records. Writes paper/figures/fig_*.pdf. Usage: figures_results.py [ROOT]
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "paper", "figures")
FAM_COL = {"N": "#000000", "AD": "#999999", "DK": "#56B4E9", "RA": "#009E73", "RB": "#66C2A5", "HU": "#CC79A7",
           "HF": "#E69F00", "HB": "#0072B2", "MX": "#8C564B", "LS": "#BCBD22", "RD": "#8C564B", "Q0": "#D55E00",
           "L0": "#D55E00", "RDROP": "#7F7F7F", "DB": "#17BECF"}
VERDICT_MARK = {"Better": "o", "Worse": "s", "Equivalent": "D", "Inconclusive": "x", "calibration only": "^"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fam(tag):
    return tag.split("@")[0]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)


def forest(an):
    rows = []
    for k, t in an["confirmatory"]["tests"].items():
        if t.get("available"):
            enc = k.endswith("E")
            ci = t["ci95_two_level"] if enc else t["ci95_signflip"]
            rows.append((f"{k}: {t['a']} $-$ {t['b']}", t["mean_diff"], ci, enc, t.get("verdict", "")))
    for k, t in an["secondary"].items():
        if isinstance(t, dict) and t.get("available") and "mean_diff" in t:
            enc = "ci95_two_level" in t
            ci = t["ci95_two_level"] if enc else t.get("ci95_signflip", t.get("ci90_t"))
            rows.append((f"F2 {k}", t["mean_diff"], ci, enc, t.get("verdict_F2", "")))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * max(len(rows), 6) + 1.2), constrained_layout=True)
    for ax, enc, margin, unit in ((axes[0], True, 0.015, "test Brier difference"), (axes[1], False, 0.010, "test NLL difference (nats/token)")):
        sub = [r for r in rows if r[3] == enc]
        ys = np.arange(len(sub))[::-1]
        ax.axvspan(-margin, margin, color="0.92", zorder=0)
        ax.axvline(0, color="0.5", lw=0.8)
        for y, (label, m, ci, _, verdict) in zip(ys, sub):
            ax.plot(ci, [y, y], color="#0072B2", lw=1.6)
            ax.plot([m], [y], VERDICT_MARK.get(verdict, "o"), color="#0072B2", ms=5, mfc="white" if verdict == "Inconclusive" else "#0072B2")
            ax.text(1.01, y, verdict, transform=ax.get_yaxis_transform(), fontsize=7, va="center")
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in sub], fontsize=7.5)
        ax.set_xlabel(f"{unit}, first arm minus second\n(↓ lower is better: negative favours the first arm)", fontsize=8)
        ax.set_title(("Encoder" if enc else "Language model") + f": 95% CI, margin $\\pm${margin}", fontsize=9, loc="left")
        style(ax)
    fig.savefig(os.path.join(OUT, "fig_forest.pdf"))


# Arms at the same dose sit on almost the same x, so some labels go to the left of their point.
LEFT_LABELS_ENC = {"RA@0.10", "DB@0.10", "RA@0.05", "DK@0.05"}


def _label(ax, tag, x, y, left=False):
    ax.annotate(tag, (x, y), fontsize=6.5, xytext=(-6 if left else 6, 0), textcoords="offset points",
                ha="right" if left else "left", va="center", color="0.15")


def loss_vs_displacement(an, cv, uns):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    enc = an["descriptive"]["S1"]
    for tag, d in enc.items():
        f = fam(tag)
        if f not in FAM_COL or f in ("Q0", "L0", "MX", "LS", "RDROP"):
            continue
        units = [u for k, u in cv["S1"].get(tag, {}).items() if k.endswith("_1")]
        dbar = np.mean([(u.get("budget") or {}).get("realized_tv", 0.0) for u in units]) if units else 0.0
        axes[0].errorbar(dbar, d["brier"][0], yerr=d["brier"][1] / np.sqrt(max(1, d["n_runs"])), fmt="o", color=FAM_COL[f], ms=5)
        _label(axes[0], tag, dbar, d["brier"][0], left=tag in LEFT_LABELS_ENC)
    axes[0].set_xlabel("realized mean displacement during training")
    axes[0].set_ylabel("test Brier, mean $\\pm$ s.e., 10 folds\n(↓ lower is better)")
    lm = an["descriptive"]["S2"]
    for tag, d in lm.items():
        f = fam(tag)
        if f not in FAM_COL or f == "RD":
            continue
        units = list(cv["S2"].get(tag, {}).values())
        dbar = np.mean([(u.get("budget") or {}).get("realized_tv", 0.0) for u in units]) if units else 0.0
        axes[1].errorbar(dbar, d["final"][0], yerr=d["final"][1] / np.sqrt(max(1, d["n_runs"])), fmt="o", color=FAM_COL[f], ms=5)
        _label(axes[1], tag, dbar, d["final"][0], left=dbar >= 0.19)
    axes[1].set_xlabel("realized mean displacement during training")
    axes[1].set_ylabel("test NLL at final step (nats/token)\n(↓ lower is better)")
    for ax in axes:
        style(ax)
    fig.savefig(os.path.join(OUT, "fig_loss_vs_displacement.pdf"))


def lm_curves():
    runs = {}
    for path in glob.glob(os.path.join(ROOT, "studies", "S2", "*", "runs", "S2_*.json")):
        r = load(path)
        tag = r["spec"]["run_id"][3:].rsplit("_s", 1)[0]
        runs.setdefault(tag, []).append(r)
    winner = os.path.join(ROOT, "studies", "S2_base2", "decisions_G1.json")
    if os.path.exists(winner):
        w = load(winner)["lr_winner"]
        p = os.path.join(ROOT, "studies", "S2_base2", "runs", f"{w}.json")
        if os.path.exists(p):
            runs.setdefault("N", []).append(load(p))
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.4), constrained_layout=True)
    for tag, rs in sorted(runs.items()):
        f = fam(tag)
        if f not in FAM_COL and f != "HBsg" or len(rs) < 10:
            continue  # only arms run on all ten seeds (finals and extensions); single tuning runs are omitted
        steps = [e["step"] for e in rs[0]["validation"]]
        vals = np.array([[e["val_nll"] for e in r["validation"]] for r in rs if len(r["validation"]) == len(steps)])
        ax.plot(steps, vals.mean(0), color=FAM_COL.get(f, FAM_COL["HB"]), lw=1.3,
                ls="--" if f == "HBsg" or tag.endswith("@0.05") else "-", label=f"{tag} (n={len(vals)})")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation NLL (nats/token)\n(↓ lower is better)")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    style(ax)
    fig.savefig(os.path.join(OUT, "fig_lm_curves.pdf"))


def strip(uns):
    tags = sorted({v["tag"] for v in uns["S1"].values() if v["rep"] == 1})
    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    for i, tag in enumerate(tags):
        acc = [100 * v["test"]["accuracy"] for v in uns["S1"].values() if v["tag"] == tag and v["rep"] == 1]
        x = i + np.linspace(-0.18, 0.18, len(acc))
        ax.plot(x, acc, "o", ms=3.5, color=FAM_COL.get(fam(tag), "0.3"), alpha=0.85)
        ax.plot([i - 0.3, i + 0.3], [np.mean(acc)] * 2, color="k", lw=1)
    ax.axhline(55.2, color="#D55E00", lw=0.9, ls="--")
    ax.text(len(tags) - 0.5, 55.6, "majority class (55.2%)", fontsize=7, color="#D55E00", ha="right")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("PubMedQA test accuracy (%)\n(↑ higher is better)")
    style(ax)
    fig.savefig(os.path.join(OUT, "fig_strip.pdf"))


def evidence(an):
    enc = an["descriptive"]["S1"]
    tags = [t for t in sorted(enc) if fam(t) in FAM_COL]
    keys = (("evidence_use_gap", "evidence-use gap"), ("flip_rate", "flip rate, mismatched abstract"),
            ("donor_following", "follows donor label"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for ax, (k, title) in zip(axes, keys):
        vals = [enc[t][k] for t in tags]
        ax.barh(range(len(tags)), vals, color=[FAM_COL[fam(t)] for t in tags])
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels(tags, fontsize=7)
        ax.set_title(title, fontsize=9, loc="left")
        style(ax)
    fig.savefig(os.path.join(OUT, "fig_evidence.pdf"))


def main():
    os.makedirs(OUT, exist_ok=True)
    an = load(os.path.join(ROOT, "results", "analysis.json"))
    uns = load(os.path.join(ROOT, "results", "unsealed.json"))
    cv = load(os.path.join(ROOT, "results", "cv_report.json"))
    forest(an)
    loss_vs_displacement(an, cv, uns)
    lm_curves()
    strip(uns)
    evidence(an)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
