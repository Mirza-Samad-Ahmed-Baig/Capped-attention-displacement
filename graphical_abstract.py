"""Graphical abstract (Elsevier: readable at 5 x 13 cm; at least 1328 x 531 px). Three panels:
(1) the CapDrop rule, displacement d against deleted mass m at the LM operating point; (2) the LM census tails at
matched mean displacement 0.10 (exact, outcome-independent); (3) the LM test results against CapDrop, read from
results/analysis.json. Writes paper/figures/graphical_abstract.{pdf,png}."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figures_census as fc  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
COL = fc.COL  # HU #CC79A7, HF #E69F00, HB #0072B2 (Okabe-Ito; validated, direct labels relieve the contrast WARN)
BETTER, WORSE, INK, MUTED = "#0072B2", "#D55E00", "#222222", "#6b6b6b"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "font.family": "DejaVu Sans"})


def load(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return json.load(f)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9.5)


def panel_rule(ax, rec):
    m = np.linspace(0, 0.8, 200)
    tau, gamma = rec["HB"]["tau"], rec["HF"]["gamma"]
    ax.plot(m, m, color=COL["HU"], lw=2, ls=(0, (4, 2)))
    ax.plot(m, gamma * m, color=COL["HF"], lw=2)
    ax.plot(m, np.minimum(m, tau), color=COL["HB"], lw=2.6)
    ax.text(0.64, 0.74, "no cap", color=INK, fontsize=9.5, ha="right")
    ax.text(0.79, gamma * 0.79 + 0.03, "fixed mix", color=INK, fontsize=9.5, ha="right")
    ax.text(0.79, tau + 0.03, "CapDrop", color=COL["HB"], fontsize=10.5, ha="right", weight="bold")
    ax.text(0.02, 0.72, r"$\tilde a=(1-\alpha)\,a+\alpha\, b$" "\n" r"$\alpha=\min(1,\ \tau/m)$",
            fontsize=10.5, color=INK, va="top")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("attention mass deleted, $m$")
    ax.set_ylabel("row displacement, $d$")
    ax.set_title("1  New: cap how far each row moves", loc="left", color=INK)
    style(ax)


def panel_tails(ax, rec):
    census = load("results", "s0", "lm_census.json")
    pooled = {a: ([], []) for a in ("HU", "HF", "HB")}
    for layer in sorted(census, key=int):
        a_top = np.array(census[layer]["a_top"], dtype=np.float64)
        n = np.array(census[layer]["n"], dtype=np.int64)
        for arm in pooled:
            prob, d = fc.displacement(arm, rec, {arm: fc.patterns(a_top, n, rec[arm]["p"])})
            pooled[arm][0].append(prob)
            pooled[arm][1].append(d)
    grid = np.linspace(0, 0.6, 400)
    ends = {}
    for arm, ls, lw in (("HU", (0, (4, 2)), 2), ("HF", "-", 2), ("HB", "-", 2.6)):
        s = fc.survival(np.concatenate(pooled[arm][0]), np.concatenate(pooled[arm][1]), grid)
        ax.plot(grid, s, color=COL[arm], lw=lw, ls=ls)
        ends[arm] = s
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.1)
    ax.set_xlim(0, 0.6)
    i = np.searchsorted(grid, 0.30)
    ax.text(0.31, ends["HU"][i] * 1.25, "no cap", color=INK, fontsize=9.5)
    j = np.searchsorted(grid, 0.45)
    ax.text(0.45, ends["HF"][j] * 0.3, "fixed mix", color=INK, fontsize=9.5, ha="center")
    ax.text(rec["HB"]["tau"] + 0.01, 1.5e-3, "CapDrop:\nnothing above $\\tau$", color=COL["HB"], fontsize=9.5,
            weight="bold")
    ax.set_xlabel("row displacement, $x$")
    ax.set_ylabel("share of rows with $d > x$")
    ax.set_title("2  Same mean dose (0.10), different tails", loc="left", color=INK)
    style(ax)


def panel_results(ax, a):
    t, s, e8 = a["confirmatory"]["tests"], a["secondary"], a["extensions"]["E8"]
    rows = [("fixed-mix deletion", t["H1-L"]["mean_diff"], t["H1-L"]["ci95_signflip"], "Better"),
            ("tuned Relaxed Attention", t["H2-L"]["mean_diff"], t["H2-L"]["ci95_signflip"], "Better"),
            ("uncapped deletion", s["S-HBvsHU-L"]["mean_diff"], s["S-HBvsHU-L"]["ci95_signflip"], "Better"),
            ("tuned DropKey (2x dose)", t["H3-L"]["mean_diff"], t["H3-L"]["ci95_signflip"], "Worse")]
    ax.axvspan(-0.010, 0.010, color="#e9e9e9", lw=0)
    ax.axvline(0, color=MUTED, lw=0.8)
    for i, (lab, m, ci, v) in enumerate(rows):
        y = len(rows) - 1 - i
        c = BETTER if v == "Better" else WORSE
        ax.plot(ci, [y, y], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(m, y, "o", ms=9, color=c, mec="white", mew=1.5)
        ax.text(m, y + 0.22, f"{m:+.3f}".replace("-", "−"), ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.text(-0.012, -0.85, "← CapDrop better", ha="right", va="center", fontsize=10, color=BETTER, weight="bold")
    ax.text(0.012, -0.85, "CapDrop worse →", ha="left", va="center", fontsize=10, color=WORSE, weight="bold")
    ax.set_ylim(-1.2, len(rows) - 0.4)
    ax.set_xlim(-0.06, 0.075)
    ax.set_yticks(range(len(rows) - 1, -1, -1))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-0.04, -0.02, 0, 0.02, 0.04, 0.06])
    ax.set_xlabel("test NLL, CapDrop minus other (nats/token)")
    ax.set_title("3  Result: language model, 10 seeds", loc="left", color=INK)
    style(ax)
    ax.tick_params(left=False)
    return e8


def main():
    pred = load("results", "predictions.json")
    rec = pred["lm"]["levels"]["0.10"]
    a = load("results", "analysis.json")
    fig = plt.figure(figsize=(13.3, 5.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.1], left=0.05, right=0.985, top=0.88, bottom=0.25, wspace=0.5)
    gs = outer[0].subgridspec(1, 2, wspace=0.32)
    panel_rule(fig.add_subplot(gs[0]), rec)
    panel_tails(fig.add_subplot(gs[1]), rec)
    panel_results(fig.add_subplot(outer[1]), a)
    fig.suptitle("Regularize attention by how far it actually moves, not by its drop rate", x=0.05, ha="left",
                 fontsize=14, weight="bold", color=INK)
    fig.text(0.05, 0.06,
             "PubMedQA fine-tuning (BiomedBERT, 450 questions per fold): no attention regularizer, CapDrop included, "
             "improved on the unregularized model.\nAll arms matched on realized mean displacement; sealed test sets "
             "scored once; Holm-corrected exact tests; every run reported.",
             fontsize=10.5, color=INK, va="bottom")
    out = os.path.join(ROOT, "paper", "figures")
    fig.savefig(os.path.join(out, "graphical_abstract.pdf"))
    fig.savefig(os.path.join(out, "graphical_abstract.png"), dpi=200)  # close any viewer holding the file first


if __name__ == "__main__":
    main()
