"""Figure 1 (outcome-independent): displacement laws on the census rows at l = 0.10.

Left: survival function P(d > x) of the realized per-row displacement for unbudgeted top-4 deletion (HU),
fixed-alpha deletion (HF) and CapDrop (HB), exact over all 16 deletion patterns of every census row.
Middle: mean realized displacement per layer (calibration pools layers). Right: P(m > tau) per layer for HB.
Reads results/s0/{encoder,lm}_census.json and results/predictions.json only (CPU). Writes
paper/figures/fig_census.pdf and results/census_layers.json.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
COL = {"HU": "#CC79A7", "HF": "#E69F00", "HB": "#0072B2"}
NAME = {"HU": "HU (unbudgeted)", "HF": "HF (fixed $\\alpha$)", "HB": "HB (CapDrop)"}


def patterns(a_top, n, p, k=4):
    """Per row and deletion pattern: probability and deleted mass m (argmax restored if all visible deleted)."""
    top = a_top[:, :k]
    elig = np.arange(k)[None, :] < n[:, None]
    n_elig = elig.sum(1)
    probs, masses = [], []
    for pattern in range(2 ** k):
        bits = np.array([(pattern >> i) & 1 for i in range(k)], dtype=bool)
        valid = ~(bits[None, :] & ~elig).any(1)
        drop = bits[None, :] & elig
        n_drop = drop.sum(1)
        prob = p ** n_drop * (1 - p) ** (n_elig - n_drop) * valid
        fallback = (n_drop == n) & (n > 0)
        drop = drop.copy()
        drop[:, 0] &= ~fallback
        probs.append(prob)
        masses.append((top * drop).sum(1))
    return np.stack(probs, 1), np.stack(masses, 1)  # [R, 16]


def displacement(arm, rec, probs_masses):
    prob, m = probs_masses[arm]
    if arm == "HB":
        d = np.minimum(m, rec["HB"]["tau"])
    elif arm == "HF":
        d = rec["HF"]["gamma"] * m
    else:
        d = m
    return prob, d


def survival(prob, d, grid):
    w, x = prob.ravel(), d.ravel()
    keep = w > 0
    w, x = w[keep] / w[keep].sum(), x[keep]
    order = np.argsort(x)
    x, w = x[order], w[order]
    tail = 1.0 - np.cumsum(w)
    idx = np.searchsorted(x, grid, side="right") - 1
    return np.where(idx >= 0, tail[np.clip(idx, 0, None)], 1.0)


def main():
    pred = json.load(open(os.path.join(ROOT, "results", "predictions.json"), encoding="utf-8"))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), constrained_layout=True)
    summary = {}
    for row, (regime, label) in enumerate((("encoder", "Encoder"), ("lm", "LM"))):
        census = json.load(open(os.path.join(ROOT, "results", "s0", f"{regime}_census.json"), encoding="utf-8"))
        rec = pred[regime]["levels"]["0.10"]
        layers = sorted(census, key=int)
        per_layer = {a: [] for a in ("HU", "HF", "HB")}
        active = []
        pooled = {a: ([], []) for a in per_layer}
        for layer in layers:
            a_top = np.array(census[layer]["a_top"], dtype=np.float64)
            n = np.array(census[layer]["n"], dtype=np.int64)
            pm = {"HU": patterns(a_top, n, rec["HU"]["p"]), "HF": patterns(a_top, n, rec["HF"]["p"]),
                  "HB": patterns(a_top, n, rec["HB"]["p"])}
            for arm in per_layer:
                prob, d = displacement(arm, rec, pm)
                per_layer[arm].append(float((prob * d).sum(1).mean()))
                pooled[arm][0].append(prob)
                pooled[arm][1].append(d)
            prob, m = pm["HB"]
            active.append(float((prob * (m > rec["HB"]["tau"] + 1e-12)).sum(1).mean()))
        grid = np.linspace(0, 0.8, 400)
        ax = axes[row, 0]
        for arm in ("HU", "HF", "HB"):
            prob, d = np.concatenate(pooled[arm][0]), np.concatenate(pooled[arm][1])
            ax.plot(grid, survival(prob, d, grid), color=COL[arm], lw=1.8, label=NAME[arm])
        ax.axvline(rec["HB"]["tau"], color=COL["HB"], lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.05)
        ax.set_xlabel("realized displacement $x$")
        ax.set_ylabel("$\\Pr(d > x)$")
        ax.set_title(f"{label}: tail at $\\ell=0.10$", fontsize=9.5, loc="left")
        if row == 0:
            ax.legend(frameon=False, fontsize=8)
        ax = axes[row, 1]
        xs = np.arange(len(layers))
        for arm, style in (("HF", "-"), ("HU", (0, (3, 2))), ("HB", "-")):
            ax.plot(xs, per_layer[arm], marker="o" if arm != "HU" else None, ms=3.5, lw=1.4, ls=style, color=COL[arm],
                    label=NAME[arm], zorder=3 if arm == "HU" else 2)
        ax.axhline(0.10, color="0.4", lw=0.8, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(l) + 1) for l in layers], fontsize=7.5)
        ax.set_xlabel("layer")
        ax.set_ylabel("mean displacement $\\mathbb{E}d$")
        ax.set_title("per-layer mean displacement (HU = HF)", fontsize=9.5, loc="left")
        ax = axes[row, 2]
        ax.bar(xs, active, color=COL["HB"], width=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(l) + 1) for l in layers], fontsize=7.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("layer")
        ax.set_ylabel("$\\Pr(m > \\tau)$")
        ax.set_title("CapDrop: rows where the cap binds", fontsize=9.5, loc="left")
        for axx in axes[row]:
            axx.spines[["top", "right"]].set_visible(False)
            axx.tick_params(labelsize=7.5)
        summary[regime] = {"layers": [int(l) + 1 for l in layers], "mean_displacement": per_layer, "P_active_HB": active}
    os.makedirs(os.path.join(ROOT, "paper", "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "paper", "figures", "fig_census.pdf"))
    fig.savefig(os.path.join(ROOT, "paper", "figures", "fig_census.png"), dpi=110)
    with open(os.path.join(ROOT, "results", "census_layers.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    for regime, s in summary.items():
        print(regime, {a: [round(v, 3) for v in vals] for a, vals in s["mean_displacement"].items()},
              "P_active", [round(v, 2) for v in s["P_active_HB"]])


if __name__ == "__main__":
    main()
