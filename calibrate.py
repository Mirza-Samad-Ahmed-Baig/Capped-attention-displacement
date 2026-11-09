"""Census -> calibrated arm parameters and registered predictions (PROTOCOL A §3.0, B1-B2).

Input: a census JSON {layer: {a_top, n, sum_a2, tv_uniform, entropy, p_max}} (rows pooled over
layers with equal weight, as sampled). For each level l in LEVELS, every family is set to a
realized mean displacement E[d] = l using the exact expectations of census.py:
  AD  p = l (exact: E[d] = p)
  DK  p_max with the linear depth schedule, E[m] = p - p^n a_max per layer
  RA  gamma = l / mean TV(a, U)
  RB  tau with mean min(TV(a, U), tau) = l
  HU  p_lo with mean E[m](p_lo) = l (k = 4, unbudgeted)
  HB  p_hi = 0.5, raised to the smallest p in {0.6, ..., 1.0} with mean E[m](p) >= 1.5 l;
      tau with mean E[min(m, tau)] = l
  HF  same p_hi, gamma = l / mean E[m](p_hi)
Predictions (B2): delta = (S2(HF) - S2(HB)) / l^2 at l = 0.10; registered H1 expectation.
"""
import json
import sys

import torch

import census

LEVELS = (0.05, 0.10, 0.20)
K = 4


def load(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    layers = sorted(raw, key=int)
    t = {key: torch.cat([torch.tensor(raw[l][key], dtype=torch.float64) for l in layers])
         for key in ("a_top", "n", "sum_a2", "tv_uniform", "p_max")}
    t["layer"] = torch.cat([torch.full((len(raw[l]["n"]),), float(i), dtype=torch.float64) for i, l in enumerate(layers)])
    t["num_layers"] = len(layers)
    return t


def bisect(f, target, lo=0.0, hi=1.0, iters=60):
    """Solve f(x) = target for nondecreasing f on [lo, hi]; returns (x, reachable)."""
    if f(hi) < target - 1e-12:
        return hi, False
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi, True


def hard(t, p, tau=1.0, alpha_fixed=None):
    return census.hard_from_top(t["a_top"], t["n"], p, K, tau, alpha_fixed=alpha_fixed)


def mean(x):
    return float(x.mean())


def calibrate(t, levels=LEVELS):
    kappa_stats = hard(t, 0.1)
    out = {"rows": int(t["n"].numel()), "layers": t["num_layers"],
           "concentration": {"mean_top4": mean(t["a_top"][:, :4].sum(-1)), "mean_top8": mean(t["a_top"].sum(-1)),
                             "median_top4": float(t["a_top"][:, :4].sum(-1).median()),
                             "mean_p_max": mean(t["p_max"]), "mean_tv_uniform": mean(t["tv_uniform"])},
           "kappa_p0.1_k4": mean(kappa_stats["Em2"]) / max(1e-300, mean(kappa_stats["Em"])),
           "levels": {}}
    L = t["num_layers"]
    for level in levels:
        rec = {"AD": {"p": level, "Ed": level,
                      "Ed2": mean(census.dropout_from_stats(t["sum_a2"], level)["Ed2"]), "B": 0.0}}

        def dk(pmax):
            p = pmax * (1 - t["layer"] / L)
            pn = p ** t["n"]
            return mean(p - pn * t["p_max"])
        pmax, ok = bisect(dk, level)
        p_l = pmax * (1 - t["layer"] / L)
        rec["DK"] = {"p_max": pmax, "reachable": ok, "Ed": dk(pmax), "schedule": "dropkey_linear",
                     "Ed2": mean(p_l * p_l + p_l * (1 - p_l) * t["sum_a2"] - p_l ** t["n"] * (1 - (1 - t["p_max"]) ** 2))}
        tvu = t["tv_uniform"]
        gamma = min(1.0, level / mean(tvu))
        rec["RA"] = {"gamma": gamma, "reachable": gamma < 1.0, "Ed": mean(gamma * tvu), "Ed2": mean((gamma * tvu) ** 2),
                     "B": mean(gamma * tvu)}
        tau_rb, ok = bisect(lambda x: mean(torch.minimum(tvu, torch.full_like(tvu, x))), level)
        clip = torch.minimum(tvu, torch.full_like(tvu, tau_rb))
        rec["RB"] = {"tau": tau_rb, "reachable": ok, "Ed": mean(clip), "Ed2": mean(clip ** 2), "B": mean(clip)}
        p_lo, ok = bisect(lambda p: mean(hard(t, p)["Em"]), level)
        hu = hard(t, p_lo)
        rec["HU"] = {"p": p_lo, "reachable": ok, "Ed": mean(hu["Ed"]), "Ed2": mean(hu["Ed2"]), "B": mean(hu["B"])}
        p_hi = 0.5
        for cand in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            p_hi = cand
            if mean(hard(t, cand)["Em"]) >= 1.5 * level:
                break
        em_hi = mean(hard(t, p_hi)["Em"])
        feasible = em_hi >= 1.5 * level
        tau, ok = bisect(lambda x: mean(hard(t, p_hi, x)["Ed"]), level)
        hb = hard(t, p_hi, tau)
        rec["HB"] = {"p": p_hi, "tau": tau, "reachable": ok and feasible, "Ed": mean(hb["Ed"]), "Ed2": mean(hb["Ed2"]),
                     "B": mean(hb["B"]), "P_active": mean(hb["P_active"]), "excess": mean(hb["excess"]),
                     "Em_at_p": em_hi}
        g = min(1.0, level / em_hi)
        hf = hard(t, p_hi, 1.0, alpha_fixed=g)
        rec["HF"] = {"p": p_hi, "gamma": g, "reachable": g < 1.0, "Ed": mean(hf["Ed"]), "Ed2": mean(hf["Ed2"]),
                     "B": mean(hf["B"])}
        rec["delta"] = (rec["HF"]["Ed2"] - rec["HB"]["Ed2"]) / level ** 2
        out["levels"][f"{level:.2f}"] = rec
    return out


def dropkey_budget_expected(t, p, tau):
    """E[min(m, tau)] for independent deletion of every visible key (budgeted DropKey, E3).
    Exact over each row's top-8 keys (256 deletion patterns); the remaining keys' deleted mass is
    approximated as normal with mean p*(1-A8) and variance p(1-p)*(sum a^2 - sum top8^2) (many small
    keys; central-limit approximation). The all-deleted fallback is ignored (probability p^n)."""
    from math import erf, exp, pi, sqrt
    top = t["a_top"]
    kk = top.shape[1]
    rest_mean = p * (1 - top.sum(-1)).clamp_min(0)
    rest_var = (p * (1 - p) * (t["sum_a2"] - (top * top).sum(-1)).clamp_min(0))
    total = torch.zeros(top.shape[0], dtype=torch.float64)
    for pattern in range(2 ** kk):
        bits = torch.tensor([(pattern >> i) & 1 for i in range(kk)], dtype=torch.float64)
        k1 = int(bits.sum())
        prob = (p ** k1) * ((1 - p) ** (kk - k1))
        mu = (top * bits).sum(-1) + rest_mean
        sd = rest_var.sqrt().clamp_min(1e-12)
        d = (mu - tau) / sd
        Phi = 0.5 * (1 + torch.erf(d / 2 ** 0.5))
        phi = torch.exp(-0.5 * d * d) / (2 * torch.pi) ** 0.5
        excess = (mu - tau) * Phi + sd * phi
        total += prob * (mu - excess)
    return float(total.mean())


def dropkey_budget_tau(t, level, p=0.5):
    tau, ok = bisect(lambda x: dropkey_budget_expected(t, p, x), level)
    return {"p": p, "tau": tau, "reachable": ok, "Ed": dropkey_budget_expected(t, p, tau),
            "method": "exact top-8 + normal tail approximation"}


def predictions(encoder_census, lm_census):
    enc, lm = calibrate(load(encoder_census)), calibrate(load(lm_census))
    d_e, d_l = enc["levels"]["0.10"]["delta"], lm["levels"]["0.10"]["delta"]
    return {"encoder": enc, "lm": lm,
            "registered": {
                "c": 0.5,
                "H1-E": "HB better" if d_e >= 0.5 else "equivalent",
                "H1-L": "HB not better (equivalent or worse)" if d_l >= 0.5 else "equivalent",
                "delta_encoder": d_e, "delta_lm": d_l,
                "kappa_premise_kE_ge_2kL": enc["kappa_p0.1_k4"] >= 2 * lm["kappa_p0.1_k4"],
                "note": "tests are two-sided; these are registered expectations (PROTOCOL B2)"}}


if __name__ == "__main__":
    print(json.dumps(calibrate(load(sys.argv[1])), indent=1))
