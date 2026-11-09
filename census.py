"""Exact displacement census for attention perturbations on fixed scores.

For one attention row a (visible keys S, n = |S|) and a stochastic perturbed row ã, define
  d = TV(a, ã) = ½‖ã − a‖₁                       realized displacement
  D = E[d],  S2 = E[d²]                            mean and second moment
  B = ½‖E[ã] − a‖₁                                 train/eval mean shift
For hard (top-k) deletion also m (deleted mass), P(m > τ), κ = E[m²]/E[m], and the clipped
excess E[(m − τ)₊] = TV between budgeted and unbudgeted rows (identity TV(b, ã) = (m − τ)₊).

Exactness:
  dropout (post-softmax, 1/(1-p) rescale): closed form; E[ã] = a so B = 0, D = p.
  dropkey (independent deletion of every visible key, argmax fallback): closed-form D and S2;
      B is estimated by Monte Carlo (reported with its sample count).
  relaxed / relaxed_budget: deterministic.
  hard / hard_budget: exact enumeration of the 2^k deletion patterns (k <= 8), including the
      all-deleted fallback, for D, S2, B, P(m > τ) and the clipped excess.
Concentration: top-k mass A_k, max weight, Shannon entropy, TV(a, U).
"""
import torch

import attnreg


def concentration(scores, allowed, k=4):
    x, allowed, active, masked = attnreg._prepare(scores, allowed)
    a = masked.softmax(-1).masked_fill(~allowed, 0)
    n = allowed.sum(-1)
    top = a.topk(min(k, a.shape[-1]), dim=-1).values.sum(-1)
    ent = -(a * a.clamp_min(1e-300).log()).sum(-1)
    u = attnreg._uniform(allowed, a.dtype)
    tvu = 0.5 * (u - a).abs().sum(-1)
    return {"active": active.squeeze(-1), "n": n, "top_mass": top, "p_max": a.max(-1).values,
            "entropy": ent, "tv_uniform": tvu}


def hard_exact(scores, allowed, p, k, tau):
    """Exact per-row statistics of (budgeted) top-k deletion by pattern enumeration."""
    if k > 8:
        raise ValueError("k <= 8 required")
    x, allowed, active, masked = attnreg._prepare(scores.double(), allowed)
    a = masked.softmax(-1).masked_fill(~allowed, 0)
    kk = min(k, x.shape[-1])
    idx = masked.topk(kk, dim=-1).indices
    eligible = allowed.gather(-1, idx)
    n_elig = eligible.sum(-1)
    shape = eligible.shape
    stats = {name: torch.zeros(shape[:-1], dtype=torch.float64, device=x.device)
             for name in ("Em", "Em2", "Ed", "Ed2", "P_active", "excess")}
    mean_row = torch.zeros_like(a)
    for pattern in range(2 ** kk):
        bits = torch.tensor([(pattern >> i) & 1 for i in range(kk)], dtype=torch.bool, device=x.device)
        valid = ~(bits & ~eligible).any(-1)
        n_drop = (bits & eligible).sum(-1)
        prob = (p ** n_drop.double()) * ((1 - p) ** (n_elig - n_drop).double()) * valid.double()
        draw = torch.where(bits, torch.zeros(shape, dtype=torch.float32, device=x.device),
                           torch.ones(shape, dtype=torch.float32, device=x.device)).expand(shape)
        if p == 0:
            draw = torch.ones(shape, dtype=torch.float32, device=x.device)
        out, m, real = attnreg.hard_budget(x, allowed, p if p > 0 else 0.0, k, tau, draw=draw, return_stats=True)
        stats["Em"] += prob * m
        stats["Em2"] += prob * m * m
        stats["Ed"] += prob * real
        stats["Ed2"] += prob * real * real
        stats["P_active"] += prob * (m > tau + 1e-12).double()
        stats["excess"] += prob * (m - tau).clamp_min(0)
        mean_row += prob.unsqueeze(-1) * out
    stats["B"] = 0.5 * (mean_row - a).abs().sum(-1)
    stats["kappa"] = stats["Em2"] / stats["Em"].clamp_min(1e-300)
    stats["active"] = active.squeeze(-1)
    return stats


def dropout_exact(scores, allowed, p):
    x, allowed, active, masked = attnreg._prepare(scores.double(), allowed)
    a = masked.softmax(-1).masked_fill(~allowed, 0)
    # d = ½ Σ a_j c_j with c_j = 1 (dropped, prob p) or p/(1-p) (kept, prob 1-p)
    ec = 2 * p
    ec2 = p + (p * p / (1 - p)) if p < 1 else float("inf")
    sum_a2 = (a * a).sum(-1)
    ed = 0.5 * ec * a.sum(-1)
    ed2 = 0.25 * (sum_a2 * ec2 + (a.sum(-1) * ec) ** 2 - sum_a2 * ec * ec)
    return {"Ed": ed, "Ed2": ed2, "B": torch.zeros_like(ed), "active": active.squeeze(-1)}


def dropkey_exact(scores, allowed, p, mc_samples=0, generator=None):
    """Closed-form E[m], E[m²] for independent deletion of all visible keys with fallback;
    optional Monte Carlo estimate of the mean shift B."""
    x, allowed, active, masked = attnreg._prepare(scores.double(), allowed)
    a = masked.softmax(-1).masked_fill(~allowed, 0)
    n = allowed.sum(-1).double()
    amax = a.max(-1).values
    sum_a2 = (a * a).sum(-1)
    pn = p ** n
    em = p - pn * amax
    em2 = p * p + p * (1 - p) * sum_a2 - pn * (1 - (1 - amax) ** 2)
    act = active.squeeze(-1)
    em, em2 = torch.where(act, em, torch.zeros_like(em)), torch.where(act, em2, torch.zeros_like(em2))
    out = {"Ed": em, "Ed2": em2, "active": act}
    if mc_samples:
        acc = torch.zeros_like(a)
        for _ in range(mc_samples):
            u = torch.rand(a.shape, generator=generator, dtype=torch.float32, device=a.device)
            acc += attnreg.dropkey_candidate(x, allowed, p, draw=u)
        out["B_mc"] = 0.5 * (acc / mc_samples - a).abs().sum(-1)
        out["mc_samples"] = mc_samples
    return out


def relaxed_exact(scores, allowed, gamma=None, tau=None):
    c = concentration(scores.double(), allowed)
    r = c["tv_uniform"]
    d = gamma * r if tau is None else torch.minimum(r, torch.full_like(r, tau))
    return {"Ed": d, "Ed2": d * d, "B": d, "active": c["active"]}


def row_mean(values, active):
    w = active.double()
    return float((values.double() * w).sum() / w.sum().clamp_min(1))


# ---------------------------------------------------------------------------------------
# Sufficient-statistics census. For top-k deletion (budgeted or not) every displacement
# statistic, including the mean shift B, depends on a row only through its sorted top-k
# probabilities and its number of visible keys n; dropout/dropkey need n, max and sum a^2;
# relaxed needs TV(a, U). An observer therefore stores these per sampled row once, and any
# (p, tau, gamma) grid is then evaluated exactly offline.
# ---------------------------------------------------------------------------------------

def hard_from_top(a_top, n, p, k, tau, alpha_fixed=None):
    """Exact statistics of (budgeted) top-k deletion from sufficient statistics.

    a_top: [R, K>=k] row probabilities sorted descending (zeros beyond n); n: [R] visible keys.
    Returns dict of [R] tensors: Em, Em2, Ed, Ed2, P_active, excess, B.
    """
    a_top = a_top.double()
    R = a_top.shape[0]
    kk = min(k, a_top.shape[1])
    top = a_top[:, :kk]
    n = n.to(torch.long)
    eligible = torch.arange(kk)[None, :] < n[:, None]
    n_elig = eligible.sum(-1)
    A = (top * eligible).sum(-1)
    out = {key: torch.zeros(R, dtype=torch.float64) for key in ("Em", "Em2", "Ed", "Ed2", "P_active", "excess")}
    e_factor_top = torch.zeros(R, kk, dtype=torch.float64)
    e_factor_rest = torch.zeros(R, dtype=torch.float64)
    for pattern in range(2 ** kk):
        bits = torch.tensor([(pattern >> i) & 1 for i in range(kk)], dtype=torch.bool)
        valid = ~(bits & ~eligible).any(-1)
        drop = bits & eligible
        n_drop = drop.sum(-1)
        prob = (p ** n_drop.double()) * ((1 - p) ** (n_elig - n_drop).double()) * valid.double()
        fallback = (n_drop == n) & (n > 0)
        drop = drop.clone()
        drop[:, 0] &= ~fallback
        m = (top * drop).sum(-1)
        if alpha_fixed is None:
            alpha = torch.where(m > 0, (tau / m.clamp_min(1e-300)).clamp(max=1), torch.zeros_like(m))
        else:
            alpha = torch.where(m > 0, torch.full_like(m, float(alpha_fixed)), torch.zeros_like(m))
        kept_gain = (1 - alpha) + alpha / (1 - m).clamp_min(1e-300)
        real = alpha * m
        out["Em"] += prob * m
        out["Em2"] += prob * m * m
        out["Ed"] += prob * real
        out["Ed2"] += prob * real * real
        out["P_active"] += prob * (m > tau + 1e-12).double()
        out["excess"] += prob * (m - tau).clamp_min(0)
        factor = torch.where(drop, (1 - alpha)[:, None].expand(-1, kk), kept_gain[:, None].expand(-1, kk))
        e_factor_top += prob[:, None] * factor
        e_factor_rest += prob * kept_gain
    rest = (1 - A).clamp_min(0)
    out["B"] = 0.5 * ((top * eligible * (e_factor_top - 1).abs()).sum(-1) + rest * (e_factor_rest - 1).abs())
    out["kappa"] = out["Em2"] / out["Em"].clamp_min(1e-300)
    return out


def dropout_from_stats(sum_a2, p):
    ec, ec2 = 2 * p, p + p * p / (1 - p)
    sum_a2 = sum_a2.double()
    return {"Ed": torch.full_like(sum_a2, p), "Ed2": 0.25 * (sum_a2 * ec2 + ec * ec - sum_a2 * ec * ec),
            "B": torch.zeros_like(sum_a2)}


def dropkey_from_stats(n, a_max, sum_a2, p):
    n, a_max, sum_a2 = n.double(), a_max.double(), sum_a2.double()
    pn = p ** n
    return {"Ed": p - pn * a_max, "Ed2": p * p + p * (1 - p) * sum_a2 - pn * (1 - (1 - a_max) ** 2)}


def relaxed_from_stats(tv_uniform, gamma=None, tau=None):
    r = tv_uniform.double()
    d = gamma * r if tau is None else torch.minimum(r, torch.full_like(r, tau))
    return {"Ed": d, "Ed2": d * d, "B": d}


class RowSampler:
    """Observer that samples per-layer rows' sufficient statistics (CPU float64).

    Attach with `reg.observer = sampler` on every AttnReg (eval mode for dose calibration).
    Query rows at padding positions are skipped: for a key-padding mask [B,1,1,K], query j of
    example b is valid iff key j is visible; for other masks all rows with a visible key count.
    """

    def __init__(self, max_rows_per_layer=2048, rows_per_call=256, k_max=8, full_rows_per_layer=32, seed=0):
        self.max_rows, self.per_call, self.k_max, self.full_max = max_rows_per_layer, rows_per_call, k_max, full_rows_per_layer
        self.gen = torch.Generator().manual_seed(seed)
        self.data = {}

    def __call__(self, layer, scores, allowed):
        store = self.data.setdefault(layer, {"a_top": [], "n": [], "sum_a2": [], "tv_uniform": [], "entropy": [],
                                             "p_max": [], "full": [], "rows": 0})
        if store["rows"] >= self.max_rows:
            return
        x, full, active, masked = attnreg._prepare(scores, allowed)
        a = masked.softmax(-1).masked_fill(~full, 0)
        if allowed.shape[-2] == 1 and allowed.dim() == scores.dim():
            qvalid = torch.broadcast_to(allowed.squeeze(-2).unsqueeze(-1), scores.shape[:-1] + (1,)).squeeze(-1)
        else:
            qvalid = active.squeeze(-1)
        qvalid = qvalid & active.squeeze(-1)
        rows = a[qvalid]
        keep_mask = full[qvalid]
        if rows.shape[0] == 0:
            return
        take = min(self.per_call, self.max_rows - store["rows"], rows.shape[0])
        idx = torch.randperm(rows.shape[0], generator=self.gen)[:take].to(rows.device)
        rows, keep_mask = rows[idx].double(), keep_mask[idx]
        kk = min(self.k_max, rows.shape[-1])
        top = rows.topk(kk, dim=-1).values
        if kk < self.k_max:
            top = torch.nn.functional.pad(top, (0, self.k_max - kk))
        n = keep_mask.sum(-1)
        u = keep_mask.double() / n.clamp_min(1).double()[:, None]
        store["a_top"].append(top.cpu())
        store["n"].append(n.cpu())
        store["sum_a2"].append((rows * rows).sum(-1).cpu())
        store["tv_uniform"].append((0.5 * (u - rows).abs().sum(-1)).cpu())
        store["entropy"].append((-(rows * rows.clamp_min(1e-300).log()).sum(-1)).cpu())
        store["p_max"].append(rows.max(-1).values.cpu())
        if len(store["full"]) < self.full_max:
            for r, km in zip(rows[: self.full_max - len(store["full"])].cpu(), keep_mask.cpu()):
                store["full"].append(r[km])
        store["rows"] += take

    def tensors(self, layer):
        s = self.data[layer]
        return {key: torch.cat(s[key]) for key in ("a_top", "n", "sum_a2", "tv_uniform", "entropy", "p_max")}
