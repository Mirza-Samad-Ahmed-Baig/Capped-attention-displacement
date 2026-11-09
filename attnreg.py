"""Attention-probability regularizers for the unified study (version 2).

This file is new; the archived reference implementation in strengthened/attentiondrop.py
is kept byte-identical because 80 historical runs hash it. The reference is imported
only by the tests, as an oracle.

All regularizers take pre-softmax scores (already divided by sqrt(d)), a boolean
`allowed` mask broadcastable to the scores (True = visible key), and return row
distributions in float32 (float64 inputs stay float64). Callers cast the result to the
value dtype before the value product. Fully masked rows return zeros. In evaluation
mode every regularizer returns the plain masked softmax and consumes no randomness.

Kinds
-----
none            masked softmax
dropout         standard post-softmax element dropout with 1/(1-p) rescaling (rows do
                not sum to one), as in BERT/HF `attention_probs_dropout_prob`
dropkey         pre-softmax Bernoulli(p) key masking with renormalization (DropKey-style;
                layer schedule supplied by the caller through `p`), argmax restored if a
                row would be emptied
relaxed         Relaxed Attention: (1-gamma) a + gamma U_allowed, fixed gamma
hard            unbudgeted top-k hard deletion (tau = 1): the realized candidate b
hard_budget     Budgeted AttentionDrop, hard candidate: (1-alpha) a + alpha b with
                alpha = min(1, tau / TV(a, b)); closed form, exact in distribution
hard_budget_sg  extension E8 (PROTOCOL B10): hard_budget with alpha = min(1, tau/TV) computed from the
                deleted mass under stop-gradient, so forward values equal hard_budget exactly and only
                the gradient path through alpha is removed
dropkey_budget  DropKey candidate under the same TV budget (targeting ablation)
relaxed_budget  row-adaptive Relaxed Attention: b = U_allowed, alpha = min(1, tau/TV(a,U))
                (deterministic; the budgeted generalization of Relaxed Attention)

Budget-activity statistics (candidate displacement r, realized displacement, and the
fraction of rows where r > tau) are accumulated on-device without host synchronization
when `record` is enabled, and read with `stats()`.
"""
import math

import torch
from torch import nn

KINDS = ("none", "dropout", "dropkey", "relaxed", "hard", "hard_budget", "hard_mix",
         "dropkey_budget", "relaxed_budget", "hard_budget_sg")


def _prepare(scores, allowed):
    if allowed.dtype != torch.bool:
        raise TypeError("allowed must be boolean (True means visible)")
    allowed = torch.broadcast_to(allowed, scores.shape)
    x = scores if scores.dtype == torch.float64 else scores.float()
    active = allowed.any(-1, keepdim=True)
    masked = torch.where(active, x.masked_fill(~allowed, -torch.inf), torch.zeros_like(x))
    return x, allowed, active, masked


class NoiseSource:
    """Counter-based perturbation noise: uniforms are a pure function of
    (run seed, optimizer step, layer, pass, call), independent of the global RNG.

    Consequences: (1) gradient-checkpoint recomputation sees identical masks; (2) arms that
    share a seed and data order receive common random numbers (e.g. budgeted and
    unbudgeted deletion at the same step/layer compare the same uniforms against p);
    (3) R-Drop's two views use pass 0 and pass 1.
    """

    def __init__(self, seed):
        self.seed = int(seed)
        self.step = 0
        self.pass_index = 0
        self.micro = 0

    def set_step(self, step, pass_index=0, micro=0):
        """Set the noise key for the next forward pass. The microbatch index keeps microbatches
        of one optimizer step independent. Do not change the key between a forward and its
        backward when gradient checkpointing is on: recomputation reads the current key."""
        self.step, self.pass_index, self.micro = int(step), int(pass_index), int(micro)

    @staticmethod
    def _mix(*values):
        x = 0x9E3779B97F4A7C15
        for v in values:
            x = (x ^ (int(v) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
            x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            x ^= x >> 31
        return x & 0x7FFFFFFFFFFFFFFF

    def uniform(self, layer, shape, device, stream=0):
        g = torch.Generator(device=device)
        g.manual_seed(self._mix(self.seed, self.step, layer, self.pass_index, self.micro, stream))
        return torch.rand(shape, generator=g, device=device)


def masked_softmax(scores, allowed):
    _, allowed, _, masked = _prepare(scores, allowed)
    return masked.softmax(-1).masked_fill(~allowed, 0)


def _uniform(allowed, dtype):
    counts = allowed.sum(-1, keepdim=True).clamp_min(1).to(dtype)
    return allowed.to(dtype) / counts


def capped_alpha(r, tau):
    """alpha = min(1, tau / r), written so the backward pass is finite everywhere.

    The division is evaluated only where r > tau (then r >= tau > 0 is not tiny); elsewhere alpha
    is the constant 1. Values and gradients equal min(1, tau/r) for r > 0; at r = 0 the mixture
    is the unperturbed row for any alpha. (A clamp of tau/r would give 0 * inf = NaN in backward
    for tiny r.)
    """
    over = r > tau
    denom = torch.where(over, r, torch.ones_like(r))
    return torch.where(over, tau / denom, torch.ones_like(r))


def hard_budget(scores, allowed, p, k, tau, draw=None, return_stats=False, alpha_fixed=None, detach_alpha=False):
    """Closed-form budgeted hard deletion.

    For deletion set D with mass m = sum_{j in D} a_j, the candidate b is the softmax over
    kept keys and TV(a, b) = m. Hence, with alpha = min(1, tau/m),
      kept j:    a~_j = exp(x_j - lse_kept) [(1 - alpha) exp(lse_kept - lse_all) + alpha]
      deleted j: a~_j = (1 - alpha) a_j.
    Only the top-k visible keys are eligible, so draws are needed for k entries per row.
    `draw` (shape [..., Q, min(k, K)], uniform [0,1)) supplies explicit noise (tests, common random numbers).
    `alpha_fixed` (HF arm) replaces the capped rule by a constant alpha on perturbed rows.
    `detach_alpha` (E8 arm) computes the capped alpha from the deleted mass under stop-gradient: values are
    unchanged and no gradient flows through alpha.
    """
    x, allowed, active, masked = _prepare(scores, allowed)
    kk = min(k, x.shape[-1])
    top_vals, idx = masked.topk(kk, dim=-1)                    # idx[..., 0] is the argmax anchor
    with torch.no_grad():
        eligible = allowed.gather(-1, idx)
        u = torch.rand(eligible.shape, device=x.device) if draw is None else draw
        drop = eligible & (u < p)
        empty = (drop.sum(-1, keepdim=True) == allowed.sum(-1, keepdim=True)) & active
        drop[..., :1] &= ~empty                                # restore anchor if row would empty
    lse_all = masked.logsumexp(-1, keepdim=True)
    kept_logits = masked.scatter(-1, idx, top_vals.masked_fill(drop, -torch.inf))
    lse_kept = kept_logits.logsumexp(-1, keepdim=True)
    a_top = (top_vals - lse_all).exp().masked_fill(~eligible, 0)
    m = (a_top * drop).sum(-1, keepdim=True)                   # = TV(a, b)
    if alpha_fixed is None:
        alpha = capped_alpha(m.detach() if detach_alpha else m, tau)
    else:
        alpha = torch.where(m > 0, torch.full_like(m, float(alpha_fixed)), torch.zeros_like(m))
    gain = (1 - alpha) * (lse_kept - lse_all).exp() + alpha
    out = (kept_logits - lse_kept).exp() * gain
    out = out.scatter_add(-1, idx, (1 - alpha) * a_top * drop)
    out = torch.where(active, out, torch.zeros_like(out)).masked_fill(~allowed, 0)
    if return_stats:
        return out, m.squeeze(-1), (alpha * m).squeeze(-1)
    return out


def dropkey_candidate(scores, allowed, p, draw=None):
    """DropKey-style candidate: independent Bernoulli(p) deletion of every visible key."""
    x, allowed, active, masked = _prepare(scores, allowed)
    with torch.no_grad():
        u = torch.rand(x.shape, device=x.device) if draw is None else draw
        keep = allowed & ~(u < p)
        empty = ~keep.any(-1, keepdim=True) & active
        anchor = masked.argmax(-1, keepdim=True)
        keep = keep | (torch.zeros_like(allowed).scatter(-1, anchor, True) & empty)
    kept = torch.where(active, x.masked_fill(~keep, -torch.inf), torch.zeros_like(x))
    return kept.softmax(-1).masked_fill(~keep, 0)


def budget_mix(a, b, tau):
    r = 0.5 * (b - a).abs().sum(-1, keepdim=True)
    alpha = capped_alpha(r, tau)
    return (1 - alpha) * a + alpha * b, r.squeeze(-1), (alpha * r).squeeze(-1)


class AttnReg(nn.Module):
    """One attention-probability regularizer; holds no parameters."""

    def __init__(self, kind="none", p=0.1, k=4, tau=0.1, gamma=0.1, record=False):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"Unknown kind {kind!r}")
        if not (0 <= p <= 1 and 0 <= tau <= 1 and 0 <= gamma <= 1):
            raise ValueError("p, tau and gamma must lie in [0, 1]")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if kind == "dropout" and p >= 1:
            raise ValueError("dropout requires p < 1")
        self.kind, self.p, self.k, self.tau, self.gamma = kind, p, k, tau, gamma
        self.record = record
        self.layer = 0
        self.noise = None  # NoiseSource for common random numbers; None uses the global RNG
        self.observer = None  # optional callable(layer, scores, allowed) for census hooks
        self.reset_stats()

    def _draw(self, shape, device):
        if self.noise is None:
            return torch.rand(shape, device=device)
        return self.noise.uniform(self.layer, shape, device)

    def extra_repr(self):
        return f"kind={self.kind}, p={self.p}, k={self.k}, tau={self.tau}, gamma={self.gamma}"

    def reset_stats(self):
        self._n = self._cand = self._real = self._active = self._real2 = None

    def _accumulate(self, allowed, cand, real):
        rows = torch.broadcast_to(allowed, cand.shape + allowed.shape[-1:]).any(-1)
        w = rows.to(cand.dtype)
        vals = torch.stack([w.sum(), (cand * w).sum(), (real * w).sum(),
                            ((cand > self.tau + 1e-12).to(cand.dtype) * w).sum(),
                            (real * real * w).sum()]).detach().double()
        if self._n is None:
            self._n, self._cand, self._real, self._active, self._real2 = vals.unbind()
        else:
            self._n, self._cand, self._real, self._active, self._real2 = (
                self._n + vals[0], self._cand + vals[1], self._real + vals[2], self._active + vals[3],
                self._real2 + vals[4])

    def stats(self):
        """Row-averaged candidate/realized TV and budget-activation rate since last reset."""
        if self._n is None or float(self._n) == 0:
            return {"rows": 0}
        n = float(self._n)
        return {"rows": n, "candidate_tv": float(self._cand) / n,
                "realized_tv": float(self._real) / n, "active_rate": float(self._active) / n,
                "realized_tv_sq": float(self._real2) / n}

    def forward(self, scores, allowed):
        if self.observer is not None:
            with torch.no_grad():
                self.observer(self.layer, scores.detach(), allowed)
        if not self.training or self.kind == "none":
            return masked_softmax(scores, allowed)
        kind = self.kind
        if kind == "dropout":
            a = masked_softmax(scores, allowed)
            if self.p == 0:
                return a
            with torch.no_grad():
                keep = (self._draw(a.shape, a.device) >= self.p).to(a.dtype)
            out = a * keep / (1 - self.p)
            if self.record:
                self._accumulate(allowed, 0.5 * (out - a).abs().sum(-1), 0.5 * (out - a).abs().sum(-1))
            return out
        if kind in ("hard", "hard_budget", "hard_mix", "hard_budget_sg"):
            tau = 1.0 if kind == "hard" else self.tau
            shape = torch.broadcast_shapes(scores.shape, allowed.shape)[:-1] + (min(self.k, scores.shape[-1]),)
            draw = self._draw(shape, scores.device)
            out, cand, real = hard_budget(scores, allowed, self.p, self.k, tau, draw=draw, return_stats=True,
                                          alpha_fixed=self.gamma if kind == "hard_mix" else None,
                                          detach_alpha=kind == "hard_budget_sg")
            if self.record:
                self._accumulate(allowed, cand, real)
            return out
        a = masked_softmax(scores, allowed)
        if kind in ("dropkey", "dropkey_budget"):
            b = dropkey_candidate(scores, allowed, self.p,
                                  draw=self._draw(torch.broadcast_shapes(scores.shape, allowed.shape), scores.device))
            if kind == "dropkey":
                if self.record:
                    r = 0.5 * (b - a).abs().sum(-1)
                    self._accumulate(allowed, r, r)
                return b
            out, cand, real = budget_mix(a, b, self.tau)
        else:  # relaxed, relaxed_budget
            u = _uniform(torch.broadcast_to(allowed, a.shape), a.dtype)
            if kind == "relaxed":
                out = (1 - self.gamma) * a + self.gamma * u
                cand = 0.5 * (u - a).abs().sum(-1)
                real = self.gamma * cand
            else:
                out, cand, real = budget_mix(a, u, self.tau)
            out = out.masked_fill(~torch.broadcast_to(allowed, a.shape), 0)
        if self.record:
            self._accumulate(allowed, cand, real)
        return out


def depth_schedule(p_max, layer, num_layers):
    """DropKey's linearly decreasing drop ratio: p_max at the first layer, 0 after the last."""
    if num_layers <= 1:
        return p_max
    return p_max * (1 - layer / num_layers)


def expected_hard_stats(scores, allowed, p, k, tau):
    """Exact per-row E[m], E[min(m, tau)] and P(m > tau) for top-k Bernoulli deletion.

    Enumerates all 2^k deletion patterns (including the all-deleted fallback). Intended for
    post-training diagnostics on fixed representations; k must be small (<= 8).
    """
    if k > 8:
        raise ValueError("Exact enumeration is limited to k <= 8")
    x, allowed, active, masked = _prepare(scores, allowed)
    kk = min(k, x.shape[-1])
    top_vals, idx = masked.topk(kk, dim=-1)
    eligible = allowed.gather(-1, idx)
    a_top = (top_vals - masked.logsumexp(-1, keepdim=True)).exp().masked_fill(~eligible, 0)
    n_allowed = allowed.sum(-1, keepdim=True)
    em = torch.zeros(a_top.shape[:-1], dtype=a_top.dtype, device=a_top.device)
    eclip = torch.zeros_like(em)
    pact = torch.zeros_like(em)
    for pattern in range(2 ** kk):
        bits = torch.tensor([(pattern >> i) & 1 for i in range(kk)], dtype=torch.bool, device=x.device)
        drop = eligible & bits
        n_drop = drop.sum(-1)
        n_elig = eligible.sum(-1)
        # probability: eligible positions follow the pattern; ineligible pattern bits must be 0
        valid = ~(bits & ~eligible).any(-1)
        prob = (p ** n_drop.to(a_top.dtype)) * ((1 - p) ** (n_elig - n_drop).to(a_top.dtype))
        prob = prob * valid.to(a_top.dtype)
        fallback = (n_drop.unsqueeze(-1) == n_allowed).squeeze(-1) & active.squeeze(-1)
        drop = drop.clone()
        drop[..., 0] &= ~fallback
        m = (a_top * drop).sum(-1)
        em = em + prob * m
        eclip = eclip + prob * torch.minimum(m, torch.full_like(m, tau))
        pact = pact + prob * (m > tau + 1e-12).to(m.dtype)
    return em, eclip, pact
