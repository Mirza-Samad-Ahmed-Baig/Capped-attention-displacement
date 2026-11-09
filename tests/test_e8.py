"""Tests for the extension-E8 arm (PROTOCOL B10): hard_budget_sg = Budgeted AttentionDrop with the capped
alpha computed under stop-gradient. Forward values must equal hard_budget exactly; gradients must equal an
independent closed-form reference (1 - alpha) a + alpha b with alpha treated as a constant."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
UNIFIED = os.path.dirname(HERE)
sys.path.insert(0, UNIFIED)
import env  # noqa: E402

env.block_arrow()
import torch  # noqa: E402

import attnreg  # noqa: E402

torch.set_num_threads(2)


def random_case(seed, dtype):
    g = torch.Generator().manual_seed(seed)
    b, h, s = 2, 3, int(torch.randint(2, 40, (1,), generator=g))
    scale = [1.0, 5.0, 30.0][seed % 3]
    scores = (torch.randn(b, h, s, s, generator=g) * scale).to(dtype)
    lengths = torch.randint(1, s + 1, (b,), generator=g)
    allowed = (torch.arange(s)[None, :] < lengths[:, None])[:, None, None, :]
    if seed % 5 == 0:
        allowed = allowed & torch.ones(s, s, dtype=torch.bool).tril()
    return g, scores, allowed


def reference(scores, allowed, p, k, tau, draw, detach):
    """Direct mixture (1 - alpha) a + alpha b over full rows, written independently of attnreg's
    logsumexp form; alpha = min(1, tau / m) with m the deleted attention mass."""
    allowed = torch.broadcast_to(allowed, scores.shape)
    masked = scores.masked_fill(~allowed, -torch.inf)
    kk = min(k, scores.shape[-1])
    _, idx = masked.topk(kk, dim=-1)
    eligible = allowed.gather(-1, idx)
    drop = eligible & (draw < p)
    empty = drop.sum(-1, keepdim=True) == allowed.sum(-1, keepdim=True)
    drop[..., :1] &= ~empty
    dropped = torch.zeros(scores.shape, dtype=scores.dtype).scatter(-1, idx, drop.to(scores.dtype)) > 0.5
    a = masked.softmax(-1)
    b = scores.masked_fill(~(allowed & ~dropped), -torch.inf).softmax(-1)
    m = (a * dropped).sum(-1, keepdim=True)
    mm = m.detach() if detach else m
    alpha = torch.where(mm > tau, tau / torch.where(mm > tau, mm, torch.ones_like(mm)), torch.ones_like(mm))
    return ((1 - alpha) * a + alpha * b).masked_fill(~allowed, 0)


def grads(fn, scores, weight):
    x = scores.clone().requires_grad_(True)
    out = fn(x)
    (out * weight).sum().backward()
    return out.detach(), x.grad


class TestE8StopGradient(unittest.TestCase):
    P, K = 0.5, 4

    def draws(self, g, scores):
        return torch.rand(scores.shape[:-1] + (min(self.K, scores.shape[-1]),), generator=g, dtype=scores.dtype)

    def test_forward_and_stats_equal_hard_budget(self):
        for seed in range(40):
            for dtype in (torch.float32, torch.float64):
                g, scores, allowed = random_case(seed, dtype)
                draw = self.draws(g, scores)
                for tau in (0.02, 0.1, 0.3):
                    hb = attnreg.hard_budget(scores, allowed, self.P, self.K, tau, draw=draw, return_stats=True)
                    sg = attnreg.hard_budget(scores, allowed, self.P, self.K, tau, draw=draw, return_stats=True,
                                             detach_alpha=True)
                    for u, v in zip(hb, sg):
                        self.assertTrue(torch.equal(u, v), (seed, dtype, tau))

    def test_reference_reproduces_hard_budget_gradient(self):
        """Validates the reference itself: without detach it must match the existing HB gradient."""
        for seed in range(30):
            g, scores, allowed = random_case(seed, torch.float64)
            draw = self.draws(g, scores)
            w = torch.randn(scores.shape, generator=g, dtype=torch.float64)
            for tau in (0.02, 0.1, 0.3):
                o1, g1 = grads(lambda x: attnreg.hard_budget(x, allowed, self.P, self.K, tau, draw=draw), scores, w)
                o2, g2 = grads(lambda x: reference(x, allowed, self.P, self.K, tau, draw, detach=False), scores, w)
                torch.testing.assert_close(o1, o2, rtol=0, atol=1e-12)
                torch.testing.assert_close(g1, g2, rtol=0, atol=1e-10)

    def test_gradient_matches_stop_gradient_reference(self):
        for seed in range(30):
            g, scores, allowed = random_case(seed, torch.float64)
            draw = self.draws(g, scores)
            w = torch.randn(scores.shape, generator=g, dtype=torch.float64)
            for tau in (0.02, 0.1, 0.3):
                o1, g1 = grads(lambda x: attnreg.hard_budget(x, allowed, self.P, self.K, tau, draw=draw,
                                                             detach_alpha=True), scores, w)
                o2, g2 = grads(lambda x: reference(x, allowed, self.P, self.K, tau, draw, detach=True), scores, w)
                torch.testing.assert_close(o1, o2, rtol=0, atol=1e-12)
                torch.testing.assert_close(g1, g2, rtol=0, atol=1e-10)

    def test_gradients_differ_only_where_the_cap_is_active(self):
        differed = False
        for seed in range(30):
            g, scores, allowed = random_case(seed, torch.float64)
            draw = self.draws(g, scores)
            w = torch.randn(scores.shape, generator=g, dtype=torch.float64)
            tau = 0.05
            _, g_hb = grads(lambda x: attnreg.hard_budget(x, allowed, self.P, self.K, tau, draw=draw), scores, w)
            _, g_sg = grads(lambda x: attnreg.hard_budget(x, allowed, self.P, self.K, tau, draw=draw,
                                                          detach_alpha=True), scores, w)
            _, m, _ = attnreg.hard_budget(scores, allowed, self.P, self.K, tau, draw=draw, return_stats=True)
            inactive = (m <= tau).unsqueeze(-1).expand_as(g_hb)  # alpha == 1 is constant on these rows
            torch.testing.assert_close(g_hb[inactive], g_sg[inactive], rtol=0, atol=1e-12)
            if (g_hb - g_sg).abs().max() > 1e-6:
                differed = True
        self.assertTrue(differed, "the alpha gradient path was never exercised")

    def test_tau_one_is_identical_to_hard_budget_in_value_and_gradient(self):
        """With tau = 1 the cap never binds, so alpha == 1 is constant. Values are bit-identical; gradients agree
        to fp64 rounding only (measured <= 5e-15: the two autograd graphs sum the same terms in another order)."""
        for seed in range(10):
            g, scores, allowed = random_case(seed, torch.float64)
            draw = self.draws(g, scores)
            w = torch.randn(scores.shape, generator=g, dtype=torch.float64)
            o1, g1 = grads(lambda x: attnreg.hard_budget(x, allowed, self.P, self.K, 1.0, draw=draw), scores, w)
            o2, g2 = grads(lambda x: attnreg.hard_budget(x, allowed, self.P, self.K, 1.0, draw=draw,
                                                         detach_alpha=True), scores, w)
            self.assertTrue(torch.equal(o1, o2))
            torch.testing.assert_close(g1, g2, rtol=0, atol=1e-12)

    def test_module_kind_shares_noise_and_stats_with_hard_budget(self):
        g, scores, allowed = random_case(7, torch.float32)
        outs, stats = [], []
        for kind in ("hard_budget", "hard_budget_sg"):
            reg = attnreg.AttnReg(kind, p=self.P, k=self.K, tau=0.1, record=True).train()
            reg.noise = attnreg.NoiseSource(2001)
            reg.noise.set_step(5, micro=1)
            reg.layer = 3
            outs.append(reg(scores, allowed))
            stats.append(reg.stats())
        self.assertTrue(torch.equal(outs[0], outs[1]))
        self.assertEqual(stats[0], stats[1])
        reg = attnreg.AttnReg("hard_budget_sg", p=self.P, k=self.K, tau=0.1).eval()
        self.assertTrue(torch.equal(reg(scores, allowed), attnreg.masked_softmax(scores, allowed)))


def synthetic_uns(shift, noise, seed=0, drop_seed=None):
    """Synthetic S2 records for HB@0.10 and HBsg@0.10 (seeds 1-10); HBsg = HB + shift + noise."""
    import numpy as np
    rng = np.random.default_rng(seed)
    s2 = {}
    for s in range(1, 11):
        base = 4.0 + rng.normal(0, 0.05)
        for tag, sh in (("HB@0.10", 0.0), ("HBsg@0.10", shift)):
            if tag == "HBsg@0.10" and s == drop_seed:
                continue
            e = rng.normal(0, noise)
            s2[f"S2_{tag}_s{s}"] = {"tag": tag, "seed": s, "final": {"nll_per_token": base + sh + e},
                                    "best": {"nll_per_token": base + sh + e}}
    return {"S2": s2}


class TestE8Analysis(unittest.TestCase):
    def setUp(self):
        import analyze_ext
        self.ae = analyze_ext

    def test_detects_better_and_worse(self):
        better = self.ae.e8(synthetic_uns(-0.05, 0.005))
        worse = self.ae.e8(synthetic_uns(+0.05, 0.005))
        self.assertEqual((better["a"], better["b"], better["units"]), ("HBsg@0.10", "HB@0.10", 10))
        self.assertEqual(better["verdict"], "Better")
        self.assertLess(better["mean_diff"], 0)
        self.assertEqual(worse["verdict"], "Worse")
        self.assertEqual(better["label"], "descriptive (unadjusted)")

    def test_equivalent_within_margin(self):
        self.assertEqual(self.ae.e8(synthetic_uns(0.0, 0.001))["verdict"], "Equivalent")

    def test_missing_seed_is_unavailable(self):
        self.assertFalse(self.ae.e8(synthetic_uns(0.0, 0.001, drop_seed=4))["available"])

    def test_e6_reported_not_run_for_budget(self):
        self.assertFalse(self.ae.E6_STATUS["run"])
        self.assertTrue(self.ae.E6_STATUS["reason"].startswith("not run (budget)"))


if __name__ == "__main__":
    unittest.main()
