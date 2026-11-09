import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import env  # noqa: E402

env.block_arrow()
import torch  # noqa: E402

import attnreg  # noqa: E402
import census  # noqa: E402

torch.set_num_threads(2)


def case(seed, rows=3, keys=7, scale=2.0, pad=2):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(1, 1, rows, keys, generator=g, dtype=torch.float64) * scale
    allowed = torch.ones(1, 1, 1, keys, dtype=torch.bool)
    if pad:
        allowed[..., -pad:] = False
    return scores, allowed


def mc(scores, allowed, reg, n):
    big = scores.expand(1, n, *scores.shape[2:])
    a = attnreg.masked_softmax(scores, allowed)
    out = reg(big, allowed)
    d = 0.5 * (out - a).abs().sum(-1)
    return d.mean(1), (d * d).mean(1), 0.5 * (out.mean(1) - a[:, 0]).abs().sum(-1)


class CensusTest(unittest.TestCase):
    N = 200000

    def test_hard_exact_matches_monte_carlo(self):
        torch.manual_seed(0)
        for tau in (0.05, 0.2, 1.0):
            scores, allowed = case(1)
            ex = census.hard_exact(scores, allowed, 0.3, 4, tau)
            reg = attnreg.AttnReg("hard_budget", p=0.3, k=4, tau=tau).train()
            d, d2, b = mc(scores, allowed, reg, self.N)
            self.assertLess((d - ex["Ed"][:, 0]).abs().max().item(), 3e-3, tau)
            self.assertLess((d2 - ex["Ed2"][:, 0]).abs().max().item(), 3e-3, tau)
            self.assertLess((b - ex["B"][:, 0]).abs().max().item(), 3e-3, tau)

    def test_hard_exact_agrees_with_attnreg_enumeration_and_identities(self):
        scores, allowed = case(2, rows=5, keys=9, pad=5)  # includes n <= k rows (fallback)
        ex = census.hard_exact(scores, allowed, 0.4, 4, 0.1)
        em, eclip, pact = attnreg.expected_hard_stats(scores, allowed, 0.4, 4, 0.1)
        self.assertLess((ex["Em"] - em).abs().max().item(), 1e-12)
        self.assertLess((ex["Ed"] - eclip).abs().max().item(), 1e-12)
        self.assertLess((ex["P_active"] - pact).abs().max().item(), 1e-12)
        # E[min(m,tau)] + E[(m-tau)+] = E[m]
        self.assertLess((ex["Ed"] + ex["excess"] - ex["Em"]).abs().max().item(), 1e-12)
        # tail bound E[min(m,tau)^2] <= tau E[min(m,tau)]
        self.assertTrue((ex["Ed2"] <= 0.1 * ex["Ed"] + 1e-15).all())

    def test_marginal_dose_identity(self):
        scores, allowed = case(3)
        h = 1e-6
        for tau in (0.03, 0.12, 0.3):
            lo = census.hard_exact(scores, allowed, 0.5, 4, tau - h)["Ed"]
            hi = census.hard_exact(scores, allowed, 0.5, 4, tau + h)["Ed"]
            pa = census.hard_exact(scores, allowed, 0.5, 4, tau)["P_active"]
            self.assertLess(((hi - lo) / (2 * h) - pa).abs().max().item(), 1e-5)

    def test_dropout_closed_form(self):
        torch.manual_seed(1)
        scores, allowed = case(4)
        ex = census.dropout_exact(scores, allowed, 0.1)
        self.assertLess((ex["Ed"] - 0.1).abs().max().item(), 1e-12)
        d, d2, b = mc(scores, allowed, attnreg.AttnReg("dropout", p=0.1).train(), self.N)
        self.assertLess((d2 - ex["Ed2"][:, 0]).abs().max().item(), 5e-4)
        self.assertLess(b.max().item(), 5e-3)

    def test_dropkey_closed_form(self):
        torch.manual_seed(2)
        scores, allowed = case(5, rows=3, keys=5, pad=2)  # n = 3 keys: fallback matters at p=0.6
        ex = census.dropkey_exact(scores, allowed, 0.6)
        d, d2, _ = mc(scores, allowed, attnreg.AttnReg("dropkey", p=0.6).train(), self.N)
        self.assertLess((d - ex["Ed"][:, 0]).abs().max().item(), 3e-3)
        self.assertLess((d2 - ex["Ed2"][:, 0]).abs().max().item(), 3e-3)

    def test_relaxed_exact(self):
        scores, allowed = case(6)
        a = attnreg.masked_softmax(scores, allowed)
        out = attnreg.AttnReg("relaxed", gamma=0.2).train()(scores, allowed)
        d = 0.5 * (out - a).abs().sum(-1)
        self.assertLess((census.relaxed_exact(scores, allowed, gamma=0.2)["Ed"] - d).abs().max().item(), 1e-12)
        out = attnreg.AttnReg("relaxed_budget", tau=0.05).train()(scores, allowed)
        d = 0.5 * (out - a).abs().sum(-1)
        self.assertLess((census.relaxed_exact(scores, allowed, tau=0.05)["Ed"] - d).abs().max().item(), 1e-12)


class FixedAlphaCensusTest(unittest.TestCase):
    def test_hard_mix_census_matches_monte_carlo(self):
        torch.manual_seed(5)
        scores, allowed = case(7)
        a = attnreg.masked_softmax(scores, allowed)[0, 0]
        n = torch.broadcast_to(allowed, scores.shape)[0, 0].sum(-1)
        got = census.hard_from_top(a.topk(min(8, a.shape[-1]), -1).values, n, 0.5, 4, 1.0, alpha_fixed=0.3)
        reg = attnreg.AttnReg("hard_mix", p=0.5, k=4, gamma=0.3).train()
        d, d2, b = mc(scores, allowed, reg, 200000)
        self.assertLess((d - got["Ed"]).abs().max().item(), 3e-3)
        self.assertLess((d2 - got["Ed2"]).abs().max().item(), 3e-3)
        self.assertLess((b - got["B"]).abs().max().item(), 3e-3)

    def test_convex_order_capped_has_smallest_second_moment_at_matched_mean(self):
        for seed in range(8):
            scores, allowed = case(seed, rows=6, keys=12, scale=3.0, pad=0)
            a = attnreg.masked_softmax(scores, allowed)[0, 0]
            top = a.topk(8, -1).values
            n = torch.full((top.shape[0],), 12)
            p = 0.6
            em = census.hard_from_top(top, n, p, 4, 1.0)["Em"].mean()
            target = 0.5 * em
            gamma = float(target / em)
            hf = census.hard_from_top(top, n, p, 4, 1.0, alpha_fixed=gamma)
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if census.hard_from_top(top, n, p, 4, mid)["Ed"].mean() < target:
                    lo = mid
                else:
                    hi = mid
            hb = census.hard_from_top(top, n, p, 4, hi)
            self.assertLess(abs(hb["Ed"].mean() - hf["Ed"].mean()), 1e-9)
            self.assertLessEqual(hb["Ed2"].mean().item(), hf["Ed2"].mean().item() + 1e-12)


class SufficientStatisticsTest(unittest.TestCase):
    def rows(self, seed):
        scores, allowed = case(seed, rows=6, keys=9, pad=[0, 5, 6][seed % 3])
        a = attnreg.masked_softmax(scores, allowed)[0, 0]
        full = torch.broadcast_to(allowed, scores.shape)[0, 0]
        top = a.topk(8, dim=-1).values
        return scores, allowed, top, full.sum(-1), a

    def test_hard_from_top_matches_full_row_enumeration(self):
        for seed in range(9):
            for p, k, tau in [(0.3, 4, 0.1), (0.7, 4, 0.05), (0.5, 3, 1.0), (0.2, 6, 0.2)]:
                scores, allowed, top, n, _ = self.rows(seed)
                ref = census.hard_exact(scores, allowed, p, k, tau)
                got = census.hard_from_top(top, n, p, k, tau)
                for key in ("Em", "Em2", "Ed", "Ed2", "P_active", "excess", "B"):
                    self.assertLess((got[key] - ref[key][0, 0]).abs().max().item(), 1e-12, (seed, p, k, tau, key))

    def test_closed_forms_from_stats(self):
        for seed in range(6):
            scores, allowed, top, n, a = self.rows(seed)
            sum_a2 = (a * a).sum(-1)
            d = census.dropout_exact(scores, allowed, 0.15)
            self.assertLess((census.dropout_from_stats(sum_a2, 0.15)["Ed2"] - d["Ed2"][0, 0]).abs().max().item(), 1e-14)
            k = census.dropkey_exact(scores, allowed, 0.4)
            got = census.dropkey_from_stats(n, a.max(-1).values, sum_a2, 0.4)
            self.assertLess((got["Ed"] - k["Ed"][0, 0]).abs().max().item(), 1e-14)
            self.assertLess((got["Ed2"] - k["Ed2"][0, 0]).abs().max().item(), 1e-14)

    def test_row_sampler_on_bert_skips_padding_queries(self):
        env.patch_transformers_imports()
        from transformers import BertConfig, BertModel
        import bert_adapter
        config = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                            intermediate_size=32, max_position_embeddings=32, attn_implementation="eager")
        torch.manual_seed(0)
        model = BertModel(config, add_pooling_layer=False).eval()
        bert_adapter.install(model, lambda i, n: attnreg.AttnReg("none"))
        sampler = census.RowSampler(max_rows_per_layer=10000, rows_per_call=10000)
        for reg in bert_adapter.regularizers(model):
            reg.observer = sampler
        ids = torch.randint(1, 50, (2, 10))
        mask = torch.ones(2, 10, dtype=torch.long)
        mask[1, 4:] = 0
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        stats = sampler.tensors(0)
        self.assertEqual(stats["n"].numel(), 2 * (10 + 4))  # heads x valid queries
        self.assertEqual(sorted(set(stats["n"].tolist())), [4, 10])
        mass = stats["a_top"].sum(-1)
        small = stats["n"] <= 8
        self.assertLess((mass[small] - 1).abs().max().item(), 1e-6)   # all visible keys captured
        self.assertTrue((mass[~small] < 1 + 1e-9).all())               # top-8 of 10 keys


if __name__ == "__main__":
    unittest.main(verbosity=2)
