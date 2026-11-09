"""Tests for unified/attnreg.py against the archived reference implementation."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
UNIFIED = os.path.dirname(HERE)
sys.path.insert(0, UNIFIED)
import env  # noqa: E402

env.block_arrow()
import torch  # noqa: E402

sys.path.insert(0, os.path.join(env.WORKSPACE, "strengthened"))
import attentiondrop as ref  # noqa: E402  (archived oracle; never modified)
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


class ReferenceEquivalence(unittest.TestCase):
    def test_hard_budget_matches_reference_with_identical_draws(self):
        real_rand = torch.rand
        for seed in range(300):
            dtype = torch.float64 if seed % 2 else torch.float32
            g, scores, allowed = random_case(seed, dtype)
            p = [0.1, 0.5, 1.0][seed % 3 if seed % 7 else 2]
            k = [1, 4, 8, 64][seed % 4]
            tau = [0.1, 0.5, 1.0][(seed // 3) % 3]
            draw = real_rand(scores.shape, generator=g, dtype=torch.float32)
            torch.rand = lambda *a, **kw: draw
            try:
                a = ref.masked_softmax(scores, allowed)
                b = ref.hard_candidate(scores, allowed, p, k)
                expected = ref.budget_mix(a, b, tau)
            finally:
                torch.rand = real_rand
            full = torch.broadcast_to(allowed, scores.shape)
            idx = scores.masked_fill(~full, -torch.inf).topk(min(k, scores.shape[-1]), dim=-1).indices
            got = attnreg.hard_budget(scores, allowed, p, k, tau, draw=draw.gather(-1, idx))
            tol = 1e-5 if dtype == torch.float32 else 1e-12
            self.assertLess((got - expected).abs().max().item(), tol, (seed, p, k, tau))

    def test_hard_budget_gradient_matches_reference(self):
        real_rand = torch.rand
        g = torch.Generator().manual_seed(1)
        scores = torch.randn(1, 2, 6, 6, generator=g, dtype=torch.float64, requires_grad=True)
        allowed = torch.ones(1, 1, 1, 6, dtype=torch.bool)
        allowed[..., 5] = False
        draw = real_rand(scores.shape, generator=g)
        weights = torch.randn(1, 2, 6, 6, generator=g, dtype=torch.float64)
        torch.rand = lambda *a, **kw: draw
        try:
            expected = ref.budget_mix(ref.masked_softmax(scores, allowed),
                                      ref.hard_candidate(scores, allowed, 0.5, 4), 0.1)
            ref_grad = torch.autograd.grad((expected * weights).sum(), scores)[0]
        finally:
            torch.rand = real_rand
        idx = scores.detach().masked_fill(~allowed, -torch.inf).topk(4, dim=-1).indices
        got = attnreg.hard_budget(scores, allowed, 0.5, 4, 0.1, draw=draw.gather(-1, idx))
        got_grad = torch.autograd.grad((got * weights).sum(), scores)[0]
        self.assertLess((ref_grad - got_grad).abs().max().item(), 1e-12)

    def test_masked_softmax_matches_reference(self):
        for seed in range(20):
            _, scores, allowed = random_case(seed, torch.float64)
            self.assertLess((attnreg.masked_softmax(scores, allowed)
                             - ref.masked_softmax(scores, allowed)).abs().max().item(), 1e-14)


class Invariants(unittest.TestCase):
    KINDS = ["none", "dropkey", "relaxed", "hard", "hard_budget", "dropkey_budget", "relaxed_budget"]

    def test_support_normalization_and_padding_rows(self):
        for seed in range(40):
            _, scores, allowed = random_case(seed, torch.float64)
            allowed = allowed.clone()
            allowed[0] = False  # an entirely padded example (all rows inactive)
            full = torch.broadcast_to(allowed, scores.shape)
            active = full.any(-1)
            for kind in self.KINDS:
                reg = attnreg.AttnReg(kind, p=0.3, k=4, tau=0.2, gamma=0.2).train()
                out = reg(scores, allowed)
                self.assertTrue(torch.isfinite(out).all(), kind)
                self.assertEqual(out.masked_select(~full).abs().max().item() if (~full).any() else 0.0, 0.0, kind)
                sums = out.sum(-1)
                self.assertLess((sums[active] - 1).abs().max().item(), 1e-12, kind)
                self.assertEqual(sums[~active].abs().max().item() if (~active).any() else 0.0, 0.0, kind)
                self.assertGreaterEqual(out.min().item(), 0.0, kind)

    def test_budget_caps_realized_displacement(self):
        for seed in range(40):
            _, scores, allowed = random_case(seed, torch.float64)
            a = attnreg.masked_softmax(scores, allowed)
            for kind in ["hard_budget", "dropkey_budget", "relaxed_budget"]:
                tau = [0.05, 0.1, 0.3][seed % 3]
                out = attnreg.AttnReg(kind, p=0.5, k=4, tau=tau).train()(scores, allowed)
                tv = 0.5 * (out - a).abs().sum(-1)
                self.assertLessEqual(tv.max().item(), tau + 1e-12, kind)

    def test_relaxed_budget_is_exact_row_adaptive_relaxation(self):
        _, scores, allowed = random_case(3, torch.float64)
        a = attnreg.masked_softmax(scores, allowed)
        full = torch.broadcast_to(allowed, scores.shape)
        u = full.double() / full.sum(-1, keepdim=True).clamp_min(1)
        r = 0.5 * (u - a).abs().sum(-1)
        out = attnreg.AttnReg("relaxed_budget", tau=0.1).train()(scores, allowed)
        tv = 0.5 * (out - a).abs().sum(-1)
        self.assertLess((tv - torch.minimum(r, torch.full_like(r, 0.1))).abs().max().item(), 1e-12)
        fixed = attnreg.AttnReg("relaxed", gamma=0.25).train()(scores, allowed)
        expect = (0.75 * a + 0.25 * u).masked_fill(~full, 0)
        self.assertLess((fixed - expect).abs().max().item(), 1e-15)

    def test_hard_tv_equals_deleted_mass_and_is_min_with_tau(self):
        for seed in range(30):
            _, scores, allowed = random_case(seed, torch.float64)
            a = attnreg.masked_softmax(scores, allowed)
            torch.manual_seed(seed)
            out, cand, real = attnreg.hard_budget(scores, allowed, 0.5, 4, 0.1, return_stats=True)
            tv = 0.5 * (out - a).abs().sum(-1)
            self.assertLess((tv - real).abs().max().item(), 1e-12)
            self.assertLess((real - torch.minimum(cand, torch.full_like(cand, 0.1))).abs().max().item(), 1e-12)

    def test_evaluation_identity_consumes_no_randomness(self):
        _, scores, allowed = random_case(7, torch.float32)
        plain = attnreg.masked_softmax(scores, allowed)
        for kind in attnreg.KINDS:
            reg = attnreg.AttnReg(kind, p=0.5, tau=0.2, gamma=0.3).eval()
            state = torch.get_rng_state()
            out = reg(scores, allowed)
            self.assertTrue(torch.equal(state, torch.get_rng_state()), kind)
            self.assertTrue(torch.equal(out, plain), kind)

    def test_dropout_is_unbiased_elementwise(self):
        torch.manual_seed(0)
        scores = torch.randn(1, 1, 1, 8, dtype=torch.float64)
        allowed = torch.ones(1, 1, 1, 8, dtype=torch.bool)
        reg = attnreg.AttnReg("dropout", p=0.2).train()
        mean = torch.stack([reg(scores.expand(1, 1, 4000, 8), allowed) for _ in range(5)]).mean(dim=(0, 1, 2, 3))
        a = attnreg.masked_softmax(scores, allowed)[0, 0, 0]
        self.assertLess((mean - a).abs().max().item(), 0.01)

    def test_zero_strength_equals_plain(self):
        _, scores, allowed = random_case(11, torch.float64)
        plain = attnreg.masked_softmax(scores, allowed)
        for kind, kw in [("hard_budget", {"tau": 0.0}), ("relaxed", {"gamma": 0.0}), ("dropkey", {"p": 0.0}),
                         ("dropout", {"p": 0.0}), ("hard", {"p": 0.0}), ("relaxed_budget", {"tau": 0.0})]:
            out = attnreg.AttnReg(kind, **kw).train()(scores, allowed)
            self.assertLess((out - plain).abs().max().item(), 1e-12, kind)

    def test_low_precision_inputs_compute_in_float32(self):
        _, scores, allowed = random_case(4, torch.float32)
        for dtype in (torch.float16, torch.bfloat16):
            out = attnreg.AttnReg("hard_budget").train()(scores.to(dtype), allowed)
            self.assertEqual(out.dtype, torch.float32)


class ExactExpectations(unittest.TestCase):
    def test_expected_mass_matches_closed_form_corollary(self):
        for seed in range(30):
            _, scores, allowed = random_case(seed, torch.float64)
            p, k, tau = 0.3, 4, 0.1
            em, eclip, pact = attnreg.expected_hard_stats(scores, allowed, p, k, tau)
            a = attnreg.masked_softmax(scores, allowed)
            full = torch.broadcast_to(allowed, scores.shape)
            n = full.sum(-1)
            top = a.topk(min(k, a.shape[-1]), dim=-1).values.sum(-1)
            amax = a.max(-1).values
            corollary = p * top - (n <= k).double() * (p ** n.double()) * amax
            corollary = torch.where(n > 0, corollary, torch.zeros_like(corollary))
            self.assertLess((em - corollary).abs().max().item(), 1e-12)
            self.assertTrue(((eclip <= em + 1e-15) & (eclip <= tau + 1e-15)).all())
            self.assertTrue(((pact >= 0) & (pact <= 1 + 1e-15)).all())

    def test_expected_stats_match_monte_carlo(self):
        torch.manual_seed(0)
        scores = torch.randn(1, 1, 3, 6, dtype=torch.float64) * 2
        allowed = torch.ones(1, 1, 1, 6, dtype=torch.bool)
        em, eclip, pact = attnreg.expected_hard_stats(scores, allowed, 0.4, 3, 0.15)
        n = 40000
        big = scores.expand(1, n, 3, 6)
        _, cand, real = attnreg.hard_budget(big, allowed, 0.4, 3, 0.15, return_stats=True)
        self.assertLess((cand.mean(1) - em).abs().max().item(), 0.01)
        self.assertLess((real.mean(1) - eclip).abs().max().item(), 0.01)
        self.assertLess(((cand > 0.15).double().mean(1) - pact).abs().max().item(), 0.02)


class CommonRandomNumbers(unittest.TestCase):
    def test_noise_is_counter_based(self):
        a, b = attnreg.NoiseSource(3), attnreg.NoiseSource(3)
        a.set_step(10); b.set_step(10)
        torch.manual_seed(0)
        x = a.uniform(2, (4, 5), "cpu")
        torch.manual_seed(1)
        y = b.uniform(2, (4, 5), "cpu")
        self.assertTrue(torch.equal(x, y))
        b.set_step(11)
        self.assertFalse(torch.equal(x, b.uniform(2, (4, 5), "cpu")))
        b.set_step(10, pass_index=1)
        self.assertFalse(torch.equal(x, b.uniform(2, (4, 5), "cpu")))
        self.assertFalse(torch.equal(x, a.uniform(3, (4, 5), "cpu")))

    def test_budgeted_and_unbudgeted_share_deletion_sets(self):
        _, scores, allowed = random_case(8, torch.float64)
        results = {}
        for kind in ("hard", "hard_budget"):
            reg = attnreg.AttnReg(kind, p=0.4, k=4, tau=0.05, record=True).train()
            reg.noise = attnreg.NoiseSource(5)
            reg.noise.set_step(3)
            reg.layer = 1
            reg(scores, allowed)
            results[kind] = reg.stats()
        self.assertAlmostEqual(results["hard"]["candidate_tv"], results["hard_budget"]["candidate_tv"], places=12)
        self.assertLessEqual(results["hard_budget"]["realized_tv"], 0.05 + 1e-12)


class TinyDeletedMassGradient(unittest.TestCase):
    def test_backward_is_finite_when_deleted_mass_is_tiny(self):
        # row with one dominant key; the deleted top-k keys carry ~1e-30 mass (regression: 0*inf NaN)
        scores = torch.tensor([[[[80.0, 10.0, 9.0, 8.0, 7.0, 0.0]]]], requires_grad=True)
        allowed = torch.ones(1, 1, 1, 6, dtype=torch.bool)
        draw = torch.tensor([[[[0.9, 0.0, 0.0, 0.0]]]])  # keep the anchor, delete keys 2-4
        for tau in (0.1, 1.0):
            out, m, real = attnreg.hard_budget(scores, allowed, 0.5, 4, tau, draw=draw, return_stats=True)
            self.assertGreater(m.item(), 0.0)
            self.assertLess(m.item(), 1e-20)
            (out * torch.arange(6.0)).sum().backward()
            self.assertTrue(torch.isfinite(scores.grad).all(), tau)
            scores.grad = None
        a = attnreg.masked_softmax(scores, allowed)
        b = torch.tensor([[[[1.0, 0, 0, 0, 0, 0]]]]) * (1 - 1e-30) + a * 1e-30
        mixed, _, _ = attnreg.budget_mix(a, b, 0.05)
        mixed.sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_capped_alpha_equals_min_rule(self):
        r = torch.tensor([0.0, 1e-35, 0.03, 0.1, 0.5, 1.0], dtype=torch.float64)
        for tau in (0.0, 0.1, 1.0):
            alpha = attnreg.capped_alpha(r, tau)
            ref_alpha = torch.where(r > 0, torch.clamp(tau / torch.where(r > 0, r, torch.ones_like(r)), max=1), alpha)
            self.assertTrue(torch.allclose(alpha, ref_alpha), tau)
            self.assertTrue(torch.allclose(alpha * r, torch.minimum(r, torch.full_like(r, tau))), tau)


class FixedAlphaDeletion(unittest.TestCase):
    def test_hard_mix_matches_reference_candidate_at_fixed_alpha(self):
        real_rand = torch.rand
        for seed in range(60):
            g, scores, allowed = random_case(seed, torch.float64)
            draw = real_rand(scores.shape, generator=g, dtype=torch.float32)
            torch.rand = lambda *a, **kw: draw
            try:
                a = ref.masked_softmax(scores, allowed)
                b = ref.hard_candidate(scores, allowed, 0.5, 4)
            finally:
                torch.rand = real_rand
            gamma = 0.3
            m = 0.5 * (b - a).abs().sum(-1, keepdim=True)
            expected = torch.where(m > 0, (1 - gamma) * a + gamma * b, a)
            full = torch.broadcast_to(allowed, scores.shape)
            idx = scores.masked_fill(~full, -torch.inf).topk(min(4, scores.shape[-1]), dim=-1).indices
            got = attnreg.hard_budget(scores, allowed, 0.5, 4, 1.0, draw=draw.gather(-1, idx), alpha_fixed=gamma)
            self.assertLess((got - expected).abs().max().item(), 1e-12, seed)

    def test_hard_mix_shares_deletions_with_hard_budget(self):
        _, scores, allowed = random_case(12, torch.float64)
        stats = {}
        for kind in ("hard_budget", "hard_mix"):
            reg = attnreg.AttnReg(kind, p=0.5, k=4, tau=0.05, gamma=0.2, record=True).train()
            reg.noise = attnreg.NoiseSource(4)
            reg.noise.set_step(2)
            reg(scores, allowed)
            stats[kind] = reg.stats()
        self.assertAlmostEqual(stats["hard_budget"]["candidate_tv"], stats["hard_mix"]["candidate_tv"], places=12)
        self.assertAlmostEqual(stats["hard_mix"]["realized_tv"], 0.2 * stats["hard_mix"]["candidate_tv"], places=12)
        self.assertGreater(stats["hard_mix"]["realized_tv_sq"], 0)


class Recording(unittest.TestCase):
    def test_stats_accumulate_row_averages(self):
        _, scores, allowed = random_case(2, torch.float64)
        reg = attnreg.AttnReg("hard_budget", p=0.5, k=4, tau=0.1, record=True).train()
        torch.manual_seed(3)
        reg(scores, allowed)
        torch.manual_seed(3)
        _, cand, real = attnreg.hard_budget(scores, allowed, 0.5, 4, 0.1, return_stats=True)
        full = torch.broadcast_to(allowed, scores.shape)
        rows = full.any(-1)
        stats = reg.stats()
        self.assertEqual(stats["rows"], rows.sum().item())
        self.assertAlmostEqual(stats["candidate_tv"], cand[rows].mean().item(), places=12)
        self.assertAlmostEqual(stats["realized_tv"], real[rows].mean().item(), places=12)
        self.assertAlmostEqual(stats["active_rate"], (cand[rows] > 0.1).double().mean().item(), places=12)


class BertAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env.patch_transformers_imports()
        from transformers import BertConfig, BertModel
        import bert_adapter
        cls.bert_adapter = bert_adapter
        cls.config = BertConfig(vocab_size=97, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                intermediate_size=64, max_position_embeddings=64,
                                attn_implementation="eager")
        cls.BertModel = BertModel

    def make(self, kind="none", **kw):
        torch.manual_seed(0)
        model = self.BertModel(self.config, add_pooling_layer=False)
        self.bert_adapter.install(model, lambda i, n: attnreg.AttnReg(kind, **kw))
        return model

    def inputs(self):
        g = torch.Generator().manual_seed(5)
        ids = torch.randint(1, 97, (3, 12), generator=g)
        mask = torch.ones(3, 12, dtype=torch.long)
        mask[1, 8:] = 0
        mask[2, 3:] = 0
        return ids, mask

    def test_none_matches_stock_eager_and_sdpa_in_eval(self):
        ids, mask = self.inputs()
        torch.manual_seed(0)
        stock = self.BertModel(self.config, add_pooling_layer=False).eval()
        ours = self.make("hard_budget", p=0.5, tau=0.2).eval()
        ours.load_state_dict(stock.state_dict())
        with torch.no_grad():
            a = stock(input_ids=ids, attention_mask=mask).last_hidden_state
            b = ours(input_ids=ids, attention_mask=mask).last_hidden_state
        valid = mask.bool()
        self.assertLess((a - b)[valid].abs().max().item(), 1e-5)

    def test_padding_keys_receive_no_attention_in_training(self):
        ids, mask = self.inputs()
        model = self.make("hard_budget", p=0.5, tau=0.3).train()
        out = model(input_ids=ids, attention_mask=mask, output_attentions=True)
        for probs in out.attentions:
            self.assertEqual(probs[1, :, :, 8:].abs().max().item(), 0.0)
            self.assertEqual(probs[2, :, :, 3:].abs().max().item(), 0.0)

    def test_padded_tokens_do_not_change_valid_outputs(self):
        ids, mask = self.inputs()
        model = self.make("dropkey_budget", p=0.3, tau=0.2).eval()
        ids2 = ids.clone()
        ids2[2, 3:] = 5
        with torch.no_grad():
            a = model(input_ids=ids, attention_mask=mask).last_hidden_state
            b = model(input_ids=ids2, attention_mask=mask).last_hidden_state
        self.assertLess((a[2, :3] - b[2, :3]).abs().max().item(), 1e-6)

    def test_gradient_checkpointing_replays_identical_masks(self):
        ids, mask = self.inputs()
        grads = []
        for checkpoint in (False, True):
            model = self.make("hard_budget", p=0.5, tau=0.2).train()
            if checkpoint:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            torch.manual_seed(123)
            loss = model(input_ids=ids, attention_mask=mask).last_hidden_state.pow(2).mean()
            loss.backward()
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
        self.assertLess((grads[0] - grads[1]).abs().max().item(), 1e-5)

    def test_checkpointing_with_counter_noise_matches_no_checkpointing(self):
        ids, mask = self.inputs()
        grads = []
        from transformers import BertConfig
        config = BertConfig(**{**self.config.to_dict(), "hidden_dropout_prob": 0.0}, attn_implementation="eager")
        for checkpoint in (False, True):
            torch.manual_seed(0)
            model = self.BertModel(config, add_pooling_layer=False)
            self.bert_adapter.install(model, lambda i, n: attnreg.AttnReg("hard_budget", p=0.5, tau=0.2))
            model.train()
            noise = attnreg.NoiseSource(77)
            noise.set_step(5)
            self.bert_adapter.set_noise(model, noise)
            if checkpoint:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            torch.manual_seed(999 if checkpoint else 1)   # global RNG must be irrelevant
            loss = model(input_ids=ids, attention_mask=mask).last_hidden_state.pow(2).mean()
            loss.backward()
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
        self.assertLess((grads[0] - grads[1]).abs().max().item(), 1e-5)

    def test_stats_collection(self):
        ids, mask = self.inputs()
        model = self.make("hard_budget", p=0.5, tau=0.1, record=True).train()
        model(input_ids=ids, attention_mask=mask)
        stats = self.bert_adapter.collect_stats(model)
        self.assertEqual(len(stats), 2)
        self.assertTrue(all(s["rows"] > 0 and 0 <= s["active_rate"] <= 1 for s in stats))


if __name__ == "__main__":
    unittest.main(verbosity=2)
