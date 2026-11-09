import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import env  # noqa: E402

env.block_arrow()
import torch  # noqa: E402
from torch import nn  # noqa: E402

import training_utils as tu  # noqa: E402


class BertAdamTest(unittest.TestCase):
    def test_first_step_has_no_bias_correction(self):
        p = nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64))
        g = torch.tensor([0.3, -0.1, 2.0], dtype=torch.float64)
        opt = tu.BertAdam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
        p.grad = g.clone()
        before = p.detach().clone()
        opt.step()
        m, v = 0.1 * g, 0.001 * g * g
        expected = before - 0.1 * (m / (v.sqrt() + 1e-6) + 0.01 * before)
        self.assertLess((p.detach() - expected).abs().max().item(), 1e-12)

    def test_differs_from_bias_corrected_adamw(self):
        p1 = nn.Parameter(torch.ones(3, dtype=torch.float64))
        p2 = nn.Parameter(torch.ones(3, dtype=torch.float64))
        a = tu.BertAdam([p1], lr=0.1)
        b = torch.optim.AdamW([p2], lr=0.1, weight_decay=0.0, eps=1e-6)
        for q in (p1, p2):
            q.grad = torch.full((3,), 0.5, dtype=torch.float64)
        a.step(); b.step()
        self.assertGreater((p1 - p2).abs().max().item(), 1e-3)


class MixoutTest(unittest.TestCase):
    def test_zero_probability_is_identity(self):
        lin = nn.Linear(4, 3)
        mix = tu.MixLinear(lin, 0.0).train()
        x = torch.randn(5, 4)
        self.assertTrue(torch.allclose(mix(x), lin(x)))

    def test_expectation_equals_current_weight(self):
        torch.manual_seed(0)
        lin = nn.Linear(6, 5).double()
        mix = tu.MixLinear(lin, 0.7).train()
        with torch.no_grad():
            lin.weight.add_(torch.randn_like(lin.weight))  # move away from the pretrained target
        x = torch.eye(6, dtype=torch.float64)
        outs = torch.stack([mix(x) for _ in range(20000)]).mean(0)
        self.assertLess((outs - lin(x)).abs().max().item(), 0.05)

    def test_masked_entries_equal_target(self):
        lin = nn.Linear(3, 2).double()
        mix = tu.MixLinear(lin, 0.5).train()
        target = mix.target.clone().double()
        with torch.no_grad():
            lin.weight.fill_(7.0)
        torch.manual_seed(4)
        mask = (torch.rand(lin.weight.shape) < 0.5).double()
        torch.manual_seed(4)
        out = mix(torch.eye(3, dtype=torch.float64))
        w = ((1 - mask) * 7.0 + mask * target - 0.5 * target) / 0.5
        self.assertLess((out - (torch.eye(3, dtype=torch.float64) @ w.T + lin.bias)).abs().max().item(), 1e-12)
        mix.eval()
        self.assertTrue(torch.allclose(mix(torch.eye(3, dtype=torch.float64)), lin(torch.eye(3, dtype=torch.float64))))

    def test_apply_mixout_replaces_linears_once(self):
        model = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Sequential(nn.Linear(3, 2)))
        replaced = tu.apply_mixout(model, 0.1)
        self.assertEqual(replaced, ["0", "2.0"])
        self.assertEqual(sum(isinstance(m, tu.MixLinear) for m in model.modules()), 2)


class LossesTest(unittest.TestCase):
    def test_rdrop_symmetric_and_zero_for_identical_views(self):
        torch.manual_seed(1)
        a, b = torch.randn(4, 3), torch.randn(4, 3)
        y = torch.tensor([0, 1, 2, 1])
        l1, _, kl1 = tu.rdrop_loss(a, b, y, 1.0)
        l2, _, kl2 = tu.rdrop_loss(b, a, y, 1.0)
        self.assertAlmostEqual(l1.item(), l2.item(), places=6)
        _, _, kl0 = tu.rdrop_loss(a, a, y, 1.0)
        self.assertAlmostEqual(kl0.item(), 0.0, places=7)
        self.assertGreater(kl1.item(), 0)

    def test_kd_loss(self):
        torch.manual_seed(2)
        s, t = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
        self.assertAlmostEqual(tu.kd_loss(t, t, 2.0).item(), 0.0, places=12)
        pt, ls = (t / 2).softmax(-1), (s / 2).log_softmax(-1)
        manual = 4 * (pt * (pt.log() - ls)).sum(-1).mean()
        self.assertAlmostEqual(tu.kd_loss(s, t, 2.0).item(), manual.item(), places=10)


class ScheduleAndGroupsTest(unittest.TestCase):
    def test_linear_warmup_decay(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=1.0)
        sched = tu.linear_warmup_decay(opt, total_steps=100, warmup_fraction=0.1)
        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step(); sched.step()
        self.assertAlmostEqual(lrs[0], 0.1)
        self.assertAlmostEqual(lrs[9], 1.0)
        self.assertAlmostEqual(lrs[10], 1.0)
        self.assertAlmostEqual(lrs[99], 1 / 90)

    def test_no_decay_for_bias_and_layernorm(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = nn.Linear(3, 3)
                self.LayerNorm = nn.LayerNorm(3)
        groups = tu.parameter_groups(M(), 0.01)
        self.assertEqual(len(groups[0]["params"]), 1)
        self.assertEqual(len(groups[1]["params"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
