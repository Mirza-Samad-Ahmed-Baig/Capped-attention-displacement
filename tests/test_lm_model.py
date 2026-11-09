import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import env  # noqa: E402

env.block_arrow()
import torch  # noqa: E402

import attnreg  # noqa: E402
import lm_model  # noqa: E402

sys.path.insert(0, os.path.join(env.WORKSPACE, "strengthened"))


class LMTest(unittest.TestCase):
    def test_parameter_count_matches_benchmarked_shape(self):
        m = lm_model.LM(16384, width=384, layers=6, heads=6, context=256)
        # L(12W^2 + 13W) + 2VW + TW + 2W (archived formula; untied head without bias)
        W, L, V, T = 384, 6, 16384, 256
        expected = L * (12 * W * W + 13 * W) + 2 * V * W + T * W + 2 * W
        self.assertEqual(lm_model.parameter_count(m), expected)

    def test_causality_in_train_and_eval_with_counter_noise(self):
        torch.manual_seed(0)
        for kind in ("none", "hard_budget", "dropkey", "relaxed", "dropout", "relaxed_budget"):
            m = lm_model.LM(50, width=32, layers=2, heads=4, context=16,
                            reg_factory=lambda i, n, k=kind: attnreg.AttnReg(k, p=0.4, tau=0.2, gamma=0.2))
            x = torch.randint(0, 50, (2, 16))
            y = x.clone()
            y[:, 10:] = torch.randint(0, 50, (2, 6))
            for train in (True, False):
                m.train(train)
                noise = attnreg.NoiseSource(1)
                m.set_noise(noise)
                noise.set_step(3)
                a = m(x)
                noise.set_step(3)
                b = m(y)
                self.assertLess((a[:, :10] - b[:, :10]).abs().max().item(), 1e-6, (kind, train))

    def test_loss_matches_full_cross_entropy(self):
        torch.manual_seed(1)
        m = lm_model.LM(40, width=16, layers=1, heads=2, context=8).eval()
        x = torch.randint(0, 40, (3, 8))
        t = torch.randint(0, 40, (3, 8))
        full = torch.nn.functional.cross_entropy(m(x).reshape(-1, 40).float(), t.reshape(-1))
        self.assertAlmostEqual(m.loss(x, t, chunk=5).item(), full.item(), places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
