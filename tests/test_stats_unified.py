import itertools
import os
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stats_unified as su  # noqa: E402


class PairedTests(unittest.TestCase):
    def test_sign_flip_matches_brute_force_and_minimum_p(self):
        d = np.array([0.3, -0.1, 0.2, 0.25, 0.05, 0.4, -0.02, 0.1, 0.15, 0.2])
        obs = abs(d.mean())
        brute = np.mean([abs(np.mean(np.array(s) * d)) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=10)])
        self.assertAlmostEqual(su.sign_flip_test(d), brute, places=12)
        self.assertAlmostEqual(su.sign_flip_test(np.ones(10)), 2 / 1024, places=12)
        self.assertAlmostEqual(su.sign_flip_test(np.ones(10), "greater"), 1 / 1024, places=12)

    def test_sign_flip_ci_contains_mean_and_excludes_far_shift(self):
        d = np.random.default_rng(0).normal(0.1, 0.05, 10)
        lo, hi = su.sign_flip_ci(d, step=1e-3)
        self.assertLess(lo, d.mean())
        self.assertGreater(hi, d.mean())
        self.assertGreater(lo, -0.2)

    def test_holm(self):
        self.assertEqual([round(x, 6) for x in su.holm([0.01, 0.04, 0.03, 0.2])], [0.04, 0.09, 0.09, 0.2])

    def test_nadeau_bengio_wider_than_naive(self):
        d = np.random.default_rng(1).normal(0, 1, 10)
        nb = su.nadeau_bengio_ci(d, 450, 50)
        naive = su.t_ci(d, 0.95)
        self.assertGreater(nb[1] - nb[0], naive[1] - naive[0])

    def test_tost_and_verdict(self):
        d = np.random.default_rng(2).normal(0, 0.001, 10)
        self.assertTrue(su.tost(d, 0.015)["equivalent"])
        self.assertEqual(su.verdict(0.01, (-0.02, -0.01), 0.015, -0.015), "Better")
        self.assertEqual(su.verdict(0.3, (-0.005, 0.004), 0.015, 0.0), "Equivalent")
        self.assertEqual(su.verdict(0.3, (-0.03, 0.02), 0.015, 0.0), "Inconclusive")

    def test_two_level_bootstrap_centers_on_difference(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0.5, 0.1, (10, 200))
        b = a + 0.05 + rng.normal(0, 0.05, (10, 200))
        lo, hi = su.two_level_bootstrap(a, b, reps=2000)
        self.assertTrue(lo <= (a - b).mean() <= hi)
        self.assertLess(hi - lo, 0.05)

    def test_page_l_detects_trend(self):
        t = np.tile(np.arange(4), (10, 1)) + np.random.default_rng(4).normal(0, 0.1, (10, 4))
        L, p = su.page_l(t, reps=2000)
        self.assertEqual(L, 10 * (1 + 4 + 9 + 16))
        self.assertLess(p, 0.01)

    def test_mcnemar_and_brown_forsythe(self):
        self.assertAlmostEqual(su.mcnemar_exact(0, 10), 2 * 0.5 ** 10, places=12)
        s, p = su.brown_forsythe([1, 2, 3, 4], [1, 5, 9, 13])
        self.assertAlmostEqual(p, stats.levene([1, 2, 3, 4], [1, 5, 9, 13], center="median").pvalue)


class Metrics(unittest.TestCase):
    def test_uniform_and_perfect(self):
        labels = np.array([0, 1, 2, 0])
        m = su.classification_metrics(np.zeros((4, 3)), labels)
        self.assertAlmostEqual(m["brier"], 2 / 3, places=12)
        self.assertAlmostEqual(m["nll"], np.log(3), places=12)
        m = su.classification_metrics(np.eye(3)[labels] * 50, labels)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["auroc"], 1.0)

    def test_temperature_recovers_scale(self):
        rng = np.random.default_rng(5)
        z = rng.normal(0, 1, (4000, 3))
        labels = np.array([rng.choice(3, p=su.softmax(row[None])[0]) for row in z])
        t = su.fit_temperature(z * 2.0, labels)
        self.assertAlmostEqual(t, 2.0, delta=0.15)

    def test_macro_auroc_matches_pairwise_definition(self):
        rng = np.random.default_rng(6)
        p = su.softmax(rng.normal(0, 1, (120, 3)))
        y = rng.integers(0, 3, 120)
        vals = []
        for c in range(3):
            pos, neg = p[y == c, c], p[y != c, c]
            wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
            vals.append(wins / (len(pos) * len(neg)))
        self.assertAlmostEqual(su.macro_auroc(p, y), float(np.mean(vals)), places=12)

    def test_evidence_use(self):
        full = np.eye(3)[[0, 1, 2, 0]] * 5
        qo = np.eye(3)[[0, 0, 0, 0]] * 5
        mm = np.eye(3)[[1, 1, 2, 2]] * 5
        out = su.evidence_use(full, qo, [mm], [0, 1, 2, 0], [[1, 0, 0, 2]])
        self.assertAlmostEqual(out["evidence_use_gap"], 1.0 - 0.5)
        self.assertAlmostEqual(out["flip_rate"], 0.5)
        self.assertAlmostEqual(out["donor_label_following"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
