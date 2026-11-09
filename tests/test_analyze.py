import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze  # noqa: E402


def fake_uns(effect=0.0, lm_effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    s1, s2 = {}, {}
    base = rng.uniform(0.3, 0.7, 500)
    for tag, shift in (("HB@0.10", effect), ("HF@0.10", 0.0), ("RA@0.05", 0.0), ("N", 0.0)):
        for f in range(10):
            fold_noise = rng.normal(0, 0.002)
            b = np.clip(base + shift + fold_noise + rng.normal(0, 0.05, 500), 0, 2)
            s1[f"S1_{tag}_f{f}"] = {"tag": tag, "fold": f, "rep": 1, "per_example_brier": b.tolist(),
                                   "per_example_brier_ts": (b * 0.98).tolist(),
                                   "correct": (rng.random(500) < 0.6).astype(int).tolist()}
    for tag, shift in (("HB@0.10", lm_effect), ("HF@0.10", 0.0)):
        for s in range(1, 11):
            v = 3.0 + shift + rng.normal(0, 0.003)
            s2[f"S2_{tag}_s{s}"] = {"tag": tag, "seed": s, "final": {"nll_per_token": v}, "best": {"nll_per_token": v - 0.01}}
    return {"S1": s1, "S2": s2}


class AnalyzeTest(unittest.TestCase):
    def test_detects_encoder_effect_and_null(self):
        t = analyze.encoder_contrast(fake_uns(effect=-0.05), "HB@0.10", "HF@0.10")
        self.assertLess(t["mean_diff"], -0.04)
        self.assertLess(t["p_raw"], 0.01)
        self.assertLess(t["ci95_two_level"][1], 0)
        n = analyze.encoder_contrast(fake_uns(effect=0.0, seed=1), "HB@0.10", "HF@0.10")
        self.assertGreater(n["p_raw"], 0.01)
        self.assertTrue(-0.015 < n["ci90_two_level"][0] and n["ci90_two_level"][1] < 0.015)

    def test_lm_contrast(self):
        t = analyze.lm_contrast(fake_uns(lm_effect=0.02), "HB@0.10", "HF@0.10")
        self.assertGreater(t["mean_diff"], 0.015)
        self.assertLess(t["p_final"], 0.01)
        self.assertLess(t["p_best"], 0.01)

    def test_missing_units_reported_unavailable(self):
        u = fake_uns()
        del u["S1"]["S1_HB@0.10_f3"]
        self.assertFalse(analyze.encoder_contrast(u, "HB@0.10", "HF@0.10")["available"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
