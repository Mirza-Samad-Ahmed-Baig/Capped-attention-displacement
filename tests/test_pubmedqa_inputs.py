import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import env  # noqa: E402

env.block_arrow()
env.seal()
env.patch_transformers_imports()
from transformers import AutoTokenizer  # noqa: E402

import pubmedqa_inputs as pi  # noqa: E402

TOKENIZER_DIR = os.path.join(env.WORKSPACE, "domain_distill", "models", "biomedbert")


class InputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        with open(os.path.join(env.ROOT, "data", "pqal_pool.jsonl"), encoding="utf-8") as f:
            cls.rows = [json.loads(line) for line in f]
        with open(os.path.join(env.ROOT, "data", "pqal_pool_sections.json"), encoding="utf-8") as f:
            sections = json.load(f)
        for r in cls.rows:
            r["section_labels"] = sections[r["id"]]

    def test_conclusion_never_enters_non_privileged_inputs(self):
        row = dict(self.rows[0])
        row["long_answer"] = "zebracorn sentinelword conclusion"
        sentinel = set(self.tok.encode("zebracorn sentinelword", add_special_tokens=False))
        for mode in pi.MODES:
            donor = self.rows[1] if mode == "mismatched" else None
            enc = pi.encode(row, self.tok, mode=mode, donor=donor)
            self.assertFalse(sentinel & set(enc["input_ids"]) - set(self.tok.encode(row["question"] + " ".join(row["contexts"]), add_special_tokens=False)), mode)
        priv = pi.encode(row, self.tok, privileged=True)
        self.assertTrue(sentinel <= set(priv["input_ids"]))

    def test_question_kept_and_limit_respected(self):
        for row in self.rows[:200]:
            for single in (True, False):
                enc = pi.encode(row, self.tok, limit=384, single_segment=single)
                q = self.tok.encode(row["question"], add_special_tokens=False)
                self.assertEqual(enc["input_ids"][1:1 + len(q)], q)
                self.assertLessEqual(len(enc["input_ids"]), 384)
                self.assertEqual(len(enc["input_ids"]), len(enc["token_type_ids"]))
                if single:
                    self.assertEqual(set(enc["token_type_ids"]), {0})
                else:
                    self.assertEqual(enc["token_type_ids"][:len(q) + 2], [0] * (len(q) + 2))
                    self.assertEqual(set(enc["token_type_ids"][len(q) + 2:]), {1})

    def test_privileged_conclusion_is_complete(self):
        for row in self.rows[:100]:
            enc = pi.encode(row, self.tok, privileged=True, limit=256)
            concl = self.tok.encode(row["long_answer"], add_special_tokens=False)
            self.assertEqual(enc["input_ids"][-len(concl) - 1:-1], concl)
            self.assertLessEqual(len(enc["input_ids"]), 256)

    def test_question_only_and_mismatched(self):
        row, donor = self.rows[2], self.rows[3]
        qo = pi.encode(row, self.tok, mode="question_only")
        self.assertEqual(qo["evidence_tokens"], 0)
        mm = pi.encode(row, self.tok, mode="mismatched", donor=donor)
        dm = pi.encode(donor, self.tok)
        self.assertEqual(mm["input_ids"][mm["question_tokens"] + 2:], dm["input_ids"][dm["question_tokens"] + 2:])
        with self.assertRaises(ValueError):
            pi.encode(row, self.tok, mode="mismatched", donor=row)

    def test_stratified_derangement(self):
        labels = [pi.label_index(r["label"]) for r in self.rows]
        lengths = [len(" ".join(r["contexts"])) for r in self.rows]
        donors = pi.stratified_derangement(labels, lengths, seed=7)
        self.assertTrue(all(labels[d] != labels[i] for i, d in enumerate(donors)))
        self.assertEqual(donors, pi.stratified_derangement(labels, lengths, seed=7))
        self.assertNotEqual(donors, pi.stratified_derangement(labels, lengths, seed=8))

    def test_microbatches_cover_and_respect_budget(self):
        lengths = [10, 500, 200, 64, 512, 300, 90, 17, 250, 400, 33, 128, 256, 511, 5, 77]
        batches = pi.microbatches(list(range(16)), lengths, max_tokens=1024, max_size=8)
        flat = sorted(i for b in batches for i in b)
        self.assertEqual(flat, list(range(16)))
        for b in batches:
            self.assertTrue(len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 1024)
            self.assertLessEqual(len(b), 8)

    def test_results_deleted_removes_only_results_sections(self):
        row = next(r for r in self.rows if "RESULTS" in r["section_labels"] and len(r["contexts"]) > 1)
        enc = pi.encode(row, self.tok, mode="results_deleted", limit=4096)
        kept = [c for c, lab in zip(row["contexts"], row["section_labels"]) if lab != "RESULTS"]
        expect = self.tok.encode(" ".join(kept), add_special_tokens=False) if kept else []
        self.assertEqual(enc["evidence_tokens"], len(self.tok.encode(chr(10).join(kept), add_special_tokens=False)))
        dropped = [c for c, lab in zip(row["contexts"], row["section_labels"]) if lab == "RESULTS"]
        self.assertTrue(dropped)
        self.assertLess(enc["evidence_tokens"], pi.encode(row, self.tok, limit=4096)["evidence_tokens"])

    def test_sealed_test_file_cannot_be_opened(self):
        with self.assertRaises(PermissionError):
            open(os.path.join(env.WORKSPACE, "domain_distill", "data", "test.json"), encoding="utf-8").close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
