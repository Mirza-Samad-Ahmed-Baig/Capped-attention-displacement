"""Outcome-blind smoke registries for the extension code paths (seed 999; results in the scratch area
and never analysed): privileged teacher with exports -> phase-K KD student (unlabelled pool, eval_pool,
checkpoint) -> phase-L student (init checkpoint, CE + KD) ; SciFact test override ; BioLinkBERT."""
import json
import os
import sys

import runlib

ROOT = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("SMOKE_DIR", os.path.join(os.environ.get("TEMP", "."), "unified_smoke_ext"))
POOL = os.path.join(ROOT, "data", "pqal_pool.jsonl")
PQAU = os.path.join(ROOT, "data", "pqa_unlabeled.jsonl")
SF = os.path.join(ROOT, "data", "scifact")
ids = [json.loads(l)["id"] for l in open(POOL, encoding="utf-8")]
u_ids = json.load(open(os.path.join(ROOT, "data", "pqau_splits.json"), encoding="utf-8"))["kd"][:48]
BASE = {"lr": 2e-5, "epochs": 1, "batch": 16, "warmup": 0.1, "weight_decay": 0.01, "single_segment": True,
        "max_len": 512, "max_tokens": 2048}
LAYERS = [0, 2, 4, 6, 8, 10]


def spec(rid, arm, recipe, **kw):
    s = {"run_id": rid, "study": "smoke", "seed": 999, "arm": arm, "recipe": {**BASE, **recipe}, "pool": POOL,
         "train_ids": ids[:32], "eval_ids": ids[32:44], "diagnostics": {"derangements": 1}}
    s.update(kw)
    return s


def declare(name, runs):
    runlib.Registry(os.path.join(SMOKE, name)).declare(runs, "SMOKE")
    print(os.path.join(SMOKE, name))


def ref(stage, rid, export):
    path = os.path.join(SMOKE, stage, "runs", f"{rid}.json")
    return {"teacher_result": path, "teacher_result_sha256": runlib.sha256_file(path), "export": export, "temperature": 2.0}


def main(stage):
    if stage == "t":
        declare("t", [spec("smk_T", {"kind": "none"}, {"attn_impl": "sdpa", "sdpa_attention_dropout": 0.1, "privileged": True},
                           evaluate_test=False,
                           export_logits={"kd": {"pool": PQAU, "ids": u_ids, "privileged": True},
                                          "train": {"pool": POOL, "ids": ids[:32], "privileged": True},
                                          "dev_priv": {"pool": POOL, "ids": ids[32:44], "privileged": True}})])
    elif stage == "k":
        declare("k", [spec("smk_K", {"kind": "none"}, {"attn_impl": "sdpa", "keep_layers": LAYERS, "lr": 5e-5, "batch": 32,
                                                       "ce_weight": 0.0, "soft_targets": {**ref("t", "smk_T", "kd"), "weight": 1.0}},
                           pool=PQAU, train_ids=u_ids, eval_pool=POOL, evaluate_test=False,
                           save_checkpoint=os.path.join(SMOKE, "k_ckpt.pt"))])
    elif stage == "l":
        ck = json.load(open(os.path.join(SMOKE, "k", "runs", "smk_K.json"), encoding="utf-8"))["checkpoint"]
        declare("l", [spec("smk_L_HB", {"kind": "hard_budget", "p": 0.5, "k": 4, "tau": 0.1},
                           {"keep_layers": LAYERS, "init_checkpoint": ck, "soft_targets": {**ref("t", "smk_T", "train"), "weight": 0.5}})])
    elif stage == "sf":
        rows = [json.loads(l)["id"] for l in open(os.path.join(SF, "train_pairs.jsonl"), encoding="utf-8")]
        declare("sf", [spec("smk_SF", {"kind": "hard_budget", "p": 0.5, "k": 4, "tau": 0.1}, {},
                            pool=os.path.join(SF, "train_pairs.jsonl"), train_ids=rows[:32], eval_ids=rows[32:44],
                            test_inputs=os.path.join(SF, "sealed_inputs", "dev_inputs.jsonl"),
                            test_donors=os.path.join(SF, "sealed_inputs", "dev_donors.json"))])
    elif stage == "e5":
        declare("e5", [spec("smk_E5", {"kind": "hard_budget", "p": 0.5, "k": 4, "tau": 0.1}, {})])


if __name__ == "__main__":
    main(sys.argv[1])
