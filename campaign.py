"""Declare study registries exactly as PROTOCOL.md specifies (runs are declared before execution).

Usage: campaign.py STAGE
  s2_base      S2 base-LR runs: N, lr in {5e-4, 1e-3, 2e-3}, seed 2001
Later stages are declared after the decisions they depend on (G1, predictions.json, G_E).
"""
import json
import os
import sys

import runlib

ROOT = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.join(ROOT, "studies")
FREEZE = runlib.sha256_file(os.path.join(ROOT, "FREEZE.json"))

LM_RECIPE = {"vocab": 16384, "width": 384, "layers": 6, "heads": 6, "context": 256, "batch": 32, "steps": 977,
             "warmup_steps": 49, "betas": [0.9, 0.95], "weight_decay": 0.1, "min_lr_fraction": 0.1,
             "unique_tokens": 1_000_000, "eval_every": 98, "dropout": 0.1}


def lm_spec(run_id, seed, arm, lr, extra=None, **kw):
    spec = {"run_id": run_id, "study": "S2", "seed": seed, "arm": arm,
            "recipe": {**LM_RECIPE, "lr": lr, **(extra or {})}}
    spec.update(kw)
    return spec


def s2_base():
    folder = os.path.join(STUDIES, "S2_base")
    runs = []
    for lr in (5e-4, 1e-3, 2e-3):
        tag = f"{lr:.0e}".replace("-0", "-")
        runs.append(lm_spec(f"S2_base_N_lr{tag}_s1", 2001, {"kind": "none"}, lr,
                            census_at=[0.1, 0.25, 0.5, 1.0],
                            save_checkpoint=os.path.join(folder, "ckpt", f"S2_base_N_lr{tag}_s1.pt")))
    os.makedirs(os.path.join(folder, "ckpt"), exist_ok=True)
    runlib.Registry(folder).declare(runs, FREEZE)
    print(json.dumps([r["run_id"] for r in runs]))


POOL = os.path.join(ROOT, "data", "pqal_pool.jsonl")
FOLDS = json.load(open(os.path.join(ROOT, "data", "pqal_folds.json"), encoding="utf-8"))["folds"]
ENC_RECIPE = {"lr": 2e-5, "epochs": 20, "batch": 16, "warmup": 0.1, "weight_decay": 0.01, "clip": 1.0,
              "single_segment": True, "max_len": 512, "max_tokens": 2048, "hidden_dropout": 0.1, "head_dropout": 0.1,
              "grad_ckpt": True}
S1 = os.path.join(STUDIES, "S1")


def enc_spec(run_id, fold, arm, extra=None, seed=None, **kw):
    train = [i for j, f in enumerate(FOLDS) if j != fold for i in f]
    spec = {"run_id": run_id, "study": "S1", "fold": fold, "seed": seed if seed is not None else 1000 + fold,
            "arm": arm, "recipe": {**ENC_RECIPE, **(extra or {})}, "pool": POOL, "train_ids": train,
            "eval_ids": FOLDS[fold], "diagnostics": {"derangements": 3}}
    spec.update(kw)
    return spec


def predictions():
    with open(os.path.join(ROOT, "results", "predictions.json"), encoding="utf-8") as f:
        return json.load(f)


def enc_arm(family, level_rec, level):
    """Attention arm spec for an encoder family at a calibrated level."""
    r = level_rec
    return {"AD": {"kind": "dropout", "p": r["AD"]["p"]},
            "DK": {"kind": "dropkey", "p": r["DK"]["p_max"], "layer_schedule": "dropkey_linear"},
            "RA": {"kind": "relaxed", "gamma": r["RA"]["gamma"]},
            "RB": {"kind": "relaxed_budget", "tau": r["RB"]["tau"]},
            "HU": {"kind": "hard", "p": r["HU"]["p"], "k": 4},
            "HF": {"kind": "hard_mix", "p": r["HF"]["p"], "k": 4, "gamma": r["HF"]["gamma"]},
            "HB": {"kind": "hard_budget", "p": r["HB"]["p"], "k": 4, "tau": r["HB"]["tau"]}}[family]


def declare(folder, runs):
    os.makedirs(folder, exist_ok=True)
    reg = runlib.Registry(os.path.join(folder))
    reg.declare(runs, FREEZE)
    print(json.dumps({"folder": folder, "runs": len(runs)}))


def s1_n():
    """N arm (no attention regularization, SDPA path) on all 10 folds; allowed before predictions.json."""
    runs = [enc_spec(f"S1_N_f{i}", i, {"kind": "none"}, {"attn_impl": "sdpa"},
                     save_checkpoint=os.path.join(S1, "ckpt", "S1_N_f0.pt") if i == 0 else None)
            for i in range(10)]
    for r in runs:
        if r["save_checkpoint"] is None:
            del r["save_checkpoint"]
    os.makedirs(os.path.join(S1, "ckpt"), exist_ok=True)
    declare(os.path.join(S1, "n"), runs)


def s1_gate():
    """AD@0.10 on folds 0-1 (gate G_E together with N folds 0-1)."""
    lv = predictions()["encoder"]["levels"]["0.10"]
    declare(os.path.join(S1, "gate"), [enc_spec(f"S1_AD@0.10_f{i}", i, enc_arm("AD", lv, 0.10)) for i in (0, 1)])


LEVELS = (0.05, 0.10, 0.20)
MX_P = (0.5, 0.7, 0.9)
LS_EPS = (0.05, 0.10, 0.20)
RD_P = (0.2, 0.3, 0.4)
ENC_FAMILIES = ("AD", "DK", "RA", "HU")


def lv(x):
    return f"{x:.2f}"


def s1_arm(family, level):
    """(run-name tag, arm, recipe extra) for an encoder family at a level."""
    if family == "MX":
        return f"MX@{level}", {"kind": "none"}, {"attn_impl": "sdpa", "mixout_p": level}
    if family == "LS":
        return f"LS@{level}", {"kind": "none"}, {"attn_impl": "sdpa", "label_smoothing": level}
    rec = predictions()["encoder"]["levels"][lv(level)]
    return f"{family}@{lv(level)}", enc_arm(family, rec, level), {}


def s1_tuning():
    """Folds 0-1: AD/DK/RA/HU at 3 levels (AD@0.10 already in the gate), MX, LS; HB@{0.05,0.20}
    (descriptive); core cells HB/HF/RB@0.10."""
    runs = []
    for fold in (0, 1):
        for fam in ENC_FAMILIES:
            for level in LEVELS:
                if fam == "AD" and level == 0.10:
                    continue
                tag, arm, extra = s1_arm(fam, level)
                runs.append(enc_spec(f"S1_{tag}_f{fold}", fold, arm, extra))
        for fam, grid in (("MX", MX_P), ("LS", LS_EPS)):
            for level in grid:
                tag, arm, extra = s1_arm(fam, level)
                runs.append(enc_spec(f"S1_{tag}_f{fold}", fold, arm, extra))
        for fam, level in (("HB", 0.05), ("HB", 0.20), ("HB", 0.10), ("HF", 0.10), ("RB", 0.10)):
            tag, arm, extra = s1_arm(fam, level)
            runs.append(enc_spec(f"S1_{tag}_f{fold}", fold, arm, extra))
    declare(os.path.join(S1, "tuning"), runs)


def s1_finals():
    """Folds 2-9 for selected levels and core cells; completions; Q0 and L0 on folds 0-9."""
    dec = json.load(open(os.path.join(S1, "decisions_levels.json"), encoding="utf-8"))
    runs = []
    cells = {("HB", 0.10), ("HF", 0.10), ("RB", 0.10), ("RA", 0.10), ("HU", 0.10)}
    for fam, level in dec["selected"].items():
        cells.add((fam, level))
    for fold in range(2, 10):
        for fam, level in sorted(cells, key=str):
            tag, arm, extra = s1_arm(fam, level)
            runs.append(enc_spec(f"S1_{tag}_f{fold}", fold, arm, extra))
    for fold in range(10):
        runs.append(enc_spec(f"S1_Q0_f{fold}", fold, {"kind": "none"}, {"attn_impl": "sdpa", "train_mode": "question_only"}))
        runs.append(enc_spec(f"S1_L0_f{fold}", fold, {"kind": "dropout", "p": 0.1},
                             {"lr": 1e-4, "epochs": 5, "lora": {"r": 8, "alpha": 16}, "single_segment": False}))
    declare(os.path.join(S1, "finals"), runs)


def s2_arm(family, level):
    if family == "RD":
        return f"RD@{level}", {"kind": "none"}, {"dropout": level}
    rec = predictions()["lm"]["levels"][lv(level)]
    return f"{family}@{lv(level)}", enc_arm(family, rec, level), {}


def s2_lr():
    return json.load(open(os.path.join(STUDIES, "S2_base2", "decisions_G1.json"), encoding="utf-8"))["lr"]


def s2_recipe_extra():
    return {"unique_tokens": 500_000}


def s2_tuning():
    lr = s2_lr()
    runs = []
    for fam in ("AD", "DK", "RA", "HU"):
        for level in LEVELS:
            tag, arm, extra = s2_arm(fam, level)
            runs.append(lm_spec(f"S2_{tag}_s1", 2001, arm, lr, {**s2_recipe_extra(), **extra}))
    for level in RD_P:
        tag, arm, extra = s2_arm("RD", level)
        runs.append(lm_spec(f"S2_{tag}_s1", 2001, arm, lr, {**s2_recipe_extra(), **extra}))
    lm20 = predictions()["lm"]["levels"]["0.20"]
    for fam, level in (("HB", 0.05), ("HB", 0.20), ("HB", 0.10), ("HF", 0.10), ("RB", 0.10)):
        if (fam, level) == ("HB", 0.20) and lm20["HB"]["tau"] >= 1.0 and lm20["HB"]["p"] == lm20["HU"]["p"]:
            continue  # identical to HU@0.20 (p = 1, no effective cap): run once and shared (PROTOCOL B15)
        tag, arm, extra = s2_arm(fam, level)
        runs.append(lm_spec(f"S2_{tag}_s1", 2001, arm, lr, {**s2_recipe_extra(), **extra}))
    declare(os.path.join(STUDIES, "S2", "tuning"), runs)


def s2_finals():
    lr = s2_lr()
    dec = json.load(open(os.path.join(STUDIES, "S2", "decisions_levels.json"), encoding="utf-8"))
    cells = {("HB", 0.10), ("HF", 0.10), ("RB", 0.10), ("RA", 0.10), ("HU", 0.10), ("N", None)}
    for fam, level in dec["selected"].items():
        cells.add((fam, level))
    runs = []
    for s in range(2, 11):
        for fam, level in sorted(cells, key=str):
            if fam == "N":
                runs.append(lm_spec(f"S2_N_s{s}", 2000 + s, {"kind": "none"}, lr, s2_recipe_extra()))
                continue
            tag, arm, extra = s2_arm(fam, level)
            runs.append(lm_spec(f"S2_{tag}_s{s}", 2000 + s, arm, lr, {**s2_recipe_extra(), **extra}))
    declare(os.path.join(STUDIES, "S2", "finals"), runs)


def p1():
    """P1 top-up (PROTOCOL B5): second seed 4200+i for HB@0.10 and HF@0.10 on all 10 folds."""
    runs = []
    for fold in range(10):
        for fam in ("HB", "HF"):
            tag, arm, extra = s1_arm(fam, 0.10)
            runs.append(enc_spec(f"S1_{tag}_f{fold}_s2", fold, arm, extra, seed=4200 + fold))
    declare(os.path.join(S1, "p1"), runs)


def e2():
    """E2 fragile-recipe ladder (PROTOCOL A 3.5): BertAdam (no bias correction), lr 2e-5, 3 epochs,
    two segment IDs; F-AD (p=0.1), F-HB@0.10, F-MX (Mixout 0.9 + attention dropout 0.1);
    20 runs per arm = folds 0-9 x seeds 4000+i and 4100+i."""
    fragile = {"optimizer": "bertadam", "lr": 2e-5, "epochs": 3, "single_segment": False}
    hb = s1_arm("HB", 0.10)[1]
    arms = {"F-AD": ({"kind": "dropout", "p": 0.1}, {}),
            "F-HB@0.10": (hb, {}),
            "F-MX": ({"kind": "none"}, {"attn_impl": "sdpa", "sdpa_attention_dropout": 0.1, "mixout_p": 0.9})}
    runs = []
    for fold in range(10):
        for base_seed, r in ((4000, 1), (4100, 2)):
            for name, (arm, extra) in arms.items():
                spec = enc_spec(f"E2_{name}_f{fold}_r{r}", fold, arm, {**fragile, **extra}, seed=base_seed + fold)
                spec["study"] = "E2"
                runs.append(spec)
    declare(os.path.join(STUDIES, "E2"), runs)


E5_MODEL = os.path.join(ROOT, "models", "biolinkbert_base")
PQAU = os.path.join(ROOT, "data", "pqa_unlabeled.jsonl")
SCIFACT = os.path.join(ROOT, "data", "scifact")
STUDENT_LAYERS = [0, 2, 4, 6, 8, 10]


def e5():
    """E5: BioLinkBERT-base (Apache-2.0, b71f5d70): N, AD@0.20 (AD* of S1), HU@0.10, HB@0.10 x 10 folds.
    HU/HB parameters from the BioLinkBERT census (results/predictions_e5.json); levels transferred."""
    lv10 = json.load(open(os.path.join(ROOT, "results", "predictions_e5.json"), encoding="utf-8"))["levels"]["0.10"]
    arms = {"N": ({"kind": "none"}, {"attn_impl": "sdpa"}), "AD@0.20": ({"kind": "dropout", "p": 0.20}, {}),
            "HU@0.10": (enc_arm("HU", lv10, 0.10), {}), "HB@0.10": (enc_arm("HB", lv10, 0.10), {})}
    runs = []
    for fold in range(10):
        for tag, (arm, extra) in arms.items():
            spec = enc_spec(f"E5_{tag}_f{fold}", fold, arm, extra)
            spec["study"] = "E5"
            runs.append(spec)
    declare(os.path.join(STUDIES, "E5"), runs)


def e4():
    """E4: SciFact oracle pairs, 3 labels, BiomedBERT, 10 epochs, stabilized recipe; arms N, RA@RA*_E,
    HU@0.10, HB@0.10 (S1 encoder calibration) x seeds 5001-5010. Train on the 90% of training pairs whose
    claim hash is not 0 mod 10; the 10% calibration split fits temperatures and dev diagnostics only.
    The 339 dev pairs are the one-shot sealed test."""
    import hashlib
    rows = [json.loads(l) for l in open(os.path.join(SCIFACT, "train_pairs.jsonl"), encoding="utf-8")]
    calib = lambda rid: int(hashlib.sha256(f"e4:{rid.split('_')[0]}".encode()).hexdigest(), 16) % 10 == 0
    train_ids = [r["id"] for r in rows if not calib(r["id"])]
    eval_ids = [r["id"] for r in rows if calib(r["id"])]
    dec = json.load(open(os.path.join(S1, "decisions_levels.json"), encoding="utf-8"))
    pred = predictions()["encoder"]["levels"]
    arms = {"N": ({"kind": "none"}, {"attn_impl": "sdpa"}),
            f"RA@{dec['RA_star']:.2f}": (enc_arm("RA", pred[lv(dec["RA_star"])], dec["RA_star"]), {}),
            "HU@0.10": (enc_arm("HU", pred["0.10"], 0.10), {}), "HB@0.10": (enc_arm("HB", pred["0.10"], 0.10), {})}
    runs = []
    for k in range(1, 11):
        for tag, (arm, extra) in arms.items():
            runs.append({"run_id": f"E4_{tag}_f0_s{k}", "study": "E4", "fold": 0, "seed": 5000 + k, "arm": arm,
                         "recipe": {**ENC_RECIPE, "epochs": 10, **extra}, "pool": os.path.join(SCIFACT, "train_pairs.jsonl"),
                         "train_ids": train_ids, "eval_ids": eval_ids, "diagnostics": {"derangements": 3},
                         "test_inputs": os.path.join(SCIFACT, "sealed_inputs", "dev_inputs.jsonl"),
                         "test_donors": os.path.join(SCIFACT, "sealed_inputs", "dev_donors.json")})
    declare(os.path.join(STUDIES, "E4"), runs)


def es3_teachers():
    """E-S3 teachers: privileged input (question + evidence + conclusion), stabilized recipe, SDPA with
    attention dropout 0.1, fold-i training set, seed 3200+i; never evaluated on the test. Exports logits
    on the KD subset (PQA-U), the fold training set and (for gate G3) the fold dev set, all privileged."""
    kd_ids = json.load(open(os.path.join(ROOT, "data", "pqau_splits.json"), encoding="utf-8"))["kd"]
    runs = []
    for fold in range(10):
        spec = enc_spec(f"ES3_T_f{fold}", fold, {"kind": "none"},
                        {"attn_impl": "sdpa", "sdpa_attention_dropout": 0.1, "privileged": True}, seed=3200 + fold)
        spec.update({"study": "ES3", "evaluate_test": False,
                     "export_logits": {"kd": {"pool": PQAU, "ids": kd_ids, "privileged": True},
                                       "train": {"pool": POOL, "ids": spec["train_ids"], "privileged": True},
                                       "dev_priv": {"pool": POOL, "ids": spec["eval_ids"], "privileged": True}}})
        runs.append(spec)
    declare(os.path.join(STUDIES, "ES3", "teachers"), runs)


def teacher_ref(fold, export):
    path = os.path.join(STUDIES, "ES3", "teachers", "runs", f"ES3_T_f{fold}.json")
    return {"teacher_result": path, "teacher_result_sha256": runlib.sha256_file(path), "export": export, "temperature": 2.0}


def es3_phase_k():
    """Phase K per fold: 6-layer student (BiomedBERT embeddings + layers 0,2,4,6,8,10), soft-label KD only
    (T = 2) for 1 epoch over the 10k KD subset, lr 5e-5, batch 32, 6% warmup, SDPA; seed 3000+i."""
    kd_ids = json.load(open(os.path.join(ROOT, "data", "pqau_splits.json"), encoding="utf-8"))["kd"]
    runs = []
    for fold in range(10):
        spec = enc_spec(f"ES3_K_f{fold}", fold, {"kind": "none"},
                        {"attn_impl": "sdpa", "sdpa_attention_dropout": 0.1, "keep_layers": STUDENT_LAYERS, "lr": 5e-5,
                         "epochs": 1, "batch": 32, "warmup": 0.06, "ce_weight": 0.0,
                         "soft_targets": {**teacher_ref(fold, "kd"), "weight": 1.0}}, seed=3000 + fold)
        spec.update({"study": "ES3", "pool": PQAU, "train_ids": kd_ids, "eval_pool": POOL, "evaluate_test": False,
                     "save_checkpoint": os.path.join(STUDIES, "ES3", "ckpt", f"ES3_K_f{fold}.pt")})
        runs.append(spec)
    os.makedirs(os.path.join(STUDIES, "ES3", "ckpt"), exist_ok=True)
    declare(os.path.join(STUDIES, "ES3", "phase_k"), runs)


def es3_phase_l():
    """Phase L per fold: fold training set, CE + 0.5 * T^2 KL (privileged-teacher logits, T = 2), stabilized
    recipe (20 epochs), init from the phase-K student; arms S-N, S-HU@0.10, S-HB@0.10, S-HF@0.10 with the
    student census calibration; plus E7 S-noKD (no phase K, no KD, CE only). Seeds 3100+i."""
    lv10 = json.load(open(os.path.join(ROOT, "results", "predictions_student.json"), encoding="utf-8"))["levels"]["0.10"]
    runs = []
    for fold in range(10):
        kpath = os.path.join(STUDIES, "ES3", "phase_k", "runs", f"ES3_K_f{fold}.json")
        ck = json.load(open(kpath, encoding="utf-8"))["checkpoint"]
        base = {"keep_layers": STUDENT_LAYERS, "init_checkpoint": ck, "soft_targets": {**teacher_ref(fold, "train"), "weight": 0.5}}
        arms = {"S-N": ({"kind": "none"}, {"attn_impl": "sdpa"}), "S-HU@0.10": (enc_arm("HU", lv10, 0.10), {}),
                "S-HB@0.10": (enc_arm("HB", lv10, 0.10), {}), "S-HF@0.10": (enc_arm("HF", lv10, 0.10), {})}
        for tag, (arm, extra) in arms.items():
            spec = enc_spec(f"ES3_{tag}_f{fold}", fold, arm, {**base, **extra}, seed=3100 + fold)
            spec["study"] = "ES3"
            runs.append(spec)
        spec = enc_spec(f"ES3_S-noKD_f{fold}", fold, {"kind": "none"}, {"attn_impl": "sdpa", "keep_layers": STUDENT_LAYERS},
                        seed=3100 + fold)
        spec["study"] = "ES3"
        runs.append(spec)
    declare(os.path.join(STUDIES, "ES3", "phase_l"), runs)


RDROP_ALPHA = (0.1, 0.5, 1.0)


def e1_tuning():
    """E1 (PROTOCOL A 3.5): R-Drop in S1, noise from attention dropout p=0.1 (+ hidden dropout), two passes;
    alpha in {0.1, 0.5, 1} on folds 0-1."""
    runs = [enc_spec(f"S1_RDROP@{a}_f{f}", f, {"kind": "dropout", "p": 0.1}, {"rdrop_alpha": a})
            for f in (0, 1) for a in RDROP_ALPHA]
    declare(os.path.join(S1, "e1_tuning"), runs)


def e1_finals():
    import select as sel
    runs = sel.load_runs(os.path.join(S1, "e1_tuning"))
    score = {a: sum(sel.brier(runs[f"S1_RDROP@{a}_f{f}"]["dev"]["original"], runs[f"S1_RDROP@{a}_f{f}"]["eval_labels"])
                    for f in (0, 1)) / 2 for a in RDROP_ALPHA}
    alpha = sel.choose_level(score)
    deleted = []  # PROTOCOL B12.3: sealed test files of unselected alphas are deleted unopened
    sealed = os.path.join(S1, "e1_tuning", "sealed")
    for a in RDROP_ALPHA:
        if a == alpha:
            continue
        for f in (0, 1):
            path = os.path.join(sealed, f"S1_RDROP@{a}_f{f}.json")
            if os.path.exists(path):
                os.remove(path)
                deleted.append(os.path.basename(path))
    runlib.write_json_once(os.path.join(S1, "decisions_E1.json"), {"rule": "E1 alpha: lowest mean dev Brier folds 0-1, ties -> lower alpha",
                                                                   "selected_alpha": alpha, "deleted_unopened": deleted})
    runs = [enc_spec(f"S1_RDROP@{alpha}_f{f}", f, {"kind": "dropout", "p": 0.1}, {"rdrop_alpha": alpha}) for f in range(2, 10)]
    declare(os.path.join(S1, "e1_finals"), runs)
    print("selected alpha", alpha)


def e3():
    """E3: budgeted DropKey (uniform candidate, p=0.5, tau calibrated to realized mean 0.10) vs HB@0.10:
    S1 10 folds and S2 10 seeds (predictions_e3.json)."""
    pe3 = json.load(open(os.path.join(ROOT, "results", "predictions_e3.json"), encoding="utf-8"))
    enc_arm_db = {"kind": "dropkey_budget", "p": pe3["encoder"]["p"], "tau": pe3["encoder"]["tau"]}
    lm_arm_db = {"kind": "dropkey_budget", "p": pe3["lm"]["p"], "tau": pe3["lm"]["tau"]}
    runs = [enc_spec(f"S1_DB@0.10_f{f}", f, enc_arm_db) for f in range(10)]
    declare(os.path.join(S1, "e3"), runs)
    lr = s2_lr()
    lm_runs = [lm_spec(f"S2_DB@0.10_s{s}", 2000 + s, lm_arm_db, lr, s2_recipe_extra()) for s in range(1, 11)]
    declare(os.path.join(STUDIES, "S2", "e3"), lm_runs)


def e8():
    """E8 (PROTOCOL B10): alpha-detached HB@0.10 in the LM (attnreg kind hard_budget_sg: forward values equal
    HB exactly, no gradient through alpha). Same p, k, tau, recipe, learning rate and seeds 2001-2010 as the S2
    HB@0.10 runs (seed 1 in S2/tuning, seeds 2-10 in S2/finals), so each pair shares data order and noise."""
    lr = s2_lr()
    _, arm, extra = s2_arm("HB", 0.10)
    arm_sg = {**arm, "kind": "hard_budget_sg"}
    runs = [lm_spec(f"S2_HBsg@0.10_s{s}", 2000 + s, arm_sg, lr, {**s2_recipe_extra(), **extra}) for s in range(1, 11)]
    declare(os.path.join(STUDIES, "S2", "e8"), runs)


def s2_base2():
    """G1 fallback (PROTOCOL A 3.3): corpus halved to >=500k tokens, 16 epochs (977 steps), base tuning redone once."""
    global LM_RECIPE
    LM_RECIPE = {**LM_RECIPE, "unique_tokens": 500_000}
    folder = os.path.join(STUDIES, "S2_base2")
    runs = []
    for lr in (5e-4, 1e-3, 2e-3):
        tag = f"{lr:.0e}".replace("-0", "-")
        runs.append(lm_spec(f"S2_base2_N_lr{tag}_s1", 2001, {"kind": "none"}, lr, census_at=[0.1, 0.25, 0.5, 1.0],
                            save_checkpoint=os.path.join(folder, "ckpt", f"S2_base2_N_lr{tag}_s1.pt")))
    os.makedirs(os.path.join(folder, "ckpt"), exist_ok=True)
    runlib.Registry(folder).declare(runs, FREEZE)
    print(json.dumps([r["run_id"] for r in runs]))


if __name__ == "__main__":
    {"s2_base": s2_base, "s2_base2": s2_base2, "s1_n": s1_n, "s1_gate": s1_gate, "s1_tuning": s1_tuning, "s1_finals": s1_finals, "s2_tuning": s2_tuning, "s2_finals": s2_finals, "p1": p1, "e2": e2, "e5": e5, "e4": e4, "es3_teachers": es3_teachers, "es3_phase_k": es3_phase_k, "es3_phase_l": es3_phase_l, "e1_tuning": e1_tuning, "e1_finals": e1_finals, "e3": e3, "e8": e8}[sys.argv[1]]()
