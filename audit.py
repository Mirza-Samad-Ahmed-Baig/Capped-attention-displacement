"""Audits (PROTOCOL A §4 audit row, §5 `audit.py`; B14 R14). Outcome-blind: sealed test outputs are
compared by SHA-256 only, and the console prints PASS/FAIL only (values go to results/audit.json).

Subcommands (run after the core campaign, before the extensions: PROTOCOL A §4 order):
  choose            2 audit runs per core study (S1/finals, S2/finals): the first two run ids ranked by
                    sha256(FREEZE.json sha256 + ":" + run_id). -> results/audit_choice.json (write-once)
  declare           audit studies studies/audit/{S1,S2}: each chosen spec twice (_detA, _detB), unchanged
                    except run_id, study and a model-only checkpoint for _detA
  run FOLDER KIND   executes an audit study (KIND encoder|lm) with deterministic training for the whole
                    run: torch deterministic algorithms, MATH SDPA, cuDNN deterministic, and the cuBLAS
                    workspace config that env.setup_model_process sets before CUDA initialises
  evalreplay KIND   bit-exact evaluation replays from saved checkpoints (encoder: S1_N_f0 and the S1 _detA
                    runs; LM: the S2_base2 winner and the S2 _detA runs)
  report            -> results/audit.json (write-once)

Criteria.
  Deterministic retrain replay (R14): _detA and _detB must agree within 1e-6 on every recorded dev logit
    (encoder) or validation NLL and training-loss trace entry (LM), and their sealed test files must
    be identical (LM: equal SHA-256; encoder: byte-identical once the embedded run_id is swapped,
    compared in-process with only the boolean kept).
  Non-deterministic retrain check (R14): |dev metric(original) - dev metric(_detA)| <= the between-unit
    SD of the N arm (encoder: dev Brier over folds 0-9; LM: final validation NLL over seeds 1-10).
  Evaluation replay: encoder dev logits identical to the record and sealed test hash identical; LM (the
    checkpoints are fp16): two evaluations bit-identical to each other, and the replayed validation NLL
    within 1e-3 of the recorded final value.
"""
import copy
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.join(ROOT, "studies")
AUDIT = os.path.join(STUDIES, "audit")
CHOICE = os.path.join(ROOT, "results", "audit_choice.json")
CORE = {"S1": os.path.join(STUDIES, "S1", "finals"), "S2": os.path.join(STUDIES, "S2", "finals")}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def freeze_sha():
    return hashlib.sha256(open(os.path.join(ROOT, "FREEZE.json"), "rb").read()).hexdigest()


def choose():
    import runlib
    key = freeze_sha()
    out = {"rule": "first two run ids by sha256(FREEZE sha + ':' + run_id)", "freeze_sha256": key, "chosen": {}}
    for study, folder in CORE.items():
        reg = runlib.Registry(folder)
        if reg.pending():
            raise SystemExit(f"{folder} has pending runs; audits start after the core")
        ids = [r["run_id"] for r in reg.load()["runs"]]
        ids.sort(key=lambda rid: hashlib.sha256(f"{key}:{rid}".encode()).hexdigest())
        out["chosen"][study] = ids[:2]
    print(runlib.write_json_once(CHOICE, out))
    print(json.dumps(out["chosen"]))


def declare():
    import runlib
    choice = load(CHOICE)["chosen"]
    for study, ids in choice.items():
        folder = os.path.join(AUDIT, study)
        os.makedirs(os.path.join(folder, "ckpt"), exist_ok=True)
        specs = {r["run_id"]: r for r in runlib.Registry(CORE[study]).load()["runs"]}
        runs = []
        for rid in ids:
            for rep in ("A", "B"):
                s = copy.deepcopy(specs[rid])
                s["run_id"], s["study"] = f"{rid}_det{rep}", "audit"
                s.pop("save_checkpoint", None)
                if rep == "A":
                    s["save_checkpoint"] = os.path.join(folder, "ckpt", f"{rid}_detA.pt")
                runs.append(s)
        runlib.Registry(folder).declare(runs, runlib.sha256_file(os.path.join(ROOT, "FREEZE.json")))
        print(json.dumps({"folder": folder, "runs": len(runs)}))


def deterministic(torch):
    """Force deterministic kernels for the whole process; runner calls that switch them off are ignored."""
    original = torch.use_deterministic_algorithms
    original(True)
    torch.use_deterministic_algorithms = lambda mode, **kw: original(True, **kw)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def run(folder, kind):
    runner = __import__("encoder_runner" if kind == "encoder" else "lm_runner")
    deterministic(runner.torch)
    sys.argv = [f"{runner.__name__}.py", os.path.abspath(folder)]
    runner.main()


def max_abs_diff(a, b):
    if isinstance(a, dict):
        if set(a) != set(b):
            return float("inf")
        return max([max_abs_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, list):
        if len(a) != len(b):
            return float("inf")
        return max([max_abs_diff(x, y) for x, y in zip(a, b)] or [0.0])
    return abs(float(a) - float(b))


def brier(logits, labels):
    import math
    total = 0.0
    for row, y in zip(logits, labels):
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        total += sum((ei / s - (1.0 if c == y else 0.0)) ** 2 for c, ei in enumerate(e))
    return total / len(labels)


def sealed_equal_modulo_run_id(sealed_dir, a, b):
    """Encoder sealed files embed their own run_id, so _detA and _detB can never share a raw hash.
    Compare A's bytes with its run_id replaced by B's against B's bytes, after checking each file against
    the hash its run recorded. Only the boolean is kept; no logit is printed or stored."""
    ra, rb = a["spec"]["run_id"], b["spec"]["run_id"]
    pa, pb = os.path.join(sealed_dir, f"{ra}.json"), os.path.join(sealed_dir, f"{rb}.json")
    ba, bb = open(pa, "rb").read(), open(pb, "rb").read()
    if hashlib.sha256(ba).hexdigest() != a["test_sealed_sha256"] or hashlib.sha256(bb).hexdigest() != b["test_sealed_sha256"]:
        return False
    return ba.replace(json.dumps(ra).encode(), json.dumps(rb).encode(), 1) == bb


def sd(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def evalreplay(kind):
    import runlib
    out = {}
    if kind == "encoder":
        import encoder_runner as er
        worker = er.Worker(er.DEFAULT_MODEL)
        sealed = os.path.join(AUDIT, "S1", "sealed_evalreplay")
        targets = [(os.path.join(STUDIES, "S1", "n", "runs", "S1_N_f0.json"), os.path.join(STUDIES, "S1", "ckpt", "S1_N_f0.pt"))]
        targets += [(p, p.replace(os.sep + "runs" + os.sep, os.sep + "ckpt" + os.sep).replace(".json", ".pt"))
                    for p in sorted(glob.glob(os.path.join(AUDIT, "S1", "runs", "*_detA.json")))]
        for rec_path, ckpt in targets:
            rec = load(rec_path)
            spec = rec["spec"]
            if runlib.sha256_file(ckpt) != rec["checkpoint"]["sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {ckpt}")
            # Replay with the SDPA backend the run evaluated with: audit _detA runs ran under deterministic()
            # (MATH only); original runs used the default backends.
            math_only = spec["run_id"].endswith("_detA")
            er.torch.backends.cuda.enable_flash_sdp(not math_only)
            er.torch.backends.cuda.enable_mem_efficient_sdp(not math_only)
            er.torch.backends.cuda.enable_math_sdp(True)
            model, _, _ = worker.build(spec)
            state = er.torch.load(ckpt, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected or any(not k.endswith(".target") for k in missing):
                raise RuntimeError(f"state mismatch for {ckpt}")
            model = model.cuda()
            recipe = spec["recipe"]
            enc_kw = dict(limit=recipe.get("max_len", 512), single_segment=recipe.get("single_segment", True), privileged=False)
            max_tokens = recipe.get("max_tokens", 2048)
            pool = worker.load_records(spec.get("eval_pool", spec["pool"]))
            eval_rows = [pool[i] for i in spec["eval_ids"]]
            by_id = {r["id"]: r for r in eval_rows}
            dev_donors = [[by_id[i] for i in d] for d in rec["dev_donors"]]
            dev = worker.conditions(model, eval_rows, dev_donors, enc_kw, max_tokens * 2)
            test_rows, donor_lists = worker.test_set(spec.get("test_inputs", er.TEST_INPUTS), spec.get("test_donors", er.TEST_DONORS))
            tid = {r["id"]: r for r in test_rows}
            test_donors = [[tid[i] for i in lst] for lst in donor_lists[:spec.get("diagnostics", {"derangements": 3}).get("derangements", 3)]]
            test = worker.conditions(model, test_rows, test_donors, enc_kw, max_tokens * 2)
            digest = runlib.write_json_once(os.path.join(sealed, f"{spec['run_id']}.json"),
                                            {"run_id": spec["run_id"], "ids": [r["id"] for r in test_rows], "logits": test})
            out[spec["run_id"]] = {"dev_max_abs_diff": max_abs_diff(dev, rec["dev"]),
                                   "sealed_hash_equal": digest == rec["test_sealed_sha256"]}
            del model
            er.torch.cuda.empty_cache()
    else:
        import lm_runner as lr
        import lm_model
        import numpy as np
        worker = lr.Worker()
        targets = []
        base = os.path.join(STUDIES, "S2_base2")
        winner = load(os.path.join(base, "decisions_G1.json"))["lr_winner"]
        targets.append(os.path.join(base, "runs", f"{winner}.json"))
        targets += sorted(glob.glob(os.path.join(AUDIT, "S2", "runs", "*_detA.json")))
        for rec_path in targets:
            rec = load(rec_path)
            r = rec["spec"]["recipe"]
            ck = rec["checkpoint"]
            if runlib.sha256_file(ck["path"]) != ck["sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {ck['path']}")
            vals = []
            for _ in range(2):
                model = lm_model.LM(r["vocab"], width=r["width"], layers=r["layers"], heads=r["heads"], context=r["context"],
                                    dropout=r.get("dropout", 0.1))
                model.load_state_dict({k: v.float() for k, v in lr.torch.load(ck["path"], map_location="cpu", weights_only=True).items()})
                model = model.cuda()
                val = worker.val[: int(r.get("val_tokens", len(worker.val)))]
                vals.append(worker.evaluate(model, val, r["context"]))
                del model
                lr.torch.cuda.empty_cache()
            out[rec["spec"]["run_id"]] = {"replays_bit_identical": vals[0] == vals[1],
                                          "val_nll_abs_diff_vs_record": abs(vals[0] - rec["validation"][-1]["val_nll"])}
    path = os.path.join(ROOT, "results", f"audit_evalreplay_{kind}.json")
    print(runlib.write_json_once(path, out))
    for rid, v in out.items():
        ok = (v["dev_max_abs_diff"] == 0.0 and v["sealed_hash_equal"]) if kind == "encoder" else \
             (v["replays_bit_identical"] and v["val_nll_abs_diff_vs_record"] <= 1e-3)
        print(rid, "PASS" if ok else "FAIL")


def report():
    import runlib
    choice = load(CHOICE)["chosen"]
    res = {"criteria": __doc__.split("Criteria.")[1].strip(), "retrain": {}, "evalreplay": {}}
    n_s1 = [load(p) for p in sorted(glob.glob(os.path.join(STUDIES, "S1", "n", "runs", "S1_N_f*.json")))]
    tol_e = sd([brier(r["dev"]["original"], r["eval_labels"]) for r in n_s1])
    n_s2 = [load(p) for p in sorted(glob.glob(os.path.join(STUDIES, "S2", "finals", "runs", "S2_N_s*.json")))]
    base = os.path.join(STUDIES, "S2_base2")
    winner = load(os.path.join(base, "decisions_G1.json"))["lr_winner"]
    n_s2.append(load(os.path.join(base, "runs", f"{winner}.json")))  # seed 1 of the N arm
    tol_l = sd([r["validation"][-1]["val_nll"] for r in n_s2])
    for study, ids in choice.items():
        folder = os.path.join(AUDIT, study, "runs")
        for rid in ids:
            orig = load(os.path.join(CORE[study], "runs", f"{rid}.json"))
            a, b = load(os.path.join(folder, f"{rid}_detA.json")), load(os.path.join(folder, f"{rid}_detB.json"))
            if study == "S1":
                det = max_abs_diff(a["dev"], b["dev"])
                hashes = sealed_equal_modulo_run_id(os.path.join(AUDIT, study, "sealed"), a, b)
                diff = abs(brier(orig["dev"]["original"], orig["eval_labels"]) - brier(a["dev"]["original"], a["eval_labels"]))
                tol = tol_e
            else:
                det = max(max_abs_diff([e["val_nll"] for e in a["validation"]], [e["val_nll"] for e in b["validation"]]),
                          max_abs_diff([t[1] for t in a["trace"]], [t[1] for t in b["trace"]]))
                hashes = a["sealed_sha256"] == b["sealed_sha256"]
                diff = abs(orig["validation"][-1]["val_nll"] - a["validation"][-1]["val_nll"])
                tol = tol_l
            res["retrain"][rid] = {"deterministic_max_abs_diff": det, "deterministic_sealed_hashes_equal": hashes,
                                   "deterministic_pass": det <= 1e-6 and hashes,
                                   "nondeterministic_abs_diff": diff, "tolerance_N_between_unit_sd": tol,
                                   "nondeterministic_pass": diff <= tol}
            print(rid, "deterministic", "PASS" if det <= 1e-6 and hashes else "FAIL",
                  "| non-deterministic", "PASS" if diff <= tol else "FAIL")
    for kind in ("encoder", "lm"):
        p = os.path.join(ROOT, "results", f"audit_evalreplay_{kind}.json")
        if os.path.exists(p):
            res["evalreplay"][kind] = load(p)
    res["audit_py_sha256"] = runlib.sha256_file(os.path.abspath(__file__))
    print(runlib.write_json_once(os.path.join(ROOT, "results", "audit.json"), res))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "choose":
        choose()
    elif cmd == "declare":
        declare()
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3])
    elif cmd == "evalreplay":
        evalreplay(sys.argv[2])
    elif cmd == "report":
        report()
    else:
        raise SystemExit("unknown command")
