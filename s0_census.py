"""S0 census and calibration (PROTOCOL A §3.1 item 4; B2, B3). Eval mode only; no outcomes.

Subcommands:
  splits            PQA-U minus the 2 decontamination exclusions, ranked by
                    sha256("unified-u-v1:" + pmid): first 256 = census inputs, next 10,000 =
                    KD subset (E-S3). Writes data/pqau_splits.json (write-once).
  encoder           pretrained BiomedBERT census (RowSampler, 2,048 rows/layer) + value samples
                    on the 256 census inputs -> results/s0/encoder_census.json, encoder_values.pt
  lm RUN_JSON CKPT  pools the 10/25/50/100% census snapshots of the winning N base-LR run and
                    collects value samples from its saved checkpoint on 64 train windows
                    -> results/s0/lm_census.json, lm_values.pt
  predict           calibrate both regimes, value analysis, registered predictions
                    -> results/predictions.json (write-once; its hash is recorded)
"""
import hashlib
import json
import os
import sys

import env

torch = env.setup_model_process(threads=2)
import numpy as np  # noqa: E402

import attnreg  # noqa: E402
import calibrate  # noqa: E402
import census  # noqa: E402
import census_output  # noqa: E402
import runlib  # noqa: E402

DATA = os.path.join(env.ROOT, "data")
OUT = os.path.join(env.ROOT, "results", "s0")
MODEL = os.path.join(env.WORKSPACE, "domain_distill", "models", "biomedbert")


def splits():
    path = os.path.join(DATA, "pqau_splits.json")
    excluded = set(json.load(open(os.path.join(DATA, "decontamination.json"), encoding="utf-8"))
                   ["subsets"]["pqa_unlabeled"]["excluded_ids"])
    ids = []
    with open(os.path.join(DATA, "pqa_unlabeled.jsonl"), encoding="utf-8") as f:
        for line in f:
            pid = json.loads(line)["id"]
            if pid not in excluded:
                ids.append(pid)
    ids.sort(key=lambda p: hashlib.sha256(f"unified-u-v1:{p}".encode()).hexdigest())
    rec = {"rule": "sha256('unified-u-v1:'+pmid) order, 2 decontamination exclusions removed",
           "census": ids[:256], "kd": ids[256:10256], "excluded": sorted(excluded)}
    print(runlib.write_json_once(path, rec))


def encoder(model_dir=MODEL, name="encoder", device="cuda"):
    import pubmedqa_inputs as pi
    import bert_adapter
    from transformers import AutoModel, AutoTokenizer
    sp = json.load(open(os.path.join(DATA, "pqau_splits.json"), encoding="utf-8"))
    want = set(sp["census"])
    rows = {}
    with open(os.path.join(DATA, "pqa_unlabeled.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                rows[r["id"]] = r
    rows = [rows[i] for i in sp["census"]]
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    enc = [pi.encode(r, tok, limit=512, single_segment=True) for r in rows]
    if device == "cuda":
        env.cap_vram()
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, use_safetensors=False, weights_only=True,
                                      add_pooling_layer=False, attn_implementation="eager").to(device).eval()
    regs = bert_adapter.install(model, lambda i, n: attnreg.AttnReg("none"))
    rs = census.RowSampler(max_rows_per_layer=2048, rows_per_call=64, seed=1)
    vs = census_output.ValueSampler(rows_per_layer=256, seed=2)
    for reg in regs:
        reg.observer, reg.value_observer = rs, vs
    with torch.no_grad():
        for s in range(0, len(enc), 8):
            idx = list(range(s, min(len(enc), s + 8)))
            batch = pi.collate(enc, idx, tok.pad_token_id, device)
            vs.special = (batch["input_ids"] == tok.cls_token_id) | (batch["input_ids"] == tok.sep_token_id)
            with torch.autocast(device, dtype=torch.bfloat16):
                model(**batch)
    os.makedirs(OUT, exist_ok=True)
    data = {str(l): {k: v.tolist() for k, v in rs.tensors(l).items()} for l in rs.data}
    print(runlib.write_json_once(os.path.join(OUT, f"{name}_census.json"), data))
    torch.save({"data": vs.data, "num_layers": len(regs)}, os.path.join(OUT, f"{name}_values.pt"))


def student(run_json, out_name="student"):
    """Census of the fold-0 phase-K student (E-S3): its saved checkpoint, 6 retained layers, eval mode,
    the same 256 census inputs; writes results/s0/student_census.json and results/predictions_student.json."""
    import pubmedqa_inputs as pi
    import bert_adapter
    from torch import nn
    from transformers import AutoModel, AutoTokenizer
    run = json.load(open(run_json, encoding="utf-8"))
    ck = run["checkpoint"]
    if runlib.sha256_file(ck["path"]) != ck["sha256"]:
        raise RuntimeError("student checkpoint hash mismatch")
    keep = run["spec"]["recipe"]["keep_layers"]
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=False, weights_only=True,
                                      add_pooling_layer=False, attn_implementation="eager")
    model.encoder.layer = nn.ModuleList([model.encoder.layer[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    state = torch.load(ck["path"], map_location="cpu", weights_only=True)
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"student state mismatch {missing} {unexpected}")
    sp = json.load(open(os.path.join(DATA, "pqau_splits.json"), encoding="utf-8"))
    want = set(sp["census"])
    rows = {}
    with open(os.path.join(DATA, "pqa_unlabeled.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                rows[r["id"]] = r
    enc = [pi.encode(rows[i], tok, limit=512, single_segment=True) for i in sp["census"]]
    env.cap_vram()
    model = model.cuda().eval()
    regs = bert_adapter.install(model, lambda i, n: attnreg.AttnReg("none"))
    rs = census.RowSampler(max_rows_per_layer=2048, rows_per_call=64, seed=1)
    for reg in regs:
        reg.observer = rs
    with torch.no_grad():
        for s in range(0, len(enc), 8):
            batch = pi.collate(enc, list(range(s, min(len(enc), s + 8))), tok.pad_token_id, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(**batch)
    data = {str(l): {k: v.tolist() for k, v in rs.tensors(l).items()} for l in rs.data}
    path = os.path.join(OUT, f"{out_name}_census.json")
    print(runlib.write_json_once(path, data))
    print(runlib.write_json_once(os.path.join(env.ROOT, "results", f"predictions_{out_name}.json"),
                                 calibrate.calibrate(calibrate.load(path))))


def calibrate_only(census_name, out_name):
    """Calibration of an existing census file (used for the E5 BioLinkBERT census)."""
    path = os.path.join(OUT, f"{census_name}_census.json")
    print(runlib.write_json_once(os.path.join(env.ROOT, "results", f"predictions_{out_name}.json"),
                                 calibrate.calibrate(calibrate.load(path))))


def lm(run_json, ckpt):
    import lm_model
    run = json.load(open(run_json, encoding="utf-8"))
    folder = os.path.dirname(os.path.dirname(run_json))
    pooled = {}
    for step, digest in run["census_files"].items():
        path = os.path.join(folder, "census", f"{run['spec']['run_id']}_step{step}.json")
        if runlib.sha256_file(path) != digest:
            raise RuntimeError("census snapshot hash mismatch")
        snap = json.load(open(path, encoding="utf-8"))
        for layer, rec in snap.items():
            dst = pooled.setdefault(layer, {k: [] for k in rec})
            for k, v in rec.items():
                dst[k] += v
    os.makedirs(OUT, exist_ok=True)
    print(runlib.write_json_once(os.path.join(OUT, "lm_census.json"), pooled))
    r = run["spec"]["recipe"]
    if runlib.sha256_file(ckpt) != run["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    model = lm_model.LM(r["vocab"], width=r["width"], layers=r["layers"], heads=r["heads"], context=r["context"],
                        dropout=r.get("dropout", 0.1))
    model.load_state_dict({k: v.float() for k, v in torch.load(ckpt, map_location="cpu", weights_only=True).items()})
    env.cap_vram()
    model = model.cuda().eval()
    vs = census_output.ValueSampler(rows_per_layer=256, seed=3)
    for reg in model.regularizers():
        reg.value_observer = vs
    train = np.memmap(os.path.join(DATA, "lm", "train.bin"), dtype=np.uint16, mode="r")
    prefix = np.asarray(train[:run["prefix_tokens"]])
    gen = torch.Generator().manual_seed(4)
    starts = torch.randint(0, len(prefix) - r["context"] - 1, (64,), generator=gen).numpy()
    x = torch.from_numpy(np.stack([prefix[s:s + r["context"]] for s in starts]).astype(np.int64)).cuda()
    with torch.no_grad():
        for chunk in x.split(8):
            vs.special = (chunk == 0)  # EOT id 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.hidden(chunk)
    torch.save({"data": vs.data, "num_layers": r["layers"]}, os.path.join(OUT, "lm_values.pt"))


def predict():
    pred = calibrate.predictions(os.path.join(OUT, "encoder_census.json"), os.path.join(OUT, "lm_census.json"))
    for regime in ("encoder", "lm"):
        vals = torch.load(os.path.join(OUT, f"{regime}_values.pt"), weights_only=False)
        vs = census_output.ValueSampler()
        vs.data = vals["data"]
        pred[regime]["value_census_l0.10"] = census_output.analyse(vs, pred[regime]["levels"]["0.10"], vals["num_layers"])
    path = os.path.join(env.ROOT, "results", "predictions.json")
    digest = runlib.write_json_once(path, pred)
    print(json.dumps(pred["registered"], indent=1))
    print("predictions.json", digest)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "encoder_model":
        encoder(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "student":
        student(sys.argv[2])
    elif cmd == "calibrate_only":
        calibrate_only(sys.argv[2], sys.argv[3])
    else:
        {"splits": splits, "encoder": encoder, "predict": predict}.get(cmd, lambda: lm(sys.argv[2], sys.argv[3]))()
