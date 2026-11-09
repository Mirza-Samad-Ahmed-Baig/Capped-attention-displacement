"""Persistent-worker runner for the encoder studies (S1, P1, E2, E5, and the E-S3 teachers).

A run is a JSON spec declared in the study registry before execution (runlib.Registry).
The worker loads the pinned encoder weights ONCE into CPU memory; for each run it rebuilds
the model, installs the attention arm (and optional Mixout / legacy LoRA), trains with the
declared recipe, evaluates the dev fold under five input conditions, and writes the same
conditions on the label-free official test inputs into the sealed folder (write-once).

Data order (PROTOCOL B14): `epochs` seeded permutations of the training ids are concatenated
into one stream and cut into batches of `batch` (the last batch may be smaller); the LR
schedule spans exactly that many optimizer steps. `seed` fixes head init, data order and the
counter-based perturbation noise, so arms sharing a seed share batches, head init and common
random numbers. The console shows loss health only; no dev or test metric is printed.
"""
import json
import math
import os
import sys
import time

import env

torch = env.setup_model_process(threads=2)
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import attnreg  # noqa: E402
import bert_adapter  # noqa: E402
import pubmedqa_inputs as pi  # noqa: E402
import runlib  # noqa: E402
import training_utils as tu  # noqa: E402

DEFAULT_MODEL = os.path.join(env.WORKSPACE, "domain_distill", "models", "biomedbert")
TEST_INPUTS = os.path.join(env.ROOT, "data", "sealed_inputs", "pqal_test_inputs.jsonl")
TEST_DONORS = os.path.join(env.ROOT, "data", "sealed_inputs", "test_donors.json")
POOL_SECTIONS = os.path.join(env.ROOT, "data", "pqal_pool_sections.json")
SOURCE_FILES = ["env.py", "attnreg.py", "bert_adapter.py", "pubmedqa_inputs.py", "runlib.py",
                "training_utils.py", "encoder_runner.py"]


def source_hashes():
    return {name: runlib.sha256_file(os.path.join(env.ROOT, name)) for name in SOURCE_FILES}


class Classifier(nn.Module):
    def __init__(self, encoder, classes=3, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state[:, 0]
        return self.head(self.dropout(h.float()))


class LoRALinear(nn.Module):
    """Legacy Track-2 adapter (L0): frozen base + fp32 low-rank update, B initialised to zero."""

    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.a), self.b) * self.scale).to(x.dtype)


def aio_load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stream_batches(n, epochs, batch, seed):
    gen = torch.Generator().manual_seed(seed + 10000)
    stream = []
    for _ in range(epochs):
        stream += torch.randperm(n, generator=gen).tolist()
    return [stream[i:i + batch] for i in range(0, len(stream), batch)]


class Worker:
    def __init__(self, model_dir=DEFAULT_MODEL):
        from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
        self.BertConfig, self.BertModel = BertConfig, BertModel
        self.model_dir = model_dir
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        base = AutoModel.from_pretrained(model_dir, local_files_only=True, use_safetensors=False, weights_only=True,
                                         add_pooling_layer=False, attn_implementation="eager")
        self.config_dict = base.config.to_dict()
        self.base_state = {k: v.detach().clone().float() for k, v in base.state_dict().items()}
        weights = os.path.join(model_dir, "pytorch_model.bin")
        self.base_sha256 = runlib.sha256_file(weights)
        del base
        self.records, self.cache = {}, {}
        self.source_sha256 = source_hashes()  # hashed once at process start: the code actually executed
        self.test_sets = {}
        self.test_rows, self.test_donors = self.test_set(TEST_INPUTS, TEST_DONORS)
        with open(POOL_SECTIONS, encoding="utf-8") as f:
            self.pool_sections = json.load(f)
        self.vram_cap = env.cap_vram()

    def test_set(self, inputs_path, donors_path):
        key = (inputs_path, donors_path)
        if key not in self.test_sets:
            with open(inputs_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            with open(donors_path, encoding="utf-8") as f:
                donors = json.load(f)
            assert donors["order"] == [r["id"] for r in rows]
            self.test_sets[key] = (rows, donors["donors"])
        return self.test_sets[key]

    def load_records(self, path):
        if path not in self.records:
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            for r in rows:
                if r["id"] in self.pool_sections:
                    r["section_labels"] = self.pool_sections[r["id"]]
            self.records[path] = {r["id"]: r for r in rows}
        return self.records[path]

    def encode(self, pairs, mode, **kw):
        key = (mode, tuple(sorted(kw.items())))
        cache = self.cache.setdefault(key, {})
        out = []
        for row, donor in pairs:
            ck = (row["id"], donor["id"] if donor is not None else None)
            if ck not in cache:
                cache[ck] = pi.encode(row, self.tok, mode=mode, donor=donor, **kw)
            out.append(cache[ck])
        return out

    # ------------------------------------------------------------------------------------
    def build(self, spec):
        recipe, arm = spec["recipe"], spec["arm"]
        impl = recipe.get("attn_impl", "eager")
        if impl == "sdpa" and arm["kind"] != "none":
            raise ValueError("SDPA path is only allowed for attention kind 'none'")
        cfg = dict(self.config_dict)
        cfg["hidden_dropout_prob"] = recipe.get("hidden_dropout", 0.1)
        cfg["attention_probs_dropout_prob"] = recipe.get("sdpa_attention_dropout", 0.0) if impl == "sdpa" else 0.0
        config = self.BertConfig(**cfg, attn_implementation=impl)
        torch.manual_seed(spec["seed"])
        encoder = self.BertModel(config, add_pooling_layer=False)
        missing, unexpected = encoder.load_state_dict(self.base_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"state mismatch: missing={missing} unexpected={unexpected}")
        keep = recipe.get("keep_layers")
        if keep:
            encoder.encoder.layer = nn.ModuleList([encoder.encoder.layer[i] for i in keep])
            encoder.config.num_hidden_layers = len(keep)
        for layer in (encoder.encoder.layer[-recipe["reinit_top"]:] if recipe.get("reinit_top") else []):
            layer.apply(encoder._init_weights)
        model = Classifier(encoder, dropout=recipe.get("head_dropout", 0.1))
        g = torch.Generator().manual_seed(spec["seed"])
        with torch.no_grad():
            model.head.weight.copy_(torch.randn(model.head.weight.shape, generator=g) * 0.02)
            model.head.bias.zero_()
        init = recipe.get("init_checkpoint")
        if init:
            if runlib.sha256_file(init["path"]) != init["sha256"]:
                raise RuntimeError("initial checkpoint hash mismatch")
            model.load_state_dict(torch.load(init["path"], map_location="cpu", weights_only=True), strict=True)
        regs = []
        if impl == "eager":
            def factory(i, n):
                p = arm.get("p", 0.1)
                if arm.get("layer_schedule") == "dropkey_linear":
                    p = attnreg.depth_schedule(p, i, n)
                return attnreg.AttnReg(arm["kind"], p=p, k=arm.get("k", 4), tau=arm.get("tau", 0.1),
                                       gamma=arm.get("gamma", 0.1), record=True)
            regs = bert_adapter.install(encoder, factory)
        noise = attnreg.NoiseSource(spec["seed"])
        for reg in regs:
            reg.noise = noise
        if recipe.get("mixout_p"):
            tu.apply_mixout(encoder, recipe["mixout_p"], include=lambda name: name.startswith("encoder."))
        lora = recipe.get("lora")
        if lora:
            for p in encoder.parameters():
                p.requires_grad_(False)
            encoder.to(torch.bfloat16)
            for layer in encoder.encoder.layer:
                att = layer.attention.self
                att.query = LoRALinear(att.query, lora["r"], lora["alpha"])
                att.value = LoRALinear(att.value, lora["r"], lora["alpha"])
        if recipe.get("grad_ckpt", True):
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model.cuda(), noise, regs

    def predict(self, model, encoded, max_tokens):
        model.eval()
        torch.use_deterministic_algorithms(True)
        lengths = [len(e["input_ids"]) for e in encoded]
        logits = [None] * len(encoded)
        with torch.no_grad():
            for mb in pi.microbatches(list(range(len(encoded))), lengths, max_tokens=max_tokens, max_size=32):
                inputs = pi.collate(encoded, mb, self.tok.pad_token_id, "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inputs).float().cpu()
                for j, i in enumerate(mb):
                    logits[i] = [round(v, 7) for v in out[j].tolist()]
        torch.use_deterministic_algorithms(False)
        return logits

    def conditions(self, model, rows, donors_list, enc_kw, max_tokens):
        out = {"original": self.predict(model, self.encode([(r, None) for r in rows], "original", **enc_kw), max_tokens),
               "question_only": self.predict(model, self.encode([(r, None) for r in rows], "question_only", **enc_kw), max_tokens),
               "results_deleted": self.predict(model, self.encode([(r, None) for r in rows], "results_deleted", **enc_kw), max_tokens)}
        for d, donors in enumerate(donors_list):
            out[f"mismatched_{d}"] = self.predict(model, self.encode(list(zip(rows, donors)), "mismatched", **enc_kw), max_tokens)
        return out

    # ------------------------------------------------------------------------------------
    def run(self, spec, out_path, sealed_dir):
        t_start = time.perf_counter()
        recipe = spec["recipe"]
        pool = self.load_records(spec["pool"])
        train_rows = [pool[i] for i in spec["train_ids"]]
        eval_pool = self.load_records(spec.get("eval_pool", spec["pool"]))
        eval_rows = [eval_pool[i] for i in spec["eval_ids"]]
        enc_kw = dict(limit=recipe.get("max_len", 512), single_segment=recipe.get("single_segment", True))
        train_mode = recipe.get("train_mode", "original")
        train_enc = self.encode([(r, None) for r in train_rows], train_mode,
                                privileged=recipe.get("privileged", False), **enc_kw)
        # Knowledge distillation (E-S3): soft targets from a teacher run's exported logits.
        kd = recipe.get("soft_targets")
        teacher = None
        if kd:
            src = aio_load(kd["teacher_result"])
            if runlib.sha256_file(kd["teacher_result"]) != kd["teacher_result_sha256"]:
                raise RuntimeError("teacher result hash mismatch")
            exp = src["exported_logits"][kd["export"]]
            by_id = dict(zip(exp["ids"], exp["logits"]))
            teacher = torch.tensor([by_id[r["id"]] for r in train_rows], dtype=torch.float32)
        ce_weight = recipe.get("ce_weight", 1.0)
        kd_weight = kd["weight"] if kd else 0.0
        if ce_weight > 0:
            labels = torch.tensor([pi.label_index(r["label"]) for r in train_rows])
        else:
            labels = torch.zeros(len(train_rows), dtype=torch.long)  # unlabeled transfer set: CE disabled
        lengths = [len(e["input_ids"]) for e in train_enc]
        model, noise, regs = self.build(spec)
        params = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": [p for p in g["params"] if p.requires_grad], "weight_decay": g["weight_decay"]}
                  for g in tu.parameter_groups(model, recipe.get("weight_decay", 0.01))]
        if recipe.get("optimizer", "adamw") == "bertadam":
            opt = tu.BertAdam(groups, lr=recipe["lr"], eps=1e-6)
        else:
            opt = torch.optim.AdamW(groups, lr=recipe["lr"], betas=(0.9, 0.999), eps=1e-6)
        batches = stream_batches(len(train_rows), recipe["epochs"], recipe.get("batch", 16), spec["seed"])
        sched = tu.linear_warmup_decay(opt, len(batches), recipe.get("warmup", 0.1))
        max_tokens = recipe.get("max_tokens", 2048)
        rdrop, smoothing = recipe.get("rdrop_alpha", 0.0), recipe.get("label_smoothing", 0.0)
        per_epoch = math.ceil(len(batches) / recipe["epochs"])
        trace, budget_trace = [], []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_train = time.perf_counter()
        model.train()
        pending = []  # (step, loss tensor, grad-norm tensor): flushed every 25 steps to avoid per-step GPU syncs

        def flush():
            if not pending:
                return
            vals = torch.stack([torch.stack([l, g]) for _, l, g in pending]).float().cpu().tolist()
            for (s, _, _), (l, g) in zip(pending, vals):
                if not (math.isfinite(l) and math.isfinite(g)):
                    raise FloatingPointError(f"non-finite loss or gradient at step {s}")
                trace.append([s, round(l, 6), round(g, 6)])
            pending.clear()

        for step, idx in enumerate(batches):
            loss_sum = torch.zeros((), device="cuda")
            budget = max_tokens // 2 if rdrop else max_tokens
            for micro, mb in enumerate(pi.microbatches(idx, lengths, max_tokens=budget, max_size=len(idx))):
                inputs = pi.collate(train_enc, mb, self.tok.pad_token_id, "cuda")
                y = labels[mb].cuda()
                if rdrop:
                    inputs = {k: torch.cat([v, v]) for k, v in inputs.items()}
                noise.set_step(step, 0, micro)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**inputs)
                if rdrop:
                    loss, _, _ = tu.rdrop_loss(logits[:len(mb)], logits[len(mb):], y, rdrop)
                    loss = loss * len(mb) / len(idx)
                else:
                    loss = logits.new_zeros((), dtype=torch.float32)
                    if ce_weight > 0:
                        loss = loss + ce_weight * F.cross_entropy(logits.float(), y, reduction="sum",
                                                                  label_smoothing=smoothing) / len(idx)
                    if kd_weight > 0:
                        loss = loss + kd_weight * tu.kd_loss(logits, teacher[mb].cuda(), kd["temperature"]) * len(mb) / len(idx)
                loss.backward()
                loss_sum = loss_sum + loss.detach()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, recipe.get("clip", 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            pending.append((step, loss_sum, grad_norm.detach()))
            if len(pending) >= 25 or step == len(batches) - 1:
                flush()
            if regs and ((step + 1) % per_epoch == 0 or step == len(batches) - 1):
                budget_trace.append({"step": step + 1, "layers": [r.stats() for r in regs]})
                for r in regs:
                    r.reset_stats()
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - t_train
        eval_kw = dict(enc_kw, privileged=False)
        diag = spec.get("diagnostics", {"derangements": 3})
        eval_labels = [pi.label_index(r["label"]) for r in eval_rows]
        ev_len = [len(" ".join(r["contexts"])) for r in eval_rows]
        dev_donors = [[eval_rows[j] for j in pi.stratified_derangement(eval_labels, ev_len, seed=d + 1)]
                      for d in range(diag.get("derangements", 3))]
        dev = self.conditions(model, eval_rows, dev_donors, eval_kw, max_tokens * 2)
        test_hash = None
        if spec.get("evaluate_test", True):
            test_rows, donor_lists = self.test_set(spec.get("test_inputs", TEST_INPUTS), spec.get("test_donors", TEST_DONORS))
            by_id = {r["id"]: r for r in test_rows}
            test_donors = [[by_id[i] for i in lst] for lst in donor_lists[:diag.get("derangements", 3)]]
            test = self.conditions(model, test_rows, test_donors, eval_kw, max_tokens * 2)
            test_hash = runlib.write_json_once(os.path.join(sealed_dir, f"{spec['run_id']}.json"),
                                               {"run_id": spec["run_id"], "ids": [r["id"] for r in test_rows],
                                                "logits": test})
        exported = {}
        for name, ref in spec.get("export_logits", {}).items():
            src = self.load_records(ref["pool"])
            rows = [src[i] for i in ref["ids"]]
            enc = self.encode([(r, None) for r in rows], "original", **dict(enc_kw, privileged=ref.get("privileged", False)))
            exported[name] = {"ids": ref["ids"], "logits": self.predict(model, enc, max_tokens * 2)}
        checkpoint = None
        if spec.get("save_checkpoint"):
            path = spec["save_checkpoint"]
            state = {k: v.detach().float().cpu() for k, v in model.state_dict().items() if not k.endswith(".target")}
            torch.save(state, path + ".partial")
            os.replace(path + ".partial", path)
            checkpoint = {"path": path, "sha256": runlib.sha256_file(path)}
        result = {"spec": spec, "spec_sha256": runlib.sha256_json(spec), "utc": runlib.utc(),
                  "source_sha256": self.source_sha256, "base_weights_sha256": self.base_sha256,
                  "steps": len(batches), "train_seconds": train_seconds, "total_seconds": time.perf_counter() - t_start,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "vram_cap_bytes": self.vram_cap,
                  "trace": trace, "budget_trace": budget_trace, "eval_ids": spec["eval_ids"],
                  "eval_labels": eval_labels, "dev": dev,
                  "dev_donors": [[r["id"] for r in d] for d in dev_donors], "test_sealed_sha256": test_hash,
                  "exported_logits": exported, "checkpoint": checkpoint,
                  "trainable_parameters": sum(p.numel() for p in params),
                  "total_parameters": sum(p.numel() for p in model.parameters())}
        runlib.write_json_once(out_path, result)
        del model, opt
        torch.cuda.empty_cache()
        return result


def validate_spec(spec):
    required = {"run_id", "study", "seed", "arm", "recipe", "pool", "train_ids", "eval_ids"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"spec missing {sorted(missing)}")
    if set(spec["train_ids"]) & set(spec["eval_ids"]):
        raise ValueError("train and eval ids overlap")
    if spec["arm"]["kind"] not in attnreg.KINDS:
        raise ValueError("unknown arm kind")
    for key in ("lr", "epochs"):
        if key not in spec["recipe"]:
            raise ValueError(f"recipe missing {key}")


def main():
    """Usage: encoder_runner.py STUDY_FOLDER [MODEL_DIR]  (executes pending registry runs in order)."""
    folder = os.path.abspath(sys.argv[1])
    model_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    registry = runlib.Registry(folder)
    ledger = runlib.Ledger(os.path.join(folder, "ledger.jsonl"))
    pending = registry.pending()
    if not pending:
        print("nothing pending", flush=True)
        return
    for spec in pending:
        validate_spec(spec)
    import shutil
    if shutil.disk_usage(folder).free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free disk")
    worker = Worker(model_dir)
    sealed = os.path.join(folder, "sealed")
    with runlib.GpuLogger(os.path.join(folder, "gpu_log.csv")):
        for spec in pending:
            if os.path.exists(os.path.join(env.ROOT, "STOP")):
                print(f"[{runlib.utc()}] STOP file present: exiting cleanly before {spec['run_id']}", flush=True)
                break
            t0 = time.perf_counter()
            print(f"[{runlib.utc()}] start {spec['run_id']}", flush=True)
            result = worker.run(spec, registry.result_path(spec["run_id"]), sealed)
            ledger.add(spec["run_id"], time.perf_counter() - t0)
            print(f"[{runlib.utc()}] done {spec['run_id']} steps={result['steps']} final_train_loss="
                  f"{result['trace'][-1][1]:.4f} train_s={result['train_seconds']:.0f}", flush=True)


if __name__ == "__main__":
    main()
