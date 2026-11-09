"""Runner for the data-limited from-scratch biomedical LM regime (S2, E6, E8).

Data (PROTOCOL Part A §3.3 and B14): LM-train = the shortest prefix of data/lm/train.bin
(documents in stored hash order) with >= `unique_tokens` tokens. For epoch e a random offset
o_e in [0, context) is drawn and non-overlapping (context+1)-token windows starting at o_e are
shuffled with a CPU generator (seed + 10000 + e); epochs are concatenated and truncated to
steps x batch windows. Arms with the same seed see identical windows; perturbation noise is
counter-based (common random numbers). Validation NLL on data/lm/val.bin (disjoint blocks,
every token scored) at every `eval_every` steps and at the final step; best-validation
weights are kept in CPU RAM. The sealed test is scored (sealed_eval.lm_test) at the final and
best-validation weights; only hashes are recorded.
"""
import copy
import json
import math
import os
import sys
import time

import env

torch = env.setup_model_process(threads=2)
import numpy as np  # noqa: E402

import attnreg  # noqa: E402
import census  # noqa: E402
import lm_model  # noqa: E402
import runlib  # noqa: E402
import sealed_eval  # noqa: E402

LM_DIR = os.path.join(env.ROOT, "data", "lm")
# Arms whose training step exceeds the 3.2 GiB of VRAM free on this GPU run with per-block activation
# checkpointing (FREEZE_ADDENDUM 2026-09-29). Probed on random tokens: loss and every gradient are
# bit-identical with and without it (RNG state preserved, counter-based noise), so only memory changes.
CHECKPOINT_KINDS = ("relaxed_budget", "dropkey_budget")


def checkpoint_blocks(model):
    from torch.utils.checkpoint import checkpoint
    for blk in model.blocks:
        fn = blk.forward
        blk.forward = (lambda f: (lambda x: checkpoint(f, x, use_reentrant=False) if torch.is_grad_enabled() else f(x)))(fn)
SOURCE_FILES = ["env.py", "attnreg.py", "census.py", "lm_model.py", "runlib.py", "sealed_eval.py", "lm_runner.py"]


def source_hashes():
    return {name: runlib.sha256_file(os.path.join(env.ROOT, name)) for name in SOURCE_FILES}


def prefix_end(unique_tokens):
    with open(os.path.join(LM_DIR, "train_docs.json"), encoding="utf-8") as f:
        docs = json.load(f)
    for pid, start, length in docs:
        if start + length >= unique_tokens:
            return start + length
    raise ValueError("unique_tokens exceeds the train stream")


def epoch_windows(n_tokens, context, batch, steps, seed):
    windows = []
    e = 0
    while len(windows) < steps * batch:
        gen = torch.Generator().manual_seed(seed + 10000 + e)
        offset = int(torch.randint(0, context, (1,), generator=gen))
        starts = np.arange(offset, n_tokens - context - 1 + 1, context)
        perm = torch.randperm(len(starts), generator=gen).numpy()
        windows.extend(starts[perm].tolist())
        e += 1
    return np.asarray(windows[: steps * batch], dtype=np.int64).reshape(steps, batch), e


class Worker:
    def __init__(self):
        self.train = np.memmap(os.path.join(LM_DIR, "train.bin"), dtype=np.uint16, mode="r")
        self.val = np.asarray(np.memmap(os.path.join(LM_DIR, "val.bin"), dtype=np.uint16, mode="r"))
        self.vram_cap = env.cap_vram()
        self.source_sha256 = source_hashes()  # hashed once at process start

    def evaluate(self, model, tokens, context, batch=16):
        model.eval()
        n_blocks = (len(tokens) - 1) // context
        total, count = 0.0, 0
        with torch.no_grad():
            for s in range(0, n_blocks, batch):
                idx = range(s, min(n_blocks, s + batch))
                x = torch.stack([torch.from_numpy(tokens[i * context:(i + 1) * context].astype(np.int64)) for i in idx])
                y = torch.stack([torch.from_numpy(tokens[i * context + 1:(i + 1) * context + 1].astype(np.int64)) for i in idx])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model.loss(x.cuda(), y.cuda())
                total += float(loss) * y.numel()
                count += y.numel()
        model.train()
        return total / count

    def census_snapshot(self, model, prefix, context, gen_seed, windows=64):
        sampler = census.RowSampler(max_rows_per_layer=2048, rows_per_call=2048, seed=gen_seed)
        for reg in model.regularizers():
            reg.observer = sampler
        gen = torch.Generator().manual_seed(gen_seed)
        starts = torch.randint(0, len(prefix) - context - 1, (windows,), generator=gen).numpy()
        x = torch.from_numpy(np.stack([prefix[s:s + context] for s in starts]).astype(np.int64)).cuda()
        model.eval()
        with torch.no_grad():
            for chunk in x.split(16):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model.hidden(chunk)
        model.train()
        for reg in model.regularizers():
            reg.observer = None
        return {str(layer): {k: v.tolist() for k, v in sampler.tensors(layer).items()} for layer in sampler.data}

    def run(self, spec, out_path, sealed_dir, census_dir):
        t_start = time.perf_counter()
        r, arm = spec["recipe"], spec["arm"]
        end = prefix_end(int(r["unique_tokens"]))
        prefix = np.asarray(self.train[:end])
        counts = np.bincount(prefix.astype(np.int64), minlength=r["vocab"])
        val = self.val[: int(r.get("val_tokens", len(self.val)))]
        ctx, batch, steps = r["context"], r["batch"], int(r["steps"])
        windows, epochs_used = epoch_windows(len(prefix), ctx, batch, steps, spec["seed"])
        torch.manual_seed(spec["seed"])

        def factory(i, n):
            p = arm.get("p", 0.1)
            if arm.get("layer_schedule") == "dropkey_linear":  # as in encoder_runner (fixed 2026-09-29)
                p = attnreg.depth_schedule(p, i, n)
            return attnreg.AttnReg(arm["kind"], p=p, k=arm.get("k", 4), tau=arm.get("tau", 0.1),
                                   gamma=arm.get("gamma", 0.1), record=True)
        model = lm_model.LM(r["vocab"], width=r["width"], layers=r["layers"], heads=r["heads"], context=ctx,
                            reg_factory=factory, dropout=r.get("dropout", 0.1)).cuda()
        if arm["kind"] in CHECKPOINT_KINDS:
            checkpoint_blocks(model)
        init_sha = runlib.sha256_json({k: float(v.double().sum()) for k, v in model.state_dict().items()})
        noise = attnreg.NoiseSource(spec["seed"])
        model.set_noise(noise)
        decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and not n.startswith(("tok", "pos"))]
        no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and not n.startswith(("tok", "pos")))]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": r["weight_decay"]},
                                 {"params": no_decay, "weight_decay": 0.0}],
                                lr=r["lr"], betas=tuple(r.get("betas", (0.9, 0.95))), eps=1e-8)
        warm, floor = int(r["warmup_steps"]), r.get("min_lr_fraction", 0.1)

        def lr_factor(step):
            if step < warm:
                return (step + 1) / warm
            t = (step - warm) / max(1, steps - warm)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
        eval_every = int(r.get("eval_every", 98))
        census_at = sorted({max(1, int(round(f * steps))) for f in spec.get("census_at", [])})
        trace, evals, budget_trace, census_files = [], [], [], {}
        best = {"val_nll": float("inf"), "step": None, "state": None}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_train, side_seconds = time.perf_counter(), 0.0
        model.train()
        first_batch_sha = runlib.sha256_json(windows[0].tolist())
        for step in range(steps):
            w = torch.from_numpy(np.stack([prefix[s:s + ctx + 1] for s in windows[step]]).astype(np.int64)).cuda()
            noise.set_step(step, 0, 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(w[:, :-1], w[:, 1:])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), r.get("clip", 1.0)))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if step % 25 == 0 or step == steps - 1:
                trace.append([step, round(float(loss.detach()), 6), round(grad_norm, 6)])
            done = step + 1
            if done % eval_every == 0 or done == steps:
                ts = time.perf_counter()
                v = self.evaluate(model, val, ctx)
                evals.append({"step": done, "val_nll": v})
                budget_trace.append({"step": done, "layers": [reg.stats() for reg in model.regularizers()]})
                for reg in model.regularizers():
                    reg.reset_stats()
                if v < best["val_nll"]:
                    best = {"val_nll": v, "step": done,
                            "state": {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}}
                side_seconds += time.perf_counter() - ts
            if done in census_at:
                ts = time.perf_counter()
                path = os.path.join(census_dir, f"{spec['run_id']}_step{done}.json")
                census_files[str(done)] = runlib.write_json_once(path, self.census_snapshot(model, prefix, ctx, spec["seed"] + done))
                side_seconds += time.perf_counter() - ts
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - t_train - side_seconds
        sealed = {}
        stream = spec.get("test_stream", "test")
        if spec.get("evaluate_test", True):
            sealed["final"] = sealed_eval.lm_test(model, ctx, counts, os.path.join(sealed_dir, f"{spec['run_id']}_final.json"),
                                                  "final", stream=stream)
            final_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            model.load_state_dict(best["state"])
            sealed["best"] = sealed_eval.lm_test(model, ctx, counts, os.path.join(sealed_dir, f"{spec['run_id']}_best.json"),
                                                 "best_validation", stream=stream)
            model.load_state_dict(final_state)
        checkpoint = None
        if spec.get("save_checkpoint"):
            path = spec["save_checkpoint"]
            torch.save({k: v.detach().half().cpu() for k, v in model.state_dict().items()}, path + ".partial")
            os.replace(path + ".partial", path)
            checkpoint = {"path": path, "sha256": runlib.sha256_file(path), "dtype": "float16"}
        result = {"spec": spec, "spec_sha256": runlib.sha256_json(spec), "utc": runlib.utc(),
                  "source_sha256": self.source_sha256,
                  "data_sha256": {s: runlib.sha256_file(os.path.join(LM_DIR, f"{s}.bin")) for s in ("train", "val")},
                  "prefix_tokens": int(end), "epochs_used": epochs_used, "initial_state_sha": init_sha,
                  "first_batch_sha": first_batch_sha,
                  "parameters": lm_model.parameter_count(model),
                  "non_embedding_parameters": lm_model.parameter_count(model, include_embeddings=False),
                  "train_seconds": train_seconds, "side_seconds": side_seconds,
                  "total_seconds": time.perf_counter() - t_start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "vram_cap_bytes": self.vram_cap, "trace": trace, "validation": evals,
                  "best_validation_step": best["step"], "budget_trace": budget_trace,
                  "census_files": census_files, "sealed_sha256": sealed, "checkpoint": checkpoint,
                  "tokens_seen": steps * batch * ctx}
        runlib.write_json_once(out_path, result)
        del model, opt, best
        torch.cuda.empty_cache()
        return result


def main():
    folder = os.path.abspath(sys.argv[1])
    registry = runlib.Registry(folder)
    ledger = runlib.Ledger(os.path.join(folder, "ledger.jsonl"))
    pending = registry.pending()
    if not pending:
        print("nothing pending", flush=True)
        return
    import shutil
    if shutil.disk_usage(folder).free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free disk")
    worker = Worker()
    sealed, census_dir = os.path.join(folder, "sealed"), os.path.join(folder, "census")
    with runlib.GpuLogger(os.path.join(folder, "gpu_log.csv")):
        for spec in pending:
            if os.path.exists(os.path.join(env.ROOT, "STOP")):
                print(f"[{runlib.utc()}] STOP file present: exiting cleanly before {spec['run_id']}", flush=True)
                break
            t0 = time.perf_counter()
            print(f"[{runlib.utc()}] start {spec['run_id']}", flush=True)
            result = worker.run(spec, registry.result_path(spec["run_id"]), sealed, census_dir)
            ledger.add(spec["run_id"], time.perf_counter() - t0)
            print(f"[{runlib.utc()}] done {spec['run_id']} train_s={result['train_seconds']:.0f} "
                  f"final_train_loss={result['trace'][-1][1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
