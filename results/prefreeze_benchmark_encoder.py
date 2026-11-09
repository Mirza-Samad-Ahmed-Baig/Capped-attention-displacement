"""Pre-freeze re-benchmark with the production unified/ library on real PQA-L lengths.
Throughput only (no outcomes); random head, no evaluation, labels unused except as CE targets."""
import json, os, sys, time
sys.path.insert(0, r"C:\Attentiondrop__Attention_dropS_\unified")
import env
torch = env.setup_model_process(threads=2)
from transformers import AutoModel, AutoTokenizer
import attnreg, bert_adapter, pubmedqa_inputs as pi, training_utils as tu

cap = env.cap_vram(reserve_bytes=0)
MODEL = os.path.join(env.WORKSPACE, "domain_distill", "models", "biomedbert")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
rows = [json.loads(l) for l in open(os.path.join(env.ROOT, "data", "pqal_pool.jsonl"), encoding="utf-8")]
enc = [pi.encode(r, tok, limit=512, single_segment=True) for r in rows]
lengths = [len(e["input_ids"]) for e in enc]
priv = [pi.encode(r, tok, limit=512, privileged=True) for r in rows]
plen = [len(e["input_ids"]) for e in priv]
labels = torch.tensor([pi.label_index(r["label"]) for r in rows])
stats = {"n": len(lengths), "mean_len": sum(lengths) / len(lengths), "max_len": max(lengths),
         "truncated": sum(e["truncated"] for e in enc), "p90": sorted(lengths)[int(.9 * len(lengths))],
         "priv_mean_len": sum(plen) / len(plen), "priv_truncated": sum(e["truncated"] for e in priv)}
print("LENGTHS", json.dumps(stats), flush=True)

ARMS = [("sdpa_none", None), ("none", dict(kind="none")), ("dropout", dict(kind="dropout", p=0.1)),
        ("dropkey", dict(kind="dropkey", p=0.1)), ("relaxed", dict(kind="relaxed", gamma=0.1)),
        ("hard_budget", dict(kind="hard_budget", p=0.5, k=4, tau=0.1)),
        ("dropkey_budget", dict(kind="dropkey_budget", p=0.3, tau=0.1)),
        ("relaxed_budget", dict(kind="relaxed_budget", tau=0.1))]
extra = sys.argv[1:] or []
results = []
steps = 12
for name, spec in ARMS:
    for max_tokens in (4096, 2048):
        torch.manual_seed(0)
        try:
            model = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=False,
                                              add_pooling_layer=False, weights_only=True,
                                              attn_implementation="sdpa" if spec is None else "eager").cuda()
            if spec is not None:
                bert_adapter.install(model, lambda i, n, s=spec: attnreg.AttnReg(**s))
                noise = attnreg.NoiseSource(1); bert_adapter.set_noise(model, noise)
            head = torch.nn.Linear(768, 3).cuda()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            params = list(model.parameters()) + list(head.parameters())
            opt = torch.optim.AdamW(tu.parameter_groups(model, 0.01) + [{"params": head.parameters(), "weight_decay": 0.01}], lr=2e-5)
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0)).tolist()
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t0 = None; seen = 0
            for step in range(steps + 2):
                if step == 2:
                    torch.cuda.synchronize(); t0 = time.perf_counter(); seen = 0
                batch = order[(step * 16) % 480:(step * 16) % 480 + 16]
                if spec is not None:
                    noise.set_step(step)
                for mb in pi.microbatches(batch, lengths, max_tokens=max_tokens, max_size=16):
                    inputs = pi.collate(enc, mb, tok.pad_token_id, "cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        hs = model(**inputs).last_hidden_state[:, 0]
                        logits = head(hs.float())
                    loss = torch.nn.functional.cross_entropy(logits.float(), labels[mb].cuda(), reduction="sum") / len(batch)
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                seen += len(batch)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            r = {"arm": name, "max_tokens": max_tokens, "ex_per_s": seen / dt, "peak_mb": torch.cuda.max_memory_allocated() / 2**20,
                 "min_per_epoch_450": 450 / (seen / dt) / 60, "min_per_epoch_61k": 61249 / (seen / dt) / 60}
        except torch.OutOfMemoryError:
            r = {"arm": name, "max_tokens": max_tokens, "oom": True}
        print("RESULT", json.dumps(r), flush=True)
        results.append(r)
        del model, head, opt
        torch.cuda.empty_cache()
json.dump({"lengths": stats, "results": results, "cap_bytes": cap},
          open(os.path.join(os.path.dirname(__file__), "bench_arms.json"), "w"), indent=2)
