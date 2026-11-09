"""Gate G0 (PROTOCOL A §3.1 item 1): engineering equivalence checks before the freeze.

1. Track-1 reproduction: recompute exact E[m], E[min(m, tau)], P(m > tau) for all 20 archived
   budgeted-hard checkpoints (first 8 validation blocks, eval mode) with unified/census.py and
   compare layer by layer with strengthened/results/budget_diagnostics.json.
2. One-step training equivalence: closed-form hard_budget vs the archived oracle
   (strengthened/attentiondrop.py) with identical injected draws, on one LM batch and one
   BiomedBERT microbatch: loss and all gradients within 1e-5 (fp32).
3. Extreme rows: m >= 0.99 with alpha = 1, fp32 and bf16 inputs: outputs and gradients match.
Writes results/g0.json (write-once). CPU only (reads archived Parquet, so arrow is allowed).
"""
import json
import os
import sys

import env

env.block_arrow()   # pyarrow segfaults when loaded after torch; parquet is read in a subprocess
env.seal()
env.patch_transformers_imports()
import torch  # noqa: E402

STRENGTHENED = os.path.join(env.WORKSPACE, "strengthened")
sys.path.insert(0, STRENGTHENED)
import attentiondrop as ref  # noqa: E402

import attnreg  # noqa: E402
import census  # noqa: E402
import runlib  # noqa: E402

torch.set_num_threads(int(os.environ.get("G0_THREADS", "4")))


WIKI_READER = r'''
import sys, pyarrow.parquet as pq
records = pq.read_table(sys.argv[1], columns=["text"])["text"].to_pylist()
open(sys.argv[2], "wb").write("\n".join(records).encode("utf-8"))
'''


def wiki_validation():
    """Replicates run_wikitext.prepare() for the validation split in a torch-free subprocess."""
    import subprocess
    import tempfile
    parquet = os.path.join(STRENGTHENED, "data", "wikitext2", "validation.parquet")
    manifest = json.load(open(os.path.join(STRENGTHENED, "results", "wikitext_factorial", "manifest.json"), encoding="utf-8"))
    recorded = json.dumps(manifest)
    digest = runlib.sha256_file(parquet)
    if digest not in recorded:
        raise RuntimeError("WikiText validation parquet hash not found in the archived manifest")
    out = os.path.join(tempfile.gettempdir(), "g0_wiki_validation.bin")
    subprocess.run([sys.executable.replace("pythonw", "python"), "-c", WIKI_READER, parquet, out], check=True)
    data = open(out, "rb").read()
    os.remove(out)
    return torch.tensor(list(data), dtype=torch.long)


def track1():
    import run_lm
    archived = json.load(open(os.path.join(STRENGTHENED, "results", "budget_diagnostics.json"), encoding="utf-8"))
    raw = open(os.path.join(STRENGTHENED, "data", "tinyshakespeare.txt"), "rb").read()
    text = raw.decode("utf-8")
    wiki = {"validation": wiki_validation()}
    worst = 0.0
    compared = 0
    for rec in archived["records"]:
        study, method, seed = rec["study"], rec["method"], rec["seed"]
        manifest = json.load(open(os.path.join(STRENGTHENED, "results", study, "manifest.json"), encoding="utf-8"))
        args = manifest["args"]
        path = os.path.join(STRENGTHENED, "results", study, f"{method}_seed{seed}", "final.pt")
        if runlib.sha256_file(path) != rec["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if study == "lm_pilot":
            mapping = state["vocab"]
            val = torch.tensor([mapping.get(c, 0) for c in text[int(.9 * len(text)):int(.95 * len(text))]])
            vocab = len(mapping) + 1
        elif study == "wikitext_factorial":
            val, vocab = wiki["validation"], 256
        else:
            part = raw[int(.9 * len(raw)):int(.95 * len(raw))]
            val, vocab = torch.tensor(list(part)), 256
        ctx, tau, p = args.get("context", 128), args.get("tau", .1), args.get("p", .1)
        model = run_lm.LM(vocab, args.get("width", 96), args.get("heads", 4), args.get("layers", 2), ctx, "hard", tau, p)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        ours = []

        def hook(module, inputs):
            logits, allowed = inputs
            full = torch.broadcast_to(allowed, logits.shape)
            a = attnreg.masked_softmax(logits.double(), allowed)
            top = a.topk(min(8, a.shape[-1]), dim=-1).values.reshape(-1, min(8, a.shape[-1]))
            n = full.sum(-1).reshape(-1)
            st = census.hard_from_top(top, n, p, 4, tau)
            ours.append({"expected_candidate_tv": float(st["Em"].mean()), "expected_budgeted_tv": float(st["Ed"].mean()),
                         "probability_budget_active": float(st["P_active"].mean())})
        handles = [blk.ad.register_forward_pre_hook(hook) for blk in model.blocks]
        with torch.no_grad():
            model(val[:8 * ctx].reshape(8, ctx))
        for h in handles:
            h.remove()
        for mine, theirs in zip(ours, rec["layers"]):
            for key in mine:
                worst = max(worst, abs(mine[key] - theirs[key]))
                compared += 1
    return {"records": len(archived["records"]), "values_compared": compared, "max_abs_diff": worst,
            "passed": worst < 1e-9}


class OracleHB(torch.nn.Module):
    """Reference budgeted hard AttentionDrop driven by an injected full-shape uniform tensor."""

    def __init__(self, p, k, tau, draws):
        super().__init__()
        self.p, self.k, self.tau, self.draws = p, k, tau, draws  # draws: list consumed in call order
        self.layer, self.noise, self.observer, self.value_observer = 0, None, None, None

    def forward(self, scores, allowed):
        u = self.draws.pop(0)
        real = torch.rand
        torch.rand = lambda *a, **kw: u
        try:
            a = ref.masked_softmax(scores, allowed)
            b = ref.hard_candidate(scores, allowed, self.p, self.k)
            return ref.budget_mix(a, b, self.tau)
        finally:
            torch.rand = real


class InjectedHB(attnreg.AttnReg):
    def __init__(self, p, k, tau, draws):
        super().__init__("hard_budget", p=p, k=k, tau=tau)
        self.draws = draws

    def forward(self, scores, allowed):
        u = self.draws.pop(0)
        full = torch.broadcast_to(allowed, scores.shape)
        idx = scores.float().masked_fill(~full, -torch.inf).topk(min(self.k, scores.shape[-1]), dim=-1).indices
        return attnreg.hard_budget(scores, allowed, self.p, self.k, self.tau, draw=u.gather(-1, idx))


def one_step(build, forward, n_layers, shape_fn, p=0.5, k=4, tau=0.1):
    torch.manual_seed(0)
    draws = [torch.rand(shape_fn(i)) for i in range(n_layers)]
    grads, losses = [], []
    for variant in ("oracle", "closed"):
        torch.manual_seed(1)
        model = build(lambda i, n: (OracleHB if variant == "oracle" else InjectedHB)(p, k, tau, [draws[i].clone()]))
        model.train()
        loss = forward(model)
        loss.backward()
        losses.append(float(loss))
        grads.append(torch.cat([q.grad.flatten() for q in model.parameters() if q.grad is not None]))
    return {"loss_diff": abs(losses[0] - losses[1]), "max_grad_diff": float((grads[0] - grads[1]).abs().max()),
            "passed": abs(losses[0] - losses[1]) < 1e-5 and float((grads[0] - grads[1]).abs().max()) < 1e-5}


def lm_step():
    import lm_model
    ctx, b = 32, 4
    x = torch.randint(0, 200, (b, ctx + 1), generator=torch.Generator().manual_seed(3))

    def build(factory):
        m = lm_model.LM(200, width=64, layers=3, heads=4, context=ctx, reg_factory=factory, dropout=0.0)
        m.blocks = torch.nn.ModuleList(m.blocks)
        return m
    return one_step(build, lambda m: m.loss(x[:, :-1], x[:, 1:]), 3, lambda i: (b, 4, ctx, ctx))


def bert_step():
    from transformers import BertConfig, BertModel
    import bert_adapter
    cfg = BertConfig(vocab_size=300, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
                     intermediate_size=128, max_position_embeddings=64, hidden_dropout_prob=0.0,
                     attention_probs_dropout_prob=0.0, attn_implementation="eager")
    ids = torch.randint(1, 300, (3, 24), generator=torch.Generator().manual_seed(4))
    mask = torch.ones(3, 24, dtype=torch.long)
    mask[1, 15:] = 0

    def build(factory):
        torch.manual_seed(1)
        m = BertModel(cfg, add_pooling_layer=False)
        targets = [mod for mod in m.modules() if type(mod).__name__ == "BertSelfAttention"]
        swapped = type("RegBertSelfAttention", (bert_adapter.RegularizedBertSelfAttention, type(targets[0])), {})
        for i, mod in enumerate(targets):
            mod.__class__ = swapped
            mod.attnreg = factory(i, len(targets))
        return m
    return one_step(build, lambda m: m(input_ids=ids, attention_mask=mask).last_hidden_state.pow(2).mean(), 3,
                    lambda i: (3, 4, 24, 24))


def extreme_rows():
    out = {}
    for dtype in (torch.float32, torch.bfloat16):
        g = torch.Generator().manual_seed(5)
        scores = torch.randn(2, 3, 6, 10, generator=g) * 0.5
        scores[..., 0] += 9.0  # top-1 mass >= 0.99
        scores = scores.to(dtype).requires_grad_(True)
        allowed = torch.ones(1, 1, 1, 10, dtype=torch.bool)
        u = torch.zeros(2, 3, 6, 10)  # delete every eligible key (p = 1, k = 1)
        real = torch.rand
        torch.rand = lambda *a, **kw: u
        try:
            a = ref.masked_softmax(scores, allowed)
            expected = ref.budget_mix(a, ref.hard_candidate(scores, allowed, 1.0, 1), 1.0)
        finally:
            torch.rand = real
        got, m, _ = attnreg.hard_budget(scores, allowed, 1.0, 1, 1.0, draw=u[..., :1], return_stats=True)
        w = torch.randn(expected.shape, generator=g)
        ge = torch.autograd.grad((expected * w).sum(), scores, retain_graph=False)[0]
        gg = torch.autograd.grad((got * w).sum(), scores)[0]
        out[str(dtype)] = {"min_m": float(m.min()), "max_out_diff": float((got - expected).abs().max()),
                           "max_grad_diff": float((gg - ge).abs().max().float()),
                           "passed": float(m.min()) >= 0.99 and float((got - expected).abs().max()) < 1e-5
                           and float((gg - ge).abs().max().float()) < 1e-4}
    return out


def main():
    out_path = os.path.join(env.ROOT, "results", os.environ.get("G0_OUT", "g0.json"))
    record = {"utc": runlib.utc(), "track1": track1(), "lm_one_step": lm_step(), "bert_one_step": bert_step(),
              "extreme_rows": extreme_rows()}
    record["passed"] = (record["track1"]["passed"] and record["lm_one_step"]["passed"] and record["bert_one_step"]["passed"]
                        and all(v["passed"] for v in record["extreme_rows"].values()))
    print(json.dumps(record, indent=1))
    runlib.write_json_once(out_path, record)


if __name__ == "__main__":
    main()
