"""Outcome-blind probe (random tokens): peak VRAM with and without per-block activation checkpointing,
and whether checkpointing changes the loss or the gradients (same seed, same noise, same dropout RNG)."""
import sys

sys.path.insert(0, r"C:\Attentiondrop__Attention_dropS_\unified")
import env  # noqa: E402

torch = env.setup_model_process(threads=2)
from torch.utils.checkpoint import checkpoint  # noqa: E402

import attnreg  # noqa: E402
import lm_model  # noqa: E402

env.cap_vram()
arms = {"RB": {"kind": "relaxed_budget", "tau": 0.10362714148988628},
        "DB": {"kind": "dropkey_budget", "p": 0.5, "tau": 0.10051269661594905},
        "HB": {"kind": "hard_budget", "p": 0.8, "k": 4, "tau": 0.1627}}


def build(arm):
    def factory(i, n):
        return attnreg.AttnReg(arm["kind"], p=arm.get("p", 0.1), k=arm.get("k", 4), tau=arm.get("tau", 0.1),
                               gamma=arm.get("gamma", 0.1), record=True)
    torch.manual_seed(999)
    model = lm_model.LM(16384, width=384, layers=6, heads=6, context=256, reg_factory=factory, dropout=0.1).cuda()
    noise = attnreg.NoiseSource(999)
    model.set_noise(noise)
    return model, noise


def wrap(model):
    for blk in model.blocks:
        fn = blk.forward
        blk.forward = (lambda f: (lambda x: checkpoint(f, x, use_reentrant=False) if torch.is_grad_enabled() else f(x)))(fn)


def step(model, noise, ckpt):
    if ckpt:
        wrap(model)
    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(device="cpu").manual_seed(5)
    w = torch.randint(0, 16384, (32, 257), generator=g).cuda()
    noise.set_step(0, 0, 0)
    torch.manual_seed(123)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model.loss(w[:, :-1], w[:, 1:])
    loss.backward()
    torch.cuda.synchronize()
    grads = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.grad is not None])
    stats = model.regularizers()[0].stats()
    return float(loss), grads.cpu(), torch.cuda.max_memory_allocated() / 2**30, stats


for name, arm in arms.items():
    out = {}
    for ckpt in (False, True):
        try:
            model, noise = build(arm)
            out[ckpt] = step(model, noise, ckpt)
        except torch.OutOfMemoryError:
            out[ckpt] = None
        del model
        torch.cuda.empty_cache()
    msg = [name]
    for ckpt in (False, True):
        msg.append(f"ckpt={ckpt}: " + ("OOM" if out[ckpt] is None else f"peak {out[ckpt][2]:.3f} GiB loss {out[ckpt][0]:.6f}"))
    if out[False] and out[True]:
        d = (out[False][1] - out[True][1]).abs().max().item()
        msg.append(f"max|grad diff| {d:.2e}; realized_tv {out[False][3]['realized_tv']:.6f} vs {out[True][3]['realized_tv']:.6f}")
    print(" | ".join(msg), flush=True)
