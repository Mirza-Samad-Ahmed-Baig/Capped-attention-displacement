"""Optimizers, Mixout, R-Drop and schedules for the encoder studies.

- `BertAdam`: Adam WITHOUT bias correction and with decoupled weight decay, as in the
  original BERT release. It is the "fragile" optimizer of the fine-tuning-instability
  literature (Mosbach et al., 2021; Zhang et al., 2021); the stabilized recipe uses
  torch.optim.AdamW, which is bias-corrected.
- `apply_mixout`: Mixout (Lee et al., 2020) on nn.Linear weights. Each training forward
  draws M ~ Bernoulli(p) and uses (1/(1-p)) * ((1-M) W + M U - p U), where U is a frozen
  fp32 copy of the pretrained weight, so E[W~] = W. Biases are not mixed (as in the
  reference implementation). Masks come from the global RNG, which torch checkpointing
  restores on recomputation.
- `rdrop_loss`: mean CE of two stochastic views plus alpha times the mean symmetric KL.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class BertAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-5, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                update = m / (v.sqrt() + group["eps"])
                if group["weight_decay"]:
                    update = update + group["weight_decay"] * p
                p.add_(update, alpha=-group["lr"])
        return loss


def parameter_groups(model, weight_decay):
    """Decay matrices only; no decay for biases and LayerNorm weights (standard BERT)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "LayerNorm" in name or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def linear_warmup_decay(optimizer, total_steps, warmup_fraction=0.1):
    warmup = max(1, int(round(warmup_fraction * total_steps)))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class MixLinear(nn.Module):
    def __init__(self, linear, p):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("mixout p must lie in [0, 1)")
        self.inner = linear
        self.p = p
        self.register_buffer("target", linear.weight.detach().clone().float(), persistent=False)

    def forward(self, x):
        w = self.inner.weight
        if self.training and self.p > 0:
            with torch.no_grad():
                mask = (torch.rand(w.shape, device=w.device) < self.p).to(w.dtype)
            target = self.target.to(w.dtype)
            w = ((1 - mask) * w + mask * target - self.p * target) / (1 - self.p)
        return F.linear(x, w, self.inner.bias)


def apply_mixout(model, p, include=lambda name: True):
    """Replace nn.Linear submodules (whose qualified name passes `include`) by MixLinear."""
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if type(child) is nn.Linear and include(full):
                setattr(parent, child_name, MixLinear(child, p))
                replaced.append(full)
    return replaced


def _wide(x):
    return x if x.dtype == torch.float64 else x.float()


def rdrop_loss(logits1, logits2, labels, alpha):
    ce = 0.5 * (F.cross_entropy(_wide(logits1), labels) + F.cross_entropy(_wide(logits2), labels))
    lp, lq = _wide(logits1).log_softmax(-1), _wide(logits2).log_softmax(-1)
    sym_kl = 0.5 * ((lp.exp() - lq.exp()) * (lp - lq)).sum(-1).mean()
    return ce + alpha * sym_kl, ce.detach(), sym_kl.detach()


def kd_loss(student_logits, teacher_logits, temperature):
    """T^2 * KL(softmax(t/T) || softmax(s/T)), mean over rows; teacher detached."""
    t = _wide(teacher_logits.detach()) / temperature
    s = _wide(student_logits) / temperature
    return temperature ** 2 * F.kl_div(s.log_softmax(-1), t.log_softmax(-1), log_target=True,
                                       reduction="batchmean")
