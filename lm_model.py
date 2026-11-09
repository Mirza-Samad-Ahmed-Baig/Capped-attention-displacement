"""Small pre-LN causal LM whose attention probabilities pass through attnreg.AttnReg.

Architecture follows the archived strengthened/run_lm.LM (pre-LN, learned absolute
positions, fused qkv, GELU 4x MLP, untied output head, N(0, 0.02) init) so the new study
extends the earlier one; the file is new so archived hashes stay valid. Attention is always
materialized (every arm, including "none", uses the same path, so wall-clock comparisons are
fair). Residual/MLP dropout is a fixed per-regime setting, never an arm.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from attnreg import AttnReg


class Block(nn.Module):
    def __init__(self, width, heads, context, reg, dropout=0.0):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.drop = nn.Dropout(dropout)
        self.attnreg = reg
        self.register_buffer("causal", torch.ones(context, context, dtype=torch.bool).tril(), persistent=False)

    def forward(self, x):
        b, t, w = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if getattr(self.attnreg, "value_observer", None) is not None:
            with torch.no_grad():
                self.attnreg.value_observer(self.attnreg.layer, scores.detach(), self.causal[:t, :t], v.detach())
        probs = self.attnreg(scores, self.causal[:t, :t])
        y = torch.matmul(probs.to(v.dtype), v).transpose(1, 2).reshape(b, t, w)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.mlp(self.ln2(x)))


class LM(nn.Module):
    def __init__(self, vocab, width=384, layers=6, heads=6, context=256, reg_factory=None, dropout=0.0):
        super().__init__()
        reg_factory = reg_factory or (lambda i, n: AttnReg("none"))
        self.context = context
        self.tok = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(context, width)
        self.blocks = nn.ModuleList(Block(width, heads, context, reg_factory(i, layers), dropout) for i in range(layers))
        for i, blk in enumerate(self.blocks):
            blk.attnreg.layer = i
        self.ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def regularizers(self):
        return [blk.attnreg for blk in self.blocks]

    def set_noise(self, noise):
        for reg in self.regularizers():
            reg.noise = noise

    def hidden(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.ln(x)

    def loss(self, idx, targets, chunk=2048):
        """Mean next-token NLL (nats) over all target positions, computed in fp32 chunks."""
        h = self.hidden(idx).reshape(-1, self.head.in_features)
        y = targets.reshape(-1)
        total = h.new_zeros((), dtype=torch.float32)
        for s in range(0, h.shape[0], chunk):
            logits = self.head(h[s:s + chunk]).float()
            total = total + F.cross_entropy(logits, y[s:s + chunk], reduction="sum")
        return total / y.numel()

    def forward(self, idx):
        return self.head(self.hidden(idx))


def parameter_count(model, include_embeddings=True):
    total = sum(p.numel() for p in model.parameters())
    if include_embeddings:
        return total
    return total - model.tok.weight.numel() - model.pos.weight.numel() - model.head.weight.numel()
