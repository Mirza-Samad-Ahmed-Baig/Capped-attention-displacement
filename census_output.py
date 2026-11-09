"""Value-weighted census (PROTOCOL B3): does displacement d track the head-output change?

For each family at its calibrated l = 0.10 parameters and sampled attention rows (eval-mode
scores and values of a fixed model), compute on 64 Monte Carlo draws per row:
  d = TV(a, a~)  and  ||dz|| = ||(a~ - a) V||,  relative size ||dz|| / ||a V||,
the pooled Spearman correlation rho(d, ||dz||), and, for deletion kinds, the share of deleted
mass that falls on special keys ([CLS]/[SEP] in BERT; EOT in the LM).
Registered rule: pooled rho < 0.5 => the displacement-accounting premise is reported refuted.
"""
import torch

import attnreg


def spearman(x, y):
    rx = x.argsort().argsort().double()
    ry = y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-300))


class ValueSampler:
    """value_observer collecting (scores row, allowed row, V, special-key mask) samples."""

    def __init__(self, rows_per_layer=256, seed=0):
        self.rows, self.gen, self.data = rows_per_layer, torch.Generator().manual_seed(seed), {}
        self.special = None  # bool [B, K] set by the caller before each forward

    def __call__(self, layer, scores, allowed, v):
        store = self.data.setdefault(layer, [])
        have = sum(s[0].shape[0] for s in store)
        if have >= self.rows:
            return
        b, h, q, k = scores.shape
        full = torch.broadcast_to(allowed, scores.shape)
        qvalid = torch.broadcast_to(allowed.squeeze(-2).unsqueeze(-1), (b, h, q, 1)).squeeze(-1) \
            if allowed.dim() == 4 and allowed.shape[-2] == 1 else full.any(-1)
        idx = qvalid.nonzero()
        take = min(self.rows - have, idx.shape[0], 64)
        pick = idx[torch.randperm(idx.shape[0], generator=self.gen)[:take].to(idx.device)]
        bi, hi, qi = pick[:, 0], pick[:, 1], pick[:, 2]
        special = self.special[bi] if self.special is not None else torch.zeros((take, k), dtype=torch.bool)
        store.append((scores[bi, hi, qi].double().cpu(), full[bi, hi, qi].cpu(), v[bi, hi].double().cpu(),
                      special.cpu()))


def family_regs(levels_rec):
    """AttnReg objects for the calibrated l = 0.10 parameters of each family."""
    r = levels_rec
    return {"AD": attnreg.AttnReg("dropout", p=r["AD"]["p"]),
            "DK": attnreg.AttnReg("dropkey", p=r["DK"]["p_max"]),  # depth schedule applied per layer below
            "RA": attnreg.AttnReg("relaxed", gamma=r["RA"]["gamma"]),
            "RB": attnreg.AttnReg("relaxed_budget", tau=r["RB"]["tau"]),
            "HU": attnreg.AttnReg("hard", p=r["HU"]["p"], k=4),
            "HF": attnreg.AttnReg("hard_mix", p=r["HF"]["p"], k=4, gamma=r["HF"]["gamma"]),
            "HB": attnreg.AttnReg("hard_budget", p=r["HB"]["p"], k=4, tau=r["HB"]["tau"])}


def analyse(sampler, levels_rec, num_layers, draws=64, seed=0):
    torch.manual_seed(seed)
    regs = family_regs(levels_rec)
    out = {}
    for name, reg in regs.items():
        reg.train()
        ds, dzs, rel, spec_share, del_mass = [], [], [], [], []
        for layer, chunks in sampler.data.items():
            if name == "DK":
                reg.p = attnreg.depth_schedule(levels_rec["DK"]["p_max"], int(layer), num_layers)
            for scores, allowed, v, special in chunks:
                s = scores[:, None, :].expand(-1, draws, -1)
                al = allowed[:, None, :].expand(-1, draws, -1)
                a = attnreg.masked_softmax(scores, allowed)
                z = torch.einsum("rk,rkd->rd", a, v)
                pert = reg(s, al)
                delta = pert - a[:, None, :]
                dz = torch.einsum("rnk,rkd->rnd", delta, v).norm(dim=-1)
                d = 0.5 * delta.abs().sum(-1)
                ds.append(d.flatten())
                dzs.append(dz.flatten())
                rel.append((dz / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)).flatten())
                if name in ("HU", "HF", "HB"):
                    removed = (a[:, None, :] - pert).clamp_min(0)
                    spec_share.append((removed * special[:, None, :].double()).sum(-1).flatten())
                    del_mass.append(removed.sum(-1).flatten())
        d, dz = torch.cat(ds), torch.cat(dzs)
        rec = {"mean_d": float(d.mean()), "mean_rel_dz": float(torch.cat(rel).mean()), "spearman_d_dz": spearman(d, dz)}
        if spec_share:
            rec["special_share_of_removed_mass"] = float(torch.cat(spec_share).sum() / torch.cat(del_mass).sum().clamp_min(1e-300))
        out[name] = rec
    pooled = [out[f]["spearman_d_dz"] for f in out]
    out["pooled_rho_mean"] = sum(pooled) / len(pooled)
    out["premise_refuted"] = out["pooled_rho_mean"] < 0.5
    return out
