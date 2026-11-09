"""Statistics for the unified study (PROTOCOL A §2, B4-B6, B10). Pure numpy/scipy, no torch.

Unit of analysis: fold (encoder) or seed (LM); `diffs` are per-unit paired differences
(arm minus comparator, lower = better for losses).
"""
import itertools
import math

import numpy as np
from scipy import stats

# ------------------------------------------------------------------ metrics (3 classes)


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def brier(probs, labels):
    y = np.eye(probs.shape[1])[np.asarray(labels)]
    return float(((probs - y) ** 2).sum(1).mean())


def nll(probs, labels):
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-15, 1)).mean())


def accuracy(probs, labels):
    return float((probs.argmax(1) == np.asarray(labels)).mean())


def macro_f1(probs, labels, classes=3):
    pred, labels = probs.argmax(1), np.asarray(labels)
    f1 = []
    for c in range(classes):
        tp = np.sum((pred == c) & (labels == c))
        denom = np.sum(pred == c) + np.sum(labels == c)
        f1.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1))


def ece(probs, labels, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == np.asarray(labels)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def macro_auroc(probs, labels, classes=3):
    labels = np.asarray(labels)
    vals = []
    for c in range(classes):
        pos, neg = probs[labels == c, c], probs[labels != c, c]
        if len(pos) == 0 or len(neg) == 0:
            continue
        # Mann-Whitney U / (n_pos n_neg), ties counted half
        ranks = stats.rankdata(np.concatenate([pos, neg]))
        u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
        vals.append(u / (len(pos) * len(neg)))
    return float(np.mean(vals))


def fit_temperature(logits, labels, lo=0.25, hi=4.0, iters=80):
    """Golden-section search on NLL over log-temperature in [lo, hi]."""
    f = lambda lt: nll(softmax(logits, math.exp(lt)), labels)
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return float(math.exp((a + b) / 2))


def classification_metrics(logits, labels, temperature=1.0):
    p = softmax(logits, temperature)
    return {"brier": brier(p, labels), "nll": nll(p, labels), "accuracy": accuracy(p, labels),
            "macro_f1": macro_f1(p, labels), "ece15": ece(p, labels), "auroc": macro_auroc(p, labels)}


def failure_and_collapse(logits, labels, majority_rate):
    p = softmax(logits)
    pred = p.argmax(1)
    share = np.bincount(pred, minlength=3).max() / len(pred)
    return {"failed": bool(accuracy(p, labels) <= majority_rate + 1e-12), "collapsed": bool(share >= 0.9),
            "top_class_share": float(share)}


def evidence_use(full_logits, qo_logits, mm_logits_list, labels, donor_labels_list):
    pf = softmax(full_logits)
    pred_full = pf.argmax(1)
    labels = np.asarray(labels)
    out = {"acc_full": accuracy(pf, labels), "acc_question_only": accuracy(softmax(qo_logits), labels)}
    out["evidence_use_gap"] = out["acc_full"] - out["acc_question_only"]
    flips, follow, kl = [], [], []
    for mm, dl in zip(mm_logits_list, donor_labels_list):
        pm = softmax(mm)
        flips.append(float((pm.argmax(1) != pred_full).mean()))
        follow.append(float((pm.argmax(1) == np.asarray(dl)).mean()))
        kl.append(float((pf * (np.log(np.clip(pf, 1e-15, 1)) - np.log(np.clip(pm, 1e-15, 1)))).sum(1).mean()))
    out.update({"flip_rate": float(np.mean(flips)), "donor_label_following": float(np.mean(follow)),
                "kl_full_to_mismatch": float(np.mean(kl))})
    return out

# ------------------------------------------------------------------ paired tests


def sign_flip_test(diffs, alternative="two-sided"):
    """Exact sign-flip (randomization) test on the mean of paired differences."""
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    means = (signs * d).mean(1)
    obs = d.mean()
    tol = 1e-12 * max(1.0, abs(obs))
    if alternative == "two-sided":
        return float(np.mean(np.abs(means) >= abs(obs) - tol))
    if alternative == "less":
        return float(np.mean(means <= obs + tol))
    return float(np.mean(means >= obs - tol))


def sign_flip_ci(diffs, alpha=0.05, step=1e-4):
    """CI by inverting the two-sided sign-flip test over a shift grid."""
    d = np.asarray(diffs, dtype=np.float64)
    span = max(np.ptp(d), abs(d.mean()), step) * 3
    grid = np.arange(d.mean() - span, d.mean() + span + step, step)
    accepted = [s for s in grid if sign_flip_test(d - s) > alpha]
    return (float(min(accepted)), float(max(accepted))) if accepted else (float("nan"), float("nan"))


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def t_ci(diffs, level=0.90):
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    return float(d.mean() - q * se), float(d.mean() + q * se)


def nadeau_bengio_ci(diffs, n_train, n_test, level=0.95):
    """Corrected resampled-t interval for CV folds with overlapping training sets."""
    d = np.asarray(diffs, dtype=np.float64)
    k = len(d)
    var = (1 / k + n_test / n_train) * d.var(ddof=1)
    q = stats.t.ppf(0.5 + level / 2, k - 1)
    return float(d.mean() - q * math.sqrt(var)), float(d.mean() + q * math.sqrt(var))


def tost(diffs, margin, level=0.90):
    lo, hi = t_ci(diffs, level)
    return {"ci90": [lo, hi], "margin": margin, "equivalent": bool(-margin < lo and hi < margin)}


def verdict(p_holm, ci_equiv, margin, direction_mean):
    """Four-way verdict (PROTOCOL A §2): Better / Worse (Holm p < .05), Equivalent (90% CI in margin)."""
    if p_holm < 0.05:
        return "Better" if direction_mean < 0 else "Worse"
    if -margin < ci_equiv[0] and ci_equiv[1] < margin:
        return "Equivalent"
    return "Inconclusive"


def two_level_bootstrap(per_run_example_a, per_run_example_b, reps=10000, seed=0, level=0.95):
    """CI for mean difference of per-example scores, resampling runs (paired) and examples (paired).

    Inputs: arrays [runs, examples] of per-example losses (e.g. Brier) for arm A and B, paired by
    run index (fold) and example index (test item)."""
    a, b = np.asarray(per_run_example_a), np.asarray(per_run_example_b)
    rng = np.random.default_rng(seed)
    r, e = a.shape
    diff = a - b
    means = np.empty(reps)
    for i in range(reps):
        ri = rng.integers(0, r, r)
        ei = rng.integers(0, e, e)
        means[i] = diff[np.ix_(ri, ei)].mean()
    return float(np.quantile(means, (1 - level) / 2)), float(np.quantile(means, 1 - (1 - level) / 2))


def page_l(table, reps=100000, seed=0):
    """Page's L trend test; table [blocks, treatments] ordered by predicted increasing response.
    Returns L and a within-block permutation p-value (one-sided, increasing trend)."""
    t = np.asarray(table, dtype=np.float64)
    ranks = np.apply_along_axis(stats.rankdata, 1, t)
    weights = np.arange(1, t.shape[1] + 1)
    L = float((ranks.sum(0) * weights).sum())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(reps):
        perm = np.array([rng.permutation(row) for row in ranks])
        if (perm.sum(0) * weights).sum() >= L - 1e-9:
            count += 1
    return L, (count + 1) / (reps + 1)


def brown_forsythe(*groups):
    res = stats.levene(*groups, center="median")
    return float(res.statistic), float(res.pvalue)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def stratified_permutation_failure(fail_a, fail_b, strata, reps=100000, seed=0, alternative="less"):
    """Difference in failure rates (A - B), permuting arm labels within strata (folds)."""
    fa, fb, s = np.asarray(fail_a, float), np.asarray(fail_b, float), np.asarray(strata)
    obs = fa.mean() - fb.mean()
    rng = np.random.default_rng(seed)
    stacked = np.stack([fa, fb], 1)
    count = 0
    for _ in range(reps):
        swap = rng.random(len(fa)) < 0.5
        x = np.where(swap[:, None], stacked[:, ::-1], stacked)
        d = x[:, 0].mean() - x[:, 1].mean()
        count += d <= obs + 1e-12 if alternative == "less" else d >= obs - 1e-12
    return float(obs), (count + 1) / (reps + 1)


def spearman_perm(x, y, reps=10000, seed=0):
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    count = sum(abs(stats.spearmanr(x, rng.permutation(y)).statistic) >= abs(rho) - 1e-12 for _ in range(reps))
    return float(rho), (count + 1) / (reps + 1)
