"""Sealed test evaluation for the LM regime (PROTOCOL Part B §B12).

Called inside a runner process. Reads the sealed LM test stream through env.sealed_access,
computes per-document NLL sums and per-frequency-bucket sums, writes them write-once to the
study's sealed folder, and returns ONLY the file hash. It never prints or returns values.
Nobody opens the sealed folder before unseal.py.
"""
import json
import os

import numpy as np
import torch
from torch.nn import functional as F

import env
import runlib

LM_DIR = os.path.join(env.ROOT, "data", "lm")
BUCKETS = (0, 1, 10, 100, 1000)  # train-prefix count thresholds: [0], [1,10), [10,100), [100,1000), [1000,inf)


def _bucket(counts):
    b = np.zeros_like(counts)
    for i, th in enumerate(BUCKETS[1:], start=1):
        b[counts >= th] = i
    return b


def lm_test(model, context, prefix_counts, out_path, tag, batch=16, stream="test"):
    """Evaluate `model` (eval mode) on the sealed test stream; write per-doc sums once.
    stream="val" exercises the identical code path on validation data (smoke tests only)."""
    test_bin = os.path.join(LM_DIR, f"{stream}.bin")
    docs_json = os.path.join(LM_DIR, f"{stream}_docs.json")
    if stream == "test":
        with env.sealed_access(test_bin, docs_json):
            tokens = np.asarray(np.memmap(test_bin, dtype=np.uint16, mode="r"))
            with open(docs_json, encoding="utf-8") as f:
                docs = json.load(f)
    elif stream == "val":
        tokens = np.asarray(np.memmap(test_bin, dtype=np.uint16, mode="r"))
        with open(docs_json, encoding="utf-8") as f:
            docs = json.load(f)
    else:
        raise ValueError(stream)
    n = len(tokens) - 1
    losses = np.zeros(n, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        starts = list(range(0, n, context))
        for s in range(0, len(starts), batch):
            chunk = starts[s:s + batch]
            width = max(min(context, n - c) for c in chunk)
            x = torch.zeros((len(chunk), width), dtype=torch.long)
            y = torch.full((len(chunk), width), -100, dtype=torch.long)
            for j, c in enumerate(chunk):
                L = min(context, n - c)
                x[j, :L] = torch.from_numpy(tokens[c:c + L].astype(np.int64))
                y[j, :L] = torch.from_numpy(tokens[c + 1:c + 1 + L].astype(np.int64))
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.hidden(x)
            logits = model.head(h).float()
            tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none",
                                  ignore_index=-100).reshape(y.shape).cpu().numpy()
            for j, c in enumerate(chunk):
                L = min(context, n - c)
                losses[c:c + L] = tok[j, :L]
    targets = tokens[1:].astype(np.int64)
    buckets = _bucket(prefix_counts[targets])
    per_doc = []
    for pid, start, length in docs:
        lo, hi = max(0, start - 1), min(n, start + length - 1)  # targets at positions start..start+length-1
        per_doc.append([pid, float(losses[lo:hi].sum()), int(hi - lo)])
    record = {"tag": tag, "context": context, "per_doc": per_doc,
              "bucket_thresholds": list(BUCKETS),
              "bucket_sum": [float(losses[buckets == i].sum()) for i in range(len(BUCKETS))],
              "bucket_count": [int((buckets == i).sum()) for i in range(len(BUCKETS))],
              "tokens_scored": int(n)}
    return runlib.write_json_once(out_path, record)
