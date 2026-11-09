"""PubMedQA encoder inputs, evidence perturbations and length-aware batching.

Input layout: [CLS] question [SEP] evidence [SEP] (privileged teacher: [CLS] question [SEP]
evidence [SEP] conclusion [SEP]). The complete question is always kept (a question that does
not fit raises); evidence is head/tail truncated (the earlier study's rule); a privileged
conclusion is never truncated. `single_segment=True` gives token_type 0 everywhere (Tinn
et al.); False gives 0 for [CLS] q [SEP] and 1 afterwards.

The PubMedQA `long_answer` (conclusion) is used ONLY when privileged=True; every other mode
excludes it (tested).
"""
import random

import torch

LABELS = ("yes", "no", "maybe")
MODES = ("original", "question_only", "mismatched", "results_deleted")


def label_index(label):
    return LABELS.index(label)


def encode(row, tokenizer, mode="original", donor=None, limit=512, single_segment=True, privileged=False):
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}")
    question = tokenizer.encode(row["question"], add_special_tokens=False)
    if mode == "original":
        evidence_text = "\n".join(row["contexts"])
    elif mode == "question_only":
        evidence_text = ""
    elif mode == "results_deleted":
        labels = row.get("section_labels")
        if labels is None or len(labels) != len(row["contexts"]):
            raise ValueError("results_deleted requires aligned section_labels")
        evidence_text = "\n".join(c for c, lab in zip(row["contexts"], labels) if lab.upper() != "RESULTS")
    else:
        if donor is None or donor["id"] == row["id"]:
            raise ValueError("Mismatched evidence requires a different donor record")
        evidence_text = "\n".join(donor["contexts"])
    evidence = tokenizer.encode(evidence_text, add_special_tokens=False) if evidence_text else []
    conclusion = []
    if privileged:
        if mode != "original":
            raise ValueError("The privileged teacher input is defined for original evidence only")
        conclusion = tokenizer.encode(row["long_answer"], add_special_tokens=False)
    fixed = len(question) + 3 + (len(conclusion) + 1 if privileged else 0)
    budget = limit - fixed
    if budget < 0:
        raise ValueError("Question (and conclusion) exceed the input limit")
    truncated = len(evidence) > budget
    if truncated:
        head = (budget + 1) // 2
        tail = budget - head
        evidence = evidence[:head] + (evidence[-tail:] if tail else [])
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    ids = [cls] + question + [sep] + evidence + [sep]
    if privileged:
        ids += conclusion + [sep]
    if single_segment:
        types = [0] * len(ids)
    else:
        types = [0] * (len(question) + 2) + [1] * (len(ids) - len(question) - 2)
    return {"input_ids": ids, "token_type_ids": types, "truncated": truncated,
            "question_tokens": len(question), "evidence_tokens": len(evidence)}


def stratified_derangement(labels, lengths, seed, quantiles=5):
    """Donor index per example: a different gold label, same evidence-length quantile when
    possible (else any different-label example). Donors are sampled with replacement,
    because label imbalance makes a label-changing permutation impossible."""
    n = len(labels)
    order = sorted(range(n), key=lambda i: (lengths[i], i))
    bucket = [0] * n
    for rank, i in enumerate(order):
        bucket[i] = min(quantiles - 1, rank * quantiles // n)
    rng = random.Random(seed)
    donors = []
    for i in range(n):
        same = [j for j in range(n) if labels[j] != labels[i] and bucket[j] == bucket[i]]
        pool = same or [j for j in range(n) if labels[j] != labels[i]]
        if not pool:
            raise ValueError("No different-label donor exists")
        donors.append(rng.choice(pool))
    return donors


def microbatches(indices, lengths, max_tokens, max_size=16):
    """Split one optimizer batch into length-sorted microbatches with padded size <= max_tokens."""
    ordered = sorted(indices, key=lambda i: lengths[i])
    batches, current = [], []
    for i in ordered:
        trial = current + [i]
        if current and (max(lengths[j] for j in trial) * len(trial) > max_tokens or len(trial) > max_size):
            batches.append(current)
            current = [i]
        else:
            current = trial
    if current:
        batches.append(current)
    for b in batches:
        if max(lengths[j] for j in b) * len(b) > max_tokens and len(b) > 1:
            raise AssertionError("microbatch exceeds token budget")
    return batches


def collate(encoded, indices, pad_id, device):
    width = max(len(encoded[i]["input_ids"]) for i in indices)
    ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    types = torch.zeros((len(indices), width), dtype=torch.long)
    mask = torch.zeros((len(indices), width), dtype=torch.long)
    for row, i in enumerate(indices):
        n = len(encoded[i]["input_ids"])
        ids[row, :n] = torch.tensor(encoded[i]["input_ids"])
        types[row, :n] = torch.tensor(encoded[i]["token_type_ids"])
        mask[row, :n] = 1
    return {"input_ids": ids.to(device, non_blocking=True), "token_type_ids": types.to(device, non_blocking=True),
            "attention_mask": mask.to(device, non_blocking=True)}
