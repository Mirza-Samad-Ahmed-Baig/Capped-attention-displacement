"""Locate run results across study folders and index them by (regime, arm tag, unit). No torch."""
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.join(ROOT, "studies")

S1_RE = re.compile(r"^S1_(?P<tag>.+)_f(?P<fold>\d+)(?:_s(?P<rep>\d+))?$")
S2_RE = re.compile(r"^S2(?:_base2)?_(?P<tag>.+)_s(?P<seed>\d+)$")


def index_runs(prefix):
    """{run_id: path} for every result JSON under studies/<prefix>*/**/runs/."""
    out = {}
    for base, dirs, files in os.walk(STUDIES):
        if os.path.basename(base) != "runs":
            continue
        rel = os.path.relpath(base, STUDIES)
        if not rel.startswith(prefix):
            continue
        for f in files:
            if f.endswith(".json"):
                rid = f[:-5]
                if rid in out:
                    raise RuntimeError(f"duplicate run id {rid}")
                out[rid] = os.path.join(base, f)
    return out


def sealed_path(result_path, suffix=""):
    folder = os.path.dirname(os.path.dirname(result_path))
    rid = os.path.basename(result_path)[:-5]
    return os.path.join(folder, "sealed", f"{rid}{suffix}.json")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def s1_units():
    """{tag: {(fold, rep): path}} for all encoder runs (rep 1 = primary seed, 2 = P1 second seed)."""
    table = {}
    for rid, path in index_runs("S1").items():
        m = S1_RE.match(rid)
        if not m:
            continue
        tag = m.group("tag")
        rep = int(m.group("rep") or 1)
        table.setdefault(tag, {})[(int(m.group("fold")), rep)] = path
    return table


def s2_units():
    """{tag: {seed_index: path}}; the base2 learning-rate winner supplies N at seed 1."""
    idx = index_runs("S2")
    table = {}
    winner = load(os.path.join(STUDIES, "S2_base2", "decisions_G1.json"))["lr_winner"]
    for rid, path in idx.items():
        if rid.startswith("S2_base_"):
            continue  # superseded by the G1 fallback (S2_base2); never analysed
        if rid.startswith("S2_base2_"):
            if rid != winner:
                continue
            table.setdefault("N", {})[1] = path
            continue
        m = S2_RE.match(rid)
        if m:
            table.setdefault(m.group("tag"), {})[int(m.group("seed"))] = path
    return table
