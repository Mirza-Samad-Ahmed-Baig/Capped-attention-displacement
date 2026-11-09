"""PROTOCOL B11 manipulation check (outcome-blind): ratios of time-averaged realized mean displacement.

Reads only the on-device budget traces (realized TV per layer and epoch) of the tuning runs; no dev or
test metric is read. Ratios HB/HF and HB/HU at l = 0.10 are pooled over the tuning units (encoder folds
0-1; LM seed 1). If a ratio is outside [0.8, 1.25], B11 prescribes one recalibration on the average of
two censuses before the finals. Usage: manipulation_check.py STUDY_DIR PREFIX OUT_JSON
"""
import glob
import json
import os
import re
import sys

import runlib

RUN = re.compile(r"^(?P<arm>[A-Z0-9]+)@(?P<level>[0-9.]+)_[fs](?P<unit>\d+)$")


def realized(path):
    rec = json.load(open(path, encoding="utf-8"))
    rows = tot = sq = cand = 0.0
    for entry in rec.get("budget_trace", []):
        for layer in entry["layers"]:
            n = layer.get("rows", 0.0)
            if not n:
                continue
            rows += n
            tot += n * layer["realized_tv"]
            sq += n * layer["realized_tv_sq"]
            cand += n * layer["candidate_tv"]
    return {"rows": rows, "realized_D": tot / rows, "realized_S2": sq / rows, "candidate_D": cand / rows} if rows else None


def check(study, prefix, out):
    per = {}
    for path in sorted(glob.glob(os.path.join(study, "runs", f"{prefix}_*.json"))):
        name = os.path.basename(path)[len(prefix) + 1:-5]
        m = RUN.match(name)
        if not m:
            continue
        r = realized(path)
        if r:
            per.setdefault(f"{m['arm']}@{m['level']}", {})[m["unit"]] = r
    pooled = {arm: {"realized_D": sum(u["realized_D"] * u["rows"] for u in v.values()) / sum(u["rows"] for u in v.values()),
                    "realized_S2": sum(u["realized_S2"] * u["rows"] for u in v.values()) / sum(u["rows"] for u in v.values()),
                    "units": sorted(v)} for arm, v in per.items()}
    hb = [k for k in pooled if k.startswith("HB@0.1")]
    ratios = {}
    if hb:
        for other in ("HF", "HU"):
            oth = [k for k in pooled if k.startswith(f"{other}@0.1")]
            if oth:
                ratios[f"HB/{other}"] = pooled[hb[0]]["realized_D"] / pooled[oth[0]]["realized_D"]
    if len(ratios) != 2:
        raise RuntimeError(f"B11 needs HB, HF and HU at 0.10 in {study}; found {sorted(pooled)}")
    result = {"rule": "PROTOCOL B11", "study": study, "utc": runlib.utc(), "pooled": pooled, "per_unit": per,
              "ratios_l0.10": ratios, "band": [0.8, 1.25],
              "recalibrate": any(not 0.8 <= v <= 1.25 for v in ratios.values())}
    print(json.dumps({"ratios_l0.10": ratios, "recalibrate": result["recalibrate"]}, indent=1))
    print(runlib.write_json_once(out, result))
    return result


if __name__ == "__main__":
    check(*sys.argv[1:4])
