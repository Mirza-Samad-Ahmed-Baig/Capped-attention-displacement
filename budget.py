"""GPU-hour accounting for the 90 h cap (PROTOCOL A section 4).

Ledger entries record each run's wall time (end utc, seconds). Thermal-governor pauses
(studies/thermal_pauses.jsonl) stretch wall time without GPU work, so the compute time of a run is its
wall time minus the part of any pause that overlaps it. Only overlaps are subtracted, so time outside
ledgered runs is never counted twice or credited back.

  budget.py used            prints compute GPU-hours used
  budget.py fits HOURS TAG  exit 0 if used + HOURS <= 90 (prints a driver log line with TAG)
"""
import glob
import json
import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
CAP = 90.0


def ts(s):
    return datetime.fromisoformat(s)


def pauses():
    path = os.path.join(ROOT, "studies", "thermal_pauses.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [(ts(r["start"]), ts(r["end"])) for r in map(json.loads, (l for l in f if l.strip()))]


def used_hours():
    ps = pauses()
    total = 0.0
    for path in glob.glob(os.path.join(ROOT, "studies", "**", "ledger.jsonl"), recursive=True):
        with open(path, encoding="utf-8") as f:
            for rec in map(json.loads, (l for l in f if l.strip())):
                end = ts(rec["utc"])
                start = end - timedelta(seconds=rec["gpu_seconds"])
                overlap = sum(max(0.0, (min(end, pe) - max(start, ps_)).total_seconds()) for ps_, pe in ps)
                total += rec["gpu_seconds"] - overlap
    return total / 3600


if __name__ == "__main__":
    if sys.argv[1] == "used":
        print(f"{used_hours():.2f}")
    elif sys.argv[1] == "fits":
        used, nxt = used_hours(), float(sys.argv[2])
        print(f"[{sys.argv[3]}] GPU-hours used {used:.1f} (compute, thermal pauses excluded); next {nxt} h; cap {CAP:.0f}")
        sys.exit(0 if used + nxt <= CAP else 1)
