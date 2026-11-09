#!/usr/bin/env bash
# Extension E8 driver (PROTOCOL B8/B10, after E3; E6 not run: budget). Declares S2/e8 if needed, then runs the
# 10 alpha-detached HB@0.10 LM runs, only if E8's expected 1.25 GPU-h fits under the 90 GPU-h cap.
# Exits cleanly on unified/STOP; stops on error.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
VENV=../domain_distill/.venv/Scripts/python.exe

stop_check() { if [ -f STOP ]; then echo "[e8] STOP present: exiting"; exit 0; fi; }
pending() { py -3.12 -c "import sys,runlib; print(len(runlib.Registry(sys.argv[1]).pending()))" "$1"; }

stop_check
[ -f studies/S2/e8/registry.json ] || py -3.12 budget.py fits 1.25 e8 || { echo "[e8] does not fit the cap: not run (budget)"; exit 0; }
[ -f studies/S2/e8/registry.json ] || py -3.12 campaign.py e8
echo "[e8] $(date -u +%FT%TZ) running studies/S2/e8"
"$VENV" lm_runner.py studies/S2/e8 >> studies/S2/e8/console.log 2>&1
stop_check
left=$(pending studies/S2/e8)
if [ "$left" != "0" ]; then echo "[e8] studies/S2/e8 still has $left pending runs: exiting"; exit 1; fi
echo "[e8] $(date -u +%FT%TZ) E8 campaign complete"
