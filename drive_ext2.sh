#!/usr/bin/env bash
# Second extension driver (PROTOCOL B8 order, after E4): E1 (R-Drop: tune alpha on folds 0-1, finals
# folds 2-9) then E3 (budgeted DropKey vs HB@0.10 in S1 and S2). Waits for drive_ext.sh to finish;
# each stage starts only if its expected cost fits under the 90 GPU-hour cap. Exits cleanly on STOP.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
VENV=../domain_distill/.venv/Scripts/python.exe

stop_check() { if [ -f STOP ]; then echo "[ext2] STOP present: exiting"; exit 0; fi; }
pending() { py -3.12 -c "import sys,runlib; print(len(runlib.Registry(sys.argv[1]).pending()))" "$1"; }
fits() {  # compute time; thermal pauses excluded
  py -3.12 budget.py fits "$1" ext2
}
run() {
  stop_check
  echo "[ext2] $(date -u +%FT%TZ) running $2"
  "$VENV" "$1" "$2" >> "$2/console.log" 2>&1
  stop_check
  local left; left=$(pending "$2")
  if [ "$left" != "0" ]; then echo "[ext2] $2 still has $left pending runs: exiting"; exit 1; fi
}

until grep -q "extension campaign complete" drive_ext.log; do stop_check; sleep 120; done
if fits 5.4; then
  [ -f studies/S1/e1_tuning/registry.json ] || py -3.12 campaign.py e1_tuning
  run encoder_runner.py studies/S1/e1_tuning
  [ -f studies/S1/e1_finals/registry.json ] || py -3.12 campaign.py e1_finals
  run encoder_runner.py studies/S1/e1_finals
fi
if fits 4.1; then
  [ -f studies/S1/e3/registry.json ] || py -3.12 campaign.py e3
  run encoder_runner.py studies/S1/e3
  run lm_runner.py studies/S2/e3
fi
echo "[ext2] $(date -u +%FT%TZ) second extension campaign complete"
