#!/usr/bin/env bash
# Core-campaign driver: S1 tuning -> S1 levels -> S1 finals -> S2 tuning -> S2 levels -> S2 finals.
# Every decision is a prespecified rule in select.py (prints decisions only). Stops on any error,
# exits cleanly if unified/STOP exists, and advances only when a stage has no pending runs.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
VENV=../domain_distill/.venv/Scripts/python.exe

stop_check() { if [ -f STOP ]; then echo "[drive] STOP present: exiting"; exit 0; fi; }
pending() { py -3.12 -c "import sys,runlib; print(len(runlib.Registry(sys.argv[1]).pending()))" "$1"; }
stage() {  # $1 runner, $2 study folder
  stop_check
  echo "[drive] $(date -u +%FT%TZ) running $2"
  "$VENV" "$1" "$2" >> "$2/console.log" 2>&1
  stop_check
  local left; left=$(pending "$2")
  if [ "$left" != "0" ]; then echo "[drive] $2 still has $left pending runs: exiting"; exit 1; fi
  echo "[drive] $(date -u +%FT%TZ) completed $2"
}

stage encoder_runner.py studies/S1/tuning
[ -f studies/S1/decisions_levels.json ] || py -3.12 select.py levels studies/S1 encoder
[ -f studies/S1/finals/registry.json ] || py -3.12 campaign.py s1_finals
stage encoder_runner.py studies/S1/finals
[ -f studies/S2/tuning/registry.json ] || py -3.12 campaign.py s2_tuning
stage lm_runner.py studies/S2/tuning
[ -f studies/S2/decisions_levels.json ] || py -3.12 select.py levels studies/S2 lm
[ -f studies/S2/finals/registry.json ] || py -3.12 campaign.py s2_finals
stage lm_runner.py studies/S2/finals
echo "[drive] $(date -u +%FT%TZ) core campaign complete"
