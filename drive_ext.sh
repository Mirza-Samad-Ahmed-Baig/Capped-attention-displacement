#!/usr/bin/env bash
# Extension driver (PROTOCOL B8 order): waits for the core campaign, smoke-tests the extension code
# paths outcome-blind, then P1 -> E2 -> E-S3 (gate G3) -> E5 -> E4. An extension starts only if its
# expected cost fits under the 90 GPU-hour cap (ledgers). Stops on error; exits cleanly on unified/STOP.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
VENV=../domain_distill/.venv/Scripts/python.exe
SMOKE_DIR="C:/Users/FP/AppData/Local/Temp/unified_smoke_ext"; export SMOKE_DIR

stop_check() { if [ -f STOP ]; then echo "[ext] STOP present: exiting"; exit 0; fi; }
pending() { py -3.12 -c "import sys,runlib; print(len(runlib.Registry(sys.argv[1]).pending()))" "$1"; }
fits() {  # $1 = expected GPU-hours of the next extension (compute time; thermal pauses excluded)
  py -3.12 budget.py fits "$1" ext
}
run() {  # $1 runner, $2 folder, [$3 model dir]
  stop_check
  echo "[ext] $(date -u +%FT%TZ) running $2"
  "$VENV" "$1" "$2" ${3:-} >> "$2/console.log" 2>&1
  stop_check
  local left; left=$(pending "$2")
  if [ "$left" != "0" ]; then echo "[ext] $2 still has $left pending runs: exiting"; exit 1; fi
}

audits() {  # PROTOCOL A §4 order: audits after the core, before the extensions (audit.py docstring)
  [ -f results/audit_choice.json ] || py -3.12 audit.py choose || return 1
  [ -f studies/audit/S1/registry.json ] || py -3.12 audit.py declare || return 1
  local st kind
  for pair in S1:encoder S2:lm; do
    st=${pair%%:*}; kind=${pair##*:}
    stop_check
    "$VENV" audit.py run "studies/audit/$st" "$kind" >> "studies/audit/$st/console.log" 2>&1 || return 1
    stop_check
    [ "$(pending "studies/audit/$st")" = "0" ] || return 1
  done
  for kind in encoder lm; do
    [ -f "results/audit_evalreplay_$kind.json" ] || "$VENV" audit.py evalreplay "$kind" >> studies/audit/evalreplay.log 2>&1 || return 1
  done
  py -3.12 audit.py report >> studies/audit/report.log 2>&1 || return 1
}

until grep -q "core campaign complete" drive.log; do stop_check; sleep 120; done
if [ ! -f results/audit.json ]; then
  echo "[ext] $(date -u +%FT%TZ) core complete; audits"
  if audits; then echo "[ext] audits complete"; else echo "[ext] audit step failed (non-fatal; see studies/audit logs)"; fi
fi
echo "[ext] $(date -u +%FT%TZ) smoke-testing extension paths"
mkdir -p "$SMOKE_DIR"
for st in t k l sf; do
  [ -f "$SMOKE_DIR/$st/registry.json" ] || py -3.12 smoke_ext.py "$st"
  mkdir -p "$SMOKE_DIR/$st"; run encoder_runner.py "$SMOKE_DIR/$st"
done
[ -f "$SMOKE_DIR/e5/registry.json" ] || py -3.12 smoke_ext.py e5
mkdir -p "$SMOKE_DIR/e5"; run encoder_runner.py "$SMOKE_DIR/e5" models/biolinkbert_base
echo "[ext] smoke passed"

if fits 5.7; then
  [ -f studies/S1/p1/registry.json ] || py -3.12 campaign.py p1
  run encoder_runner.py studies/S1/p1
fi
if fits 2.9; then
  [ -f studies/E2/registry.json ] || py -3.12 campaign.py e2
  run encoder_runner.py studies/E2
fi
if fits 7.8; then
  [ -f studies/ES3/teachers/registry.json ] || py -3.12 campaign.py es3_teachers
  run encoder_runner.py studies/ES3/teachers
  [ -f studies/ES3/teachers/decisions_G3.json ] || py -3.12 select.py g3 studies/ES3/teachers
  if grep -q "proceed with students" studies/ES3/teachers/decisions_G3.json; then
    [ -f studies/ES3/phase_k/registry.json ] || py -3.12 campaign.py es3_phase_k
    run encoder_runner.py studies/ES3/phase_k
    [ -f results/predictions_student.json ] || "$VENV" s0_census.py student studies/ES3/phase_k/runs/ES3_K_f0.json
    [ -f studies/ES3/phase_l/registry.json ] || py -3.12 campaign.py es3_phase_l
    run encoder_runner.py studies/ES3/phase_l
  fi
fi
if fits 9.1; then
  [ -f results/predictions_e5.json ] || "$VENV" s0_census.py calibrate_only biolinkbert e5
  [ -f studies/E5/registry.json ] || py -3.12 campaign.py e5
  run encoder_runner.py studies/E5 models/biolinkbert_base
fi
if fits 9.6; then
  [ -f studies/E4/registry.json ] || py -3.12 campaign.py e4
  run encoder_runner.py studies/E4
fi
echo "[ext] $(date -u +%FT%TZ) extension campaign complete"
