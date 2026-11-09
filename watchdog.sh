#!/usr/bin/env bash
# Campaign watchdog (read-only): exits with a one-line reason when a driver logs an error or stops,
# when no run finishes for 75 minutes, or when a new stage-completion line appears in a driver log.
cd "$(dirname "$0")"
start_err=$(cat drive.log drive_ext.log drive_ext2.log 2>/dev/null | grep -Ec "Traceback|Error|pending runs: exiting|failed")
start_stage=$(cat drive.log drive_ext.log drive_ext2.log 2>/dev/null | grep -Ec "completed|complete|passed")
start_tb=$(cat studies/*/console.log studies/*/*/console.log 2>/dev/null | grep -c Traceback)
last_done=$(date +%s)
last_count=$(cat studies/*/console.log studies/*/*/console.log 2>/dev/null | grep -c " done ")
while true; do
  sleep 60
  err=$(cat drive.log drive_ext.log drive_ext2.log 2>/dev/null | grep -Ec "Traceback|Error|pending runs: exiting|failed")
  if [ "$err" -gt "$start_err" ]; then echo "WATCHDOG: driver error or failure line"; tail -n 3 drive.log drive_ext.log drive_ext2.log; exit 0; fi
  stage=$(cat drive.log drive_ext.log drive_ext2.log 2>/dev/null | grep -Ec "completed|complete|passed")
  if [ "$stage" -gt "$start_stage" ]; then echo "WATCHDOG: stage completed"; tail -n 2 drive.log drive_ext.log drive_ext2.log; exit 0; fi
  count=$(cat studies/*/console.log studies/*/*/console.log 2>/dev/null | grep -c " done ")
  if [ "$count" -gt "$last_count" ]; then last_count=$count; last_done=$(date +%s); fi
  if [ $(( $(date +%s) - last_done )) -gt 4500 ]; then echo "WATCHDOG: no run finished for 75 min"; exit 0; fi
  tb=$(cat studies/*/console.log studies/*/*/console.log 2>/dev/null | grep -c Traceback)
  if [ "$tb" -gt "$start_tb" ]; then echo "WATCHDOG: new Traceback in a study console log"; exit 0; fi
done
