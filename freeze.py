"""Freeze the protocol, code and data manifests (PROTOCOL A §3.1 item 3). Write-once FREEZE.json.

Runs the unit tests first (all must pass), then hashes PROTOCOL.md, every unified/*.py and
tests/*.py, data manifests and derived split files, sealed-input files and the G0 records.
"""
import glob
import json
import os
import subprocess
import sys

import runlib

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    out = os.path.join(ROOT, "FREEZE.json")
    if os.path.exists(out):
        raise RuntimeError("FREEZE.json exists")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=ROOT,
                           capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    tail = (tests.stdout + tests.stderr).strip().splitlines()[-3:]
    if tests.returncode != 0:
        raise SystemExit("tests failed:\n" + "\n".join(tail))
    files = ["PROTOCOL.md"] + sorted(glob.glob("*.py", root_dir=ROOT)) + sorted(glob.glob("tests/*.py", root_dir=ROOT))
    files += ["data/extra_manifest.json", "data/decontamination.json", "data/pqal_pool_manifest.json",
              "data/pqal_pool.jsonl", "data/pqal_folds.json", "data/pqal_pool_sections.json",
              "data/sealed_inputs/pqal_test_inputs.jsonl", "data/sealed_inputs/test_donors.json",
              "data/lm/lm_manifest.json", "data/lm/tokenizer.json", "results/g0.json", "results/g0_rerun_threads2.json",
              "results/prefreeze_benchmark_encoder.json", "results/prefreeze_benchmark_lm.json"]
    record = {"utc": runlib.utc(), "tests": tail, "sha256": {f: runlib.sha256_file(os.path.join(ROOT, f)) for f in files},
              "note": "Interim registration record; the authors are to register PROTOCOL.md on OSF under embargo (B14)."}
    runlib.write_json_once(out, record)
    print(json.dumps({"files": len(files), "tests": tail, "protocol_sha256": record["sha256"]["PROTOCOL.md"]}, indent=1))
    print("FREEZE.json", runlib.sha256_file(out))


if __name__ == "__main__":
    main()
