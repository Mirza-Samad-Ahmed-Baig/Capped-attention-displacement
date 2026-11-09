"""One-off remediation (2026-09-29) of the LM DropKey implementation bug: lm_runner ignored the DK
arm's `layer_schedule` ("dropkey_linear"), so the three S2 DK tuning runs used p_max in every layer
(realized mean displacement ~1.7x the calibrated level). Outcome-blind: no test file is opened.

Moves, unopened, the three S2 DK tuning results and the surviving DK sealed files to
excluded/S2_DK_noschedule/ (outside studies/, so no analysis indexes them), with a registry that keeps
them in the run inventory. Supersedes studies/S2/decisions_levels.json and studies/S2/finals/registry.json
(only S2_AD@0.20_s2, unaffected by DK, had run), which drive.sh then recreates: rerun the three DK
tuning runs with the fixed runner, re-apply the unchanged selection rule, redeclare the finals.
"""
import json
import os
import shutil

import runlib

ROOT = os.path.dirname(os.path.abspath(__file__))
S2 = os.path.join(ROOT, "studies", "S2")
Q = os.path.join(ROOT, "excluded", "S2_DK_noschedule")
DK = ["S2_DK@0.05_s1", "S2_DK@0.10_s1", "S2_DK@0.20_s1"]


def main():
    if not os.path.exists(os.path.join(ROOT, "STOP")):
        raise SystemExit("run only while the campaign is stopped")
    done_finals = sorted(os.listdir(os.path.join(S2, "finals", "runs")))
    if done_finals != ["S2_AD@0.20_s2.json"]:
        raise SystemExit(f"unexpected finals results: {done_finals}")
    os.makedirs(os.path.join(Q, "runs"), exist_ok=True)
    os.makedirs(os.path.join(Q, "sealed"), exist_ok=True)
    reg = runlib.Registry(os.path.join(S2, "tuning")).load()
    specs = [r for r in reg["runs"] if r["run_id"] in DK]
    moved = []
    for rid in DK:
        src = os.path.join(S2, "tuning", "runs", f"{rid}.json")
        shutil.move(src, os.path.join(Q, "runs", f"{rid}.json"))
        moved.append(f"runs/{rid}.json")
        for suffix in ("_final", "_best"):
            sealed = os.path.join(S2, "tuning", "sealed", f"{rid}{suffix}.json")
            if os.path.exists(sealed):
                shutil.move(sealed, os.path.join(Q, "sealed", f"{rid}{suffix}.json"))  # moved unopened
                moved.append(f"sealed/{rid}{suffix}.json")
    runlib.Registry(Q).declare(specs, reg["protocol_sha256"])
    for src, dst in ((os.path.join(S2, "decisions_levels.json"), "decisions_levels_superseded.json"),
                     (os.path.join(S2, "finals", "registry.json"), "finals_registry_superseded.json")):
        shutil.move(src, os.path.join(Q, dst))
        moved.append(dst)
    note = {"utc": runlib.utc(), "reason": __doc__.split("\n\n")[0], "moved": moved,
            "status": "excluded from every analysis; reported in the run inventory; sealed files never opened"}
    runlib.write_json_once(os.path.join(Q, "README.json"), note)
    print(json.dumps(note, indent=1))


if __name__ == "__main__":
    main()
