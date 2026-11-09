"""Thermal governor (engineering only; no effect on any computation).

Every few seconds it reads the GPU temperature. At or above T_HIGH it suspends every experiment process
(encoder_runner, lm_runner, audit, measure_cost, s0_census); the GPU finishes its queued kernels and
idles. At or below T_LOW it resumes them. Suspension only stretches wall-clock time: seeds, data order,
noise counters and kernels are unchanged, and no runner logic depends on elapsed time. Each pause is
appended to studies/thermal_pauses.jsonl so budget.py can subtract it from the ledgers' wall time.

  thermal_governor.py run [T_HIGH T_LOW RUN_S REST_S]   (defaults 70 60 degC, 60 s work / 60 s rest;
                                                         stops when unified/GOVERNOR_STOP exists)
  thermal_governor.py resume               (resumes any process left suspended, e.g. after a crash)
Always stop it with the GOVERNOR_STOP file, never by killing it, so it resumes what it suspended.
"""
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "studies", "thermal_pauses.jsonl")
STATE = os.path.join(ROOT, "studies", "thermal_suspended.json")
STOPFILE = os.path.join(ROOT, "GOVERNOR_STOP")
PATTERNS = ("encoder_runner.py", "lm_runner.py", "audit.py", "measure_cost.py", "s0_census.py")
PROCESS_SUSPEND_RESUME = 0x0800
ntdll, kernel32 = ctypes.WinDLL("ntdll"), ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.OpenProcess.restype = ctypes.c_void_p
kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
ntdll.NtSuspendProcess.argtypes = (ctypes.c_void_p,)
ntdll.NtResumeProcess.argtypes = (ctypes.c_void_p,)


def utc():
    return datetime.now(timezone.utc).isoformat()


def gpu_temp():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout.strip()
    return int(out.splitlines()[0])


def experiment_pids():
    cmd = ["powershell", "-NoProfile", "-Command",
           "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    pids = []
    for line in out.splitlines():
        pid, _, cmdline = line.partition("\t")
        if pid.strip().isdigit() and any(p in cmdline for p in PATTERNS):
            pids.append(int(pid))
    return sorted(pids)


def call(pids, fn):
    done = []
    for pid in pids:
        h = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if h:
            if fn(h) == 0:
                done.append(pid)
            kernel32.CloseHandle(h)
    return done


def save_state(pids):
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump({"utc": utc(), "suspended": pids}, f)


def resume_from_state():
    if os.path.exists(STATE):
        pids = json.load(open(STATE, encoding="utf-8")).get("suspended", [])
        print("resumed", call(pids, ntdll.NtResumeProcess), flush=True)
        os.remove(STATE)


def run(t_high=70, t_low=60, run_s=60, rest_s=60, poll=3.0):
    """Suspend when the GPU reaches t_high, or after run_s seconds of work (duty cycle); resume only when at
    least rest_s seconds have passed (duty pauses) and the GPU is at or below t_low. With run_s = rest_s the
    average GPU power, and so the heat the laptop must shed, is at most half of full speed."""
    resume_from_state()
    suspended, start, peak, reason, working_since, min_end = [], None, 0, "", None, 0.0
    cache, cache_at = [], 0.0
    print(f"[governor] {utc()} running: work {run_s} s / rest >= {rest_s} s; suspend at >= {t_high} C; "
          f"resume at <= {t_low} C", flush=True)
    try:
        while not os.path.exists(STOPFILE):
            try:
                t = gpu_temp()
            except Exception as exc:  # sensor hiccup: never leave work suspended because of it
                print(f"[governor] temperature read failed: {exc}", flush=True)
                t = t_low
            now = time.time()
            if not suspended:
                if now - cache_at >= 15 or t >= t_high or (working_since and now - working_since >= run_s):
                    cache, cache_at = experiment_pids(), now  # refresh the process list at most every 15 s
                pids = cache
                if not pids:
                    working_since = None
                else:
                    working_since = working_since or now
                    due = now - working_since >= run_s
                    if t >= t_high or due:
                        suspended, start, peak = call(pids, ntdll.NtSuspendProcess), now, t
                        reason = "thermal" if t >= t_high else "duty"
                        # every rest lasts at least as long as the work stretch before it (duty <= 50%)
                        min_end = now + max(now - working_since, rest_s if reason == "duty" else 0)
                        save_state(suspended)
            else:
                peak = max(peak, t)
                if t <= t_low and now >= min_end:
                    call(suspended, ntdll.NtResumeProcess)
                    end = time.time()
                    working_since = end
                    rec = {"start": datetime.fromtimestamp(start, timezone.utc).isoformat(), "end": utc(),
                           "seconds": round(end - start, 1), "pids": suspended, "reason": reason,
                           "peak_c": peak, "resume_c": t}
                    with open(LOG, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    suspended = []
                    os.remove(STATE)
            time.sleep(poll)
    finally:
        if suspended:
            call(suspended, ntdll.NtResumeProcess)
            if os.path.exists(STATE):
                os.remove(STATE)
        print(f"[governor] {utc()} stopped; nothing left suspended", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*(int(x) for x in sys.argv[2:6]))
    elif sys.argv[1] == "resume":
        resume_from_state()
