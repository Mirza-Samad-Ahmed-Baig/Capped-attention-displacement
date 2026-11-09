"""Run bookkeeping shared by the unified-study runners (no torch import).

- write_json_once / sha256_file: write-once artifacts with hashes.
- Registry: every run is declared (write-once) before execution; results are matched to
  declarations, so no run can be silently added, dropped, or replaced.
- Ledger: append-only GPU-seconds ledger used by the prespecified cut rules.
- ThroughputGuard: flags a run whose steady-state throughput falls below a fraction of the
  pre-freeze benchmark (a symptom of silent VRAM spill or thermal/power collapse).
- GpuLogger: 1 Hz nvidia-smi log (clocks, power, temperature, memory) for the campaign.
"""
import datetime
import hashlib
import json
import os
import subprocess
import time


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json_once(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    if os.path.exists(path):
        os.remove(tmp)
        raise FileExistsError(path)
    os.replace(tmp, path)
    return sha256_file(path)


class Registry:
    """runs.json: the complete, ordered list of declared runs for one study."""

    def __init__(self, folder):
        self.folder = folder
        self.path = os.path.join(folder, "registry.json")

    def declare(self, runs, protocol_sha256):
        ids = [r["run_id"] for r in runs]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate run ids")
        if os.path.exists(self.path):
            existing = self.load()
            if existing["runs"] != runs or existing["protocol_sha256"] != protocol_sha256:
                raise RuntimeError("Registry already declared with different runs/protocol")
            return existing
        record = {"utc": utc(), "protocol_sha256": protocol_sha256, "runs": runs}
        write_json_once(self.path, record)
        return record

    def load(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def result_path(self, run_id):
        return os.path.join(self.folder, "runs", f"{run_id}.json")

    def done(self, run_id):
        return os.path.exists(self.result_path(run_id))

    def pending(self):
        return [r for r in self.load()["runs"] if not self.done(r["run_id"])]


class Ledger:
    def __init__(self, path):
        self.path = path

    def add(self, run_id, seconds, note=""):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": utc(), "run_id": run_id, "gpu_seconds": seconds, "note": note}) + "\n")

    def total_hours(self):
        if not os.path.exists(self.path):
            return 0.0
        with open(self.path, encoding="utf-8") as f:
            return sum(json.loads(line)["gpu_seconds"] for line in f if line.strip()) / 3600


class ThroughputGuard:
    def __init__(self, expected_per_s, floor=0.6, warmup_items=64):
        self.expected, self.floor, self.warmup = expected_per_s, floor, warmup_items
        self.t0, self.items = None, 0

    def start(self):
        self.t0, self.items = time.perf_counter(), 0

    def update(self, items):
        self.items += items
        if self.expected and self.items >= self.warmup:
            rate = self.items / (time.perf_counter() - self.t0)
            if rate < self.floor * self.expected:
                return rate
        return None


class GpuLogger:
    FIELDS = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu,clocks_event_reasons.active"

    def __init__(self, path):
        self.path, self.proc, self.handle = path, None, None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.handle = open(self.path, "a", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader",
                                          "-lms", "1000"], stdout=self.handle, stderr=subprocess.DEVNULL)
        except OSError:
            self.proc = None
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.handle.close()
        return False
