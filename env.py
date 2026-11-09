"""Process setup shared by every unified-study GPU/CPU runner. Import this FIRST.

- Blocks pandas/pyarrow in model processes. On this laptop, any process that loads
  pyarrow's arrow.dll together with a CUDA model crashes in arrow.dll at interpreter
  teardown (Windows event 1000, exception 0xc0000005, identical offset), writing a
  ~0.55 GB crash dump. Parquet reading is therefore confined to CPU-only data-preparation
  scripts, which call `allow_arrow()` instead.
- Disables transformers' torchvision probe: the inherited global torchvision build lacks
  an operator required by this torch build. Process-local; no global package is changed.
- Seals historical PubMedQA test/raw files against accidental opening.
- Caps the CUDA allocator so VRAM exhaustion raises OOM instead of silently spilling into
  shared system memory (WDDM), which corrupts timing and memory measurements.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(ROOT)
SEALED = [os.path.join(WORKSPACE, "domain_distill", "data", name)
          for name in ("test.json", "test_ground_truth.json", "ori_pqal.json")]
SEALED += [os.path.join(ROOT, "data", "lm", "test.bin"), os.path.join(ROOT, "data", "lm", "test_docs.json"),
           os.path.join(ROOT, "data", "raw", "pqa_labeled.parquet"),
           os.path.join(ROOT, "data", "scifact", "sealed_labels", "dev_labels.json")]
_sealed_paths = set()
_permitted = set()
SEALED_ACCESS_SCRIPTS = ("prepare_eval_inputs.py", "sealed_eval.py", "unseal.py")


def block_arrow():
    for name in ("pyarrow", "pandas"):
        if name in sys.modules and sys.modules[name] is not None:
            raise RuntimeError(f"{name} was imported before env.block_arrow()")
        sys.modules[name] = None


def allow_arrow():
    for name in ("pyarrow", "pandas"):
        if sys.modules.get(name, 0) is None:
            del sys.modules[name]


def patch_transformers_imports():
    from transformers.utils import import_utils
    import_utils._torchvision_available = False


def seal(extra=()):
    """Raise PermissionError on any Python-level open of the sealed historical files."""
    targets = {os.path.normcase(os.path.abspath(p)) for p in list(SEALED) + list(extra)}
    new = targets - _sealed_paths
    if not new:
        return
    _sealed_paths.update(new)
    if len(_sealed_paths) != len(new):
        return  # hook already installed; it reads the shared set

    def guard(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            try:
                path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
            except (TypeError, ValueError):
                return
            if path in _sealed_paths and path not in _permitted:
                raise PermissionError(f"Sealed evaluation file: {path}")

    sys.addaudithook(guard)


def cap_vram(reserve_bytes=0):
    """Limit this process to the currently free VRAM (minus a reserve); returns the cap."""
    import torch
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    fraction = max(0.05, min(1.0, (free - reserve_bytes) / total))
    torch.cuda.set_per_process_memory_fraction(fraction)
    return int(fraction * total)


def boost_process():
    """Process-local: HIGH priority and opt out of Windows power throttling (EcoQoS). Without it,
    background console processes on this hybrid-core laptop run on efficiency cores and starve the
    GPU (measured ~35% vs ~97% utilization). No system setting is changed."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    handle = k32.GetCurrentProcess()
    ok_priority = k32.SetPriorityClass(handle, 0x00000080)  # HIGH_PRIORITY_CLASS

    class PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    state = PowerThrottling(1, 0x1, 0x0)  # EXECUTION_SPEED controlled and disabled
    ok_throttle = k32.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state))
    return bool(ok_priority), bool(ok_throttle)


def setup_model_process(threads=2, seal_test=True):
    """Standard preamble for model-training/evaluation processes."""
    boost_process()
    block_arrow()
    if seal_test:
        seal()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic eval replays
    patch_transformers_imports()
    import torch
    torch.set_num_threads(threads)
    return torch


class sealed_access:
    """Temporarily permit opening specific sealed paths. Only the scripts named in
    SEALED_ACCESS_SCRIPTS may use it (checked against the caller's file name)."""

    def __init__(self, *paths):
        self.paths = {os.path.normcase(os.path.abspath(p)) for p in paths}

    def __enter__(self):
        caller = os.path.basename(sys._getframe(1).f_code.co_filename)
        if caller not in SEALED_ACCESS_SCRIPTS:
            raise PermissionError(f"sealed_access is not allowed from {caller}")
        _permitted.update(self.paths)
        return self

    def __exit__(self, *exc):
        _permitted.difference_update(self.paths)
        return False
