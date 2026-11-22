"""Matmul in C (via ctypes): here loop order and cache blocking matter.

    make                                    # builds libmm_O2.so, libmm_O3native.so, libmm_O3native512.so
    OPENBLAS_NUM_THREADS=1 python c_ladder.py --sizes 128 256 512 1024 2048 --out c_ladder.csv
    python c_ladder.py --machine --out machine.json   # STREAM triad + FMA peak probe

Each variant's result is checked against NumPy (A @ B).
"""
import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TidyCSV, best_time, cpu_model

HERE = Path(__file__).resolve().parent
DPTR = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
EXT = "dylib" if sys.platform == "darwin" else "so"


def load(build):
    lib = ctypes.CDLL(str(HERE / f"libmm_{build}.{EXT}"))
    for name in ("mm_ijk", "mm_ikj", "mm_tB"):
        getattr(lib, name).argtypes = [ctypes.c_int, DPTR, DPTR, DPTR]
        getattr(lib, name).restype = None
    lib.mm_blocked.argtypes = [ctypes.c_int, DPTR, DPTR, DPTR, ctypes.c_int]
    lib.mm_blocked.restype = None
    lib.stream_triad.argtypes = [ctypes.c_long, DPTR, DPTR, DPTR, ctypes.c_double, ctypes.c_int]
    lib.stream_triad.restype = ctypes.c_double
    lib.peak_fma.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_double)]
    lib.peak_fma.restype = ctypes.c_double
    return lib


def reps_for(N):
    return 3 if N <= 512 else 1


def ladder(builds, sizes, tiles, out, skip_ijk_above):
    tab = TidyCSV(out)
    rng = np.random.default_rng(179)
    for N in sizes:
        A, B = rng.random((N, N)), rng.random((N, N))
        ref = A @ B
        C = np.empty((N, N))
        tab.add("blas_numpy_1thread", N, best_time(lambda: np.matmul(A, B, out=C), 5 if N <= 1024 else 3))
        for build in builds:
            lib = load(build)
            jobs = [("ijk", lambda: lib.mm_ijk(N, A, B, C)), ("ikj", lambda: lib.mm_ikj(N, A, B, C)),
                    ("tB", lambda: lib.mm_tB(N, A, B, C))]
            jobs += [(f"blocked{t}", (lambda t=t: lib.mm_blocked(N, A, B, C, t))) for t in tiles]
            for name, f in jobs:
                if name == "ijk" and N > skip_ijk_above:
                    continue
                f()
                assert np.allclose(C, ref, rtol=1e-12, atol=1e-9), (build, name, N)
                tab.add(f"c_{build}_{name}", N, best_time(f, reps_for(N)))


def lscpu():
    try:
        return subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    except OSError:
        return ""


def machine(builds, out):
    info = {"cpu": cpu_model(), "lscpu": lscpu()}
    m = re.search(r"@\s*([\d.]+)\s*GHz", info["cpu"])
    ghz = float(m.group(1)) if m else None
    flags = info["lscpu"]
    lanes = 8 if "avx512f" in flags else (4 if "avx2" in flags else 2)
    fma_units = 2  # Skylake-SP/Cascade Lake Gold 6xxx: two 512-bit FMA units per core
    info["peak_theory"] = {
        "base_ghz": ghz, "fp64_lanes": lanes, "fma_units": fma_units,
        "flop_per_cycle": 2 * lanes * fma_units,
        "gflops_at_base": None if ghz is None else ghz * 2 * lanes * fma_units,
        "note": "AVX-512 heavy code runs below the base clock (license-based frequency); turbo may be higher",
    }
    n = 20_000_000  # 3 arrays x 160 MB, far larger than L3
    a, b, c = np.zeros(n), np.random.rand(n), np.random.rand(n)
    info["stream_triad"] = {}
    info["peak_fma"] = {}
    for build in builds:
        lib = load(build)
        t = lib.stream_triad(n, a, b, c, 3.0, 10)
        info["stream_triad"][build] = {"n": n, "seconds": t, "GB_per_s": 24.0 * n / t / 1e9}
        sink = ctypes.c_double()
        iters = 3_000_000_000 // 12                         # ~0.5 s per run: long enough for clock transitions
        lib.peak_fma(iters, ctypes.byref(sink))             # warm up (AVX-512 frequency licence, turbo ramp)
        t = min(lib.peak_fma(iters, ctypes.byref(sink)) for _ in range(3))
        info["peak_fma"][build] = {"iters": iters, "seconds": t, "gflops": iters * 12 * 8 * 2 / t / 1e9}
    Path(out).write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "lscpu"}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", default=["O2", "O3native", "O3native512"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1000, 1024, 2048])
    ap.add_argument("--tiles", nargs="+", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--skip-ijk-above", type=int, default=4096)
    ap.add_argument("--machine", action="store_true")
    ap.add_argument("--out", default="c_ladder.csv")
    a = ap.parse_args(argv)
    os.chdir(HERE)
    if a.machine:
        machine(a.builds, a.out)
    else:
        ladder(a.builds, a.sizes, a.tiles, a.out, a.skip_ijk_above)


if __name__ == "__main__":
    main()
