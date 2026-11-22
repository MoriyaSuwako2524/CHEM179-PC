"""Student exercise (stage 5): how wrong is FP32, and does mixed precision fix it?

    python ext/cuda/precision_study.py h2o                  # auto target (cupy -> mps -> numpy)
    python ext/cuda/precision_study.py h2o --target mps     # Apple Silicon GPU (FP32 only)
    python ext/cuda/precision_study.py c2h4 --target numpy  # CPU: FP32 *emulation*, any laptop
    python ext/cuda/precision_study.py h2o --csv my_errors.csv

For each precision mode it runs a full RHF and prints the error against the PySCF
reference energy in tests/data/reference_energies.json.
  fp64              everything double precision (skipped on MPS: no FP64 on Metal)
  fp32              J/K, matmul, eigh all in single precision
  fp32-jk-only      only J/K in single precision, matmul/eigh in FP64 (on the host)
  mixed             FP32 J/K of the density *change*, FP64 accumulation + FP64 re-anchoring
  mixed-no-reanchor the same with a single FP64 anchor at iteration 1
FP32 runs use loose SCF thresholds (dE < 1e-7, rms dD < 1e-5): FP32 noise never reaches 1e-10.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hfperf import build_basis, load_molecule, rhf, timing  # noqa: E402
from hfperf.backends.gpu import GpuBackend, best_cpu_eri_backend  # noqa: E402

LOOSE = dict(e_tol=1e-7, d_tol=1e-5)
MODES = [("fp64", dict(precision="fp64"), {}),
         ("fp32", dict(precision="fp32"), LOOSE),
         ("fp32-jk-only", dict(precision="fp32", linalg="host"), LOOSE),
         ("mixed", dict(precision="mixed"), {}),
         ("mixed-no-reanchor", dict(precision="mixed", anchor_factor=0), {})]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("molecule", nargs="?", default="h2o")
    ap.add_argument("--target", default="auto", help="auto | cupy | mps | torch-cuda | torch-cpu | numpy")
    ap.add_argument("--csv")
    a = ap.parse_args()
    e_ref = json.loads((ROOT / "tests/data/reference_energies.json").read_text())[a.molecule]["e_rhf"]
    mol = load_molecule(a.molecule)
    from hfperf.backends.gpu.xp import make_target
    make_target(a.target)   # import the device library first (native Windows: torch DLLs must load before BLAS runs)
    cpu = best_cpu_eri_backend()          # FP64 ERI tensor once (cpp -> rust -> numpy), reused by every mode
    t0 = time.perf_counter()
    G = cpu.eri(build_basis(mol)[1])
    print(f"ERI by {cpu.name} on the CPU: {time.perf_counter() - t0:.2f} s, {G.nbytes / 1e6:.1f} MB")
    rows = []
    print(f"{'mode':18s} {'E - E_ref (Eh)':>15s} {'iter':>5s} {'conv':>5s} {'FP64/FP32 J/K':>14s} {'wall s':>8s}")
    for label, opts, kw in MODES:
        try:
            be = GpuBackend(target=a.target, **opts)
        except ValueError as exc:        # e.g. fp64 on MPS
            print(f"{label:18s} skipped: {exc}")
            continue
        be.eri = lambda funcs, _G=G: _G      # GpuBackend.jk() uploads it on first use
        timing.reset()
        t0 = time.perf_counter()
        res = rhf(mol, be, verbose=False, **kw)
        wall = time.perf_counter() - t0
        s = be.gpu_stats
        rows.append(dict(molecule=a.molecule, target=be.t.name, mode=label, error_Eh=res.energy - e_ref,
                         iterations=res.iterations, converged=res.converged, fp64_builds=s["fp64_builds"],
                         fp32_builds=s["fp32_builds"], eigh_on=s["eigh_on"], wall_s=wall))
        print(f"{label:18s} {res.energy - e_ref:+15.2e} {res.iterations:5d} {str(res.converged):>5s} "
              f"{s['fp64_builds']:>6d}/{s['fp32_builds']:<7d} {wall:8.3f}")
    print(f"target: {be.describe()}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
