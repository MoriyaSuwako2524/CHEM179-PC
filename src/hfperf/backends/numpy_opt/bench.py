"""Benchmark driver for the stage-1 ladder (run on a quiet machine, single thread).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -m hfperf.backends.numpy_opt.bench --molecules h2o nh3 --rungs R2 R3 R4 R5 --out eri.json
    python -m hfperf.backends.numpy_opt.bench --molecules h2o --kernels --out kernels.json

Rungs are cumulative; everything outside the ERI (one-electron integrals, J/K, linear
algebra) uses the stage-1 defaults in every rung, so differences come from the ERI alone.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from hfperf import build_basis, load_molecule, rhf, timing
from hfperf.backends.naive.fock import jk as jk_loop
from hfperf.backends.naive.linalg import jacobi_eigh

from .backend import NumpyBackend

RUNGS = {
    "R0": ("full N^4, naive integrals", dict(eri="full", screening=0, boys="series")),
    "R1": ("+ 8-fold symmetry", dict(eri="sym8", screening=0, boys="series")),
    "R2": ("+ shell quartets, shell-pair data reuse", dict(eri="shell", screening=0, boys="series")),
    "R3": ("+ Schwarz screening 1e-12", dict(eri="shell", screening=1e-12, boys="series")),
    "R4": ("+ Boys table/Taylor/downward recursion", dict(eri="shell", screening=1e-12, boys="table")),
    "R5": ("+ NumPy vectorization over primitive pairs", dict(eri="vector", screening=1e-12, boys="table")),
}

REF = json.loads((Path(__file__).resolve().parents[4] / "tests" / "data" / "reference_energies.json").read_text())


def cpu_model():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_rung(mol_name, rung):
    desc, opts = RUNGS[rung]
    be = NumpyBackend(**opts)
    mol = load_molecule(mol_name)
    timing.reset()
    t0 = time.perf_counter()
    res = rhf(mol, be, verbose=False)
    wall = time.perf_counter() - t0
    summ = timing.summary()
    ref = REF.get(mol_name, {}).get("e_rhf")
    return {
        "rung": rung, "description": desc, "options": be.opts, "molecule": mol_name,
        "nbf": len(build_basis(mol)[1]), "scf_seconds": wall,
        "kernels": {k: v["total_s"] for k, v in summ.items()},
        "eri_seconds": summ["ERI"]["total_s"], "stats": be.stats,
        "energy": res.energy, "energy_error": None if ref is None else res.energy - ref,
        "converged": res.converged, "iterations": res.iterations,
    }


def time_kernels(mol_name, loop_jk=True, jacobi=True):
    """J/K loop vs einsum vs BLAS, and Jacobi vs LAPACK, on the converged density / Fock matrix."""
    mol = load_molecule(mol_name)
    _, funcs = build_basis(mol)
    be = NumpyBackend()
    res = rhf(mol, be, verbose=False)
    G = be.eri(funcs)
    D = res.density
    out = {"molecule": mol_name, "nbf": len(funcs)}

    def best(f, reps):
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
        return min(ts), r

    tb, (Jb, Kb) = best(lambda: be.jk(G, D), 5)
    te, (Je, Ke) = best(lambda: NumpyBackend(jk="einsum").jk(G, D), 3)
    out["jk_blas_s"], out["jk_einsum_s"] = tb, te
    out["jk_einsum_maxdiff"] = float(max(abs(Je - Jb).max(), abs(Ke - Kb).max()))
    if loop_jk:
        tl, (Jl, Kl) = best(lambda: jk_loop(G, D), 1)
        out["jk_loop_s"] = tl
        out["jk_loop_maxdiff"] = float(max(abs(Jl - Jb).max(), abs(Kl - Kb).max()))
    # Roothaan matrix F' = X^T F X, same as inside the SCF
    S = be.overlap(funcs)
    s, U = np.linalg.eigh(S)
    X = (U / np.sqrt(s)) @ U.T
    J, K = Jb, Kb
    H = be.kinetic(funcs) + be.nuclear(funcs, mol)
    Fp = X.T @ (H + J - 0.5 * K) @ X
    tL, (wL, _) = best(lambda: np.linalg.eigh(Fp), 5)
    out["eigh_lapack_s"] = tL
    if jacobi:
        tJ, (wJ, _) = best(lambda: jacobi_eigh(Fp), 1)
        out["eigh_jacobi_s"] = tJ
        out["eigh_maxdiff"] = float(abs(wJ - wL).max())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", nargs="+", default=["h2o"])
    ap.add_argument("--rungs", nargs="+", default=list(RUNGS))
    ap.add_argument("--kernels", action="store_true", help="time J/K and eigh variants instead of the ladder")
    ap.add_argument("--no-loop-jk", action="store_true")
    ap.add_argument("--out", default="bench_stage1.json")
    a = ap.parse_args(argv)
    meta = {"cpu": cpu_model(), "numpy": np.__version__, "python": platform.python_version(),
            "host": platform.node()}
    rows = []
    for m in a.molecules:
        if a.kernels:
            r = time_kernels(m, loop_jk=not a.no_loop_jk)
            r["cpu"] = meta["cpu"]
            rows.append(r)
            print(json.dumps(r), flush=True)
        else:
            for rung in a.rungs:
                r = run_rung(m, rung)
                r["cpu"] = meta["cpu"]
                rows.append(r)
                print(f"{m:8s} {rung} nbf={r['nbf']:4d} ERI {r['eri_seconds']:9.3f} s  SCF {r['scf_seconds']:9.3f} s  "
                      f"dE={r['energy_error']:.1e}  quartets={r['stats']['quartets']} "
                      f"contracted={r['stats']['contracted']}", flush=True)
                Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
