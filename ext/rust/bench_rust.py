"""Benchmark the Rust backends: ERI time, J/K time, total SCF time, energy error.

    python ext/rust/bench_rust.py --variants naive sym8_noreuse sym8_reuse sym8_reuse_unchecked \
        --molecules h2o nh3 ch4 c2h4 benzene --out results/rust/run1

Writes <out>.json (full records) and <out>.csv (one row per variant x molecule).
Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 for single-core numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from hfperf import build_basis, load_molecule, rhf, timing
from hfperf.backends.rust import RustBackend, RustNaiveBackend, RustSym8Backend, RustUncheckedBackend

VARIANTS = {
    "naive": RustNaiveBackend,                     # eri_naive: all N^4, recursive E/R
    "sym8_noreuse": RustSym8Backend,               # eri_sym8(reuse=False)
    "sym8_reuse": RustBackend,                     # eri_sym8(reuse=True), bounds-checked (default)
    "sym8_reuse_unchecked": RustUncheckedBackend,  # same, get_unchecked in ERI and J/K
    "sym8_reuse_strict": lambda: RustBackend(linalg="naive", onee="naive"),  # only ERI+JK in Rust
}
REF = json.loads((Path(__file__).resolve().parents[2] / "tests/data/reference_energies.json").read_text())


def machine():
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        model = next((l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")), "")
    except OSError:
        lscpu, model = "", platform.processor()
    return {"cpu_model": model, "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "hostname": platform.node(),
            "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                           "MKL_NUM_THREADS", "SLURM_CPUS_PER_TASK")},
            "lscpu": lscpu}


def jk_micro(G, D, repeats=3):
    import hfperf_rust
    out = {}
    for mode in ("iter", "checked", "unchecked"):
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            hfperf_rust.jk(G, D, mode=mode)
            ts.append(time.perf_counter() - t0)
        out[mode] = min(ts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["sym8_reuse"], choices=list(VARIANTS))
    ap.add_argument("--molecules", nargs="+", default=["h2o"])
    ap.add_argument("--jk-micro", action="store_true", help="also time the 3 J/K flavours at convergence")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mach = machine()
    print("CPU:", mach["cpu_model"], flush=True)
    records = []
    for mol_name in a.molecules:
        for var in a.variants:
            mol = load_molecule(mol_name)
            nbf = len(build_basis(mol)[1])
            be = VARIANTS[var]()
            cache = {}
            if a.jk_micro:  # keep the tensor the SCF computed for the J/K micro-benchmark
                eri0 = be.eri
                be.eri = lambda funcs: cache.setdefault("G", eri0(funcs))
            timing.reset()
            t0 = time.perf_counter()
            res = rhf(mol, be, verbose=False)
            wall = time.perf_counter() - t0
            k = timing.summary()
            ref = REF.get(mol_name, {}).get("e_rhf")
            rec = {"variant": var, "backend": be.name, "molecule": mol_name, "nbf": nbf,
                   "energy": res.energy, "e_ref": ref, "abs_err": abs(res.energy - ref) if ref else None,
                   "converged": res.converged, "iterations": res.iterations,
                   "eri_s": k["ERI"]["total_s"], "jk_s": k["JK"]["total_s"], "jk_calls": k["JK"]["calls"],
                   "stv_s": k["S,T,V"]["total_s"], "scf_wall_s": wall, "kernels": k,
                   "eri_stats": be.last_eri_stats}
            if a.jk_micro:
                rec["jk_micro_s"] = jk_micro(cache["G"], np.ascontiguousarray(res.density))
                cache.clear()
            records.append(rec)
            print(f"{mol_name:12s} {var:22s} nbf={nbf:4d} ERI {rec['eri_s']:10.3f} s  JK {rec['jk_s']:8.3f} s "
                  f"SCF {wall:10.3f} s  E={res.energy:.10f}  err={rec['abs_err']:.1e}  "
                  f"{rec.get('jk_micro_s', '')}", flush=True)
            # write after every run so a time-out still leaves data
            out.with_suffix(".json").write_text(json.dumps({"machine": mach, "records": records}, indent=1))
            cols = ["variant", "molecule", "nbf", "eri_s", "jk_s", "jk_calls", "stv_s", "scf_wall_s",
                    "iterations", "energy", "e_ref", "abs_err", "converged"]
            with out.with_suffix(".csv").open("w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(cols + ["eri_computed", "eri_screened", "jk_iter_s", "jk_checked_s",
                                    "jk_unchecked_s", "cpu_model"])
                for r in records:
                    st = r["eri_stats"] or {}
                    jm = r.get("jk_micro_s", {})
                    wr.writerow([r[c] for c in cols] + [st.get("computed"), st.get("screened"),
                                                        jm.get("iter"), jm.get("checked"),
                                                        jm.get("unchecked"), mach["cpu_model"]])
            del be, res


if __name__ == "__main__":
    main()
