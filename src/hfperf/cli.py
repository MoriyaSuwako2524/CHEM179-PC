"""Command line entry point:  hfperf h2o --backend naive"""
from __future__ import annotations

import argparse
import json
import platform
import time

from . import timing
from .backends import available, get_backend
from .molecule import available_molecules, load_molecule
from .scf import rhf


def _parse_opt(text):
    key, _, val = text.partition("=")
    low = val.lower()
    if low in ("true", "false"):
        return key, low == "true"
    for cast in (int, float):
        try:
            return key, cast(val)
        except ValueError:
            pass
    return key, val


def main(argv=None):
    ap = argparse.ArgumentParser(description="RHF/6-31G* with swappable kernels")
    ap.add_argument("molecule", help=f"bundled name ({', '.join(available_molecules())}) or an .xyz file")
    ap.add_argument("--backend", default="naive", help=f"one of {available()}")
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="backend constructor option, repeatable (e.g. --opt eri=sym8 --opt reuse=false)")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--no-diis", action="store_true")
    ap.add_argument("--e-tol", type=float, default=1e-10, help="energy convergence (loosen for FP32 GPU runs)")
    ap.add_argument("--d-tol", type=float, default=1e-8, help="rms density convergence")
    ap.add_argument("--json", help="write energy + timing breakdown to this file")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    opts = dict(_parse_opt(o) for o in a.opt)
    backend = get_backend(a.backend, **opts)
    timing.reset()
    t0 = time.perf_counter()
    res = rhf(load_molecule(a.molecule, a.charge), backend, diis=not a.no_diis, verbose=not a.quiet,
              e_tol=a.e_tol, d_tol=a.d_tol)
    wall = time.perf_counter() - t0
    print(f"E(RHF) = {res.energy:.10f} Eh   converged={res.converged}")
    print(f"wall time {wall:.3f} s\n" + timing.report())
    if a.json:
        rec = {"molecule": a.molecule, "backend": a.backend, "options": opts, "energy": res.energy,
               "converged": res.converged, "iterations": res.iterations, "wall_s": wall,
               "kernels": timing.summary(), "machine": {"cpu": platform.processor(),
                                                        "platform": platform.platform()}}
        with open(a.json, "w") as f:
            json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
