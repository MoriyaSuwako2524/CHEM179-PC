"""Strong-scaling driver for the threaded backends (stage 4).

    python -m hfperf.parallel.scaling eri  benzene --backend cpp-omp    --threads 1 2 4 8
    python -m hfperf.parallel.scaling eri  benzene --backend cpp-omp    --threads 1 2 4 8 --opt schedule=static --opt chunk=0
    python -m hfperf.parallel.scaling eri  benzene --backend rust-rayon --threads 1 2 4 8 --opt mode=block
    python -m hfperf.parallel.scaling jk   benzene --backend cpp-omp    --threads 1 2 4 8
    python -m hfperf.parallel.scaling fock benzene --backend cpp-omp    --threads 1 2 4 8 --opt accumulate=atomic
    python -m hfperf.parallel.scaling scf  benzene --backend rust-rayon --threads 1 2 4 8

What is timed
    eri   one call of backend.eri(): the N^4 tensor from the unique quartets (sym8 + Schwarz)
    jk    one J/K build from the stored tensor (memory-bandwidth bound)
    fock  one integral-direct J/K build (no tensor), fock=direct; the density is the
          core-Hamiltonian guess.  With --opt store=true the first build fills the cache and
          a second build (from the cache) is timed separately ("cached" rows).
    scf   a full RHF run; the energy is checked against tests/data/reference_energies.json
Every row records the per-thread busy times -> load imbalance = max/mean.
Laptop: use --threads 1 2 4 8 (or up to your core count) and the molecules h2o..benzene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np

from .. import timing
from ..backends import get_backend
from ..basis import build_basis
from ..cli import _parse_opt
from ..molecule import load_molecule
from ..scf import rhf
from .analysis import amdahl_fit, karp_flatt, load_imbalance, speedup_table

REF_ENERGIES = Path(__file__).resolve().parents[3] / "tests" / "data" / "reference_energies.json"


def machine_info() -> dict:
    info = {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count(), "numpy": np.__version__}
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        info["cpu"] = platform.processor() or platform.machine()
    info["env"] = {k: v for k, v in os.environ.items()
                   if k.startswith(("OMP_", "RAYON_", "OPENBLAS_", "MKL_", "SLURM_CPUS", "SLURM_JOB_ID", "GOMP_"))}
    return info


def ref_energy(name):
    try:
        return json.loads(REF_ENERGIES.read_text())[name]["e_rhf"]
    except (OSError, KeyError):
        return None


def core_guess_density(mol, funcs):
    from ..backends.numpy_opt import NumpyBackend
    nb = NumpyBackend()
    S = nb.overlap(funcs)
    H = nb.kinetic(funcs) + nb.nuclear(funcs, mol)
    s, U = np.linalg.eigh(S)
    X = U @ np.diag(s ** -0.5) @ U.T
    _, Cp = np.linalg.eigh(X.T @ H @ X)
    C = X @ Cp
    nocc = mol.n_electrons // 2
    return np.ascontiguousarray(2.0 * C[:, :nocc] @ C[:, :nocc].T)


def digest(a) -> str:
    return hashlib.blake2b(np.ascontiguousarray(a).view(np.uint8), digest_size=12).hexdigest()


def run(kind, molname, backend, threads, opts, repeat=1, ref="auto"):
    mol = load_molecule(molname)
    _, funcs = build_basis(mol)
    nbf = len(funcs)
    rows = []
    extra = {}
    if kind in ("jk", "fock"):
        D = core_guess_density(mol, funcs)
    if kind == "jk":
        G = get_backend(backend, **{**opts, "threads": max(threads)}).eri(funcs)
    if kind == "fock":
        use_tensor = ref == "tensor" or (ref == "auto" and nbf <= 110)
        if use_tensor:
            from ..backends.numpy_opt import NumpyBackend
            Gref = get_backend("cpp-omp", threads=max(threads)).eri(funcs) if backend == "cpp-omp" else \
                get_backend("rust-rayon", threads=max(threads)).eri(funcs)
            Jr, Kr = NumpyBackend().jk(Gref, D)
            del Gref
            extra["fock_reference"] = "tensor J/K (BLAS)"
        else:
            Jr = Kr = None
            extra["fock_reference"] = "first thread count of this run"
    first_digest = None
    for t in threads:
        for rep in range(repeat):
            be = get_backend(backend, **{**opts, "threads": t})
            rec = {"kind": kind, "molecule": molname, "nbf": nbf, "backend": backend, "threads": t,
                   "repeat": rep, "options": {**opts, "threads": t}}
            if kind == "eri":
                t0 = time.perf_counter()
                G = be.eri(funcs)
                rec["wall_s"] = time.perf_counter() - t0
                info = dict(be.last_eri_stats)
                dg = digest(G)
                first_digest = first_digest or dg
                rec["bitwise_equal_to_first"] = dg == first_digest
                del G
            elif kind == "jk":
                t0 = time.perf_counter()
                for _ in range(3):
                    J, K = be.jk(G, D)
                rec["wall_s"] = (time.perf_counter() - t0) / 3
                info = {}
            elif kind == "fock":
                be2 = get_backend(backend, **{**opts, "threads": t, "fock": "direct"})
                t0 = time.perf_counter()
                djk = be2.eri(funcs)
                rec["setup_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                J, K = be2.jk(djk, D)
                rec["wall_s"] = time.perf_counter() - t0
                info = dict(be2.last_jk_stats)
                if Jr is None:
                    Jr, Kr = J, K
                rec["max_abs_dJ"] = float(np.abs(J - Jr).max())
                rec["max_abs_dK"] = float(np.abs(K - Kr).max())
                if opts.get("store"):
                    t0 = time.perf_counter()
                    J2, K2 = be2.jk(djk, D)
                    rec["cached_wall_s"] = time.perf_counter() - t0
                    rec["cached_info"] = {k: v for k, v in be2.last_jk_stats.items() if k != "busy"}
                    rec["cached_imbalance"] = load_imbalance(be2.last_jk_stats.get("busy", []))
                    rec["cached_max_abs_dK"] = float(np.abs(K2 - Kr).max())
                del djk
            else:  # scf
                timing.reset()
                t0 = time.perf_counter()
                res = rhf(mol, be, verbose=False)
                rec["wall_s"] = time.perf_counter() - t0
                rec["kernels"] = timing.summary()
                rec["energy"] = res.energy
                rec["iterations"] = res.iterations
                rec["converged"] = res.converged
                e_ref = ref_energy(molname)
                rec["abs_err_vs_ref"] = abs(res.energy - e_ref) if e_ref is not None else None
                info = dict(be.last_eri_stats)
            if "busy" in info:
                rec["imbalance"] = load_imbalance(info["busy"])
            rec["info"] = info
            rows.append(rec)
            print(f"{kind:4s} {molname:11s} {backend:10s} T={t:3d} wall={rec['wall_s']:9.4f} s"
                  + (f"  imbalance={rec['imbalance']:.2f}" if "imbalance" in rec else "")
                  + (f"  dE={rec['abs_err_vs_ref']:.1e}" if rec.get("abs_err_vs_ref") is not None else ""),
                  flush=True)
    best = {}
    for r in rows:
        best[r["threads"]] = min(best.get(r["threads"], np.inf), r["wall_s"])
    ts = sorted(best)
    summary = {"speedup": speedup_table(ts, [best[t] for t in ts])}
    if len(ts) >= 3 and 1 in ts:
        summary["amdahl"] = amdahl_fit(ts, [best[t] for t in ts])
        summary["karp_flatt"] = karp_flatt(ts, [best[t] for t in ts])
    return {"rows": rows, "summary": summary, **extra}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["eri", "jk", "fock", "scf"])
    ap.add_argument("molecule")
    ap.add_argument("--backend", default="cpp-omp", choices=["cpp-omp", "rust-rayon"])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--ref", choices=["auto", "tensor", "first"], default="auto")
    ap.add_argument("--label", default="", help="free text stored in the JSON (e.g. 'OMP_PROC_BIND=close')")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    opts = dict(_parse_opt(o) for o in a.opt)
    out = {"machine": machine_info(), "label": a.label, "argv": vars(a),
           **run(a.kind, a.molecule, a.backend, a.threads, opts, a.repeat, a.ref)}
    for r in out["summary"]["speedup"]:
        print(f"  p={r['p']:3d}  T={r['time_s']:9.4f} s  S={r['speedup']:6.2f}  E={r['efficiency']:5.2f}")
    if "amdahl" in out["summary"]:
        am = out["summary"]["amdahl"]
        print(f"  Amdahl fit: serial fraction {am['serial_fraction']:.4f}  -> max speedup {am['max_speedup']:.1f}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
