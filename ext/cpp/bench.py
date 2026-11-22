"""Benchmarks for stage 2a (C++).  Each call writes one JSON record.

    python ext/cpp/bench.py scf   h2o --variant sym8      # full RHF, per-kernel timing
    python ext/cpp/bench.py eri   h2o --variant sym8-noreuse --repeat 3
    python ext/cpp/bench.py flags h2o                     # O0 / O2 / O3 / native modules
    python ext/cpp/bench.py jk    h2o c2h4 benzene        # loop order of the J/K contraction
    python ext/cpp/bench.py python-scf h2o                # stage-0 baseline (pure Python)
    python ext/cpp/bench.py eri-slab benzene --i0 0 --i1 26  # eri_naive for rows i0..i1-1 only

Variants:  naive         C++ port of the stage-0 algorithm (no symmetry)
           sym8          8-fold symmetry + Schwarz screening + pair-data reuse (CppBackend default)
           sym8-noreuse  8-fold symmetry + Schwarz screening, naive per-quartet kernel
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hfperf import build_basis, load_molecule, rhf, timing  # noqa: E402
from hfperf.backends.cpp import CppBackend, load_module  # noqa: E402

VARIANTS = {"naive": dict(eri_algorithm="naive"),
            "sym8": dict(eri_algorithm="sym8", reuse=True),
            "sym8-noreuse": dict(eri_algorithm="sym8", reuse=False)}
REF = ROOT / "tests" / "data"


def machine() -> dict:
    cpu = platform.processor()
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        cpu = next(l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Model name"))
    except Exception:
        pass
    info_file = ROOT / "src" / "hfperf" / "backends" / "cpp" / "_build_info.json"
    build = json.loads(info_file.read_text()) if info_file.exists() else {}
    return {"cpu": cpu, "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "host": platform.node(), "build": build}


def ref_energy(mol):
    return json.loads((REF / "reference_energies.json").read_text()).get(mol, {}).get("e_rhf")


def ref_eri(mol):
    p = REF / f"ref_{mol}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["ERI"] if "ERI" in d.files else None


def best_of(fn, repeat):
    times, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return min(times), times, out


def write(rec, out):
    rec["machine"] = machine()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("machine", "kernels")}, indent=1))


def cmd_scf(a):
    mol = load_molecule(a.mol)
    be = CppBackend(**VARIANTS[a.variant], module=a.module, jk_algorithm=a.jk)
    timing.reset()
    t0 = time.perf_counter()
    res = rhf(mol, be, verbose=False)
    wall = time.perf_counter() - t0
    e_ref = ref_energy(a.mol)
    k = timing.summary()
    write({"kind": "scf", "molecule": a.mol, "variant": a.variant, "module": a.module, "jk_algorithm": a.jk,
           "nbf": len(build_basis(mol)[1]), "energy": res.energy, "e_ref": e_ref,
           "abs_err": None if e_ref is None else abs(res.energy - e_ref),
           "converged": res.converged, "iterations": res.iterations, "wall_s": wall,
           "eri_s": k["ERI"]["total_s"], "jk_total_s": k["JK"]["total_s"], "jk_calls": k["JK"]["calls"],
           "jk_per_call_s": k["JK"]["total_s"] / k["JK"]["calls"], "eri_stats": be.last_eri_stats,
           "kernels": k}, a.out or ROOT / "results" / "cpp" / "raw" / f"scf_{a.mol}_{a.variant}.json")


def cmd_eri(a):
    funcs = build_basis(load_molecule(a.mol))[1]
    be = CppBackend(**VARIANTS[a.variant], module=a.module)
    t, ts, G = best_of(lambda: be.eri(funcs), a.repeat)
    R = ref_eri(a.mol)
    write({"kind": "eri", "molecule": a.mol, "variant": a.variant, "module": a.module, "nbf": len(funcs),
           "eri_s_best": t, "eri_s_all": ts, "eri_stats": be.last_eri_stats,
           "max_abs_err_vs_pyscf": None if R is None else float(np.abs(G - R).max())},
          a.out or ROOT / "results" / "cpp" / "raw" / f"eri_{a.mol}_{a.variant}_{a.module}.json")


def cmd_eri_slab(a):
    """Time eri_naive for rows G[i0:i1] only and check them against the sym8 tensor."""
    funcs = build_basis(load_molecule(a.mol))[1]
    ext = load_module(a.module)
    args = CppBackend._args(funcs)
    i1 = len(funcs) if a.i1 < 0 else a.i1
    t0 = time.perf_counter()
    G = ext.eri_naive(*args, a.i0, i1)
    t = time.perf_counter() - t0
    slab = G[a.i0:i1].copy()
    del G
    Gs = ext.eri_sym8(*args, 1e-12, True)[0]
    write({"kind": "eri_slab", "molecule": a.mol, "variant": "naive", "module": a.module, "nbf": len(funcs),
           "i0": a.i0, "i1": i1, "quartets": int(slab.size), "eri_s": t,
           "max_abs_diff_vs_sym8": float(np.abs(slab - Gs[a.i0:i1]).max())},
          a.out or ROOT / "results" / "cpp" / "raw" / f"slab_{a.mol}_naive_{a.i0}_{i1}.json")


def cmd_flags(a):
    funcs = build_basis(load_molecule(a.mol))[1]
    R = ref_eri(a.mol)
    rows = []
    for prof in ("O0", "O2", "O3", "native"):
        for variant in ("naive", "sym8"):
            be = CppBackend(**VARIANTS[variant], module=f"_hfcpp_{prof}")
            t, ts, G = best_of(lambda: be.eri(funcs), a.repeat)
            rows.append({"profile": prof, "variant": variant, "eri_s_best": t, "eri_s_all": ts,
                         "max_abs_err_vs_pyscf": None if R is None else float(np.abs(G - R).max())})
            print(rows[-1])
    write({"kind": "flags", "molecule": a.mol, "nbf": len(funcs), "rows": rows},
          a.out or ROOT / "results" / "cpp" / "raw" / f"flags_{a.mol}.json")


def cmd_jk(a):
    from hfperf.backends.naive import fock

    ext = load_module(a.module)
    rng = np.random.default_rng(0)
    rows = []
    for name in a.mols:
        funcs = build_basis(load_molecule(name))[1]
        n = len(funcs)
        G = CppBackend(module=a.module).eri(funcs)
        A = rng.normal(size=(n, n))
        D = A + A.T  # symmetric test density
        t_f, _, (J1, K1) = best_of(lambda: ext.jk(G, D), a.repeat)
        t_t, _, (J2, K2) = best_of(lambda: ext.jk_twopass(G, D), a.repeat)
        t_n, _, (J4, K4) = best_of(lambda: ext.jk_naive(G, D), a.repeat)
        row = {"molecule": name, "nbf": n, "G_bytes": G.nbytes, "jk_fused_s": t_f, "jk_twopass_s": t_t,
               "jk_naive_order_s": t_n,
               "max_diff": float(max(np.abs(J1 - J2).max(), np.abs(K1 - K2).max(),
                                     np.abs(J1 - J4).max(), np.abs(K1 - K4).max()))}
        if n <= a.python_max_nbf:
            t_p, _, (J3, K3) = best_of(lambda: fock.jk(G, D), 1)
            row["jk_python_s"] = t_p
            row["max_diff_python"] = float(max(np.abs(J1 - J3).max(), np.abs(K1 - K3).max()))
        rows.append(row)
        print(row)
    write({"kind": "jk", "rows": rows}, a.out or ROOT / "results" / "cpp" / "raw" / "jk.json")


def cmd_python_scf(a):
    mol = load_molecule(a.mol)
    timing.reset()
    t0 = time.perf_counter()
    res = rhf(mol, "naive", verbose=False)
    wall = time.perf_counter() - t0
    e_ref = ref_energy(a.mol)
    k = timing.summary()
    write({"kind": "scf", "molecule": a.mol, "variant": "python-naive", "nbf": len(build_basis(mol)[1]),
           "energy": res.energy, "e_ref": e_ref, "abs_err": None if e_ref is None else abs(res.energy - e_ref),
           "converged": res.converged, "iterations": res.iterations, "wall_s": wall,
           "eri_s": k["ERI"]["total_s"], "jk_total_s": k["JK"]["total_s"], "jk_calls": k["JK"]["calls"],
           "jk_per_call_s": k["JK"]["total_s"] / k["JK"]["calls"], "kernels": k},
          a.out or ROOT / "results" / "cpp" / "raw" / f"scf_{a.mol}_python-naive.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scf", "eri"):
        p = sub.add_parser(name)
        p.add_argument("mol")
        p.add_argument("--variant", choices=list(VARIANTS), default="sym8")
        p.add_argument("--module", default="_hfcpp")
        p.add_argument("--jk", choices=["fused", "twopass", "naive"], default="fused")
        p.add_argument("--repeat", type=int, default=1)
        p.add_argument("--out")
    p = sub.add_parser("eri-slab")
    p.add_argument("mol")
    p.add_argument("--i0", type=int, default=0)
    p.add_argument("--i1", type=int, default=-1)
    p.add_argument("--module", default="_hfcpp")
    p.add_argument("--out")
    p = sub.add_parser("flags")
    p.add_argument("mol", nargs="?", default="h2o")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out")
    p = sub.add_parser("jk")
    p.add_argument("mols", nargs="+")
    p.add_argument("--module", default="_hfcpp")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--python-max-nbf", type=int, default=25)
    p.add_argument("--out")
    p = sub.add_parser("python-scf")
    p.add_argument("mol")
    p.add_argument("--out")
    a = ap.parse_args(argv)
    {"scf": cmd_scf, "eri": cmd_eri, "flags": cmd_flags, "eri-slab": cmd_eri_slab, "jk": cmd_jk, "python-scf": cmd_python_scf}[a.cmd](a)


if __name__ == "__main__":
    main()
