"""Raw JSON (results/parallel/raw) -> CSV tables (results/parallel/*.csv).

    python results/parallel/collect.py
Tables
    threads_eri.csv    one row per (variant, molecule, threads): ERI tensor time, speedup, efficiency,
                       per-thread load imbalance (max/mean busy), bitwise reproducibility
    threads_jk.csv     J/K from the stored tensor
    threads_fock.csv   integral-direct J/K build (private vs atomic; cached = semi-direct 2nd build)
    threads_scf.csv    full RHF: wall, ERI, J/K and "rest" seconds, energy error
    amdahl.csv         Amdahl fit per scaling series (serial fraction, implied max speedup)
    mp.csv             Python multiprocessing / thread-pool counterexample
    mpi.csv            MPI direct SCF: per-iteration compute / wait / allreduce / bcast / serial
"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
sys.path.insert(0, str(HERE.parents[1] / "src"))
from hfperf.parallel.analysis import amdahl_fit, karp_flatt  # noqa: E402


def write(name, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(HERE / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def variant_of(path, kind, mol):
    stem = path.stem
    return stem.split(f"thr_{kind}_{mol}_", 1)[-1] if f"thr_{kind}_{mol}_" in stem else stem


def threads_tables():
    out = {k: [] for k in ("eri", "jk", "fock", "fockstore", "scf")}
    amd = []
    for p in sorted(RAW.glob("thr_*.json")):
        d = json.loads(p.read_text())
        kind = p.stem.split("_")[1]
        rows = d["rows"]
        if not rows:
            continue
        mol = rows[0]["molecule"]
        var = variant_of(p, kind, mol)
        cpu = d["machine"].get("cpu", "")
        best = {}
        for r in rows:
            if r["threads"] not in best or r["wall_s"] < best[r["threads"]]["wall_s"]:
                best[r["threads"]] = r
        ts = sorted(best)
        t1 = best[1]["wall_s"] if 1 in best else None
        for t in ts:
            r = best[t]
            info = r.get("info", {})
            row = {"variant": var, "backend": r["backend"], "molecule": mol, "nbf": r["nbf"], "threads": t,
                   "wall_s": r["wall_s"], "speedup": t1 / r["wall_s"] if t1 else "",
                   "efficiency": t1 / r["wall_s"] / t if t1 else "", "imbalance": r.get("imbalance", ""),
                   "label": d.get("label", ""), "cpu": cpu, "file": p.name}
            if kind == "eri":
                row.update(setup_s=info.get("setup_s"), loop_s=info.get("loop_s"), gather_s=info.get("gather_s", ""),
                           quartets_computed=info.get("computed"), bitwise_equal_T1=r.get("bitwise_equal_to_first"))
            if kind in ("fock", "fockstore"):
                row.update(setup_s=r.get("setup_s"), reduce_s=info.get("reduce_s"),
                           max_abs_dJ=r.get("max_abs_dJ"), max_abs_dK=r.get("max_abs_dK"),
                           reference=d.get("fock_reference", ""))
                if "cached_wall_s" in r:
                    row.update(cached_wall_s=r["cached_wall_s"], cached_imbalance=r.get("cached_imbalance"),
                               cached_max_abs_dK=r.get("cached_max_abs_dK"))
            if kind == "scf":
                k = r["kernels"]
                eri = k.get("ERI", {}).get("total_s", 0.0)
                jk = k.get("JK", {}).get("total_s", 0.0)
                row.update(eri_s=eri, jk_s=jk, rest_s=r["wall_s"] - eri - jk, iterations=r["iterations"],
                           energy=r["energy"], abs_err_vs_ref=r["abs_err_vs_ref"])
            out[kind].append(row)
        if 1 in best and len(ts) >= 3:
            times = [best[t]["wall_s"] for t in ts]
            fit = amdahl_fit(ts, times)
            kf = karp_flatt(ts, times)
            amd.append({"kind": kind, "variant": var, "molecule": mol, "threads": " ".join(map(str, ts)),
                        "T1_s": t1, "Tmax_s": times[-1], "speedup_max_p": t1 / times[-1],
                        "serial_fraction": fit["serial_fraction"], "amdahl_max_speedup": fit["max_speedup"],
                        "fit_rms_rel_resid": fit["rms_rel_resid"], "karp_flatt_at_max_p": kf[-1], "cpu": cpu})
            if kind == "scf":  # Amdahl on the ERI / JK kernels of the same runs
                for kern in ("ERI", "JK"):
                    kt = [best[t]["kernels"][kern]["total_s"] for t in ts]
                    f2 = amdahl_fit(ts, kt)
                    amd.append({"kind": f"scf:{kern}", "variant": var, "molecule": mol,
                                "threads": " ".join(map(str, ts)), "T1_s": kt[0], "Tmax_s": kt[-1],
                                "speedup_max_p": kt[0] / kt[-1], "serial_fraction": f2["serial_fraction"],
                                "amdahl_max_speedup": f2["max_speedup"], "fit_rms_rel_resid": f2["rms_rel_resid"],
                                "karp_flatt_at_max_p": karp_flatt(ts, kt)[-1], "cpu": cpu})
    write("threads_eri.csv", out["eri"])
    write("threads_jk.csv", out["jk"])
    write("threads_fock.csv", out["fock"] + out["fockstore"])
    write("threads_scf.csv", out["scf"])
    write("amdahl.csv", amd)


def mp_table():
    rows = []
    for p in sorted(RAW.glob("mp_*.json")):
        d = json.loads(p.read_text())
        for r in d["rows"]:
            rows.append({"file": p.name, "molecule": r["molecule"], "nbf": r["nbf"], "kind": r["kind"],
                         "start_method": r.get("start_method", ""), "workers": r["workers"],
                         "total_s": r["total_s"], "setup_s": r["setup_s"], "pool_start_s": r["pool_start_s"],
                         "map_s": r["map_s"], "scatter_s": r["scatter_s"],
                         "sum_task_compute_s": r["worker_compute_s"], "result_MB": r["result_bytes"] / 1e6,
                         "initargs_MB": r.get("initargs_pickle_bytes", 0) / 1e6,
                         "max_abs_diff": r["max_abs_diff_vs_first"], "cpu": d["machine"].get("cpu", "")})
    write("mp.csv", rows)


def mpi_table():
    rows = []
    for p in sorted(RAW.glob("mpi_*.json")):
        d = json.loads(p.read_text())
        a = d["per_iter_mean_of_max"]
        nodes = len(d["hosts"])
        rows.append({"file": p.name, "molecule": d["molecule"], "nbf": d["nbf"], "nodes": nodes,
                     "ranks": d["ranks"], "threads_per_rank": d["threads_per_rank"],
                     "cores": d["ranks"] * d["threads_per_rank"], "dist": d["dist"], "store": d["store"],
                     "iterations": d["iterations"], "total_s": d["total_s"], "setup_s_max": d["setup_s_max"],
                     "compute_ms": a["compute"] * 1e3, "wait_ms": a["wait"] * 1e3,
                     "allreduce_ms": a["allreduce"] * 1e3, "bcast_ms": a["bcast"] * 1e3,
                     "serial_ms": a["serial"] * 1e3,
                     "comm_fraction": (a["allreduce"] + a["bcast"]) / max(1e-12, sum(a.values())),
                     "quartet_imbalance": d["quartets_per_rank"]["imbalance"],
                     "allreduce_kB": d["allreduce_bytes"] / 1e3, "energy": d["energy"],
                     "abs_err_vs_ref": d.get("abs_err_vs_ref"), "cpu": d["machine"].get("cpu", ""),
                     "mpi": d.get("mpi", "")})
    rows.sort(key=lambda r: (r["molecule"], r["dist"], r["store"], r["threads_per_rank"], r["cores"]))
    write("mpi.csv", rows)


if __name__ == "__main__":
    threads_tables()
    mp_table()
    mpi_table()
