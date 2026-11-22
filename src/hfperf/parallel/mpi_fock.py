"""Distributed-memory direct SCF with mpi4py (stage 4b).

    mpirun -n 4 python -m hfperf.parallel.mpi_fock benzene
    mpirun -n 4 python -m hfperf.parallel.mpi_fock benzene --dist block      # load-imbalance demo
    mpirun -n 2 python -m hfperf.parallel.mpi_fock c2h4 --threads 2         # hybrid: ranks x threads
    srun python -m hfperf.parallel.mpi_fock naphthalene --json out.json     # on a cluster

Scheme (replicated data, distributed work - the classic "parallel direct SCF"):
  * every rank holds the whole basis, density D and the result J, K (N x N each: small);
  * the unique pair rows ij are split among ranks (--dist cyclic | block | weighted, see
    distribute.py); each rank computes only the quartets (ij|kl), kl <= ij, of its rows with
    the compiled integral-direct kernel (DirectJK, C++ or Rust; --threads per rank) and
    scatters them into PARTIAL J and K;
  * one MPI_Allreduce sums the partial J/K of all ranks (2 N^2 doubles per iteration);
  * rank 0 does the cheap N x N steps (Fock, energy, DIIS, diagonalization) and broadcasts
    the new density.
Per iteration and rank we time:
  compute   the local J/K build
  wait      Barrier after the build: idle time caused by load imbalance
  allreduce the J/K reduction itself
  bcast     density broadcast (includes waiting for rank 0's serial part)
  serial    rank 0 only: Fock/DIIS/diagonalization
Communication volume per iteration is tiny (benzene: 2 x 102^2 x 8 B = 166 kB), so
communication only dominates when the compute per rank is small: small molecules, many
ranks, or --store (semi-direct: integrals cached after the first iteration).
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

from .distribute import SCHEMES, distribute


def make_engine(engine, funcs, rows, tol, store, threads):
    from ..basis import pack_basis
    b = pack_basis(funcs)
    arrs = [np.ascontiguousarray(b[k]) for k in ("centers", "lmn", "offsets", "exps", "coefs")]
    rows = [int(r) for r in rows]
    if engine == "cpp":
        from ..backends.cpp import load_module
        try:
            ext = load_module("_hfcpp_omp")
        except ImportError:
            ext = load_module("_hfcpp")  # serial build works too (threads ignored)
        djk = ext.DirectJK(*arrs, tol, store, rows, threads)
        return djk, (lambda D, t: djk.build(D, t, "private", "dynamic", 1))
    if engine == "rust":
        import hfperf_rust as rs
        djk = rs.DirectJK(*arrs, screen=tol, store=store, rows=rows, threads=threads)
        return djk, (lambda D, t: djk.build(D, threads=t, mode="fold"))
    raise ValueError("engine must be cpp or rust")


def run(molname, dist="cyclic", engine="cpp", threads=1, store=False, tol=1e-12, max_iter=60,
        e_tol=1e-10, d_tol=1e-8, barrier=True, comm=None, verbose=True):
    from mpi4py import MPI

    from ..backends.numpy_opt import NumpyBackend
    from ..basis import build_basis
    from ..molecule import load_molecule

    comm = comm or MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    t_start = MPI.Wtime()
    mol = load_molecule(molname)
    _, funcs = build_basis(mol)
    n = len(funcs)
    nocc = mol.n_electrons // 2
    npair = n * (n + 1) // 2
    rows = distribute(npair, size, rank, dist)

    t0 = MPI.Wtime()
    djk, build = make_engine(engine, funcs, rows, tol, store, threads)
    setup_s = MPI.Wtime() - t0

    # one-electron part on rank 0, broadcast (shows Bcast; costs ~N^2 doubles)
    if rank == 0:
        nb = NumpyBackend()
        S = nb.overlap(funcs)
        H = nb.kinetic(funcs) + nb.nuclear(funcs, mol)
        s, U = np.linalg.eigh(S)
        X = (U / np.sqrt(s)) @ U.T
        _, Cp = np.linalg.eigh(X.T @ H @ X)
        C = X @ Cp
        D = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    else:
        D = np.empty((n, n))
    D = np.ascontiguousarray(D)
    comm.Bcast(D, root=0)
    e_nuc = mol.nuclear_repulsion()

    focks, errors, history = [], [], []
    it_log = []
    e_old, converged = 0.0, False
    JK = np.empty((2, n, n))
    flag = np.zeros(1)
    for it in range(1, max_iter + 1):
        rec = {}
        t0 = MPI.Wtime()
        Jl, Kl = build(D, threads)
        t1 = MPI.Wtime()
        if barrier:
            comm.Barrier()
        t2 = MPI.Wtime()
        local = np.ascontiguousarray(np.stack([Jl, Kl]))
        comm.Allreduce(local, JK, op=MPI.SUM)
        t3 = MPI.Wtime()
        rec.update(compute=t1 - t0, wait=t2 - t1, allreduce=t3 - t2)
        serial = 0.0
        if rank == 0:
            ts = MPI.Wtime()
            J, K = JK
            F = H + J - 0.5 * K
            e_tot = 0.5 * float(np.einsum("ij,ji->", D, H + F)) + e_nuc
            history.append(e_tot)
            FDS = F @ D @ S
            err = FDS - FDS.T
            focks, errors = (focks + [F])[-8:], (errors + [err])[-8:]
            m = len(focks)
            if m > 1 and np.abs(err).max() > 1e-12:
                B = -np.ones((m + 1, m + 1))
                B[m, m] = 0.0
                for a in range(m):
                    for b in range(m):
                        B[a, b] = np.vdot(errors[a], errors[b])
                rhs = np.zeros(m + 1)
                rhs[m] = -1.0
                try:
                    c = np.linalg.solve(B, rhs)
                    F = sum(c[a] * focks[a] for a in range(m))
                except np.linalg.LinAlgError:
                    pass
            _, Cp = np.linalg.eigh(X.T @ F @ X)
            C = X @ Cp
            D_new = 2.0 * C[:, :nocc] @ C[:, :nocc].T
            dD = float(np.sqrt(np.mean((D_new - D) ** 2)))
            dE = abs(e_tot - e_old)
            e_old = e_tot
            D[:] = D_new
            flag[0] = 1.0 if (dE < e_tol and dD < d_tol) else 0.0
            serial = MPI.Wtime() - ts
            if verbose:
                print(f"  iter {it:3d}  E = {e_tot:.12f}  dE = {dE:.2e}  rms dD = {dD:.2e}  "
                      f"compute(r0) = {rec['compute']:.4f} s", flush=True)
        t4 = MPI.Wtime()
        comm.Bcast(D, root=0)
        comm.Bcast(flag, root=0)
        t5 = MPI.Wtime()
        rec.update(serial=serial, bcast=t5 - t4)
        it_log.append(rec)
        if flag[0] == 1.0:
            converged = True
            break

    # final energy with the converged density (one more distributed build)
    t0 = MPI.Wtime()
    Jl, Kl = build(D, threads)
    comm.Allreduce(np.ascontiguousarray(np.stack([Jl, Kl])), JK, op=MPI.SUM)
    final_build = MPI.Wtime() - t0
    energy = None
    if rank == 0:
        J, K = JK
        F = H + J - 0.5 * K
        energy = 0.5 * float(np.einsum("ij,ji->", D, H + F)) + e_nuc
    total_s = MPI.Wtime() - t_start

    st = djk.last_stats
    mine = {"rank": rank, "host": socket.gethostname(), "n_rows": len(rows),
            "n_quartets": int(djk.n_quartets), "computed": int(st["computed"]), "setup_s": setup_s,
            "iters": it_log}
    everyone = comm.gather(mine, root=0)
    if rank != 0:
        return None
    keys = ["compute", "wait", "allreduce", "bcast", "serial"]
    per_iter = []
    for k in range(len(it_log)):
        row = {}
        for key in keys:
            v = np.array([r["iters"][k][key] for r in everyone])
            row[key + "_max"], row[key + "_mean"], row[key + "_min"] = float(v.max()), float(v.mean()), float(v.min())
        per_iter.append(row)
    agg = {key: float(np.mean([p[key + "_max"] for p in per_iter[1:] or per_iter])) for key in keys}
    comp = np.array([r["computed"] for r in everyone], float)
    return {"molecule": molname, "nbf": n, "ranks": size, "threads_per_rank": threads, "dist": dist,
            "engine": engine, "store": store, "barrier": barrier,
            "hosts": sorted({r["host"] for r in everyone}), "energy": energy, "converged": converged,
            "iterations": len(it_log), "total_s": total_s, "final_build_s": final_build,
            "setup_s_max": max(r["setup_s"] for r in everyone),
            "allreduce_bytes": int(2 * n * n * 8),
            "per_iter_mean_of_max": agg, "per_iteration": per_iter,
            "quartets_per_rank": {"max": float(comp.max()), "mean": float(comp.mean()),
                                  "imbalance": float(comp.max() / comp.mean()) if comp.mean() else None},
            "ranks_detail": [{k: v for k, v in r.items() if k != "iters"} for r in everyone]}


def main(argv=None):
    from mpi4py import MPI

    from .scaling import machine_info, ref_energy

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("molecule")
    ap.add_argument("--dist", choices=SCHEMES, default="cyclic")
    ap.add_argument("--engine", choices=["cpp", "rust"], default="cpp")
    ap.add_argument("--threads", type=int, default=1, help="threads per rank (hybrid MPI x OpenMP/Rayon)")
    ap.add_argument("--store", action="store_true", help="semi-direct: cache each rank's integrals")
    ap.add_argument("--no-barrier", action="store_true", help="skip the Barrier that separates wait from comm")
    ap.add_argument("--label", default="")
    ap.add_argument("--json")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    comm = MPI.COMM_WORLD
    out = run(a.molecule, a.dist, a.engine, a.threads, a.store, barrier=not a.no_barrier, comm=comm,
              verbose=not a.quiet)
    if comm.Get_rank() == 0:
        e_ref = ref_energy(a.molecule)
        out["abs_err_vs_ref"] = abs(out["energy"] - e_ref) if e_ref is not None else None
        out["machine"] = machine_info()
        out["mpi"] = MPI.Get_library_version().splitlines()[0]
        out["label"] = a.label
        agg = out["per_iter_mean_of_max"]
        print(f"E(RHF) = {out['energy']:.10f}  |dE vs ref| = {out['abs_err_vs_ref']:.2e}  "
              f"ranks = {out['ranks']} x {out['threads_per_rank']} threads  total {out['total_s']:.3f} s")
        print("per iteration (max over ranks, mean over iterations 2..): " +
              "  ".join(f"{k} {v * 1e3:.2f} ms" for k, v in agg.items()))
        print(f"quartet imbalance (max/mean) = {out['quartets_per_rank']['imbalance']:.3f}")
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
