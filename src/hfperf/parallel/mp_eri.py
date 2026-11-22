"""The counterexample: parallelizing the *Python* (NumPy, stage-1 R5) ERI code.

    python -m hfperf.parallel.mp_eri c2h4 --kind serial
    python -m hfperf.parallel.mp_eri c2h4 --kind thread  --workers 1 2 4
    python -m hfperf.parallel.mp_eri c2h4 --kind process --workers 1 2 4 8

Work unit: one bra shell pair ab (all ket pairs cd <= ab), exactly like the compiled
threaded kernels.  Two ways to run the units concurrently from Python:

  kind="thread"   concurrent.futures.ThreadPoolExecutor.  One process, shared memory,
                  but the interpreter executes Python bytecode in one thread at a time
                  (the GIL).  NumPy releases the GIL only inside large array operations;
                  the shell-quartet kernel is many small NumPy calls glued by Python, so
                  the threads mostly wait for each other: speedup ~1 or worse.
  kind="process"  multiprocessing.Pool.  Separate interpreters, no shared GIL -> real
                  parallelism, but you pay for (a) starting the worker processes (each
                  imports NumPy/SciPy and rebuilds the Boys table; much slower with the
                  "spawn" start method used on macOS and Windows than with Linux "fork"),
                  (b) pickling the shell-pair data to every worker, and (c) pickling every
                  result block back to the parent, which then scatters them into G serially.
The function returns the tensor plus a timing breakdown of these costs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_STATE: dict = {}


def _init(pairs, Q, tau):
    from ..backends.numpy_opt import boys as boysmod
    _STATE.update(pairs=pairs, Q=Q, tau=tau, boys=boysmod.boys_all)


def _noop(_):
    return None


def _row(ab):
    """All unscreened blocks (ab|cd), cd <= ab, for one bra pair."""
    from ..backends.numpy_opt.eri import quartet_vector
    t0 = time.perf_counter()
    pairs, Q, tau, bf = _STATE["pairs"], _STATE["Q"], _STATE["tau"], _STATE["boys"]
    bra = pairs[ab]
    out = []
    for cd in range(ab + 1):
        if tau and Q[ab] * Q[cd] < tau:
            continue
        out.append((cd, quartet_vector(bra, pairs[cd], bf)))
    return ab, out, time.perf_counter() - t0


def _scatter(G, bra, ket, blk):
    a, b, c, d = bra.a, bra.b, ket.a, ket.b
    blk = blk.reshape(a.n, b.n, c.n, d.n)
    sa, sb = slice(a.start, a.start + a.n), slice(b.start, b.start + b.n)
    sc, sd = slice(c.start, c.start + c.n), slice(d.start, d.start + d.n)
    G[sa, sb, sc, sd] = blk
    G[sb, sa, sc, sd] = blk.transpose(1, 0, 2, 3)
    G[sa, sb, sd, sc] = blk.transpose(0, 1, 3, 2)
    G[sb, sa, sd, sc] = blk.transpose(1, 0, 3, 2)
    G[sc, sd, sa, sb] = blk.transpose(2, 3, 0, 1)
    G[sd, sc, sa, sb] = blk.transpose(3, 2, 0, 1)
    G[sc, sd, sb, sa] = blk.transpose(2, 3, 1, 0)
    G[sd, sc, sb, sa] = blk.transpose(3, 2, 1, 0)


def eri_parallel(funcs, workers: int = 2, kind: str = "process", start_method: str | None = None,
                 chunksize: int = 1, screening: float = 1e-12):
    """(G, info) - the stage-1 vectorized shell-quartet ERI run by a thread or process pool."""
    from ..backends.numpy_opt import boys as boysmod
    from ..backends.numpy_opt.eri import quartet_vector
    from ..backends.numpy_opt.shells import group_shells, make_pair

    info = {"kind": kind, "workers": workers, "chunksize": chunksize}
    t0 = time.perf_counter()
    shells = group_shells(funcs)
    pairs = [make_pair(shells[a], shells[b]) for a in range(len(shells)) for b in range(a + 1)]
    Q = np.array([np.sqrt(np.abs(np.diag(quartet_vector(p, p, boysmod.boys_all))).max()) for p in pairs])
    info["setup_s"] = time.perf_counter() - t0
    info["n_tasks"] = len(pairs)
    tasks = list(range(len(pairs) - 1, -1, -1))  # longest rows first: helps the pool balance

    t0 = time.perf_counter()
    if kind == "serial":
        _init(pairs, Q, screening)
        info["pool_start_s"] = 0.0
        t1 = time.perf_counter()
        results = [_row(ab) for ab in tasks]
    elif kind == "thread":
        _init(pairs, Q, screening)
        with ThreadPoolExecutor(workers) as ex:
            info["pool_start_s"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            results = list(ex.map(_row, tasks))
    elif kind == "process":
        ctx = mp.get_context(start_method) if start_method else mp.get_context()
        info["start_method"] = ctx.get_start_method()
        initargs = (pairs, Q, screening)
        info["initargs_pickle_bytes"] = len(pickle.dumps(initargs, protocol=pickle.HIGHEST_PROTOCOL))
        pool = ctx.Pool(workers, initializer=_init, initargs=initargs)
        pool.map(_noop, range(workers), chunksize=1)  # wait until every worker is up
        info["pool_start_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        results = list(pool.imap_unordered(_row, tasks, chunksize=chunksize))
        pool.close()
        pool.join()
    else:
        raise ValueError("kind must be serial, thread or process")
    info["map_s"] = time.perf_counter() - t1
    info["worker_compute_s"] = float(sum(r[2] for r in results))  # summed over tasks
    info["result_bytes"] = int(sum(blk.nbytes for _, out, _ in results for _, blk in out))

    t0 = time.perf_counter()
    n = len(funcs)
    G = np.zeros((n, n, n, n))
    for ab, out, _ in results:
        for cd, blk in out:
            _scatter(G, pairs[ab], pairs[cd], blk)
    info["scatter_s"] = time.perf_counter() - t0
    info["total_s"] = info["setup_s"] + info["pool_start_s"] + info["map_s"] + info["scatter_s"]
    return G, info


def main(argv=None):
    from ..basis import build_basis
    from ..molecule import load_molecule
    from .scaling import machine_info

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("molecule")
    ap.add_argument("--kind", choices=["serial", "thread", "process"], default="process")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--start-method", choices=["fork", "spawn", "forkserver"], default=None)
    ap.add_argument("--chunksize", type=int, default=1)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    _, funcs = build_basis(load_molecule(a.molecule))
    rows, Gref = [], None
    for w in ([1] if a.kind == "serial" else a.workers):
        G, info = eri_parallel(funcs, w, a.kind, a.start_method, a.chunksize)
        Gref = G if Gref is None else Gref
        info["max_abs_diff_vs_first"] = float(np.abs(G - Gref).max())
        info.update(molecule=a.molecule, nbf=len(funcs))
        rows.append(info)
        print(f"{a.kind:7s} workers={w:3d} total={info['total_s']:8.3f} s  start={info['pool_start_s']:6.3f}"
              f"  map={info['map_s']:8.3f}  scatter={info['scatter_s']:6.3f}  sum(task)={info['worker_compute_s']:8.3f}",
              flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps({"machine": machine_info(), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
