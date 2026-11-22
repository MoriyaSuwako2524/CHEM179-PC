"""One profiled RHF run -> a JSON record with time, CPU and memory per kernel.

    python -m hfperf.profiling h2o --backend numpy [--opt k=v] [--mode timers|memory|cprofile] [--out rec.json]

modes
  timers    per-kernel wall + CPU time (negligible overhead)          [default]
  memory    + per-kernel Python-heap peak via tracemalloc (slower)
  cprofile  + the 25 most expensive Python functions (cProfile, cumulative time)
Peak RSS (whole process, including compiled-kernel memory) is always recorded.
For line-level or native profiles use the external tools documented in
docs/stages/stage3_profiling.md (line_profiler, scalene, py-spy, perf stat).
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time

from .. import timing
from ..backends import get_backend
from ..cli import _parse_opt
from ..molecule import load_molecule
from ..scf import rhf
from .machine import machine_info, peak_rss_bytes


def profile_run(molecule: str, backend: str = "numpy", options: dict | None = None, mode: str = "timers") -> dict:
    options = options or {}
    be = get_backend(backend, **options)
    mol = load_molecule(molecule)
    timing.reset()
    timing.track_memory(mode == "memory")
    prof = cProfile.Profile() if mode == "cprofile" else None
    c0, t0 = time.process_time(), time.perf_counter()
    if prof:
        prof.enable()
    res = rhf(mol, be, verbose=False)
    if prof:
        prof.disable()
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    timing.track_memory(False)
    rec = {"molecule": molecule, "backend": backend, "options": options, "mode": mode,
           "energy": res.energy, "converged": res.converged, "iterations": res.iterations,
           "wall_s": wall, "cpu_s": cpu, "peak_rss_bytes": peak_rss_bytes(),
           "kernels": timing.summary(),
           "per_call_s": {k: list(v) for k, v in timing.TIMINGS.items()},
           "machine": machine_info()}
    if prof:
        s = io.StringIO()
        st = pstats.Stats(prof, stream=s).sort_stats("cumulative")
        st.print_stats(25)
        rec["cprofile_top"] = s.getvalue()
        rec["cprofile_functions"] = [
            {"function": f"{fn[0]}:{fn[1]}({fn[2]})", "ncalls": v[1], "tottime_s": v[2], "cumtime_s": v[3]}
            for fn, v in sorted(st.stats.items(), key=lambda kv: -kv[1][2])[:40]]
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("molecule")
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--mode", choices=["timers", "memory", "cprofile"], default="timers")
    ap.add_argument("--out", help="write the JSON record here")
    a = ap.parse_args(argv)
    rec = profile_run(a.molecule, a.backend, dict(_parse_opt(o) for o in a.opt), a.mode)
    print(f"{a.molecule} / {a.backend} {rec['options'] or ''}: E = {rec['energy']:.10f}  wall {rec['wall_s']:.3f} s"
          f"  cpu {rec['cpu_s']:.3f} s  peak RSS {(rec['peak_rss_bytes'] or 0) / 2**20:.0f} MB")
    print(timing.report())
    if "cprofile_top" in rec:
        print(rec["cprofile_top"])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
