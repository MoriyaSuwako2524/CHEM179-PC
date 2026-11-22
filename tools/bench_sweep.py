"""Run a matrix of (backend, options, molecule, threads) profiled RHF runs and
collect one tidy CSV (one row per run x kernel) plus the raw JSON records.

    python tools/bench_sweep.py sweep.json --out results/sweep [--repeat 1]

sweep.json:
  {"runs": [
     {"label": "numpy-R5", "backend": "numpy", "molecules": ["h2o", "benzene"], "threads": [1]},
     {"label": "cpp-omp", "backend": "cpp-omp", "options": {"schedule": "dynamic"},
      "molecules": ["benzene"], "threads": [1, 2, 4, 8, 16, 32], "timeout_s": 3000}
  ]}
Each run is a fresh subprocess (so peak RSS is per run) with OMP/RAYON/BLAS
thread counts pinned to `threads`.  Failures and timeouts are recorded, not hidden.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def run_one(spec, molecule, threads, outdir, repeat_idx, mode):
    tag = f"{spec['label']}__{molecule}__t{threads}__r{repeat_idx}"
    out_json = outdir / "raw" / f"{tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for v in THREAD_VARS:
        env[v] = str(threads)
    env.update({k: str(v) for k, v in spec.get("env", {}).items()})
    cmd = [sys.executable, "-m", "hfperf.profiling", molecule, "--backend", spec["backend"],
           "--mode", spec.get("mode", mode), "--out", str(out_json)]
    for k, v in spec.get("options", {}).items():
        cmd += ["--opt", f"{k}={v}"]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=spec.get("timeout_s", 3600))
        status = "ok" if p.returncode == 0 else f"error rc={p.returncode}"
        err = p.stderr[-2000:]
    except subprocess.TimeoutExpired:
        status, err = "timeout", ""
    elapsed = time.perf_counter() - t0
    rec = json.loads(out_json.read_text()) if status == "ok" and out_json.exists() else None
    print(f"{tag:55s} {status:12s} {elapsed:9.2f} s", flush=True)
    if status != "ok":
        (outdir / "raw" / f"{tag}.err").write_text(err)
    return tag, status, rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out", default=str(ROOT / "results" / "sweep"))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--mode", default="timers")
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ref = json.loads((ROOT / "tests" / "data" / "reference_energies.json").read_text())
    rows = []
    for s in spec["runs"]:
        for mol in s["molecules"]:
            for th in s.get("threads", [1]):
                for r in range(a.repeat):
                    tag, status, rec = run_one(s, mol, th, outdir, r, a.mode)
                    base = {"label": s["label"], "backend": s["backend"], "options": json.dumps(s.get("options", {})),
                            "molecule": mol, "nbf": ref.get(mol, {}).get("nbf"), "threads": th, "repeat": r,
                            "status": status}
                    if rec is None:
                        rows.append({**base, "kernel": "TOTAL"})
                        continue
                    m = rec["machine"]
                    base.update({"energy_error": rec["energy"] - ref[mol]["e_rhf"] if mol in ref else None,
                                 "iterations": rec["iterations"], "peak_rss_mb": (rec["peak_rss_bytes"] or 0) / 2**20,
                                 "cpu_model": m["cpu"], "node": m["env"].get("SLURMD_NODENAME", "")})
                    rows.append({**base, "kernel": "TOTAL", "wall_s": rec["wall_s"], "cpu_s": rec["cpu_s"], "calls": 1})
                    for k, v in rec["kernels"].items():
                        rows.append({**base, "kernel": k, "wall_s": v["total_s"], "cpu_s": v["cpu_s"], "calls": v["calls"],
                                     "peak_mem_mb": v.get("peak_mem_bytes", 0) / 2**20 if "peak_mem_bytes" in v else None})
    fields = ["label", "backend", "options", "molecule", "nbf", "threads", "repeat", "status", "kernel", "calls",
              "wall_s", "cpu_s", "peak_mem_mb", "peak_rss_mb", "iterations", "energy_error", "cpu_model", "node"]
    with open(outdir / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {outdir / 'sweep.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
