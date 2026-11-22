"""Flatten results/gpu/raw/bench_*.json into CSV tables (results/gpu/*.csv).

    python ext/cuda/collect.py [raw_dir] [out_dir]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/gpu/raw"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results/gpu"
    tabs = {k: [] for k in ("eri", "transfer", "kernels", "scf", "emulation")}
    for p in sorted(raw.glob("bench_*.json")):
        rec = json.loads(p.read_text())
        info = rec.get("info", {})
        gpu = info.get("gpu", {}).get("name", "?")
        cpu = info.get("cpu_model", "?")
        job = info.get("slurm_job")
        common = dict(gpu=gpu, cpu=cpu, job=job, source=p.name)
        for r in rec.get("eri", []):
            qr = r.get("quartet_resources", {})
            row = {k: v for k, v in r.items() if k not in ("quartet_resources", "pair_resources")}
            row.update(regs=qr.get("num_regs"), local_bytes=qr.get("local_size_bytes"),
                       occupancy=qr.get("occupancy"), active_blocks_per_sm=qr.get("active_blocks_per_sm"),
                       **common)
            tabs["eri"].append(row)
        for sec in ("transfer", "kernels"):
            for r in rec.get(sec, []):
                tabs[sec].append(dict(r, **common))
        for sec in ("scf", "emulation"):
            for r in rec.get(sec, []):
                k = r.get("kernels", {})
                g = r.get("gpu_stats", {})
                cs = r.get("cuda_stats", {})
                tabs[sec].append(dict(
                    molecule=r["molecule"], nbf=r["nbf"], mode=r["mode"], path=r["path"], energy=r["energy"],
                    err_vs_ref=r["err_vs_ref"], converged=r["converged"], iterations=r["iterations"],
                    wall_s=r["wall_s"], eri_s=k.get("ERI", {}).get("total_s"), jk_s=k.get("JK", {}).get("total_s"),
                    jk_calls=k.get("JK", {}).get("calls"), diag_s=k.get("diag", {}).get("total_s"),
                    matmul_s=k.get("matmul", {}).get("total_s"), stv_s=k.get("S,T,V", {}).get("total_s"),
                    fp64_builds=g.get("fp64_builds"), fp32_builds=g.get("fp32_builds"),
                    eri_backend=g.get("eri_backend"), eri_cpu_s=g.get("eri_cpu_s"), h2d_G_s=g.get("h2d_G_s"),
                    jk_h2d_s=g.get("jk_h2d_s"), jk_kernel_s=g.get("jk_kernel_s"), jk_d2h_s=g.get("jk_d2h_s"),
                    eigh_on=g.get("eigh_on"), cuda_quartet_kernel_s=cs.get("quartet_kernel_s"),
                    cuda_device_total_s=cs.get("device_total_s"), tolerances=str(r.get("tolerances")), **common))
    for k, rows in tabs.items():
        write(out / f"{k}.csv", rows)


if __name__ == "__main__":
    main()
