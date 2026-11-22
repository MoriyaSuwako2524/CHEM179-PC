"""Stage 5 benchmark (instructor demo on an NVIDIA GPU; also runs on a laptop with CuPy).

    python ext/cuda/bench_gpu.py --out results/gpu/raw/bench_<gpu>.json [--molecules h2o nh3 ...]

Sections (each wrapped in try/except, the JSON is rewritten after every section):
  info        GPU model, CUDA/CuPy versions, CPU model (lscpu), nvidia-smi
  eri         CUDA ERI kernel: FP64/FP32, pair order sorted/natural, block-size sweep;
              phase timings (CUDA events), registers/local memory/occupancy, accuracy
              vs the PySCF reference ERIs (h2, h2o, nh3) and FP32-vs-FP64 max error
  transfer    host<->device copy of the full ERI tensor (pageable and pinned host memory)
  kernels     J/K (matmul/einsum), eigh, matmul: device FP64/FP32 vs NumPy on the host CPU
  scf         full RHF with GpuBackend(target='cupy', eri='cuda') in fp64 / fp32 / fp32-jk-only /
              mixed / mixed-no-reanchor, + the student path (CPU ERI -> GPU J/K) for small molecules
  emulation   the same precision modes with the NumPy target (CPU FP32 emulation = what a
              student without an NVIDIA GPU measures); uses the FP64 tensor from the GPU
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hfperf import build_basis, load_molecule, rhf, timing  # noqa: E402
from hfperf.backends.gpu import GpuBackend  # noqa: E402
from hfperf.backends.gpu.cuda_eri import CudaERI, gpu_info  # noqa: E402

REF = json.loads((ROOT / "tests/data/reference_energies.json").read_text())
LOOSE = dict(e_tol=1e-7, d_tol=1e-5)          # FP32 cannot reach 1e-10 / 1e-8


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:
        return repr(exc)


def funcs_of(name):
    return build_basis(load_molecule(name))[1]


def ref_eri(name):
    p = ROOT / f"tests/data/ref_{name}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["ERI"] if "ERI" in d.files else None


def timed(fn, sync, repeats):
    best = float("inf")
    out = None
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--molecules", nargs="+", default=["h2", "h2o", "nh3", "ch4", "c2h4", "benzene"])
    ap.add_argument("--scf-molecules", nargs="+", default=None, help="default: same as --molecules")
    ap.add_argument("--student-molecules", nargs="*", default=["h2o", "nh3", "ch4", "c2h4"])
    ap.add_argument("--sections", nargs="+", default=["info", "eri", "transfer", "kernels", "scf", "emulation"])
    a = ap.parse_args()
    import cupy as cp
    sync = cp.cuda.Stream.null.synchronize

    rec: dict = {"argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(rec, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    def section(name, fn):
        if name not in a.sections:
            return
        t0 = time.perf_counter()
        try:
            rec[name] = fn()
        except Exception:
            rec[name + "_error"] = traceback.format_exc()
            print(traceback.format_exc(), flush=True)
        print(f"== section {name} done in {time.perf_counter() - t0:.1f} s", flush=True)
        save()

    # ------------------------------------------------------------------ info
    def info():
        lscpu = sh("lscpu")
        model = next((ln.split(":", 1)[1].strip() for ln in lscpu.splitlines() if ln.startswith("Model name")), "?")
        return dict(gpu=gpu_info(), cpu_model=model, lscpu=lscpu, host=platform.node(),
                    nvidia_smi=sh("nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm "
                                  "--format=csv"),
                    python=sys.version, numpy=np.__version__,
                    omp_threads=os.environ.get("OMP_NUM_THREADS"), slurm_job=os.environ.get("SLURM_JOB_ID"))
    section("info", info)

    # ------------------------------------------------------------------ ERI kernel
    G64_host: dict = {}

    def eri():
        rows = []
        CudaERI().compute(funcs_of("h2"))            # compile + warm up (both precisions)
        CudaERI(precision="fp32").compute(funcs_of("h2"))
        for name in a.molecules:
            f = funcs_of(name)
            R = ref_eri(name)
            Gref64 = None
            variants = [("fp64", "sorted", 128), ("fp32", "sorted", 128), ("fp64", "natural", 128)]
            if name in ("c2h4", "benzene"):
                variants += [("fp64", "sorted", 64), ("fp64", "sorted", 256), ("fp32", "natural", 128)]
            for prec, order, block in variants:
                eng = CudaERI(precision=prec, order=order, block=block)
                reps = 3 if len(f) < 60 else 2
                best = None
                for _ in range(reps):
                    G = eng.compute(f)
                    st = dict(eng.stats)
                    if best is None or st["quartet_kernel_s"] < best["quartet_kernel_s"]:
                        best = st
                    del G
                    cp.get_default_memory_pool().free_all_blocks()
                G = eng.compute(f)
                Gh = G.get().astype(np.float64)
                del G
                cp.get_default_memory_pool().free_all_blocks()
                row = dict(molecule=name, **{k: v for k, v in best.items() if not k.endswith("resources")},
                           quartet_resources=best["quartet_kernel_resources"],
                           pair_resources=best["pair_kernel_resources"])
                if R is not None:
                    row["max_abs_err_vs_pyscf"] = float(np.abs(Gh - R).max())
                if prec == "fp64" and order == "sorted" and block == 128:
                    Gref64 = Gh
                    G64_host[name] = Gh
                elif Gref64 is not None:
                    row["max_abs_err_vs_gpu_fp64"] = float(np.abs(Gh - Gref64).max())
                    row["max_rel_err_vs_gpu_fp64"] = float(np.abs(Gh - Gref64).max() / np.abs(Gref64).max())
                rows.append(row)
                print(f"eri {name:12s} {prec} {order:8s} blk {block:3d}: quartet kernel "
                      f"{best['quartet_kernel_s']:.4f} s  total device {best['device_total_s']:.4f} s  "
                      f"err {row.get('max_abs_err_vs_pyscf', row.get('max_abs_err_vs_gpu_fp64', float('nan'))):.2e}",
                      flush=True)
                del Gh
            save()
        return rows
    section("eri", eri)

    # ------------------------------------------------------------------ transfers
    def transfer():
        rows = []
        for name in a.molecules:
            Gh = G64_host.get(name)
            if Gh is None:
                continue
            for prec in ("fp64", "fp32"):
                host = np.ascontiguousarray(Gh, dtype=np.float32 if prec == "fp32" else np.float64)
                t_h2d, d = timed(lambda: cp.asarray(host), sync, 3)
                t_d2h, _ = timed(lambda: d.get(), sync, 3)
                # pinned (page-locked) host buffer: the DMA engine can read it directly
                pinned_mem = cp.cuda.alloc_pinned_memory(host.nbytes)
                pinned = np.frombuffer(pinned_mem, host.dtype, host.size).reshape(host.shape)
                pinned[...] = host
                t_pin, _ = timed(lambda: d.set(pinned), sync, 3)
                rows.append(dict(molecule=name, nbf=Gh.shape[0], precision=prec, bytes=host.nbytes,
                                 h2d_pageable_s=t_h2d, d2h_pageable_s=t_d2h, h2d_pinned_s=t_pin,
                                 h2d_pageable_GBps=host.nbytes / t_h2d / 1e9, h2d_pinned_GBps=host.nbytes / t_pin / 1e9,
                                 d2h_pageable_GBps=host.nbytes / t_d2h / 1e9))
                del d, pinned, pinned_mem
                cp.get_default_memory_pool().free_all_blocks()
            print(f"transfer {name} done", flush=True)
        return rows
    section("transfer", transfer)

    # ------------------------------------------------------------------ J/K, eigh, matmul
    def kernels():
        rows = []
        rng = np.random.default_rng(0)
        for name in a.molecules:
            Gh = G64_host.get(name)
            if Gh is None:
                continue
            n = Gh.shape[0]
            A = rng.normal(size=(n, n))
            D = A @ A.T / n
            Jr, Kr = np.einsum("ijkl,kl->ij", Gh, D, optimize=True), np.einsum("ikjl,kl->ij", Gh, D, optimize=True)
            reps = 20 if n < 60 else 5
            row = dict(molecule=name, nbf=n)
            # CPU NumPy on this node (threads as set by OMP_NUM_THREADS)
            from hfperf.backends.gpu.xp import NumpyTarget
            nt = NumpyTarget()
            row["cpu_jk_matmul_s"], _ = timed(lambda: nt.jk(Gh, D, "matmul"), lambda: None, reps)
            row["cpu_eigh_s"], _ = timed(lambda: np.linalg.eigh(D), lambda: None, reps)
            row["cpu_matmul_s"], _ = timed(lambda: D @ A, lambda: None, reps)
            for prec in ("fp64", "fp32"):
                dt = cp.float32 if prec == "fp32" else cp.float64
                Gd, Dd, Ad = cp.asarray(Gh, dtype=dt), cp.asarray(D, dtype=dt), cp.asarray(A, dtype=dt)
                for mode in ("matmul", "einsum"):
                    def jk():
                        if mode == "einsum":
                            return cp.einsum("ijkl,kl->ij", Gd, Dd), cp.einsum("ikjl,kl->ij", Gd, Dd)
                        J = (Gd.reshape(n * n, n * n) @ Dd.reshape(n * n)).reshape(n, n)
                        K = cp.matmul(Gd, Dd[None, :, :, None]).sum(axis=1)[:, :, 0]
                        return J, K
                    jk()
                    t, (J, K) = timed(jk, sync, reps)
                    row[f"gpu_jk_{mode}_{prec}_s"] = t
                    row[f"gpu_jk_{mode}_{prec}_maxerr"] = float(max(np.abs(J.get() - Jr).max(), np.abs(K.get() - Kr).max()))
                cp.linalg.eigh(Dd)
                row[f"gpu_eigh_{prec}_s"], _ = timed(lambda: cp.linalg.eigh(Dd), sync, reps)
                row[f"gpu_matmul_{prec}_s"], _ = timed(lambda: Dd @ Ad, sync, reps)
                # round trip of an N x N matrix (what every backend call pays)
                row[f"roundtrip_NxN_{prec}_s"], _ = timed(lambda: cp.asarray(D, dtype=dt).get(), sync, reps)
                del Gd, Dd, Ad
                cp.get_default_memory_pool().free_all_blocks()
            rows.append(row)
            print("kernels", {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        return rows
    section("kernels", kernels)

    # ------------------------------------------------------------------ SCF
    def run_scf(name, be, **kw):
        timing.reset()
        t0 = time.perf_counter()
        res = rhf(load_molecule(name), be, verbose=False, **kw)
        wall = time.perf_counter() - t0
        e_ref = REF[name]["e_rhf"]
        return dict(molecule=name, nbf=len(funcs_of(name)), energy=res.energy, err_vs_ref=res.energy - e_ref,
                    converged=res.converged, iterations=res.iterations, wall_s=wall,
                    kernels=timing.summary(), gpu_stats=dict(be.gpu_stats),
                    cuda_stats={k: v for k, v in be.cuda_stats.items() if not k.endswith("resources")},
                    backend=be.describe(), tolerances=kw or "default")

    def scf():
        rows = []
        modes = [("fp64", dict(precision="fp64"), {}),
                 ("fp32", dict(precision="fp32"), LOOSE),
                 ("fp32-jk-only", dict(precision="fp32", linalg="host"), LOOSE),
                 ("mixed", dict(precision="mixed"), {}),
                 ("mixed-no-reanchor", dict(precision="mixed", anchor_factor=0), {})]
        for name in a.scf_molecules or a.molecules:
            for label, opts, kw in modes:
                r = run_scf(name, GpuBackend(target="cupy", eri="cuda", **opts), **kw)
                r["mode"] = label
                r["path"] = "instructor (CUDA ERI)"
                rows.append(r)
                print(f"scf {name:12s} {label:18s} E err {r['err_vs_ref']:+.3e}  it {r['iterations']:3d}  "
                      f"wall {r['wall_s']:.3f} s  conv {r['converged']}", flush=True)
                cp.get_default_memory_pool().free_all_blocks()
                save()
        for name in a.student_molecules:
            for label, opts, kw in modes[:2] + modes[3:4]:
                r = run_scf(name, GpuBackend(target="cupy", eri="auto", **opts), **kw)
                r["mode"] = label
                r["path"] = "student (CPU ERI -> GPU J/K)"
                rows.append(r)
                print(f"scf-student {name:8s} {label:8s} E err {r['err_vs_ref']:+.3e} wall {r['wall_s']:.3f} s "
                      f"eri_cpu {r['gpu_stats']['eri_cpu_s']:.3f} s h2d {r['gpu_stats']['h2d_G_s']:.4f} s", flush=True)
        return rows
    section("scf", scf)

    # ------------------------------------------------------------------ CPU FP32 emulation
    def emulation():
        rows = []
        for name in a.molecules:
            Gh = G64_host.get(name)
            if Gh is None:
                continue
            for label, opts, kw in [("fp64", dict(precision="fp64"), {}), ("fp32", dict(precision="fp32"), LOOSE),
                                    ("fp32-jk-only", dict(precision="fp32", linalg="host"), LOOSE),
                                    ("mixed", dict(precision="mixed"), {}),
                                    ("mixed-no-reanchor", dict(precision="mixed", anchor_factor=0), {})]:
                be = GpuBackend(target="numpy", **opts)
                be.eri = lambda funcs, _G=Gh: _G          # reuse the validated FP64 tensor
                r = run_scf(name, be, **kw)
                r["mode"] = label
                r["path"] = "numpy target (CPU FP32 emulation)"
                rows.append(r)
                print(f"emul {name:12s} {label:18s} E err {r['err_vs_ref']:+.3e} it {r['iterations']}", flush=True)
        return rows
    section("emulation", emulation)
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()


if __name__ == "__main__":
    main()
