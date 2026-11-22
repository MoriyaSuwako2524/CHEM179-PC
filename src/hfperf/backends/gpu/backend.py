"""Stage 5 backend: J/K, matmul and eigh on a GPU (CuPy, PyTorch-MPS) or the CPU fallback.

    GpuBackend()                                   # auto: cupy -> mps -> torch-cuda -> numpy
    GpuBackend(target="mps", precision="fp32")     # Apple Silicon: FP32 only
    GpuBackend(precision="mixed")                  # FP32 J/K increments + FP64 anchors
    GpuBackend(target="cupy", eri="cuda")          # instructor demo: ERIs from the CUDA kernel

What changes vs. stage 1 (NumpyBackend)
  * the ERI tensor is computed ONCE (by the best CPU backend: cpp -> rust -> numpy, or on
    the GPU with eri="cuda") and then lives in device memory; ``eri()`` returns the device
    array, which the SCF driver only hands back to ``jk``;
  * every J/K build ships D (N^2 numbers) to the device and J, K back - the N^4 tensor
    never moves again;
  * matmul / eigh run on the device too (linalg="device") or stay in NumPy (linalg="host").
  * one-electron integrals and DIIS are unchanged (NumPy).

Precision modes
  fp64   everything in double precision (CuPy, NumPy, torch-cuda/cpu; NOT available on MPS)
  fp32   G, D, J, K, matmul and eigh in single precision: ~2x less memory traffic, and
         8x-64x more arithmetic throughput on consumer GPUs - but energy errors ~1e-5 Eh
  mixed  "incremental Fock build": an FP64 J/K at the first iteration (the *anchor*), then
         only the CHANGE of the density dD = D_n - D_{n-1} goes through FP32 J/K on the
         device and is accumulated in FP64 on the host.  The FP32 error is proportional to
         |dD|, which shrinks as the SCF converges.  A new FP64 anchor is taken whenever
         rms(dD) has dropped by ``anchor_factor`` relative to the first increment after the
         previous anchor (anchor_factor=0: never re-anchor).  matmul/eigh stay FP64
         (on the host when the device has no FP64).
"""
from __future__ import annotations

import time

import numpy as np

from ..numpy_opt import NumpyBackend
from .xp import CupyTarget, NumpyTarget, make_target


def best_cpu_eri_backend(preference: str = "auto"):
    """The fastest CPU ERI code built on this machine: cpp -> rust -> numpy."""
    order = ["cpp", "rust", "numpy"] if preference == "auto" else [preference]
    errors = []
    for name in order:
        try:
            if name == "cpp":
                from ..cpp import CppBackend
                return CppBackend()
            if name == "rust":
                from ..rust import RustBackend
                return RustBackend()
            if name == "numpy":
                return NumpyBackend()
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError("no CPU ERI backend available: " + "; ".join(errors))


class GpuBackend(NumpyBackend):
    name = "gpu"

    def __init__(self, target: str = "auto", precision: str = "auto", eri: str = "auto",
                 linalg: str = "device", jk: str = "matmul", anchor_factor: float = 1e-3,
                 block: int = 128, order: str = "sorted", screen_tol: float = 1e-12):
        super().__init__()
        self.t = make_target(target)
        if precision == "auto":
            precision = "fp64" if self.t.fp64 else "fp32"
        if precision not in ("fp64", "fp32", "mixed"):
            raise ValueError("precision must be fp64, fp32, mixed or auto")
        if precision == "fp64" and not self.t.fp64:
            raise ValueError(f"target {self.t.name} has no FP64; use precision='fp32' or 'mixed'")
        if eri not in ("auto", "cpp", "rust", "numpy", "cuda"):
            raise ValueError("eri must be auto, cpp, rust, numpy or cuda")
        if eri == "cuda" and not isinstance(self.t, CupyTarget):
            raise ValueError("eri='cuda' needs the cupy target (an NVIDIA GPU)")
        if linalg not in ("device", "host") or jk not in ("matmul", "einsum"):
            raise ValueError("linalg must be device|host and jk matmul|einsum")
        self.precision, self.eri_source, self.linalg, self.jk_mode = precision, eri, linalg, jk
        self.anchor_factor = float(anchor_factor)
        self.cuda_opts = dict(block=int(block), order=order, screen_tol=screen_tol)
        self.work = "fp64" if precision == "fp64" else "fp32"     # precision of G on the device
        self.lin = "fp32" if precision == "fp32" else "fp64"      # precision of matmul / eigh
        self.lin_on_host = linalg == "host" or (self.lin == "fp64" and not self.t.fp64)
        self._G = None
        self.cuda_stats: dict = {}
        self.reset_stats()

    def __repr__(self):
        return (f"GpuBackend(target={self.t.name!r}, precision={self.precision!r}, eri={self.eri_source!r}, "
                f"linalg={self.linalg!r}, jk={self.jk_mode!r})")

    def reset_stats(self):
        self.gpu_stats = dict(eri_cpu_s=0.0, h2d_G_s=0.0, G_bytes=0, eri_backend=None,
                              jk_calls=0, jk_h2d_s=0.0, jk_kernel_s=0.0, jk_d2h_s=0.0,
                              fp64_builds=0, fp32_builds=0, linalg_calls=0, linalg_s=0.0, eigh_on=None)

    def describe(self) -> dict:
        return dict(self.t.describe(), precision=self.precision, eri=self.eri_source, linalg=self.linalg,
                    jk=self.jk_mode, anchor_factor=self.anchor_factor)

    # ------------------------------------------------------------ ERI (computed once)
    def eri(self, funcs):
        s = self.gpu_stats
        if self.eri_source == "cuda":
            from .cuda_eri import CudaERI
            eng = CudaERI(precision="fp32" if self.precision == "fp32" else "fp64", **self.cuda_opts)
            G = eng.compute(funcs)
            self.cuda_stats = eng.stats
            s["eri_backend"] = f"cuda-{eng.precision}"
            dev = G.astype(self.t.dtype("fp32")) if self.precision == "mixed" else G
            g64 = G if self.precision == "mixed" else None
        else:
            cpu = best_cpu_eri_backend(self.eri_source)
            s["eri_backend"] = cpu.name
            t0 = time.perf_counter()
            Gh = cpu.eri(funcs)
            s["eri_cpu_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            dev = self.t.asarray(Gh, self.work)
            self.t.sync()
            s["h2d_G_s"] = time.perf_counter() - t0
            g64 = Gh if self.precision == "mixed" else None
        s["G_bytes"] = int(dev.nbytes) if hasattr(dev, "nbytes") else int(dev.numel() * dev.element_size())
        self._G = dict(key=id(dev), dev=dev, g64=g64)
        return dev

    def _entry(self, G):
        """Device copy (and FP64 source for 'mixed') of the ERI tensor passed to jk()."""
        if self._G is not None and self._G["key"] == id(G):
            return self._G
        # a tensor that eri() did not produce, e.g. a NumPy array from another backend
        self._G = dict(key=id(G), dev=self.t.asarray(G, self.work), g64=G if self.precision == "mixed" else None)
        return self._G

    # ------------------------------------------------------------ J/K
    def _jk_device(self, Gd, D, prec):
        s, t = self.gpu_stats, self.t
        t0 = time.perf_counter()
        Dd = t.asarray(D, prec)
        t.sync()
        t1 = time.perf_counter()
        J, K = t.jk(Gd, Dd, self.jk_mode)
        t.sync()
        t2 = time.perf_counter()
        Jh, Kh = t.to_numpy(J), t.to_numpy(K)
        t3 = time.perf_counter()
        s["jk_h2d_s"] += t1 - t0
        s["jk_kernel_s"] += t2 - t1
        s["jk_d2h_s"] += t3 - t2
        return Jh, Kh

    def _jk_fp64(self, e, D):
        """Exact FP64 J/K for an anchor: on the device if it holds an FP64 G, else NumPy/BLAS."""
        self.gpu_stats["fp64_builds"] += 1
        g = e["g64"]
        if isinstance(g, np.ndarray):
            return NumpyTarget().jk(g, np.asarray(D, dtype=np.float64), self.jk_mode)
        return self._jk_device(g, D, "fp64")

    def jk(self, G, D):
        e = self._entry(G)
        self.gpu_stats["jk_calls"] += 1
        D = np.asarray(D, dtype=np.float64)
        if self.precision != "mixed":
            if self.precision == "fp32":
                self.gpu_stats["fp32_builds"] += 1
            else:
                self.gpu_stats["fp64_builds"] += 1
            return self._jk_device(e["dev"], D, self.work)
        if "J" not in e:                                  # first call: FP64 anchor
            e["J"], e["K"] = self._jk_fp64(e, D)
            e["ref"] = None
        else:
            dD = D - e["Dprev"]
            rms = float(np.sqrt(np.mean(dD * dD)))
            if rms == 0.0:
                pass
            elif self.anchor_factor > 0 and e["ref"] is not None and rms < self.anchor_factor * e["ref"]:
                e["J"], e["K"] = self._jk_fp64(e, D)          # re-anchor: fresh FP64 build
                e["ref"] = None
            else:
                dJ, dK = self._jk_device(e["dev"], dD, "fp32")  # FP32 increment ...
                e["J"] = e["J"] + dJ                            # ... accumulated in FP64
                e["K"] = e["K"] + dK
                self.gpu_stats["fp32_builds"] += 1
                if e["ref"] is None:
                    e["ref"] = rms
        e["Dprev"] = D.copy()
        return e["J"].copy(), e["K"].copy()

    # ------------------------------------------------------------ dense linear algebra
    def _lin(self, fn, *mats):
        t0 = time.perf_counter()
        if self.lin_on_host:
            out = fn(NumpyTarget(), *[np.asarray(m, dtype=np.float64) for m in mats])
        else:
            dev = [self.t.asarray(m, self.lin) for m in mats]
            res = fn(self.t, *dev)
            out = tuple(self.t.to_numpy(r) for r in res) if isinstance(res, tuple) else self.t.to_numpy(res)
        self.gpu_stats["linalg_calls"] += 1
        self.gpu_stats["linalg_s"] += time.perf_counter() - t0
        return out

    def matmul(self, A, B):
        return self._lin(lambda t, a, b: t.matmul(a, b), A, B)

    def transpose(self, A):
        return np.asarray(A).T

    def eigh(self, A):
        w, V = self._lin(lambda t, a: t.eigh(a), A)
        self.gpu_stats["eigh_on"] = "host" if self.lin_on_host else self.t.eigh_on
        return w, V
