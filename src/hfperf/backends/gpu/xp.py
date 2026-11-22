"""A tiny array-namespace layer: the same J/K, matmul and eigh code on three devices.

    t = make_target("auto")      # cupy (NVIDIA) -> mps (Apple, PyTorch) -> numpy (CPU)
    Gd = t.asarray(G, "fp32")    # host -> device copy, in the requested precision
    J, K = t.jk(Gd, t.asarray(D, "fp32"))
    t.to_numpy(J)                # device -> host copy, returned as float64

Targets
  numpy        CPU, FP64 and FP32 (FP32 here is a faithful *emulation* of GPU FP32 rounding)
  cupy         NVIDIA GPU via CuPy (cuBLAS / cuSOLVER), FP64 and FP32
  mps          Apple-Silicon GPU via PyTorch Metal backend: FP32 ONLY (Metal has no FP64)
  torch-cuda   NVIDIA GPU via PyTorch (if CuPy is missing)
  torch-cpu    PyTorch on the CPU - lets you test the torch code path on any laptop
"""
from __future__ import annotations

import numpy as np

PRECISIONS = ("fp64", "fp32")


class NumpyTarget:
    name, device, fp64 = "numpy", "cpu", True

    def __init__(self):
        self.eigh_on = "device"   # where the last eigh ran ("device" means this target)

    def dtype(self, prec):
        return np.float32 if prec == "fp32" else np.float64

    def asarray(self, a, prec="fp64"):
        return np.ascontiguousarray(a, dtype=self.dtype(prec))

    def to_numpy(self, x):
        return np.asarray(x, dtype=np.float64)

    def sync(self):
        pass

    def matmul(self, A, B):
        return A @ B

    def eigh(self, A):
        return np.linalg.eigh(A)

    def jk(self, G, D, mode="matmul"):
        n = D.shape[0]
        if mode == "einsum":
            return np.einsum("ijkl,kl->ij", G, D), np.einsum("ikjl,kl->ij", G, D)
        J = (G.reshape(n * n, n * n) @ D.reshape(n * n)).reshape(n, n)
        K = np.matmul(G, D[None, :, :, None]).sum(axis=1)[:, :, 0]
        return J, K

    def describe(self):
        return {"target": self.name, "device": "CPU (NumPy)"}


class CupyTarget(NumpyTarget):
    name, fp64 = "cupy", True

    def __init__(self):
        super().__init__()
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device")
        self.cp = cp
        self.device = "cuda"

    def dtype(self, prec):
        return self.cp.float32 if prec == "fp32" else self.cp.float64

    def asarray(self, a, prec="fp64"):
        return self.cp.ascontiguousarray(self.cp.asarray(a, dtype=self.dtype(prec)))

    def to_numpy(self, x):
        return self.cp.asnumpy(x).astype(np.float64, copy=False)

    def sync(self):
        self.cp.cuda.Stream.null.synchronize()

    def eigh(self, A):
        return self.cp.linalg.eigh(A)

    def jk(self, G, D, mode="matmul"):
        cp = self.cp
        n = D.shape[0]
        if mode == "einsum":
            return cp.einsum("ijkl,kl->ij", G, D), cp.einsum("ikjl,kl->ij", G, D)
        J = (G.reshape(n * n, n * n) @ D.reshape(n * n)).reshape(n, n)
        K = cp.matmul(G, D[None, :, :, None]).sum(axis=1)[:, :, 0]
        return J, K

    def describe(self):
        p = self.cp.cuda.runtime.getDeviceProperties(self.cp.cuda.Device().id)
        name = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
        return {"target": self.name, "device": name, "cupy": self.cp.__version__}


class TorchTarget(NumpyTarget):
    """PyTorch on 'mps' (Apple), 'cuda' or 'cpu'."""

    def __init__(self, device="mps"):
        super().__init__()
        import torch
        self.torch = torch
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS backend not available on this machine")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA not available")
        self.device = device
        self.name = {"mps": "mps", "cuda": "torch-cuda", "cpu": "torch-cpu"}[device]
        self.fp64 = device != "mps"     # Metal Performance Shaders: no float64 tensors

    def dtype(self, prec):
        if prec == "fp64" and not self.fp64:
            raise ValueError("Apple MPS has no FP64: use precision='fp32' or 'mixed'")
        return self.torch.float32 if prec == "fp32" else self.torch.float64

    def asarray(self, a, prec="fp64"):
        t = self.torch
        if isinstance(a, t.Tensor):
            return a.to(device=self.device, dtype=self.dtype(prec)).contiguous()
        # cast on the host first: MPS cannot even hold a float64 tensor
        host = np.ascontiguousarray(a, dtype=np.float32 if prec == "fp32" else np.float64)
        return t.from_numpy(host).to(self.device)

    def to_numpy(self, x):
        return x.detach().to("cpu").numpy().astype(np.float64, copy=False)

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def eigh(self, A):
        # torch.linalg.eigh is not implemented for MPS in many PyTorch versions:
        # fall back to the CPU (same precision) and record that it happened.
        try:
            w, V = self.torch.linalg.eigh(A)
            self.eigh_on = "device"
            return w, V
        except (NotImplementedError, RuntimeError):
            w, V = self.torch.linalg.eigh(A.to("cpu"))
            self.eigh_on = "cpu-fallback"
            return w.to(self.device), V.to(self.device)

    def jk(self, G, D, mode="matmul"):
        t = self.torch
        n = D.shape[0]
        if mode == "einsum":
            return t.einsum("ijkl,kl->ij", G, D), t.einsum("ikjl,kl->ij", G, D)
        J = (G.reshape(n * n, n * n) @ D.reshape(n * n)).reshape(n, n)
        K = t.matmul(G, D[None, :, :, None]).sum(dim=1)[:, :, 0]
        return J, K

    def describe(self):
        d = {"target": self.name, "device": self.device, "torch": self.torch.__version__}
        if self.device == "cuda":
            d["device"] = self.torch.cuda.get_device_name(0)
        return d


def make_target(name: str = "auto"):
    """'auto' tries cupy -> mps -> torch-cuda -> numpy and returns the first that works."""
    makers = {"numpy": NumpyTarget, "cupy": CupyTarget, "mps": lambda: TorchTarget("mps"),
              "torch-cuda": lambda: TorchTarget("cuda"), "torch-cpu": lambda: TorchTarget("cpu")}
    if name != "auto":
        if name not in makers:
            raise ValueError(f"unknown target {name!r}; choose from {sorted(makers)} or 'auto'")
        return makers[name]()
    for cand in ("cupy", "mps", "torch-cuda"):
        try:
            return makers[cand]()
        except Exception:
            continue
    return NumpyTarget()
