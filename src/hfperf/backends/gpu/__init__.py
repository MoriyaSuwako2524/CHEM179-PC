"""Stage 5 backend: GPU J/K, matmul and eigh (CuPy / PyTorch-MPS / NumPy fallback),
plus a CUDA ERI kernel for NVIDIA GPUs (instructor demo, ``eri="cuda"``)."""
from .backend import GpuBackend, best_cpu_eri_backend
from .xp import make_target

__all__ = ["GpuBackend", "best_cpu_eri_backend", "make_target"]
