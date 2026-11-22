"""Backend registry.

A backend is an object with these methods (all take/return NumPy arrays):

    overlap(funcs)            -> S   (N x N)
    kinetic(funcs)            -> T   (N x N)
    nuclear(funcs, mol)       -> V   (N x N)
    eri(funcs)                -> (ij|kl) as an N x N x N x N array
    matmul(A, B)              -> A @ B
    eigh(A)                   -> (eigenvalues ascending, eigenvectors as columns)
    jk(eri, D)                -> (J, K)

Each class stage subclasses an earlier backend and overrides one or two of
these methods, so a speed-up can always be traced to exactly one change.
"""
from __future__ import annotations

from importlib import import_module

# Stage backends are registered here up front so that each stage lives in its
# own subpackage and never has to edit this file.
_REGISTRY = {
    "naive": "hfperf.backends.naive:NaiveBackend",          # stage 0: Python loops
    "numpy": "hfperf.backends.numpy_opt:NumpyBackend",      # stage 1: memory layout, BLAS, symmetry
    "cpp": "hfperf.backends.cpp:CppBackend",                # stage 2: C++ kernel via pybind11
    "rust": "hfperf.backends.rust:RustBackend",             # stage 2: Rust kernel via PyO3
    "cpp-omp": "hfperf.backends.cpp:CppOmpBackend",         # stage 4: OpenMP threads
    "rust-rayon": "hfperf.backends.rust:RustRayonBackend",  # stage 4: Rayon threads
    "gpu": "hfperf.backends.gpu:GpuBackend",                # stage 5: CuPy / PyTorch-MPS
}


def register(name: str, target: str) -> None:
    _REGISTRY[name] = target


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str = "naive", **options):
    """Instantiate a backend; ``options`` go to its constructor
    (e.g. get_backend("numpy", eri="sym8"), get_backend("cpp", reuse=False))."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {available()}")
    mod, cls = _REGISTRY[name].split(":")
    return getattr(import_module(mod), cls)(**options)
