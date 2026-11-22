"""Stage 2a: C++ kernels via pybind11 (build with ``python ext/cpp/build.py``).

Importing this package never fails; the compiled ``_hfcpp`` module is loaded
when a backend is instantiated.
"""
from .backend import CppBackend, CppOmpBackend, load_module

__all__ = ["CppBackend", "CppOmpBackend", "load_module"]
