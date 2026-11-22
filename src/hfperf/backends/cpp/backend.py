"""Stage 2a backend: ERI and J/K in C++ (pybind11); everything else is stage 0.

Only ``eri`` and ``jk`` are overridden, so the one-electron integrals, the
matrix multiplies and the Jacobi diagonalization still run as Python loops.
Comparing ``--backend naive`` with ``--backend cpp`` therefore measures the
effect of moving exactly these two kernels into a compiled language.
"""
from __future__ import annotations

from importlib import import_module

import numpy as np

from ...basis import pack_basis
from ..naive import NaiveBackend

BUILD_HINT = "build it with:  python -m pip install pybind11 && python ext/cpp/build.py"


def load_module(name: str = "_hfcpp"):
    """Import the compiled extension (``_hfcpp`` or a flag variant such as ``_hfcpp_O0``)."""
    try:
        return import_module(f"{__package__}.{name}")
    except ImportError as exc:
        raise ImportError(f"compiled module hfperf.backends.cpp.{name} not found; {BUILD_HINT}") from exc


class CppBackend(NaiveBackend):
    """C++ ERI + J/K.

    eri_algorithm : "sym8" (default; 8-fold symmetry + Schwarz screening) or
                    "naive" (same algorithm as the Python stage 0, no symmetry)
    reuse         : sym8 only - precompute primitive-pair data (True) or call the
                    naive per-quartet kernel for every unique quartet (False)
    screen_tol    : Schwarz threshold; quartets with Q_ij Q_kl < tol are set to 0
    jk_algorithm  : "fused" (default; one pass over G), "twopass" (J pass + K pass)
                    or "naive" (the Python loop order)
    module        : compiled module name, e.g. "_hfcpp_O0" for the flag study
    """

    name = "cpp"

    def __init__(self, eri_algorithm: str = "sym8", reuse: bool = True, screen_tol: float = 1e-12,
                 jk_algorithm: str = "fused", module: str = "_hfcpp"):
        if eri_algorithm not in ("sym8", "naive"):
            raise ValueError("eri_algorithm must be 'sym8' or 'naive'")
        if jk_algorithm not in ("fused", "twopass", "naive"):
            raise ValueError("jk_algorithm must be 'fused', 'twopass' or 'naive'")
        self.ext = load_module(module)
        self.eri_algorithm, self.reuse, self.screen_tol = eri_algorithm, reuse, screen_tol
        self.jk_algorithm = jk_algorithm
        self.last_eri_stats: dict = {}

    @staticmethod
    def _args(funcs):
        b = pack_basis(funcs)
        return b["centers"], b["lmn"], b["offsets"], b["exps"], b["coefs"]

    def eri(self, funcs):
        if self.eri_algorithm == "naive":
            n = len(funcs)
            self.last_eri_stats = {"computed": n ** 4, "screened": 0}
            return self.ext.eri_naive(*self._args(funcs))
        G, computed, screened = self.ext.eri_sym8(*self._args(funcs), self.screen_tol, self.reuse)
        self.last_eri_stats = {"computed": computed, "screened": screened}
        return G

    def jk(self, G, D):
        f = {"fused": self.ext.jk, "twopass": self.ext.jk_twopass, "naive": self.ext.jk_naive}[self.jk_algorithm]
        return f(G, D)


class CppOmpBackend(CppBackend):
    """Stage 4: the same kernels, shared among threads with OpenMP (module ``_hfcpp_omp``,
    built by ``python ext/cpp/build.py --openmp``).

    threads    : number of OpenMP threads (0 = OMP_NUM_THREADS or all cores)
    schedule   : "dynamic" (default) | "static" | "guided"  - how pair rows ij are handed out
    chunk      : rows per hand-out (0 = OpenMP default; static + 0 = one contiguous block per thread)
    fock       : "tensor" (default) - store the N^4 tensor once, J/K from it every iteration
                 "direct" - integral-direct: recompute the quartets in every J/K build, no tensor
    accumulate : direct only - "private" (thread-private J/K + reduction) or "atomic"
    store      : direct only - keep the computed values after the first build (semi-direct)
    rest       : "numpy" (default) - one-electron integrals and N x N linear algebra from the
                 stage-1 NumpyBackend, so the SCF time is ERI + J/K + a small serial rest;
                 "naive" - keep the stage-0 Python loops (huge serial fraction: Amdahl demo)
    After eri()/jk(): ``last_eri_stats`` / ``last_jk_stats`` hold quartet counts and the
    per-thread busy times (``busy``), from which the load imbalance max/mean is computed.
    """

    name = "cpp-omp"

    def __init__(self, threads: int = 0, schedule: str = "dynamic", chunk: int = 1, fock: str = "tensor",
                 accumulate: str = "private", store: bool = False, screen_tol: float = 1e-12,
                 rest: str = "numpy", module: str = "_hfcpp_omp"):
        if schedule not in ("dynamic", "static", "guided"):
            raise ValueError("schedule must be dynamic, static or guided")
        if fock not in ("tensor", "direct"):
            raise ValueError("fock must be 'tensor' or 'direct'")
        if accumulate not in ("private", "atomic"):
            raise ValueError("accumulate must be 'private' or 'atomic'")
        if rest not in ("numpy", "naive"):
            raise ValueError("rest must be 'numpy' or 'naive'")
        try:
            self.ext = load_module(module)
        except ImportError as exc:
            raise ImportError(f"{exc}\n(for stage 4 build the OpenMP module: python ext/cpp/build.py --openmp)") from exc
        self.eri_algorithm, self.reuse, self.screen_tol, self.jk_algorithm = "sym8", True, screen_tol, "fused"
        self.threads, self.schedule, self.chunk = int(threads), schedule, int(chunk)
        self.fock, self.accumulate, self.store, self.rest = fock, accumulate, bool(store), rest
        self.last_eri_stats: dict = {}
        self.last_jk_stats: dict = {}
        self._np = None
        if rest == "numpy":
            from ..numpy_opt import NumpyBackend
            self._np = NumpyBackend()

    def openmp_info(self) -> dict:
        return dict(self.ext.openmp_info())

    # ---- one-electron integrals and linear algebra: stage 1 (or stage 0 with rest="naive")
    def overlap(self, funcs):
        return self._np.overlap(funcs) if self._np else super().overlap(funcs)

    def kinetic(self, funcs):
        return self._np.kinetic(funcs) if self._np else super().kinetic(funcs)

    def nuclear(self, funcs, mol):
        return self._np.nuclear(funcs, mol) if self._np else super().nuclear(funcs, mol)

    def matmul(self, A, B):
        return self._np.matmul(A, B) if self._np else super().matmul(A, B)

    def transpose(self, A):
        return self._np.transpose(A) if self._np else super().transpose(A)

    def eigh(self, A):
        return self._np.eigh(A) if self._np else super().eigh(A)

    def solve(self, A, b):
        return self._np.solve(A, b) if self._np else super().solve(A, b)

    # ---- ERI / J,K with threads
    def eri(self, funcs):
        args = self._args(funcs)
        if self.fock == "direct":
            # no tensor: the returned builder object is what scf.py passes back to jk()
            djk = self.ext.DirectJK(*args, self.screen_tol, self.store, None, self.threads)
            self.last_eri_stats = {"setup_s": djk.last_stats["setup_s"], "n_quartets": djk.n_quartets}
            return djk
        G, info = self.ext.eri_sym8_par(*args, self.screen_tol, self.threads, self.schedule, self.chunk)
        self.last_eri_stats = dict(info)
        return G

    def jk(self, G, D):
        D = np.ascontiguousarray(D, dtype=np.float64)
        if self.fock == "direct":
            J, K = G.build(D, self.threads, self.accumulate, self.schedule, self.chunk)
            self.last_jk_stats = dict(G.last_stats)
            return J, K
        return self.ext.jk_par(G, D, self.threads)
