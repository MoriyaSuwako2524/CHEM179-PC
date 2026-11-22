"""RustBackend: ERIs and J/K (and S, T, V) from the compiled `hfperf_rust` module.

What changes vs. stage 0 (NaiveBackend), by default:
  * ERI, J/K and the one-electron integrals run as compiled Rust (ext/rust/src/*.rs);
  * the N x N dense linear algebra (matmul, transpose, eigh) uses NumPy/LAPACK
    (the stage-1 choice) so that the SCF total is not dominated by the pure-Python
    Jacobi diagonaliser.  ``RustBackend(linalg="naive", onee="naive")`` switches both
    back to stage 0 ("strict" mode: only ERI + J/K change, like CppBackend).

ERI variants (all return the full N^4 tensor; same interface as every backend):
  RustNaiveBackend      eri_naive              stage-0 algorithm, compiled, all N^4 quartets
  RustSym8Backend       eri_sym8(reuse=False)  same quartet routine, 8-fold symmetry + Schwarz
  RustBackend (default) eri_sym8(reuse=True)   + tabulated Hermite E per primitive pair,
                                               iterative R table; bounds-checked indexing
  RustUncheckedBackend  as default, but get_unchecked (unsafe) indexing in ERI and J/K
"""
from __future__ import annotations

import numpy as np

from ...basis import pack_basis
from ..naive import NaiveBackend

try:
    import hfperf_rust as _rs
    HAVE_RUST = True
except ImportError:  # extension not built on this machine
    _rs = None
    HAVE_RUST = False

BUILD_HINT = "the Rust extension is not built: cd ext/rust && maturin develop --release"


def _packed(funcs):
    p = pack_basis(funcs)
    return tuple(np.ascontiguousarray(p[k]) for k in ("centers", "lmn", "offsets", "exps", "coefs"))


class RustBackend(NaiveBackend):
    name = "rust"
    eri_algorithm = "sym8"   # naive | sym8
    reuse = True             # sym8: tabulate primitive-pair data (True) or per-quartet recursion
    checked = True           # bounds-checked indexing in the reuse kernel
    jk_mode = "iter"         # iter (safe, idiomatic) | checked | unchecked

    def __init__(self, screen_tol: float = 1e-12, linalg: str = "numpy", onee: str = "rust"):
        if not HAVE_RUST:
            raise ImportError(BUILD_HINT)
        if linalg not in ("numpy", "naive") or onee not in ("rust", "naive"):
            raise ValueError("linalg must be numpy|naive and onee rust|naive")
        self.screen_tol, self.linalg, self.onee = screen_tol, linalg, onee
        self.last_eri_stats: dict = {}

    # ---- one-electron integrals (naive algorithm, compiled) ----
    def overlap(self, funcs):
        return _rs.overlap(*_packed(funcs)) if self.onee == "rust" else super().overlap(funcs)

    def kinetic(self, funcs):
        return _rs.kinetic(*_packed(funcs)) if self.onee == "rust" else super().kinetic(funcs)

    def nuclear(self, funcs, mol):
        if self.onee != "rust":
            return super().nuclear(funcs, mol)
        charges = np.array(mol.charges, dtype=np.float64)
        coords = np.ascontiguousarray(np.array(mol.coords, dtype=np.float64).reshape(-1, 3))
        return _rs.nuclear(*_packed(funcs), charges, coords)

    # ---- two-electron integrals ----
    def eri(self, funcs):
        arrs = _packed(funcs)
        if self.eri_algorithm == "naive":
            n = len(funcs)
            self.last_eri_stats = {"computed": n ** 4, "screened": 0}
            return _rs.eri_naive(*arrs)
        G, nc, ns = _rs.eri_sym8(*arrs, screen=self.screen_tol, reuse=self.reuse, checked=self.checked)
        self.last_eri_stats = {"computed": nc, "screened": ns}
        return G

    def jk(self, G, D):
        return _rs.jk(np.ascontiguousarray(G), np.ascontiguousarray(D), mode=self.jk_mode)

    # ---- N x N dense linear algebra: NumPy / LAPACK unless linalg="naive" ----
    def matmul(self, A, B):
        return np.asarray(A) @ np.asarray(B) if self.linalg == "numpy" else super().matmul(A, B)

    def transpose(self, A):
        return np.ascontiguousarray(np.asarray(A).T) if self.linalg == "numpy" else super().transpose(A)

    def eigh(self, A):
        return np.linalg.eigh(np.asarray(A)) if self.linalg == "numpy" else super().eigh(A)

    # solve() (DIIS, at most 9x9) stays the naive Gaussian elimination.


class RustNaiveBackend(RustBackend):
    name = "rust-naive"
    eri_algorithm = "naive"


class RustSym8Backend(RustBackend):
    name = "rust-sym8-noreuse"
    reuse = False


class RustUncheckedBackend(RustBackend):
    name = "rust-unchecked"
    checked = False
    jk_mode = "unchecked"


class RustRayonBackend(RustBackend):
    """Stage 4: Rayon threads (ext/rust/src/par.rs).  One-electron integrals stay
    serial Rust, linear algebra NumPy/LAPACK (as RustBackend).

    threads    : worker threads (0 = RAYON_NUM_THREADS or all logical CPUs)
    mode       : "steal" (default, Rayon work stealing) | "block" (one contiguous block of
                 pair rows per thread; the analogue of OpenMP schedule(static))
    fock       : "tensor" (default; N^4 tensor stored once) | "direct" (integral-direct)
    accumulate : direct only - "fold" (thread-private J/K, fold + reduce) | "atomic"
    store      : direct only - keep computed values after the first build (semi-direct)
    The parallel ERI kernel is the bounds-checked tabulated one (RustBackend default).
    """
    name = "rust-rayon"

    def __init__(self, threads: int = 0, mode: str = "steal", fock: str = "tensor",
                 accumulate: str = "fold", store: bool = False, screen_tol: float = 1e-12,
                 linalg: str = "numpy", onee: str = "rust"):
        super().__init__(screen_tol=screen_tol, linalg=linalg, onee=onee)
        if not hasattr(_rs, "eri_sym8_par"):
            raise ImportError("hfperf_rust was built without the stage-4 Rayon functions; rebuild: "
                              "cd ext/rust && maturin develop --release")
        if mode not in ("steal", "block"):
            raise ValueError("mode must be steal | block")
        if fock not in ("tensor", "direct"):
            raise ValueError("fock must be tensor | direct")
        if accumulate not in ("fold", "private", "atomic"):
            raise ValueError("accumulate must be fold | atomic")
        self.threads, self.mode, self.fock = int(threads), mode, fock
        self.accumulate, self.store = accumulate, bool(store)
        self.last_jk_stats: dict = {}

    def eri(self, funcs):
        arrs = _packed(funcs)
        if self.fock == "direct":
            djk = _rs.DirectJK(*arrs, screen=self.screen_tol, store=self.store, rows=None,
                               threads=self.threads)
            self.last_eri_stats = {"setup_s": djk.last_stats["setup_s"], "n_quartets": djk.n_quartets}
            return djk
        G, info = _rs.eri_sym8_par(*arrs, screen=self.screen_tol, threads=self.threads, mode=self.mode)
        self.last_eri_stats = dict(info)
        return G

    def jk(self, G, D):
        D = np.ascontiguousarray(D, dtype=np.float64)
        if self.fock == "direct":
            J, K = G.build(D, threads=self.threads, mode=self.accumulate)
            self.last_jk_stats = dict(G.last_stats)
            return J, K
        return _rs.jk_par(np.ascontiguousarray(G), D, threads=self.threads)
