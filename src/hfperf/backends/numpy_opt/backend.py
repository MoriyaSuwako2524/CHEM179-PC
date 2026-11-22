"""Stage 1 backend: NumPy/BLAS/LAPACK for linear algebra and algorithmic tricks for the ERIs.

    NumpyBackend()                                    # fastest correct combination (default)
    NumpyBackend(eri="sym8")                          # one rung of the ERI ladder
    NumpyBackend(eri="shell", screening=1e-12, boys="series")

Options
    eri        "full" | "sym8" | "shell" | "vector"        (see eri.py)
    screening  Schwarz threshold tau (0 or None = off); shell-level variants only
    boys       "table" | "series"
    onee       "vector" | "naive"  one-electron integrals
    jk         "blas" | "einsum" | "loop"
    linalg     "lapack" | "naive"  matmul/eigh/solve via NumPy/LAPACK or the stage-0 loops
After eri() the attribute ``stats`` holds the work counters (quartets, contracted integrals, ...).
"""
import warnings

import numpy as np
import scipy.linalg

from ..naive import NaiveBackend
from . import eri as erimod
from . import onee


class NumpyBackend(NaiveBackend):
    name = "numpy"

    def __init__(self, eri="vector", screening=1e-12, boys="table", onee="vector", jk="blas",
                 linalg="lapack"):
        if eri not in ("full", "sym8", "shell", "vector"):
            raise ValueError(f"unknown eri variant {eri!r}")
        if screening and eri in ("full", "sym8"):
            raise ValueError("Schwarz screening needs the shell-quartet variants (eri='shell' or 'vector')")
        if boys not in ("table", "series"):
            raise ValueError(boys)
        self.opts = dict(eri=eri, screening=screening or 0.0, boys=boys, onee=onee, jk=jk, linalg=linalg)
        self.stats = erimod.new_stats()
        self._ST = None

    def __repr__(self):
        return "NumpyBackend(" + ", ".join(f"{k}={v!r}" for k, v in self.opts.items()) + ")"

    # ---- one-electron integrals
    def _st(self, funcs):
        key = id(funcs)
        if self._ST is None or self._ST[0] != key:
            self._ST = (key, onee.overlap_kinetic(funcs))
        return self._ST[1]

    def overlap(self, funcs):
        if self.opts["onee"] == "naive":
            return super().overlap(funcs)
        return self._st(funcs)[0]

    def kinetic(self, funcs):
        if self.opts["onee"] == "naive":
            return super().kinetic(funcs)
        return self._st(funcs)[1]

    def nuclear(self, funcs, mol):
        if self.opts["onee"] == "naive":
            return super().nuclear(funcs, mol)
        return onee.nuclear(funcs, mol)

    # ---- two-electron integrals
    def eri(self, funcs):
        self.stats = erimod.new_stats()
        o = self.opts
        if o["eri"] == "full":
            return erimod.eri_full(funcs, self.stats)
        if o["eri"] == "sym8":
            return erimod.eri_sym8(funcs, self.stats)
        return erimod.eri_shells(funcs, self.stats, mode=o["eri"], screening=o["screening"], boys=o["boys"])

    # ---- linear algebra
    def matmul(self, A, B):
        if self.opts["linalg"] == "naive":
            return super().matmul(A, B)
        return np.asarray(A) @ np.asarray(B)

    def transpose(self, A):
        if self.opts["linalg"] == "naive":
            return super().transpose(A)
        return np.asarray(A).T

    def eigh(self, A):
        if self.opts["linalg"] == "naive":
            return super().eigh(A)
        return np.linalg.eigh(A)

    def solve(self, A, b):
        """LU with partial pivoting (LAPACK getrf); same singularity test as stage 0."""
        if self.opts["linalg"] == "naive":
            return super().solve(A, b)
        A = np.asarray(A, dtype=float)
        scale = np.abs(A).max() or 1.0
        with warnings.catch_warnings():          # singularity is handled below, as in stage 0
            warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
            lu, piv = scipy.linalg.lu_factor(A, check_finite=False)
        if np.abs(np.diag(lu)).min() < 1e-14 * scale:
            raise ValueError("singular matrix")
        return list(scipy.linalg.lu_solve((lu, piv), np.asarray(b, dtype=float)))

    # ---- Fock build
    def jk(self, G, D):
        mode = self.opts["jk"]
        if mode == "loop":
            return super().jk(G, D)
        if mode == "einsum":
            return np.einsum("ijkl,kl->ij", G, D), np.einsum("ikjl,kl->ij", G, D)
        n = D.shape[0]
        J = (G.reshape(n * n, n * n) @ D.reshape(n * n)).reshape(n, n)     # one BLAS gemv
        # K_ij = sum_k sum_l G[i,k,j,l] D[k,l]: n^2 contiguous (j,l) blocks times row D[k]
        K = np.matmul(G, D[None, :, :, None]).sum(axis=1)[:, :, 0]         # batched gemv, no copy of G
        return J, K
