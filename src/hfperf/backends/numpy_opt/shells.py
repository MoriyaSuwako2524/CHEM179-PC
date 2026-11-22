"""Shell and shell-pair data, precomputed once and reused by every integral.

Stage 0 works function-by-function: the 6 Cartesian d functions of one shell
recompute the same Gaussian-product centers, exponentials and Hermite E
coefficients six times (and 36x for a d-d pair).  Here everything that only
depends on a *pair of shells* is computed once and stored in NumPy arrays.
"""
from dataclasses import dataclass

import numpy as np

from .hermite import hermite_E_table, hermite_list


@dataclass
class ShellData:
    index: int
    start: int            # first basis function of this shell
    l: int
    center: np.ndarray    # (3,)
    exps: np.ndarray      # (nprim,)
    coefs: np.ndarray     # (ncomp, nprim): per-component (normalized) contraction coefficients
    lmn: np.ndarray       # (ncomp, 3)

    @property
    def n(self):
        return len(self.lmn)


def group_shells(funcs) -> list:
    """Group consecutive basis functions with the same .shell index."""
    shells = []
    i = 0
    while i < len(funcs):
        j = i
        while j < len(funcs) and funcs[j].shell == funcs[i].shell:
            j += 1
        fs = funcs[i:j]
        shells.append(ShellData(len(shells), i, sum(fs[0].lmn), np.array(fs[0].center, float),
                                np.array(fs[0].exps, float), np.array([f.coefs for f in fs], float),
                                np.array([f.lmn for f in fs], int)))
        i = j
    return shells


@dataclass
class PairData:
    a: ShellData
    b: ShellData
    L: int                # la + lb
    p: np.ndarray         # (P,) total exponents of primitive pairs
    P: np.ndarray         # (3, P) product centers
    E: np.ndarray         # (na*nb, H, P)  Hermite coefficients x contraction coefficients
    Esigned: np.ndarray   # same times (-1)^(t+u+v), for use on the ket side


def make_pair(a: ShellData, b: ShellData, extra_j: int = 0) -> PairData:
    ea = np.repeat(a.exps, b.exps.size)
    eb = np.tile(b.exps, a.exps.size)
    p = ea + eb
    Pc = (ea[None, :] * a.center[:, None] + eb[None, :] * b.center[:, None]) / p
    AB = a.center - b.center
    tabs = [hermite_E_table(a.l, b.l + extra_j, AB[k], ea, eb) for k in range(3)]
    L = a.l + b.l
    H = hermite_list(L)
    ht = np.array(H, dtype=int)                                  # (nH, 3)
    # component pairs, in basis-function order (i over a, j over b)
    la = np.repeat(a.lmn, b.n, axis=0)                           # (nab, 3)
    lb = np.tile(b.lmn, (a.n, 1))
    E = (tabs[0][la[:, 0][:, None], lb[:, 0][:, None], ht[None, :, 0]]
         * tabs[1][la[:, 1][:, None], lb[:, 1][:, None], ht[None, :, 1]]
         * tabs[2][la[:, 2][:, None], lb[:, 2][:, None], ht[None, :, 2]])   # (nab, nH, P)
    cc = (a.coefs[:, None, :, None] * b.coefs[None, :, None, :]).reshape(a.n * b.n, -1)  # (nab, P)
    E = E * cc[:, None, :]
    sign = (-1.0) ** ht.sum(axis=1)
    return PairData(a, b, L, p, Pc, np.ascontiguousarray(E), np.ascontiguousarray(E * sign[None, :, None]))
