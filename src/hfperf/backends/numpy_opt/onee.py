"""One-electron integrals (S, T, V) over shell pairs, vectorized over primitive pairs."""
import math

import numpy as np

from . import boys as boysmod
from .hermite import hermite_E_table, hermite_R
from .shells import group_shells, make_pair


def _shell_pairs(funcs):
    shells = group_shells(funcs)
    for a in range(len(shells)):
        for b in range(a + 1):
            yield shells[a], shells[b]


def _overlap_kinetic_block(a, b):
    ea = np.repeat(a.exps, b.exps.size)
    eb = np.tile(b.exps, a.exps.size)
    p = ea + eb
    AB = a.center - b.center
    tabs = [hermite_E_table(a.l, b.l + 2, AB[k], ea, eb)[:, :, 0] for k in range(3)]  # (la+1, lb+3, P)
    la = np.repeat(a.lmn, b.n, axis=0)
    lb = np.tile(b.lmn, (a.n, 1))
    S1 = [tabs[k][la[:, k], lb[:, k]] for k in range(3)]                    # (nab, P) each
    T1 = []
    for k in range(3):
        j = lb[:, k]
        Sp2 = tabs[k][la[:, k], j + 2]
        Sm2 = np.where((j >= 2)[:, None], tabs[k][la[:, k], np.maximum(j - 2, 0)], 0.0)
        T1.append(eb * (2 * j[:, None] + 1) * S1[k] - 2 * eb * eb * Sp2 - 0.5 * (j * (j - 1))[:, None] * Sm2)
    pre = (math.pi / p) ** 1.5
    cc = (a.coefs[:, None, :, None] * b.coefs[None, :, None, :]).reshape(a.n * b.n, -1)
    S = (S1[0] * S1[1] * S1[2] * pre * cc).sum(1)
    T = ((T1[0] * S1[1] * S1[2] + S1[0] * T1[1] * S1[2] + S1[0] * S1[1] * T1[2]) * pre * cc).sum(1)
    return S.reshape(a.n, b.n), T.reshape(a.n, b.n)


def overlap_kinetic(funcs):
    n = len(funcs)
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    for a, b in _shell_pairs(funcs):
        s, t = _overlap_kinetic_block(a, b)
        sa, sb = slice(a.start, a.start + a.n), slice(b.start, b.start + b.n)
        S[sa, sb] = s; S[sb, sa] = s.T
        T[sa, sb] = t; T[sb, sa] = t.T
    return S, T


def nuclear(funcs, mol):
    n = len(funcs)
    V = np.zeros((n, n))
    C = np.array(mol.coords, float)                         # (Nat, 3)
    Z = np.array(mol.charges, float)
    for a, b in _shell_pairs(funcs):
        pr = make_pair(a, b)
        X = (pr.P[:, :, None] - C.T[:, None, :]).reshape(3, -1)   # (3, P*Nat)
        alpha = np.repeat(pr.p, C.shape[0])
        T = alpha * (X * X).sum(0)
        R = hermite_R(pr.L, alpha, X, boysmod.boys_all(pr.L, T))  # (H, P*Nat)
        Rz = R.reshape(R.shape[0], pr.p.size, C.shape[0]) @ Z     # (H, P)
        Rz *= -2.0 * math.pi / pr.p
        blk = np.einsum("ahp,hp->a", pr.E, Rz).reshape(a.n, b.n)
        sa, sb = slice(a.start, a.start + a.n), slice(b.start, b.start + b.n)
        V[sa, sb] = blk; V[sb, sa] = blk.T
    return V
