"""Two-electron repulsion integrals: a ladder of algorithmic tricks.

All variants return the full (N, N, N, N) tensor (the J/K code downstream is
unchanged); they differ in *how much work* is done to fill it.

  eri="full"    stage 0: every one of the N^4 contracted integrals, one at a time
  eri="sym8"    (a) 8-fold permutational symmetry (ij|kl)=(ji|kl)=(ij|lk)=(kl|ij)=...
                    -> only i>=j, k>=l, ij>=kl are computed (~N^4/8)
  eri="shell"   (b) loop over *shell quartets*; per-shell-pair data (exponent sums,
                    product centres, Hermite E tables x contraction coefficients) are
                    built once.  One Boys/R evaluation per primitive quartet serves all
                    Cartesian components of the quartet (81 for (pp|pp), 1296 for (dd|dd)).
                    Python loop over primitive pairs, small NumPy products inside.
  eri="vector"  (e) same, but the primitive-pair loops are NumPy array operations:
                    Boys function, R recursion and the E.R.E contraction (two
                    tensordot/BLAS calls) act on all P x Q primitive quartets at once.

Orthogonal switches:
  screening=tau (c) Schwarz: skip a shell quartet if Q_ab Q_cd < tau, Q_ab = max sqrt|(ab|ab)|
  boys="table"  (d) tabulated Boys function + Taylor + downward recursion (vs stage-0 series)
"""
import math

import numpy as np

from ..naive.integrals import eri_contracted
from . import boys as boysmod
from .hermite import hermite_R, hermite_R_scalar, sum_index
from .shells import group_shells, make_pair

TWO_PI_25 = 2.0 * math.pi ** 2.5


def new_stats():
    return {"quartets": 0, "contracted": 0, "primitive_quartets": 0,
            "schwarz_quartets": 0, "skipped_quartets": 0, "skipped_contracted": 0}


# ---------------------------------------------------------------- function-level variants
def eri_full(funcs, stats):
    n = len(funcs)
    G = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    G[i, j, k, l] = eri_contracted(funcs[i], funcs[j], funcs[k], funcs[l])
    stats["contracted"] += n ** 4
    return G


def eri_sym8(funcs, stats):
    n = len(funcs)
    G = np.zeros((n, n, n, n))
    cnt = 0
    for i in range(n):
        for j in range(i + 1):
            ij = i * (i + 1) // 2 + j
            for k in range(n):
                for l in range(k + 1):
                    if k * (k + 1) // 2 + l > ij:
                        continue
                    v = eri_contracted(funcs[i], funcs[j], funcs[k], funcs[l])
                    cnt += 1
                    G[i, j, k, l] = G[j, i, k, l] = G[i, j, l, k] = G[j, i, l, k] = v
                    G[k, l, i, j] = G[l, k, i, j] = G[k, l, j, i] = G[l, k, j, i] = v
    stats["contracted"] += cnt
    return G


# ---------------------------------------------------------------- shell-quartet kernels
def quartet_vector(bra, ket, boys_all):
    """(ab|cd) block for one shell quartet, vectorized over primitive quartets.

    Returns (na*nb, nc*nd).  Shapes: P, Q = number of primitive pairs; H, K = Hermite sets.
    """
    p, q = bra.p, ket.p
    ppq = p[:, None] + q[None, :]
    alpha = (p[:, None] * q[None, :] / ppq).ravel()
    X = (bra.P[:, :, None] - ket.P[:, None, :]).reshape(3, -1)          # (3, PQ)
    T = alpha * (X[0] * X[0] + X[1] * X[1] + X[2] * X[2])
    L = bra.L + ket.L
    F = boys_all(L, T)
    R = hermite_R(L, alpha, X, F)                                       # (nH(L), PQ)
    pref = (TWO_PI_25 / (p[:, None] * q[None, :] * np.sqrt(ppq))).ravel()
    M = R[sum_index(bra.L, ket.L)] * pref                               # (H, K, PQ)
    M = M.reshape(M.shape[0], M.shape[1], p.size, q.size)
    W = np.tensordot(bra.E, M, axes=([1, 2], [0, 2]))                   # (nab, K, Q)
    return np.tensordot(W, ket.Esigned, axes=([1, 2], [1, 2]))          # (nab, ncd)


def quartet_scalar(bra, ket, boys_scalar):
    """Same block, but an explicit Python loop over primitive pairs (one R tensor each)."""
    L = bra.L + ket.L
    idx = sum_index(bra.L, ket.L)
    out = np.zeros((bra.E.shape[0], ket.E.shape[0]))
    Pc, Qc = bra.P.T.tolist(), ket.P.T.tolist()
    for i, p in enumerate(bra.p.tolist()):
        Pi = Pc[i]
        Ei = bra.E[:, :, i]
        for j, q in enumerate(ket.p.tolist()):
            Qj = Qc[j]
            alpha = p * q / (p + q)
            X = (Pi[0] - Qj[0], Pi[1] - Qj[1], Pi[2] - Qj[2])
            T = alpha * (X[0] * X[0] + X[1] * X[1] + X[2] * X[2])
            F = boys_scalar(L, T)
            R = np.array(hermite_R_scalar(L, alpha, X, F))
            Rm = R[idx] * (TWO_PI_25 / (p * q * math.sqrt(p + q)))
            out += Ei @ Rm @ ket.Esigned[:, :, j].T
    return out


def _boys_scalar_series(L, T):
    from ..naive.boys import boys
    return [boys(n, T) for n in range(L + 1)]


def eri_shells(funcs, stats, mode="vector", screening=0.0, boys="table"):
    shells = group_shells(funcs)
    n = len(funcs)
    G = np.zeros((n, n, n, n))
    if mode == "vector":
        kernel = quartet_vector
        bf = boysmod.boys_all if boys == "table" else boysmod.boys_all_series
    elif mode == "shell":
        kernel = quartet_scalar
        bf = boysmod.boys_all_scalar if boys == "table" else _boys_scalar_series
    else:
        raise ValueError(mode)

    # (b) shell-pair data, built once
    pairs = [make_pair(shells[a], shells[b]) for a in range(len(shells)) for b in range(a + 1)]

    # (c) Schwarz factors from the diagonal quartets (ab|ab)
    Q = np.full(len(pairs), np.inf)
    diag = {}
    if screening:
        for ab, pr in enumerate(pairs):
            blk = kernel(pr, pr, bf)
            diag[ab] = blk
            Q[ab] = math.sqrt(np.abs(np.diag(blk)).max())
            stats["schwarz_quartets"] += 1

    for ab, bra in enumerate(pairs):
        a, b = bra.a, bra.b
        for cd in range(ab + 1):
            ket = pairs[cd]
            c, d = ket.a, ket.b
            ncon = a.n * b.n * c.n * d.n
            if screening and Q[ab] * Q[cd] < screening:
                stats["skipped_quartets"] += 1
                stats["skipped_contracted"] += ncon
                continue
            if cd == ab and ab in diag:
                blk = diag[ab]                       # already computed for the Schwarz factor
            else:
                blk = kernel(bra, ket, bf)
                stats["quartets"] += 1
                stats["contracted"] += ncon
                stats["primitive_quartets"] += bra.p.size * ket.p.size
            blk = blk.reshape(a.n, b.n, c.n, d.n)
            sa = slice(a.start, a.start + a.n)
            sb = slice(b.start, b.start + b.n)
            sc = slice(c.start, c.start + c.n)
            sd = slice(d.start, d.start + d.n)
            G[sa, sb, sc, sd] = blk
            G[sb, sa, sc, sd] = blk.transpose(1, 0, 2, 3)
            G[sa, sb, sd, sc] = blk.transpose(0, 1, 3, 2)
            G[sb, sa, sd, sc] = blk.transpose(1, 0, 3, 2)
            G[sc, sd, sa, sb] = blk.transpose(2, 3, 0, 1)
            G[sd, sc, sa, sb] = blk.transpose(3, 2, 0, 1)
            G[sc, sd, sb, sa] = blk.transpose(2, 3, 1, 0)
            G[sd, sc, sb, sa] = blk.transpose(3, 2, 1, 0)
    if screening:   # diagonal quartets were evaluated in the Schwarz pass
        for ab, pr in enumerate(pairs):
            stats["quartets"] += 1
            stats["contracted"] += pr.a.n * pr.b.n * pr.a.n * pr.b.n
            stats["primitive_quartets"] += pr.p.size ** 2
    return G
