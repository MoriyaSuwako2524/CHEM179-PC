"""McMurchie-Davidson building blocks, vectorized over primitive pairs.

Hermite index sets.  hermite_list(L) is every (t, u, v) with t + u + v <= L,
ordered by total degree, so hermite_list(L') is a *prefix* of hermite_list(L)
for L' < L.  That lets one array hold R_{tuv} for every degree at once.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def hermite_list(L: int) -> tuple:
    out = []
    for deg in range(L + 1):
        for t in range(deg, -1, -1):
            for u in range(deg - t, -1, -1):
                out.append((t, u, deg - t - u))
    return tuple(out)


@lru_cache(maxsize=None)
def hermite_index(L: int) -> dict:
    return {h: i for i, h in enumerate(hermite_list(L))}


def hermite_E_table(imax: int, jmax: int, X: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """E^{ij}_t for 0<=i<=imax, 0<=j<=jmax, 0<=t<=imax+jmax, for arrays of exponents a, b.

    X = A_x - B_x (one Cartesian direction).  Returns shape (imax+1, jmax+1, imax+jmax+2, P);
    the extra t slot is always zero so the recursion can read E_{t+1} without bounds checks.
    Recursions (Helgaker eq. 9.5.6-7):
       E^{i+1,j}_t = E^{ij}_{t-1}/(2p) + X_PA E^{ij}_t + (t+1) E^{ij}_{t+1}
       E^{i,j+1}_t = E^{ij}_{t-1}/(2p) + X_PB E^{ij}_t + (t+1) E^{ij}_{t+1}
    """
    p = a + b
    XPA = -b / p * X
    XPB = a / p * X
    oo2p = 0.5 / p
    tmax = imax + jmax
    E = np.zeros((imax + 1, jmax + 1, tmax + 2, a.size))
    E[0, 0, 0] = np.exp(-(a * b / p) * X * X)
    t1 = np.arange(1, tmax + 2)[:, None]
    for i in range(imax + 1):
        if i > 0:
            prev = E[i - 1, 0]
            E[i, 0, 0] = XPA * prev[0] + prev[1]
            E[i, 0, 1:tmax + 1] = oo2p * prev[0:tmax] + XPA * prev[1:tmax + 1] + t1[1:tmax + 1] * prev[2:tmax + 2]
        for j in range(1, jmax + 1):
            prev = E[i, j - 1]
            E[i, j, 0] = XPB * prev[0] + prev[1]
            E[i, j, 1:tmax + 1] = oo2p * prev[0:tmax] + XPB * prev[1:tmax + 1] + t1[1:tmax + 1] * prev[2:tmax + 2]
    return E


@lru_cache(maxsize=None)
def r_plan(k: int):
    """Recursion plan for one level of R^n_{tuv}: entries 1.. of hermite_list(k).

    R^n_{tuv} = X_dir R^{n+1}_{tuv - e_dir} + (m - 1) R^{n+1}_{tuv - 2 e_dir}   (m = t, u or v)
    Returns arrays (dir, idx1, idx2, coef) indexing into hermite_list(k-1).
    """
    lst = hermite_list(k)
    idx = hermite_index(max(k - 1, 0))
    dirs, i1, i2, c = [], [], [], []
    for h in lst[1:]:
        d = 0 if h[0] > 0 else (1 if h[1] > 0 else 2)
        m = h[d]
        h1 = list(h); h1[d] -= 1
        dirs.append(d); i1.append(idx[tuple(h1)])
        if m > 1:
            h2 = list(h); h2[d] -= 2
            i2.append(idx[tuple(h2)]); c.append(m - 1.0)
        else:
            i2.append(0); c.append(0.0)
    return np.array(dirs), np.array(i1), np.array(i2), np.array(c)[:, None]


def hermite_R(L: int, alpha: np.ndarray, X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """R^0_{tuv} for every (t,u,v) in hermite_list(L), vectorized over M points.

    alpha: (M,)  X: (3, M) = P - C (or P - Q)   F: (L+1, M) Boys values F_n(alpha |X|^2)
    Returns (len(hermite_list(L)), M).
    """
    m2a = -2.0 * alpha
    R = (m2a ** L * F[L])[None, :]                 # level n = L: only R_000
    for n in range(L - 1, -1, -1):
        k = L - n
        d, i1, i2, c = r_plan(k)
        new = np.empty((len(d) + 1, alpha.size))
        new[0] = m2a ** n * F[n]
        new[1:] = X[d] * R[i1] + c * R[i2]
        R = new
    return R


def hermite_R_scalar(L: int, alpha: float, X: tuple, F: list) -> list:
    """Scalar version of hermite_R for one primitive quartet (Python floats)."""
    m2a = -2.0 * alpha
    R = [m2a ** L * F[L]]
    for n in range(L - 1, -1, -1):
        d, i1, i2, c = r_plan_lists(L - n)
        new = [m2a ** n * F[n]]
        for dd, a1, a2, cc in zip(d, i1, i2, c):
            new.append(X[dd] * R[a1] + cc * R[a2])
        R = new
    return R


@lru_cache(maxsize=None)
def r_plan_lists(k: int):
    d, i1, i2, c = r_plan(k)
    return d.tolist(), i1.tolist(), i2.tolist(), c[:, 0].tolist()


@lru_cache(maxsize=None)
def sum_index(La: int, Lb: int) -> np.ndarray:
    """idx[h, g] = position of h + g in hermite_list(La+Lb), h in list(La), g in list(Lb)."""
    big = hermite_index(La + Lb)
    A, B = hermite_list(La), hermite_list(Lb)
    return np.array([[big[(h[0] + g[0], h[1] + g[1], h[2] + g[2])] for g in B] for h in A])
