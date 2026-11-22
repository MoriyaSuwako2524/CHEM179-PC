"""Boys function F_n(T) by tabulation + Taylor expansion + downward recursion.

Stage 0 calls a power series once per (n, T).  Here we

  1. tabulate F_n(T_k) once, on a grid T_k = k * DT (k = 0..KMAX), for n = 0..NTAB;
  2. for any T < TMAX evaluate only the *highest* order needed, F_L(T), by a
     6-term Taylor expansion around the nearest grid point:
        F_L(T) = sum_k F_{L+k}(T_k) (-dT)^k / k!        (dF_n/dT = -F_{n+1})
  3. get all lower orders by the (numerically stable) downward recursion
        F_{n-1}(T) = (2 T F_n(T) + exp(-T)) / (2n - 1);
  4. for T >= TMAX use F_0 = sqrt(pi/T)/2 (erf = 1 to machine precision) and the
     upward recursion, which is stable for large T.

Both a scalar (Python float) and a vectorized (NumPy array) version are given.
"""
import math

import numpy as np

DT = 0.1
TMAX = 36.0
KMAX = int(round(TMAX / DT))
NTAYLOR = 6
NMAX = 16                      # highest order supported (ERI (ff|ff) needs 12)
NTAB = NMAX + NTAYLOR + 1


def _build_table():
    """F_n(T_k) for n = 0..NTAB on the grid: series for the top order, then downward recursion."""
    T = np.arange(KMAX + 1) * DT
    N = NTAB
    term = np.full_like(T, 1.0 / (2 * N + 1))
    total = term.copy()
    for k in range(1, 400):
        term = term * 2.0 * T / (2 * N + 2 * k + 1)
        total += term
        if np.all(term <= 1e-17 * total):
            break
    e = np.exp(-T)
    tab = np.empty((N + 1, T.size))
    tab[N] = e * total
    for n in range(N, 0, -1):
        tab[n - 1] = (2 * T * tab[n] + e) / (2 * n - 1)
    return tab


TABLE = _build_table()                      # shape (NTAB+1, KMAX+1)
TABLE_T = TABLE.T.copy()                    # row k = all orders at T_k (contiguous for scalar lookups)
_INV_FACT = [1.0 / math.factorial(k) for k in range(NTAYLOR + 1)]


def boys_all_scalar(L: int, T: float) -> list:
    """[F_0(T), ..., F_L(T)] for one T (Python floats)."""
    out = [0.0] * (L + 1)
    if T < TMAX:
        k = int(T / DT + 0.5)
        row = TABLE_T[k]
        d = k * DT - T                      # = -(T - T_k)
        f = 0.0
        dk = 1.0
        for j in range(NTAYLOR + 1):
            f += row[L + j] * dk * _INV_FACT[j]
            dk *= d
        out[L] = f
        e = math.exp(-T)
        for n in range(L, 0, -1):
            out[n - 1] = (2.0 * T * out[n] + e) / (2 * n - 1)
    else:
        f = 0.5 * math.sqrt(math.pi / T)
        e = math.exp(-T)
        out[0] = f
        for n in range(1, L + 1):
            f = ((2 * n - 1) * f - e) / (2.0 * T)
            out[n] = f
    return out


def boys_all(L: int, T: np.ndarray) -> np.ndarray:
    """F_n(T) for n = 0..L and an array of T; returns shape (L+1,) + T.shape."""
    T = np.asarray(T, dtype=np.float64)
    out = np.empty((L + 1,) + T.shape)
    small = T < TMAX
    if small.all():
        Ts = T
        k = (Ts * (1.0 / DT) + 0.5).astype(np.intp)
        d = k * DT - Ts
        rows = TABLE[L:L + NTAYLOR + 1][:, k]           # (NTAYLOR+1, ...)
        f = rows[NTAYLOR] * _INV_FACT[NTAYLOR]
        for j in range(NTAYLOR - 1, -1, -1):              # Horner
            f = f * d + rows[j] * _INV_FACT[j]
        out[L] = f
        e = np.exp(-Ts)
        T2 = 2.0 * Ts
        for n in range(L, 0, -1):
            out[n - 1] = (T2 * out[n] + e) / (2 * n - 1)
        return out
    # mixed: handle the two regions separately
    out[:, small] = boys_all(L, T[small])
    Tl = T[~small]
    f = 0.5 * np.sqrt(np.pi / Tl)
    e = np.exp(-Tl)
    out[0, ~small] = f
    for n in range(1, L + 1):
        f = ((2 * n - 1) * f - e) / (2.0 * Tl)
        out[n, ~small] = f
    return out


def boys_all_series(L: int, T: np.ndarray) -> np.ndarray:
    """Same interface as boys_all, but calls the stage-0 series once per (n, T) - for the ladder."""
    from ..naive.boys import boys
    T = np.asarray(T, dtype=np.float64)
    flat = T.ravel()
    out = np.empty((L + 1, flat.size))
    for n in range(L + 1):
        out[n] = [boys(n, t) for t in flat]
    return out.reshape((L + 1,) + T.shape)
