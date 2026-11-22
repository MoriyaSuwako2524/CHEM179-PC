"""Matmul in Python: does loop order / blocking matter when the interpreter is the bottleneck?

    OPENBLAS_NUM_THREADS=1 python py_ladder.py --sizes 64 128 192 256 --out py_ladder.csv

Every variant computes C = A B for random n x n matrices and is checked against A @ B.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TidyCSV, best_time


def ijk_numpy(A, B):                       # element indexing on NumPy arrays (stage 0 style)
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def ijk_list(A, B):                        # same loops over nested Python lists
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ai, Ci = A[i], C[i]
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += Ai[k] * B[k][j]
            Ci[j] = s
    return C


def ikj_list(A, B):                        # cache-friendly order: walk rows of B and C
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ai, Ci = A[i], C[i]
        for k in range(n):
            a, Bk = Ai[k], B[k]
            for j in range(n):
                Ci[j] += a * Bk[j]
    return C


def tB_list(A, B):                         # transpose B first, then row . row dot products
    n = len(A)
    Bt = [list(col) for col in zip(*B)]
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ai, Ci = A[i], C[i]
        for j in range(n):
            Bj = Bt[j]
            s = 0.0
            for k in range(n):
                s += Ai[k] * Bj[k]
            Ci[j] = s
    return C


def blocked_list(A, B, tile=32):           # cache blocking (ikj inside tiles)
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for ii in range(0, n, tile):
        for kk in range(0, n, tile):
            for jj in range(0, n, tile):
                for i in range(ii, min(ii + tile, n)):
                    Ai, Ci = A[i], C[i]
                    for k in range(kk, min(kk + tile, n)):
                        a, Bk = Ai[k], B[k]
                        for j in range(jj, min(jj + tile, n)):
                            Ci[j] += a * Bk[j]
    return C


def row_vec(A, B):                         # one BLAS gemv per row: C[i,:] = A[i,:] @ B
    C = np.empty_like(A)
    for i in range(A.shape[0]):
        C[i, :] = A[i, :] @ B
    return C


def col_vec_strided(A, B):                 # one gemv per column on a strided view B[:, j]
    C = np.empty_like(A)
    for j in range(A.shape[1]):
        C[:, j] = A @ B[:, j]
    return C


def numpy_blas(A, B):                      # dgemm
    return A @ B


VARIANTS = {
    "py_ijk_numpy": (ijk_numpy, "np"),
    "py_ijk_list": (ijk_list, "list"),
    "py_ikj_list": (ikj_list, "list"),
    "py_tB_list": (tB_list, "list"),
    "py_blocked32_list": (blocked_list, "list"),
    "np_row_vec": (row_vec, "np"),
    "np_col_vec_strided": (col_vec_strided, "np"),
    "np_matmul_blas": (numpy_blas, "np"),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 192, 256])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out", default="py_ladder.csv")
    a = ap.parse_args(argv)
    out = TidyCSV(a.out)
    rng = np.random.default_rng(179)
    for N in a.sizes:
        A, B = rng.random((N, N)), rng.random((N, N))
        ref = A @ B
        for name in a.variants:
            f, kind = VARIANTS[name]
            args = (A, B) if kind == "np" else (A.tolist(), B.tolist())
            C = np.asarray(f(*args))
            assert np.allclose(C, ref, rtol=1e-12, atol=1e-10), name
            fast = name.startswith("np_")
            reps = 7 if fast else (3 if N <= 128 else 1)
            out.add(name, N, best_time(lambda: f(*args), reps))


if __name__ == "__main__":
    main()
