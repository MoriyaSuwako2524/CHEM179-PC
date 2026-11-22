"""Dense linear algebra written as explicit loops."""
import math

import numpy as np


def matmul(A, B):
    """C = A B with the textbook i-j-k triple loop."""
    n, m = A.shape
    m2, p = B.shape
    assert m == m2
    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def transpose(A):
    n, m = A.shape
    T = np.zeros((m, n))
    for i in range(n):
        for j in range(m):
            T[j, i] = A[i, j]
    return T


def jacobi_eigh(A, tol=1e-12, max_sweeps=100):
    """Cyclic Jacobi diagonalization of a real symmetric matrix.

    Returns eigenvalues in ascending order and eigenvectors as columns.
    Each rotation zeroes one off-diagonal element; a sweep visits all pairs.
    """
    n = A.shape[0]
    a = A.astype(float).copy()
    V = np.zeros((n, n))
    for i in range(n):
        V[i, i] = 1.0
    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += a[p, q] * a[p, q]
        if math.sqrt(off) < tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p, q]) < 1e-18 * (abs(a[p, p]) + abs(a[q, q])) + 1e-300:
                    a[p, q] = a[q, p] = 0.0  # already negligible: skip the rotation
                    continue
                theta = (a[q, q] - a[p, p]) / (2 * a[p, q])
                if abs(theta) > 1e150:  # theta^2 would overflow; t ~ 1/(2 theta)
                    t = 1.0 / (2 * theta)
                else:
                    t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):  # rotate columns p, q
                    akp, akq = a[k, p], a[k, q]
                    a[k, p] = c * akp - s * akq
                    a[k, q] = s * akp + c * akq
                for k in range(n):  # rotate rows p, q
                    apk, aqk = a[p, k], a[q, k]
                    a[p, k] = c * apk - s * aqk
                    a[q, k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k, p], V[k, q]
                    V[k, p] = c * vkp - s * vkq
                    V[k, q] = s * vkp + c * vkq
    w = [a[i, i] for i in range(n)]
    order = sorted(range(n), key=lambda i: w[i])
    W = np.array([w[i] for i in order])
    U = np.zeros((n, n))
    for jnew, jold in enumerate(order):
        for k in range(n):
            U[k, jnew] = V[k, jold]
    return W, U


def solve(A, b):
    """Gaussian elimination with partial pivoting (used by DIIS)."""
    n = len(b)
    M = [[float(A[i][j]) for j in range(n)] + [float(b[i])] for i in range(n)]
    scale = max(abs(M[i][j]) for i in range(n) for j in range(n)) or 1.0
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14 * scale:
            raise ValueError("singular matrix")
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))) / M[r][r]
    return x
