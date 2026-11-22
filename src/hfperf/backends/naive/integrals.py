"""McMurchie-Davidson one- and two-electron integrals, written as plainly
as possible: recursive functions, explicit loops, no symmetry, no caching.

References: McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978);
Helgaker, Jorgensen & Olsen, "Molecular Electronic-Structure Theory", ch. 9.
"""
import math

import numpy as np

from .boys import boys


def hermite_E(i, j, t, Qx, a, b):
    """Expansion coefficient E^{ij}_t of a 1D Gaussian product in Hermite Gaussians.

    i, j : angular momenta on centers A and B;  Qx = A_x - B_x
    a, b : exponents
    """
    p = a + b
    q = a * b / p
    if t < 0 or t > i + j:
        return 0.0
    if i == j == t == 0:
        return math.exp(-q * Qx * Qx)
    if j == 0:  # decrement i
        return (1 / (2 * p) * hermite_E(i - 1, j, t - 1, Qx, a, b)
                - q * Qx / a * hermite_E(i - 1, j, t, Qx, a, b)
                + (t + 1) * hermite_E(i - 1, j, t + 1, Qx, a, b))
    return (1 / (2 * p) * hermite_E(i, j - 1, t - 1, Qx, a, b)  # decrement j
            + q * Qx / b * hermite_E(i, j - 1, t, Qx, a, b)
            + (t + 1) * hermite_E(i, j - 1, t + 1, Qx, a, b))


def hermite_R(t, u, v, n, p, PCx, PCy, PCz, RPC):
    """Hermite Coulomb integral R^n_{tuv} (recursive)."""
    T = p * RPC * RPC
    if t == u == v == 0:
        return (-2 * p) ** n * boys(n, T)
    if t == u == 0:
        val = 0.0
        if v > 1:
            val += (v - 1) * hermite_R(t, u, v - 2, n + 1, p, PCx, PCy, PCz, RPC)
        return val + PCz * hermite_R(t, u, v - 1, n + 1, p, PCx, PCy, PCz, RPC)
    if t == 0:
        val = 0.0
        if u > 1:
            val += (u - 1) * hermite_R(t, u - 2, v, n + 1, p, PCx, PCy, PCz, RPC)
        return val + PCy * hermite_R(t, u - 1, v, n + 1, p, PCx, PCy, PCz, RPC)
    val = 0.0
    if t > 1:
        val += (t - 1) * hermite_R(t - 2, u, v, n + 1, p, PCx, PCy, PCz, RPC)
    return val + PCx * hermite_R(t - 1, u, v, n + 1, p, PCx, PCy, PCz, RPC)


def _gaussian_product_center(a, A, b, B):
    return tuple((a * A[k] + b * B[k]) / (a + b) for k in range(3))


# ---------- primitive integrals ----------

def overlap_prim(a, lmn1, A, b, lmn2, B):
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    S1 = hermite_E(l1, l2, 0, A[0] - B[0], a, b)
    S2 = hermite_E(m1, m2, 0, A[1] - B[1], a, b)
    S3 = hermite_E(n1, n2, 0, A[2] - B[2], a, b)
    return S1 * S2 * S3 * (math.pi / (a + b)) ** 1.5


def kinetic_prim(a, lmn1, A, b, lmn2, B):
    l2, m2, n2 = lmn2
    term0 = b * (2 * (l2 + m2 + n2) + 3) * overlap_prim(a, lmn1, A, b, lmn2, B)
    term1 = -2 * b * b * (overlap_prim(a, lmn1, A, b, (l2 + 2, m2, n2), B)
                          + overlap_prim(a, lmn1, A, b, (l2, m2 + 2, n2), B)
                          + overlap_prim(a, lmn1, A, b, (l2, m2, n2 + 2), B))
    term2 = -0.5 * (l2 * (l2 - 1) * overlap_prim(a, lmn1, A, b, (l2 - 2, m2, n2), B)
                    + m2 * (m2 - 1) * overlap_prim(a, lmn1, A, b, (l2, m2 - 2, n2), B)
                    + n2 * (n2 - 1) * overlap_prim(a, lmn1, A, b, (l2, m2, n2 - 2), B))
    return term0 + term1 + term2


def nuclear_prim(a, lmn1, A, b, lmn2, B, C):
    """<a| 1/|r - C| |b>  (without the -Z factor)."""
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    p = a + b
    P = _gaussian_product_center(a, A, b, B)
    RPC = math.dist(P, C)
    val = 0.0
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                val += (hermite_E(l1, l2, t, A[0] - B[0], a, b)
                        * hermite_E(m1, m2, u, A[1] - B[1], a, b)
                        * hermite_E(n1, n2, v, A[2] - B[2], a, b)
                        * hermite_R(t, u, v, 0, p, P[0] - C[0], P[1] - C[1], P[2] - C[2], RPC))
    return 2 * math.pi / p * val


def eri_prim(a, lmn1, A, b, lmn2, B, c, lmn3, C, d, lmn4, D):
    """(ab|cd) over primitive Cartesian Gaussians (chemists' notation)."""
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    l3, m3, n3 = lmn3
    l4, m4, n4 = lmn4
    p = a + b
    q = c + d
    alpha = p * q / (p + q)
    P = _gaussian_product_center(a, A, b, B)
    Q = _gaussian_product_center(c, C, d, D)
    RPQ = math.dist(P, Q)
    val = 0.0
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                for tau in range(l3 + l4 + 1):
                    for nu in range(m3 + m4 + 1):
                        for phi in range(n3 + n4 + 1):
                            val += (hermite_E(l1, l2, t, A[0] - B[0], a, b)
                                    * hermite_E(m1, m2, u, A[1] - B[1], a, b)
                                    * hermite_E(n1, n2, v, A[2] - B[2], a, b)
                                    * hermite_E(l3, l4, tau, C[0] - D[0], c, d)
                                    * hermite_E(m3, m4, nu, C[1] - D[1], c, d)
                                    * hermite_E(n3, n4, phi, C[2] - D[2], c, d)
                                    * (-1) ** (tau + nu + phi)
                                    * hermite_R(t + tau, u + nu, v + phi, 0, alpha,
                                                P[0] - Q[0], P[1] - Q[1], P[2] - Q[2], RPQ))
    return 2 * math.pi ** 2.5 / (p * q * math.sqrt(p + q)) * val


# ---------- contracted integrals over basis functions ----------

def _contract2(prim, f, g, *extra):
    s = 0.0
    for ca, a in zip(f.coefs, f.exps):
        for cb, b in zip(g.coefs, g.exps):
            s += ca * cb * prim(a, f.lmn, f.center, b, g.lmn, g.center, *extra)
    return s


def overlap_matrix(funcs):
    n = len(funcs)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = _contract2(overlap_prim, funcs[i], funcs[j])
    return S


def kinetic_matrix(funcs):
    n = len(funcs)
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            T[i, j] = _contract2(kinetic_prim, funcs[i], funcs[j])
    return T


def nuclear_matrix(funcs, mol):
    n = len(funcs)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for Z, C in zip(mol.charges, mol.coords):
                V[i, j] += -Z * _contract2(nuclear_prim, funcs[i], funcs[j], C)
    return V


def eri_contracted(f1, f2, f3, f4):
    s = 0.0
    for c1, a in zip(f1.coefs, f1.exps):
        for c2, b in zip(f2.coefs, f2.exps):
            for c3, c in zip(f3.coefs, f3.exps):
                for c4, d in zip(f4.coefs, f4.exps):
                    s += c1 * c2 * c3 * c4 * eri_prim(a, f1.lmn, f1.center, b, f2.lmn, f2.center,
                                                      c, f3.lmn, f3.center, d, f4.lmn, f4.center)
    return s


def eri_tensor(funcs):
    """All N^4 two-electron integrals, one at a time, no symmetry used."""
    n = len(funcs)
    G = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    G[i, j, k, l] = eri_contracted(funcs[i], funcs[j], funcs[k], funcs[l])
    return G
