"""Coulomb (J) and exchange (K) matrices from the full ERI tensor."""
import numpy as np


def jk(G, D):
    """J_ij = sum_kl (ij|kl) D_kl ;  K_ij = sum_kl (ik|jl) D_kl."""
    n = D.shape[0]
    J = np.zeros((n, n))
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sj = 0.0
            sk = 0.0
            for k in range(n):
                for l in range(n):
                    sj += G[i, j, k, l] * D[k, l]
                    sk += G[i, k, j, l] * D[k, l]
            J[i, j] = sj
            K[i, j] = sk
    return J, K
