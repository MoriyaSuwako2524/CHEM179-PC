"""Restricted Hartree-Fock driver.  It only calls backend methods, so the
same driver runs every stage of the course."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .basis import build_basis
from .backends import get_backend
from .molecule import Molecule
from .timing import timer


@dataclass
class SCFResult:
    energy: float
    converged: bool
    iterations: int
    mo_energies: np.ndarray
    mo_coeff: np.ndarray
    density: np.ndarray
    history: list[float] = field(default_factory=list)


def _trace_product(A, B):
    """tr(AB).  Bookkeeping, not a lesson kernel, so it uses NumPy in every stage:
    once the ERI is compiled, pure-Python O(N^2) loops here would dominate."""
    return float(np.einsum("ij,ji->", A, B))


def rhf(mol: Molecule, backend="naive", basis: str = "6-31gs", max_iter: int = 100,
        e_tol: float = 1e-10, d_tol: float = 1e-8, diis: bool = True, diis_space: int = 8,
        verbose: bool = True) -> SCFResult:
    be = get_backend(backend) if isinstance(backend, str) else backend
    _, funcs = build_basis(mol, basis)
    nbf = len(funcs)
    nocc = mol.n_electrons // 2
    if verbose:
        print(f"RHF/{basis}  {mol.name}: {nbf} basis functions, {nocc} doubly occupied, backend={be.name}")

    with timer("S,T,V"):
        S = be.overlap(funcs)
        H = be.kinetic(funcs) + be.nuclear(funcs, mol)
    with timer("ERI"):
        G = be.eri(funcs)
    e_nuc = mol.nuclear_repulsion()

    # Loewdin orthogonalization X = S^{-1/2}
    with timer("diag"):
        s, U = be.eigh(S)
    Xs = U / np.sqrt(s)[None, :]  # scale column j by s_j^{-1/2}
    with timer("matmul"):
        X = be.matmul(Xs, be.transpose(U))

    def solve_roothaan(F):
        with timer("matmul"):
            Fp = be.matmul(be.transpose(X), be.matmul(F, X))
        with timer("diag"):
            eps, Cp = be.eigh(Fp)
        with timer("matmul"):
            C = be.matmul(X, Cp)
        return eps, C

    def density(C):
        Cocc = C[:, :nocc]
        with timer("matmul"):
            return 2.0 * be.matmul(Cocc, be.transpose(Cocc))

    eps, C = solve_roothaan(H)  # core-Hamiltonian guess
    D = density(C)
    e_old = 0.0
    history = []
    focks, errors = [], []
    converged = False
    for it in range(1, max_iter + 1):
        with timer("JK"):
            J, K = be.jk(G, D)
        F = H + J - 0.5 * K
        e_elec = 0.5 * _trace_product(D, H + F)
        e_tot = e_elec + e_nuc
        history.append(e_tot)

        if diis:
            with timer("matmul"):
                FDS = be.matmul(F, be.matmul(D, S))
                err = FDS - be.transpose(FDS)
            focks.append(F)
            errors.append(err)
            focks, errors = focks[-diis_space:], errors[-diis_space:]
            m = len(focks)
            err_max = float(np.abs(err).max())
            # DIIS extrapolation; an exactly converged F gives B = 0, and nearly
            # parallel error vectors make B singular -> drop the oldest vector.
            while m > 1 and err_max > 1e-12:
                B = [[0.0] * (m + 1) for _ in range(m + 1)]
                for a in range(m):
                    for b in range(m):
                        B[a][b] = float(np.vdot(errors[a], errors[b]))
                    B[a][m] = B[m][a] = -1.0
                rhs = [0.0] * m + [-1.0]
                try:
                    c = be.solve(B, rhs)
                except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                    focks, errors = focks[1:], errors[1:]
                    m -= 1
                    continue
                F = sum(c[a] * focks[a] for a in range(m))
                break

        eps, C = solve_roothaan(F)
        D_new = density(C)
        dD = float(np.sqrt(np.mean((D_new - D) ** 2)))
        dE = abs(e_tot - e_old)
        if verbose:
            print(f"  iter {it:3d}  E = {e_tot:.12f}  dE = {dE:.2e}  rms dD = {dD:.2e}")
        D, e_old = D_new, e_tot
        if dE < e_tol and dD < d_tol:
            converged = True
            break

    # final energy with the converged density
    with timer("JK"):
        J, K = be.jk(G, D)
    F = H + J - 0.5 * K
    e_tot = 0.5 * _trace_product(D, H + F) + e_nuc
    if verbose:
        print(f"{'converged' if converged else 'NOT converged'}: E(RHF) = {e_tot:.10f} Eh")
    return SCFResult(e_tot, converged, it, eps, C, D, history)
