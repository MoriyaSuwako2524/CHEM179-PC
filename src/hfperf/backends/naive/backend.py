"""Stage 0 backend: every kernel is an explicit Python loop."""
from . import fock, integrals, linalg


class NaiveBackend:
    name = "naive"

    # one-electron integrals
    def overlap(self, funcs):
        return integrals.overlap_matrix(funcs)

    def kinetic(self, funcs):
        return integrals.kinetic_matrix(funcs)

    def nuclear(self, funcs, mol):
        return integrals.nuclear_matrix(funcs, mol)

    # two-electron integrals
    def eri(self, funcs):
        return integrals.eri_tensor(funcs)

    # linear algebra
    def matmul(self, A, B):
        return linalg.matmul(A, B)

    def transpose(self, A):
        return linalg.transpose(A)

    def eigh(self, A):
        return linalg.jacobi_eigh(A)

    def solve(self, A, b):
        return linalg.solve(A, b)

    # Fock build
    def jk(self, G, D):
        return fock.jk(G, D)
