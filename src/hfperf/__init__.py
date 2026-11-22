"""hfperf: a from-scratch RHF/6-31G* code used to teach where computing time goes.

Every expensive step (integrals, matrix multiply, diagonalization, J/K build)
is a method on a *backend* object, so each class stage replaces one kernel
while the rest of the program stays the same.
"""
from .molecule import Molecule, load_molecule
from .basis import build_basis, BasisFunction, Shell
from .scf import rhf

__all__ = ["Molecule", "load_molecule", "build_basis", "BasisFunction", "Shell", "rhf"]
