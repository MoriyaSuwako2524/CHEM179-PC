"""Contracted Cartesian Gaussian basis functions built from a BSE JSON file.

Conventions (chosen to match PySCF with ``cart=True``):
  * shells on each atom are ordered by angular momentum (s, s, s, p, p, d, ...)
  * Cartesian components:  p: x, y, z    d: xx, xy, xz, yy, yz, zz
  * every basis function is normalized to <phi|phi> = 1  (PySCF normalizes
    d_xx differently; the reference script rescales PySCF matrices to ours)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import resources

from .molecule import Molecule, SYMBOL_TO_Z


def cartesian_components(l: int) -> list[tuple[int, int, int]]:
    """(lx, ly, lz) in PySCF/libcint order: xx, xy, xz, yy, yz, zz for l=2."""
    out = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            out.append((lx, ly, l - lx - ly))
    return out


def double_factorial(n: int) -> int:
    return 1 if n <= 0 else n * double_factorial(n - 2)


def primitive_norm(alpha: float, lmn: tuple[int, int, int]) -> float:
    l, m, n = lmn
    L = l + m + n
    return ((2 * alpha / math.pi) ** 0.75 * (4 * alpha) ** (L / 2)
            / math.sqrt(double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)))


@dataclass
class Shell:
    """One contracted shell: a center, an angular momentum and a contraction."""
    atom: int
    center: tuple[float, float, float]
    l: int
    exps: list[float]
    coefs: list[float]  # as given in the basis file (for normalized primitives)


@dataclass
class BasisFunction:
    """phi(r) = sum_k coefs[k] * x^l y^m z^n exp(-exps[k] r^2), normalized."""
    center: tuple[float, float, float]
    lmn: tuple[int, int, int]
    exps: list[float]
    coefs: list[float] = field(default_factory=list)  # include primitive + contraction norms
    shell: int = -1  # index into the shell list
    label: str = ""


def _load_basis_json(name: str = "6-31gs") -> dict:
    with (resources.files("hfperf") / "data" / f"{name}.json").open() as f:
        return json.load(f)


def build_basis(mol: Molecule, name: str = "6-31gs") -> tuple[list[Shell], list[BasisFunction]]:
    """Return (shells, basis functions) for ``mol``."""
    data = _load_basis_json(name)["elements"]
    shells: list[Shell] = []
    for ia, (sym, R) in enumerate(zip(mol.symbols, mol.coords)):
        atom_shells = []
        for entry in data[str(SYMBOL_TO_Z[sym])]:
            # an "SP" entry has l = [0, 1] and two coefficient rows
            for l, c in zip(entry["l"], entry["coefficients"]):
                atom_shells.append(Shell(ia, tuple(R), l, list(entry["exponents"]), list(c)))
        atom_shells.sort(key=lambda s: s.l)  # stable sort: s shells, then p, then d
        shells.extend(atom_shells)

    funcs: list[BasisFunction] = []
    for ish, sh in enumerate(shells):
        for lmn in cartesian_components(sh.l):
            coefs = [c * primitive_norm(a, lmn) for a, c in zip(sh.exps, sh.coefs)]
            label = f"{mol.symbols[sh.atom]}{sh.atom} {'spdf'[sh.l]}" + "".join("xyz"[k] * lmn[k] for k in range(3))
            funcs.append(BasisFunction(sh.center, lmn, list(sh.exps), coefs, ish, label))
    _normalize(funcs)
    return shells, funcs


def _normalize(funcs: list[BasisFunction]) -> None:
    """Scale coefficients so that each contracted function has unit norm.

    For two primitives on the same center the overlap has a closed form:
      <x^l y^m z^n e^{-a r^2} | x^l y^m z^n e^{-b r^2}>
        = (2l-1)!!(2m-1)!!(2n-1)!! pi^{3/2} / (2^L (a+b)^{L+3/2})
    """
    for f in funcs:
        l, m, n = f.lmn
        L = l + m + n
        pref = (double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)
                * math.pi ** 1.5 / 2 ** L)
        s = 0.0
        for a, ca in zip(f.exps, f.coefs):
            for b, cb in zip(f.exps, f.coefs):
                s += ca * cb * pref / (a + b) ** (L + 1.5)
        f.coefs = [c / math.sqrt(s) for c in f.coefs]


def pack_basis(funcs: list[BasisFunction]) -> dict:
    """Flatten basis functions into contiguous NumPy arrays (structure of arrays).

    This is the data layout handed to compiled kernels (C++, Rust, CUDA):
      centers  (nbf, 3) float64      lmn     (nbf, 3) int32
      offsets  (nbf+1,) int32        primitives of function i are
      exps, coefs (nprim_total,)     exps[offsets[i]:offsets[i+1]]
    """
    import numpy as np

    nbf = len(funcs)
    offsets = np.zeros(nbf + 1, dtype=np.int32)
    for i, f in enumerate(funcs):
        offsets[i + 1] = offsets[i] + len(f.exps)
    return {
        "centers": np.array([f.center for f in funcs], dtype=np.float64).reshape(nbf, 3),
        "lmn": np.array([f.lmn for f in funcs], dtype=np.int32).reshape(nbf, 3),
        "offsets": offsets,
        "exps": np.array([e for f in funcs for e in f.exps], dtype=np.float64),
        "coefs": np.array([c for f in funcs for c in f.coefs], dtype=np.float64),
        "shell": np.array([f.shell for f in funcs], dtype=np.int32),
    }
