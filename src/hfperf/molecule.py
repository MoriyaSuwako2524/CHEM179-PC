"""Molecular geometry input (XYZ, Angstrom) and nuclear repulsion."""
from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

ANGSTROM_TO_BOHR = 1.0 / 0.52917721092  # same constant PySCF uses

SYMBOL_TO_Z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10}


@dataclass
class Molecule:
    symbols: list[str]
    coords: list[tuple[float, float, float]]  # Bohr
    charge: int = 0
    name: str = ""

    @property
    def charges(self) -> list[int]:
        return [SYMBOL_TO_Z[s] for s in self.symbols]

    @property
    def n_electrons(self) -> int:
        return sum(self.charges) - self.charge

    def nuclear_repulsion(self) -> float:
        e = 0.0
        Z, R = self.charges, self.coords
        for a in range(len(Z)):
            for b in range(a):
                e += Z[a] * Z[b] / math.dist(R[a], R[b])
        return e


def read_xyz(path: str | Path, charge: int = 0) -> Molecule:
    lines = Path(path).read_text().strip().splitlines()
    n = int(lines[0])
    symbols, coords = [], []
    for line in lines[2 : 2 + n]:
        s, x, y, z = line.split()[:4]
        symbols.append(s.capitalize())
        coords.append(tuple(float(v) * ANGSTROM_TO_BOHR for v in (x, y, z)))
    return Molecule(symbols, coords, charge, Path(path).stem)


def available_molecules() -> list[str]:
    d = resources.files("hfperf") / "data" / "molecules"
    return sorted(p.name[:-4] for p in d.iterdir() if p.name.endswith(".xyz"))


def load_molecule(name_or_path: str, charge: int = 0) -> Molecule:
    """Load a bundled molecule by name (h2o, benzene, ...) or any XYZ file."""
    p = Path(name_or_path)
    if p.suffix == ".xyz" and p.exists():
        return read_xyz(p, charge)
    f = resources.files("hfperf") / "data" / "molecules" / f"{name_or_path.lower()}.xyz"
    with resources.as_file(f) as fp:
        return read_xyz(fp, charge)
