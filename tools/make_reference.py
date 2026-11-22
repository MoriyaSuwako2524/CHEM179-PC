"""Generate PySCF reference data for the tests (run where PySCF is installed).

    python tools/make_reference.py  [--out tests/data]

PySCF is built from the *same* bundled BSE JSON as hfperf, with cart=True.
PySCF normalizes Cartesian d functions so that only d_xx-type components
have unit norm; we rescale every matrix to unit-norm basis functions
(D^-1/2 M D^-1/2 with D = diag(S)) to match hfperf.  Energies are invariant.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyscf import gto, scf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hfperf.molecule import load_molecule  # noqa: E402

ERI_MOLECULES = {"h2", "h2o", "nh3"}  # keep the repo small
ALL = ["h2", "h2o", "nh3", "ch4", "c2h4", "benzene", "naphthalene"]


def pyscf_basis():
    data = json.loads((ROOT / "src/hfperf/data/6-31gs.json").read_text())["elements"]
    sym = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
    out = {}
    for z, shells in data.items():
        lst = []
        for sh in shells:
            for l, c in zip(sh["l"], sh["coefficients"]):
                lst.append([l] + [[e, ci] for e, ci in zip(sh["exponents"], c)])
        out[sym[int(z)]] = lst
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tests/data"))
    ap.add_argument("--molecules", nargs="*", default=ALL)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    basis = pyscf_basis()
    energies = {}
    for name in args.molecules:
        m = load_molecule(name)
        atom = [[s, c] for s, c in zip(m.symbols, m.coords)]
        mol = gto.M(atom=atom, unit="Bohr", basis=basis, cart=True, charge=m.charge, verbose=0)
        S = mol.intor("int1e_ovlp")
        d = 1.0 / np.sqrt(np.diag(S))
        sc2 = np.outer(d, d)
        arrays = dict(S=S * sc2, T=mol.intor("int1e_kin") * sc2, V=mol.intor("int1e_nuc") * sc2,
                      ao_labels=np.array(mol.ao_labels()), e_nuc=mol.energy_nuc())
        if name in ERI_MOLECULES:
            eri = mol.intor("int2e")
            arrays["ERI"] = eri * np.einsum("i,j,k,l->ijkl", d, d, d, d)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        e = mf.kernel()
        arrays["e_rhf"] = e
        energies[name] = {"nbf": mol.nao, "e_rhf": e, "converged": bool(mf.converged), "e_nuc": mol.energy_nuc()}
        np.savez_compressed(out / f"ref_{name}.npz", **arrays)
        print(f"{name:12s} nbf={mol.nao:4d}  E(RHF)={e:.10f}  converged={mf.converged}", flush=True)
    (out / "reference_energies.json").write_text(json.dumps(energies, indent=1))


if __name__ == "__main__":
    main()
