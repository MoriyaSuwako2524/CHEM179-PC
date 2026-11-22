"""Stage 0 correctness: the naive backend against PySCF reference data."""
import random

import numpy as np
import pytest

from hfperf import build_basis, load_molecule, rhf
from hfperf.backends import get_backend
from hfperf.backends.naive.integrals import eri_contracted
from hfperf.backends.naive.linalg import jacobi_eigh, matmul


def test_basis_size():
    for name, nbf in [("h2", 4), ("h2o", 19), ("benzene", 102)]:
        assert len(build_basis(load_molecule(name))[1]) == nbf


def test_ao_order_matches_pyscf(ref):
    r = ref("h2o")
    _, funcs = build_basis(load_molecule("h2o"))
    ours = [f.label.split()[1][0] + f.label.split()[1][1:] for f in funcs]
    theirs = [lab.split()[2].lstrip("0123456789") for lab in r["ao_labels"]]
    assert ours == theirs


@pytest.mark.parametrize("name", ["h2", "h2o"])
def test_one_electron(ref, name):
    r = ref(name)
    be = get_backend("naive")
    mol = load_molecule(name)
    _, funcs = build_basis(mol)
    np.testing.assert_allclose(be.overlap(funcs), r["S"], atol=1e-10)
    np.testing.assert_allclose(be.kinetic(funcs), r["T"], atol=1e-10)
    np.testing.assert_allclose(be.nuclear(funcs, mol), r["V"], atol=1e-9)


def test_eri_h2_full(ref):
    r = ref("h2")
    be = get_backend("naive")
    _, funcs = build_basis(load_molecule("h2"))
    np.testing.assert_allclose(be.eri(funcs), r["ERI"], atol=1e-10)


def test_eri_h2o_sample(ref):
    """200 random quartets (the full tensor takes minutes in pure Python)."""
    r = ref("h2o")
    _, f = build_basis(load_molecule("h2o"))
    rng = random.Random(179)
    n = len(f)
    for _ in range(200):
        i, j, k, l = (rng.randrange(n) for _ in range(4))
        assert abs(eri_contracted(f[i], f[j], f[k], f[l]) - r["ERI"][i, j, k, l]) < 1e-10, (i, j, k, l)


def test_linalg():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(7, 5)), rng.normal(size=(5, 6))
    np.testing.assert_allclose(matmul(A, B), A @ B, atol=1e-12)
    M = rng.normal(size=(8, 8))
    M = M + M.T
    w, V = jacobi_eigh(M)
    np.testing.assert_allclose(w, np.linalg.eigvalsh(M), atol=1e-10)
    np.testing.assert_allclose(V.T @ M @ V, np.diag(w), atol=1e-9)


def test_rhf_h2(ref_energies):
    res = rhf(load_molecule("h2"), "naive", verbose=False)
    assert res.converged
    assert abs(res.energy - ref_energies["h2"]["e_rhf"]) < 1e-8


@pytest.mark.slow
def test_rhf_h2o(ref_energies):
    res = rhf(load_molecule("h2o"), "naive", verbose=False)
    assert abs(res.energy - ref_energies["h2o"]["e_rhf"]) < 1e-8
