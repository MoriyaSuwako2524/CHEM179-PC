"""Stage 1 correctness: every rung of the NumPy ERI ladder against PySCF reference data.

Fast enough for a laptop: everything here runs in a few seconds.
"""
import numpy as np
import pytest

from hfperf import build_basis, load_molecule, rhf
from hfperf.backends import get_backend
from hfperf.backends.naive.boys import boys as boys_series
from hfperf.backends.naive.fock import jk as jk_loop
from hfperf.backends.numpy_opt import NumpyBackend
from hfperf.backends.numpy_opt.boys import boys_all, boys_all_scalar

LADDER = [
    dict(eri="shell", screening=0, boys="series"),
    dict(eri="shell", screening=1e-12, boys="series"),
    dict(eri="shell", screening=1e-12, boys="table"),
    dict(eri="vector", screening=1e-12, boys="table"),
    dict(eri="vector", screening=0, boys="series"),
]


def test_registry_default_is_numpy_backend():
    be = get_backend("numpy")
    assert isinstance(be, NumpyBackend)
    assert be.opts["eri"] == "vector" and be.opts["boys"] == "table" and be.opts["screening"] > 0


def test_boys_table_vs_series():
    T = np.concatenate([np.linspace(0.0, 45.0, 901), [1e-14, 29.999, 30.001, 35.99, 36.0, 80.0]])
    for L in (0, 3, 8):
        F = boys_all(L, T)
        for n in range(L + 1):
            ref = np.array([boys_series(n, t) for t in T])
            np.testing.assert_allclose(F[n], ref, rtol=0, atol=1e-13)
            Fs = np.array([boys_all_scalar(L, t)[n] for t in T[::50]])
            np.testing.assert_allclose(Fs, ref[::50], rtol=0, atol=1e-13)


@pytest.mark.parametrize("name", ["h2", "h2o"])
def test_one_electron_vectorized(ref, name):
    r = ref(name)
    mol = load_molecule(name)
    _, funcs = build_basis(mol)
    be = NumpyBackend()
    np.testing.assert_allclose(be.overlap(funcs), r["S"], atol=1e-10)
    np.testing.assert_allclose(be.kinetic(funcs), r["T"], atol=1e-10)
    np.testing.assert_allclose(be.nuclear(funcs, mol), r["V"], atol=1e-9)


@pytest.mark.parametrize("opts", [dict(eri="full", screening=0), dict(eri="sym8", screening=0)] + LADDER)
def test_eri_h2_all_variants(ref, opts):
    r = ref("h2")
    _, funcs = build_basis(load_molecule("h2"))
    be = NumpyBackend(**opts)
    np.testing.assert_allclose(be.eri(funcs), r["ERI"], atol=1e-10)


def test_sym8_counts():
    _, funcs = build_basis(load_molecule("h2"))
    n = len(funcs)
    npair = n * (n + 1) // 2
    full, sym = NumpyBackend(eri="full", screening=0), NumpyBackend(eri="sym8", screening=0)
    full.eri(funcs), sym.eri(funcs)
    assert full.stats["contracted"] == n ** 4
    assert sym.stats["contracted"] == npair * (npair + 1) // 2


@pytest.mark.parametrize("opts", LADDER)
def test_eri_h2o_shell_variants(ref, opts):
    r = ref("h2o")
    _, funcs = build_basis(load_molecule("h2o"))
    be = NumpyBackend(**opts)
    np.testing.assert_allclose(be.eri(funcs), r["ERI"], atol=1e-10)
    assert be.stats["quartets"] > 0


def test_schwarz_skips_when_threshold_is_huge():
    _, funcs = build_basis(load_molecule("h2o"))
    be = NumpyBackend(screening=1e6)
    G = be.eri(funcs)
    assert be.stats["skipped_quartets"] > 0
    # the Schwarz pass computes all (ab|ab); everything else is skipped
    assert be.stats["quartets"] == be.stats["schwarz_quartets"]
    assert np.isfinite(G).all()


def test_screening_rejected_for_function_level_variants():
    with pytest.raises(ValueError):
        NumpyBackend(eri="sym8", screening=1e-12)


@pytest.mark.parametrize("mode", ["blas", "einsum"])
def test_jk_matches_loops(mode):
    _, funcs = build_basis(load_molecule("h2o"))
    G = NumpyBackend().eri(funcs)
    rng = np.random.default_rng(179)
    D = rng.normal(size=G.shape[:2])
    D = D + D.T
    J0, K0 = jk_loop(G, D)
    J, K = NumpyBackend(jk=mode).jk(G, D)
    np.testing.assert_allclose(J, J0, atol=1e-12)
    np.testing.assert_allclose(K, K0, atol=1e-12)


def test_linalg():
    be = NumpyBackend()
    rng = np.random.default_rng(0)
    M = rng.normal(size=(9, 9))
    M = M + M.T
    w, V = be.eigh(M)
    np.testing.assert_allclose(V.T @ M @ V, np.diag(w), atol=1e-12)
    x = be.solve(M.tolist(), list(range(9)))
    np.testing.assert_allclose(M @ np.array(x), np.arange(9), atol=1e-10)
    with pytest.raises(ValueError):
        be.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


@pytest.mark.parametrize("name", ["h2", "h2o"])
def test_rhf_default(ref_energies, name):
    res = rhf(load_molecule(name), NumpyBackend(), verbose=False)
    assert res.converged
    assert abs(res.energy - ref_energies[name]["e_rhf"]) < 1e-8


def test_rhf_by_name(ref_energies):
    res = rhf(load_molecule("h2o"), "numpy", verbose=False)
    assert abs(res.energy - ref_energies["h2o"]["e_rhf"]) < 1e-8
