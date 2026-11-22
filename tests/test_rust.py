"""Stage 2b correctness: Rust (PyO3) kernels against PySCF references.

Skips cleanly when the extension is not built (cd ext/rust && maturin develop --release).
"""
import numpy as np
import pytest

from hfperf import build_basis, load_molecule, rhf
from hfperf.backends.rust import (HAVE_RUST, RustBackend, RustNaiveBackend, RustSym8Backend,
                                  RustUncheckedBackend)

pytestmark = pytest.mark.skipif(not HAVE_RUST, reason="hfperf_rust not built (cd ext/rust && maturin develop --release)")

VARIANTS = [RustNaiveBackend, RustSym8Backend, RustBackend, RustUncheckedBackend]


@pytest.mark.parametrize("name", ["h2", "h2o"])
def test_one_electron(ref, name):
    r = ref(name)
    be = RustBackend()
    mol = load_molecule(name)
    _, funcs = build_basis(mol)
    np.testing.assert_allclose(be.overlap(funcs), r["S"], atol=1e-10)
    np.testing.assert_allclose(be.kinetic(funcs), r["T"], atol=1e-10)
    np.testing.assert_allclose(be.nuclear(funcs, mol), r["V"], atol=1e-9)


@pytest.mark.parametrize("cls", VARIANTS)
@pytest.mark.parametrize("name", ["h2", "h2o"])
def test_eri(ref, name, cls):
    r = ref(name)
    _, funcs = build_basis(load_molecule(name))
    np.testing.assert_allclose(cls().eri(funcs), r["ERI"], atol=1e-10, rtol=0)


def test_eri_variants_agree_nh3(ref):
    r = ref("nh3")
    _, funcs = build_basis(load_molecule("nh3"))
    for cls in (RustSym8Backend, RustBackend, RustUncheckedBackend):
        np.testing.assert_allclose(cls().eri(funcs), r["ERI"], atol=1e-10, rtol=0)


@pytest.mark.parametrize("mode", ["iter", "checked", "unchecked"])
def test_jk_modes(mode):
    import hfperf_rust
    rng = np.random.default_rng(1)
    n = 7
    G = rng.normal(size=(n, n, n, n))
    G = G + G.transpose(1, 0, 2, 3)
    G = G + G.transpose(0, 1, 3, 2)
    G = G + G.transpose(2, 3, 0, 1)           # 8-fold symmetric like real ERIs
    D = rng.normal(size=(n, n))
    D = D + D.T
    J, K = hfperf_rust.jk(G, D, mode=mode)
    np.testing.assert_allclose(J, np.einsum("ijkl,kl->ij", G, D), atol=1e-11)
    np.testing.assert_allclose(K, np.einsum("ikjl,kl->ij", G, D), atol=1e-11)


def test_bad_input_raises():
    import hfperf_rust
    with pytest.raises(ValueError):
        hfperf_rust.jk(np.zeros((3, 3, 3, 2)), np.zeros((3, 3)), mode="unchecked")
    with pytest.raises(ValueError):
        hfperf_rust.jk(np.zeros((3, 3, 3, 3)), np.zeros((3, 3)), mode="fast")


def test_schwarz_screening_counts():
    be = RustBackend()
    _, funcs = build_basis(load_molecule("h2o"))
    be.eri(funcs)
    n = len(funcs)
    npair = n * (n + 1) // 2
    st = be.last_eri_stats
    assert st["computed"] + st["screened"] == npair * (npair + 1) // 2


@pytest.mark.parametrize("cls", VARIANTS)
@pytest.mark.parametrize("name", ["h2", "h2o"])
def test_rhf_energy(ref_energies, name, cls):
    res = rhf(load_molecule(name), cls(), verbose=False)
    assert res.converged
    assert abs(res.energy - ref_energies[name]["e_rhf"]) < 1e-8


def test_strict_mode_h2o(ref_energies):
    """linalg/one-electron back to stage 0: only ERI + J/K are Rust (like CppBackend)."""
    res = rhf(load_molecule("h2o"), RustBackend(linalg="naive", onee="naive"), verbose=False)
    assert abs(res.energy - ref_energies["h2o"]["e_rhf"]) < 1e-8
