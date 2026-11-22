"""Stage 2a correctness: C++ ERI and J/K against PySCF references and stage 0.

Skipped automatically when the extension has not been built
(python ext/cpp/build.py).
"""
import random

import numpy as np
import pytest

ext = pytest.importorskip("hfperf.backends.cpp._hfcpp", reason="C++ module not built: python ext/cpp/build.py")

from hfperf import build_basis, load_molecule, rhf  # noqa: E402
from hfperf.backends import get_backend  # noqa: E402
from hfperf.backends.cpp import CppBackend  # noqa: E402
from hfperf.backends.naive import boys as pyboys  # noqa: E402
from hfperf.backends.naive import fock  # noqa: E402
from hfperf.basis import pack_basis  # noqa: E402


def funcs_of(name):
    return build_basis(load_molecule(name))[1]


def test_registry_default_is_sym8():
    be = get_backend("cpp")
    assert isinstance(be, CppBackend) and be.eri_algorithm == "sym8"


@pytest.mark.parametrize("n", [0, 1, 4, 8])
@pytest.mark.parametrize("T", [0.0, 1e-13, 0.3, 7.5, 29.9, 30.1, 85.0])
def test_boys_matches_python(n, T):
    assert ext.boys(n, T) == pytest.approx(pyboys.boys(n, T), rel=1e-14, abs=1e-300)


@pytest.mark.parametrize("name", ["h2", "h2o"])
@pytest.mark.parametrize("variant", [dict(eri_algorithm="naive"), dict(eri_algorithm="sym8", reuse=False),
                                     dict(eri_algorithm="sym8", reuse=True)])
def test_eri_full_tensor(ref, name, variant):
    G = CppBackend(**variant).eri(funcs_of(name))
    np.testing.assert_allclose(G, ref(name)["ERI"], rtol=0, atol=1e-10)


def test_eri_naive_row_slab(ref):
    b = pack_basis(funcs_of("h2o"))
    args = (b["centers"], b["lmn"], b["offsets"], b["exps"], b["coefs"])
    G = ext.eri_naive(*args, 5, 8)
    R = ref("h2o")["ERI"]
    np.testing.assert_allclose(G[5:8], R[5:8], rtol=0, atol=1e-10)
    assert not G[:5].any() and not G[8:].any()


def test_eri_nh3_sym8(ref):
    be = CppBackend()
    G = be.eri(funcs_of("nh3"))
    np.testing.assert_allclose(G, ref("nh3")["ERI"], rtol=0, atol=1e-10)
    assert be.last_eri_stats["computed"] > 0


def test_quartet_matches_python_kernel():
    from hfperf.backends.naive.integrals import eri_contracted

    f = funcs_of("h2o")
    b = pack_basis(f)
    args = (b["centers"], b["lmn"], b["offsets"], b["exps"], b["coefs"])
    rng = random.Random(179)
    for _ in range(50):
        i, j, k, l = (rng.randrange(len(f)) for _ in range(4))
        assert abs(ext.eri_quartet(*args, i, j, k, l) - eri_contracted(f[i], f[j], f[k], f[l])) < 1e-13


@pytest.mark.parametrize("jk_name", ["jk", "jk_twopass", "jk_naive"])
def test_jk_matches_python(jk_name):
    rng = np.random.default_rng(0)
    n = 7
    G = rng.normal(size=(n, n, n, n))  # no permutational symmetry, D not symmetric:
    D = rng.normal(size=(n, n))         # the kernels must not rely on either
    J, K = getattr(ext, jk_name)(G, D)
    Jr, Kr = fock.jk(G, D)
    np.testing.assert_allclose(J, Jr, atol=1e-12)
    np.testing.assert_allclose(K, Kr, atol=1e-12)


def test_jk_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ext.jk(np.zeros((3, 3, 3, 3)), np.zeros((4, 4)))


@pytest.mark.parametrize("variant", ["naive", "sym8"])
def test_rhf_h2o(ref_energies, variant):
    res = rhf(load_molecule("h2o"), CppBackend(eri_algorithm=variant), verbose=False)
    assert res.converged
    assert abs(res.energy - ref_energies["h2o"]["e_rhf"]) < 1e-8


def test_cpp_omp_registered():
    """cpp-omp was a stub in stage 2; stage 4 implements it (tests in test_parallel.py)."""
    from hfperf.backends.cpp import CppOmpBackend
    try:
        be = get_backend("cpp-omp")
    except ImportError:  # OpenMP module not built (python ext/cpp/build.py --openmp)
        return
    assert isinstance(be, CppOmpBackend)
