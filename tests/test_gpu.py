"""Stage 5 correctness: GpuBackend on every available target, and the CUDA ERI kernel.

Always runs:   NumPy target (FP64 exact, FP32 emulation, mixed precision) on h2 / h2o.
If installed:  torch-cpu (the code path the Apple MPS target uses), CuPy on an NVIDIA GPU,
               PyTorch MPS on Apple Silicon.  Missing devices -> tests are skipped.
"""
import numpy as np
import pytest

from hfperf import build_basis, load_molecule, rhf
from hfperf.backends import get_backend
from hfperf.backends.gpu import GpuBackend
from hfperf.backends.gpu.xp import make_target


def funcs_of(name):
    return build_basis(load_molecule(name))[1]


def _target_or_skip(name):
    try:
        return make_target(name)
    except Exception as exc:  # ImportError, RuntimeError (no device)
        pytest.skip(f"target {name} unavailable: {exc}")


def _has_cupy_gpu():
    try:
        make_target("cupy")
        return True
    except Exception:
        return False


needs_cupy = pytest.mark.skipif(not _has_cupy_gpu(), reason="needs CuPy and an NVIDIA GPU")


def test_registry_and_auto_target():
    be = get_backend("gpu", target="numpy")
    assert isinstance(be, GpuBackend) and be.precision == "fp64"
    assert make_target("auto").name in ("cupy", "mps", "torch-cuda", "numpy")


@pytest.mark.parametrize("target", ["numpy", "torch-cpu", "cupy", "mps"])
def test_jk_matches_reference_loops(target):
    t = _target_or_skip(target)
    rng = np.random.default_rng(0)
    n = 6
    G = rng.normal(size=(n, n, n, n))       # no permutational symmetry: kernels must not assume it
    D = rng.normal(size=(n, n))
    Jr, Kr = np.einsum("ijkl,kl->ij", G, D), np.einsum("ikjl,kl->ij", G, D)
    prec = "fp64" if t.fp64 else "fp32"
    tol = 1e-12 if prec == "fp64" else 1e-4
    for mode in ("matmul", "einsum"):
        J, K = t.jk(t.asarray(G, prec), t.asarray(D, prec), mode)
        np.testing.assert_allclose(t.to_numpy(J), Jr, atol=tol)
        np.testing.assert_allclose(t.to_numpy(K), Kr, atol=tol)


@pytest.mark.parametrize("target", ["numpy", "torch-cpu", "cupy"])
def test_rhf_h2o_fp64(ref_energies, target):
    _target_or_skip(target)
    res = rhf(load_molecule("h2o"), GpuBackend(target=target, precision="fp64"), verbose=False)
    assert res.converged
    assert abs(res.energy - ref_energies["h2o"]["e_rhf"]) < 1e-8


@pytest.mark.parametrize("target", ["numpy", "torch-cpu", "mps", "cupy"])
def test_rhf_h2o_fp32_error_is_reported_not_hidden(ref_energies, target):
    """FP32 cannot meet the 1e-8 Eh contract; it must still land within ~1e-4 Eh."""
    _target_or_skip(target)
    res = rhf(load_molecule("h2o"), GpuBackend(target=target, precision="fp32"), e_tol=1e-7, d_tol=1e-5,
              verbose=False)
    err = abs(res.energy - ref_energies["h2o"]["e_rhf"])
    assert 1e-9 < err < 1e-4      # visibly worse than FP64, but chemically small


@pytest.mark.parametrize("target", ["numpy", "torch-cpu", "mps", "cupy"])
def test_rhf_h2o_mixed_precision_meets_contract(ref_energies, target):
    _target_or_skip(target)
    be = GpuBackend(target=target, precision="mixed")
    res = rhf(load_molecule("h2o"), be, verbose=False)
    assert res.converged
    assert abs(res.energy - ref_energies["h2o"]["e_rhf"]) < 1e-8
    assert be.gpu_stats["fp32_builds"] > be.gpu_stats["fp64_builds"] >= 1


def test_mixed_without_reanchoring_is_worse_but_close(ref_energies):
    be = GpuBackend(target="numpy", precision="mixed", anchor_factor=0)
    res = rhf(load_molecule("h2o"), be, verbose=False)
    assert be.gpu_stats["fp64_builds"] == 1
    assert abs(res.energy - ref_energies["h2o"]["e_rhf"]) < 1e-5


def test_h2_all_modes(ref_energies):
    e_ref = ref_energies["h2"]["e_rhf"]
    for prec, kw, tol in [("fp64", {}, 1e-8), ("mixed", {}, 1e-8), ("fp32", dict(e_tol=1e-7, d_tol=1e-5), 1e-4)]:
        res = rhf(load_molecule("h2"), GpuBackend(target="numpy", precision=prec), verbose=False, **kw)
        assert abs(res.energy - e_ref) < tol, prec


def test_mps_refuses_fp64():
    t = _target_or_skip("mps")
    assert not t.fp64
    with pytest.raises(ValueError):
        GpuBackend(target="mps", precision="fp64")


# ---------------------------------------------------------------- CUDA ERI kernel (NVIDIA only)
@needs_cupy
@pytest.mark.parametrize("name", ["h2", "h2o", "nh3"])
@pytest.mark.parametrize("order", ["sorted", "natural"])
def test_cuda_eri_fp64_matches_reference(ref, name, order):
    from hfperf.backends.gpu.cuda_eri import CudaERI
    eng = CudaERI(precision="fp64", order=order)
    G = eng.compute(funcs_of(name)).get()
    np.testing.assert_allclose(G, ref(name)["ERI"], rtol=0, atol=1e-10)
    assert eng.stats["quartets_computed"] > 0


@needs_cupy
def test_cuda_eri_fp32_error_size(ref):
    from hfperf.backends.gpu.cuda_eri import CudaERI
    G = CudaERI(precision="fp32").compute(funcs_of("h2o")).get().astype(np.float64)
    err = np.abs(G - ref("h2o")["ERI"]).max()
    assert 1e-9 < err < 1e-4


@needs_cupy
@pytest.mark.parametrize("precision", ["fp64", "mixed"])
def test_rhf_h2o_cuda_eri(ref_energies, precision):
    res = rhf(load_molecule("h2o"), GpuBackend(target="cupy", eri="cuda", precision=precision), verbose=False)
    assert res.converged
    assert abs(res.energy - ref_energies["h2o"]["e_rhf"]) < 1e-8
