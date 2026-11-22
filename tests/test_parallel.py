"""Stage 4 (threads, processes, MPI) correctness.

Fast and local: H2 / H2O only.  Compiled variants skip cleanly when not built:
    python ext/cpp/build.py --openmp          (cpp-omp)
    cd ext/rust && maturin develop --release  (rust-rayon)
    pip install mpi4py                        (MPI driver)
Every parallel variant must give the same tensor / J,K as the serial code and the
reference RHF energy within 1e-8 Eh.
"""
import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from hfperf import build_basis, load_molecule, rhf
from hfperf.backends.naive import fock
from hfperf.basis import pack_basis
from hfperf.parallel import amdahl_fit, distribute, karp_flatt
from hfperf.parallel.distribute import imbalance


def funcs_of(name):
    return build_basis(load_molecule(name))[1]


def args_of(name):
    b = pack_basis(funcs_of(name))
    return tuple(np.ascontiguousarray(b[k]) for k in ("centers", "lmn", "offsets", "exps", "coefs"))


def core_density(name):
    from hfperf.parallel.scaling import core_guess_density
    mol = load_molecule(name)
    return core_guess_density(mol, funcs_of(name))


# ------------------------------------------------------------------ pure Python parts
@pytest.mark.parametrize("scheme", ["block", "cyclic", "weighted"])
@pytest.mark.parametrize("nw", [1, 3, 7, 64])
def test_distribute_is_a_partition(scheme, nw):
    npair = 190
    parts = [distribute(npair, nw, r, scheme) for r in range(nw)]
    allrows = np.concatenate(parts)
    assert sorted(allrows.tolist()) == list(range(npair))


def test_block_distribution_is_imbalanced():
    assert imbalance(5253, 8, "block") > 1.7
    assert imbalance(5253, 8, "cyclic") < 1.01
    assert imbalance(5253, 8, "weighted") < 1.01


def test_amdahl_fit_recovers_serial_fraction():
    p = np.array([1, 2, 4, 8, 16, 32])
    t = 100.0 * (0.05 + 0.95 / p)
    fit = amdahl_fit(p, t)
    assert abs(fit["serial_fraction"] - 0.05) < 1e-10
    assert np.allclose(karp_flatt(p, t)[1:], 0.05)


@pytest.mark.parametrize("kind,workers", [("serial", 1), ("thread", 2), ("process", 2)])
def test_mp_eri_matches_reference(ref, kind, workers):
    from hfperf.parallel.mp_eri import eri_parallel
    try:
        G, info = eri_parallel(funcs_of("h2o"), workers, kind)
    except PermissionError as exc:  # e.g. sandboxes without named pipes
        pytest.skip(f"cannot start a process pool here: {exc}")
    np.testing.assert_allclose(G, ref("h2o")["ERI"], rtol=0, atol=1e-10)
    assert info["map_s"] > 0


# ------------------------------------------------------------------ C++ / OpenMP
omp = None
if importlib.util.find_spec("hfperf.backends.cpp._hfcpp_omp") is not None:
    from hfperf.backends.cpp import load_module
    omp = load_module("_hfcpp_omp")
need_omp = pytest.mark.skipif(omp is None, reason="OpenMP module not built: python ext/cpp/build.py --openmp")


@need_omp
@pytest.mark.parametrize("threads", [1, 2, 3])
@pytest.mark.parametrize("schedule,chunk", [("dynamic", 1), ("static", 0), ("static", 1), ("guided", 0)])
def test_omp_eri_tensor(ref, threads, schedule, chunk):
    G, info = omp.eri_sym8_par(*args_of("h2o"), 1e-12, threads, schedule, chunk)
    np.testing.assert_allclose(G, ref("h2o")["ERI"], rtol=0, atol=1e-10)
    assert len(info["busy"]) == threads and info["computed"] > 0


@need_omp
def test_omp_eri_is_bitwise_reproducible():
    a = omp.eri_sym8_par(*args_of("h2o"), 1e-12, 1, "static", 0)[0]
    b = omp.eri_sym8_par(*args_of("h2o"), 1e-12, 4, "dynamic", 1)[0]
    assert np.array_equal(a, b)  # no reductions in the ERI kernel -> identical bits


@need_omp
def test_omp_jk_par_matches_python():
    rng = np.random.default_rng(0)
    G, D = rng.normal(size=(7, 7, 7, 7)), rng.normal(size=(7, 7))
    J, K = omp.jk_par(G, D, 3)
    Jr, Kr = fock.jk(G, D)
    np.testing.assert_allclose(J, Jr, atol=1e-12)
    np.testing.assert_allclose(K, Kr, atol=1e-12)


@need_omp
@pytest.mark.parametrize("mode", ["private", "atomic"])
@pytest.mark.parametrize("threads", [1, 4])
def test_omp_direct_jk(ref, mode, threads):
    D = core_density("h2o")
    Jr, Kr = fock.jk(ref("h2o")["ERI"], D)
    djk = omp.DirectJK(*args_of("h2o"), 1e-12, False, None, threads)
    J, K = djk.build(D, threads, mode, "dynamic", 1)
    np.testing.assert_allclose(J, Jr, atol=1e-10)
    np.testing.assert_allclose(K, Kr, atol=1e-10)


@need_omp
def test_omp_direct_store_and_row_subsets(ref):
    D = core_density("h2o")
    Jr, Kr = fock.jk(ref("h2o")["ERI"], D)
    djk = omp.DirectJK(*args_of("h2o"), 1e-12, True, None, 2)
    for _ in range(2):  # first build fills the cache, second reads it
        J, K = djk.build(D, 2, "private", "dynamic", 1)
        np.testing.assert_allclose(K, Kr, atol=1e-10)
    assert djk.cached
    npair = djk.npair
    Js, Ks = 0, 0
    for r in range(5):  # what 5 MPI ranks would compute; rank sums = full J/K
        part = omp.DirectJK(*args_of("h2o"), 1e-12, False, distribute(npair, 5, r, "block").tolist(), 1)
        Jp, Kp = part.build(D, 1, "private", "static", 0)
        Js, Ks = Js + Jp, Ks + Kp
    np.testing.assert_allclose(Js, Jr, atol=1e-10)
    np.testing.assert_allclose(Ks, Kr, atol=1e-10)
    empty = omp.DirectJK(*args_of("h2o"), 1e-12, False, [], 1)
    assert not empty.build(D, 1, "private", "static", 0)[0].any()


@need_omp
@pytest.mark.parametrize("opts", [dict(), dict(schedule="static", chunk=0), dict(fock="direct"),
                                  dict(fock="direct", accumulate="atomic"), dict(fock="direct", store=True),
                                  dict(rest="naive")])
def test_omp_rhf_h2o(ref_energies, opts):
    from hfperf.backends.cpp import CppOmpBackend
    res = rhf(load_molecule("h2o"), CppOmpBackend(threads=2, **opts), verbose=False)
    assert res.converged and abs(res.energy - ref_energies["h2o"]["e_rhf"]) < 1e-8


# ------------------------------------------------------------------ Rust / Rayon
try:
    import hfperf_rust as rs
    HAVE_RAYON = hasattr(rs, "eri_sym8_par")
except ImportError:
    rs, HAVE_RAYON = None, False
need_rayon = pytest.mark.skipif(not HAVE_RAYON, reason="Rust module (with Rayon) not built")


@need_rayon
@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("mode", ["steal", "block"])
def test_rayon_eri_tensor(ref, threads, mode):
    G, info = rs.eri_sym8_par(*args_of("h2o"), screen=1e-12, threads=threads, mode=mode)
    np.testing.assert_allclose(G, ref("h2o")["ERI"], rtol=0, atol=1e-10)
    assert info["computed"] > 0


@need_rayon
def test_rayon_jk_par_matches_python():
    rng = np.random.default_rng(1)
    G, D = rng.normal(size=(6, 6, 6, 6)), rng.normal(size=(6, 6))
    G = G + G.transpose(1, 0, 3, 2)  # jk_par uses J = J^T, K = K^T (true for ERIs)
    G = G + G.transpose(2, 3, 0, 1)
    D = D + D.T
    J, K = rs.jk_par(G, D, threads=2)
    Jr, Kr = fock.jk(G, D)
    np.testing.assert_allclose(J, Jr, atol=1e-11)
    np.testing.assert_allclose(K, Kr, atol=1e-11)


@need_rayon
@pytest.mark.parametrize("mode", ["fold", "atomic"])
@pytest.mark.parametrize("store", [False, True])
def test_rayon_direct_jk(ref, mode, store):
    D = core_density("h2o")
    Jr, Kr = fock.jk(ref("h2o")["ERI"], D)
    djk = rs.DirectJK(*args_of("h2o"), screen=1e-12, store=store, rows=None, threads=3)
    for _ in range(2):
        J, K = djk.build(D, threads=3, mode=mode)
        np.testing.assert_allclose(J, Jr, atol=1e-10)
        np.testing.assert_allclose(K, Kr, atol=1e-10)
    part = rs.DirectJK(*args_of("h2o"), rows=[], threads=1)
    assert not part.build(D, threads=1)[1].any()


@need_rayon
@pytest.mark.parametrize("opts", [dict(), dict(mode="block"), dict(fock="direct"),
                                  dict(fock="direct", accumulate="atomic")])
def test_rayon_rhf_h2o(ref_energies, opts):
    from hfperf.backends.rust import RustRayonBackend
    res = rhf(load_molecule("h2o"), RustRayonBackend(threads=2, **opts), verbose=False)
    assert res.converged and abs(res.energy - ref_energies["h2o"]["e_rhf"]) < 1e-8


# ------------------------------------------------------------------ MPI
HAVE_MPI = importlib.util.find_spec("mpi4py") is not None
ENGINE = "cpp" if omp is not None else ("rust" if HAVE_RAYON else None)


@pytest.mark.skipif(not HAVE_MPI or ENGINE is None, reason="needs mpi4py and a compiled DirectJK")
def test_mpi_driver_single_rank(ref_energies):
    from hfperf.parallel.mpi_fock import run
    out = run("h2o", engine=ENGINE, verbose=False)
    assert out["converged"] and abs(out["energy"] - ref_energies["h2o"]["e_rhf"]) < 1e-8


@pytest.mark.skipif(not HAVE_MPI or ENGINE is None or shutil.which("mpirun") is None,
                    reason="needs mpirun, mpi4py and a compiled DirectJK")
@pytest.mark.parametrize("dist", ["cyclic", "block"])
def test_mpi_driver_two_ranks(tmp_path, ref_energies, dist):
    out = tmp_path / "mpi.json"
    cmd = ["mpirun", "--oversubscribe", "-n", "2", sys.executable, "-m", "hfperf.parallel.mpi_fock", "h2o",
           "--engine", ENGINE, "--dist", dist, "--json", str(out), "-q"]
    # a previous test may have initialized MPI in THIS process (singleton mode); its OMPI_/PMIX_
    # variables would make the child mpirun think it is part of that job -> start it clean
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_COMM", "OMPI_MCA_orte", "PMIX_", "PMI_"))}
    env["OMP_NUM_THREADS"] = "1"
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0 and "oversubscribe" in r.stderr:  # MPICH has no --oversubscribe
        r = subprocess.run([c for c in cmd if c != "--oversubscribe"], capture_output=True, text=True, timeout=300,
                           env=env)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    import json
    res = json.loads(out.read_text())
    assert res["ranks"] == 2 and abs(res["energy"] - ref_energies["h2o"]["e_rhf"]) < 1e-8
