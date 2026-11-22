"""CUDA ERI kernel (instructor demo): CuPy RawModule around eri_md.cu.

    from hfperf.backends.gpu.cuda_eri import CudaERI
    eng = CudaERI(precision="fp64")          # or "fp32"
    G = eng.compute(funcs)                   # cupy array (N, N, N, N) on the GPU
    eng.stats                                # timings (s), counts, registers, occupancy

Work split
  host (NumPy)   list of function pairs i >= j, primitive-pair bookkeeping (offsets)
  pair_kernel    p, P, Hermite E for every primitive pair      (one thread each)
  schwarz_kernel Q_ij = sqrt|(ij|ij)|                            (one thread per pair)
  quartet_kernel (ij|kl) for every unique quartet, 8 writes      (one thread each)

``order="sorted"`` (default) reorders the pair list by angular-momentum class and
contraction length, so neighbouring threads of a warp run the same loop bounds
(less divergence); ``order="natural"`` keeps the i >= j order of the CPU code.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ...basis import pack_basis

KERNEL_SOURCE = Path(__file__).with_name("eri_md.cu")
_MODULES: dict = {}


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise ImportError("the CUDA ERI kernel needs CuPy: pip install cupy-cuda12x") from exc
    return cp


def load_module(precision: str = "fp64"):
    """Compile eri_md.cu once per precision (NVRTC; cached by CuPy on disk too)."""
    cp = _cupy()
    if precision not in _MODULES:
        defs = ("-DREAL=float", "-DREAL_IS_FLOAT=1") if precision == "fp32" else ("-DREAL=double",)
        _MODULES[precision] = cp.RawModule(code=KERNEL_SOURCE.read_text(), options=("--std=c++14",) + defs)
    return _MODULES[precision]


def pair_tables(funcs):
    """Host-side bookkeeping for the unique function pairs i >= j (NumPy, O(N^2))."""
    b = pack_basis(funcs)
    n = len(funcs)
    off, lmn = b["offsets"].astype(np.int64), b["lmn"]
    nprim = np.diff(off)
    ii, jj = np.tril_indices(n)                       # pair ij = (i, j), i >= j, row-major
    pair_i, pair_j = ii.astype(np.int32), jj.astype(np.int32)
    pair_L = (lmn[ii] + lmn[jj]).astype(np.int32)     # (npair, 3)
    nherm = np.prod(pair_L + 1, axis=1)
    npp_pair = nprim[ii] * nprim[jj]
    pair_ppoff = np.concatenate([[0], np.cumsum(npp_pair)]).astype(np.int32)
    npp = int(pair_ppoff[-1])
    pp_pair = np.repeat(np.arange(len(ii)), npp_pair).astype(np.int32)
    local = np.arange(npp) - pair_ppoff[pp_pair]       # index inside the pair's primitive block
    nb = nprim[jj][pp_pair]
    pp_a = (off[ii][pp_pair] + local // nb).astype(np.int32)
    pp_b = (off[jj][pp_pair] + local % nb).astype(np.int32)
    hsize = nherm[pp_pair]
    pp_hoff = np.concatenate([[0], np.cumsum(hsize)[:-1]]).astype(np.int64)
    return dict(b=b, n=n, npair=len(ii), npp=npp, nherm_total=int(hsize.sum()), pair_i=pair_i, pair_j=pair_j,
                pair_L=np.ascontiguousarray(pair_L), pair_ppoff=pair_ppoff, pp_pair=pp_pair, pp_a=pp_a, pp_b=pp_b,
                pp_hoff=pp_hoff, nherm=nherm, npp_pair=npp_pair)


def pair_order(tab, order: str):
    if order == "natural":
        return np.arange(tab["npair"], dtype=np.int32)
    if order == "sorted":   # same loop bounds for neighbouring threads: sort by (L class, contraction length)
        key = np.lexsort((tab["npp_pair"], tab["pair_L"][:, 2], tab["pair_L"][:, 1], tab["pair_L"][:, 0],
                          tab["pair_L"].sum(axis=1)))
        return key.astype(np.int32)
    raise ValueError("order must be 'sorted' or 'natural'")


def kernel_resources(kernel, block: int) -> dict:
    """Registers / local memory per thread and the theoretical occupancy of ``kernel``."""
    cp = _cupy()
    out = {}
    try:
        attrs = kernel.attributes
        out.update(num_regs=attrs.get("num_regs"), local_size_bytes=attrs.get("local_size_bytes"),
                   shared_size_bytes=attrs.get("shared_size_bytes"),
                   max_threads_per_block=attrs.get("max_threads_per_block"))
    except Exception as exc:  # pragma: no cover
        out["attributes_error"] = repr(exc)
    try:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        max_thr = props["maxThreadsPerMultiProcessor"]
        blocks = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(kernel.kernel.ptr, block, 0)
        out.update(block=block, active_blocks_per_sm=blocks, occupancy=blocks * block / max_thr,
                   max_threads_per_sm=max_thr, regs_per_sm=props["regsPerMultiprocessor"])
    except Exception as exc:  # pragma: no cover
        out["occupancy_error"] = repr(exc)
    return out


class CudaERI:
    def __init__(self, precision: str = "fp64", block: int = 128, order: str = "sorted",
                 screen_tol: float = 1e-12):
        if precision not in ("fp64", "fp32"):
            raise ValueError("precision must be 'fp64' or 'fp32'")
        self.precision, self.block, self.order, self.screen_tol = precision, block, order, screen_tol
        self.stats: dict = {}

    def compute(self, funcs):
        cp = _cupy()
        real = np.float32 if self.precision == "fp32" else np.float64
        t0 = time.perf_counter()
        mod = load_module(self.precision)
        kp, ks, kq = (mod.get_function(k) for k in ("pair_kernel", "schwarz_kernel", "quartet_kernel"))
        t_compile = time.perf_counter() - t0

        t0 = time.perf_counter()
        tab = pair_tables(funcs)
        order = pair_order(tab, self.order)
        t_host = time.perf_counter() - t0

        ev = [cp.cuda.Event() for _ in range(6)]
        ev[0].record()
        d = {k: cp.asarray(tab[k]) for k in ("pair_i", "pair_j", "pair_L", "pair_ppoff", "pp_pair", "pp_a", "pp_b",
                                                "pp_hoff")}
        b = tab["b"]
        centers, lmn = cp.asarray(b["centers"].astype(real)), cp.asarray(b["lmn"])
        exps, coefs = cp.asarray(b["exps"].astype(real)), cp.asarray(b["coefs"].astype(real))
        d_order = cp.asarray(order)
        n, npair, npp = tab["n"], tab["npair"], tab["npp"]
        pp_p = cp.empty(npp, dtype=real)
        pp_P = cp.empty(3 * npp, dtype=real)
        E = cp.empty(tab["nherm_total"], dtype=real)
        Q = cp.empty(npair, dtype=np.float64)
        G = cp.zeros((n, n, n, n), dtype=real)
        ev[1].record()

        blk = self.block
        kp(((npp + blk - 1) // blk,), (blk,), (np.int64(npp), d["pp_pair"], d["pp_a"], d["pp_b"], d["pp_hoff"],
                                               d["pair_i"], d["pair_j"], centers, lmn, exps, coefs, pp_p, pp_P, E))
        ev[2].record()
        ks(((npair + blk - 1) // blk,), (blk,), (np.int32(npair), d["pair_L"], d["pair_ppoff"], pp_p, pp_P,
                                                 d["pp_hoff"], E, Q))
        ev[3].record()
        nquart = npair * (npair + 1) // 2
        kq(((nquart + blk - 1) // blk,), (blk,), (np.int32(n), np.int64(nquart), d_order, d["pair_i"], d["pair_j"],
                                                  d["pair_L"], d["pair_ppoff"], pp_p, pp_P, d["pp_hoff"], E, Q,
                                                  np.float64(self.screen_tol), G))
        ev[4].record()
        ev[4].synchronize()
        q = Q.get()
        qq = np.outer(q, q)[np.tril_indices(npair)]
        screened = int((qq < self.screen_tol).sum())
        ms = cp.cuda.get_elapsed_time
        self.stats = dict(
            precision=self.precision, nbf=n, npair=npair, nprim_pairs=npp, quartets_total=nquart,
            quartets_computed=nquart - screened, quartets_screened=screened, block=blk, order=self.order,
            compile_or_load_s=t_compile, host_tables_s=t_host,
            h2d_alloc_s=ms(ev[0], ev[1]) / 1e3, pair_kernel_s=ms(ev[1], ev[2]) / 1e3,
            schwarz_kernel_s=ms(ev[2], ev[3]) / 1e3, quartet_kernel_s=ms(ev[3], ev[4]) / 1e3,
            device_total_s=ms(ev[0], ev[4]) / 1e3, G_bytes=int(G.nbytes),
            quartet_kernel_resources=kernel_resources(kq, blk),
            pair_kernel_resources=kernel_resources(kp, blk))
        return G


def gpu_info() -> dict:
    """Name, compute capability, memory and FP64 hints of the current CUDA device."""
    cp = _cupy()
    dev = cp.cuda.Device()
    p = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
    return dict(name=name, compute_capability=f"{p['major']}.{p['minor']}", sm_count=p["multiProcessorCount"],
                total_mem_GB=p["totalGlobalMem"] / 1e9, clock_MHz=p.get("clockRate", 0) / 1e3,
                cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                driver=cp.cuda.runtime.driverGetVersion())
