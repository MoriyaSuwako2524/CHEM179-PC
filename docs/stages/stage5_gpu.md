# Stage 5 - GPUs: moving the Fock build (and the integrals) to a graphics card

**Previous stage:** compiled kernels on one CPU core (C++ / Rust, `docs/stages/stage2_*.md`) and threads
(stage 4).  **This stage:** the same RHF driver (`hfperf/scf.py`, unchanged) with a backend that runs the
heavy linear algebra on a GPU, plus - as an instructor demo - the two-electron integrals themselves.

| | student path (laptop) | instructor demo (Sooner, NVIDIA data-centre GPU) |
|---|---|---|
| backend | `GpuBackend(target="auto")` | `GpuBackend(target="cupy", eri="cuda")` |
| ERI | computed once by the best CPU code (cpp -> rust -> numpy), then copied to the device | CUDA kernel (`src/hfperf/backends/gpu/eri_md.cu`), never leaves the device |
| J/K, matmul, eigh | on the device: CuPy (NVIDIA), PyTorch-MPS (Apple), NumPy fallback | CuPy (cuBLAS / cuSOLVER) |
| precision | FP64, FP32, mixed | FP64, FP32, mixed |

Code: `src/hfperf/backends/gpu/` (`xp.py` device layer, `backend.py` GpuBackend, `cuda_eri.py` + `eri_md.cu`
CUDA kernel), `tests/test_gpu.py`, `ext/cuda/` (benchmark, collection, precision study, profiling target),
`demo/sooner/` (environment + Slurm job).  Raw data: `results/gpu/raw/`, tables: `results/gpu/*.csv`.

## 1. What changed vs. the previous stage

1. **A device-array layer** (`xp.py`).  `NumpyTarget`, `CupyTarget` and `TorchTarget('mps'|'cuda'|'cpu')`
   expose the same five operations: `asarray` (host -> device copy in FP64 or FP32), `to_numpy`
   (device -> host), `matmul`, `eigh`, `jk`.  J and K are written once:
   `J = (G.reshape(N^2, N^2) @ D.reshape(N^2)).reshape(N, N)` (one matrix-vector product) and
   `K = matmul(G, D[None, :, :, None]).sum(1)` (N^2 small batched products) - the same expressions as the
   NumPy stage, now dispatched to cuBLAS or Metal.
2. **The ERI tensor lives on the device.**  `GpuBackend.eri()` computes G once and returns the *device*
   array; the SCF driver only hands it back to `jk()`.  Per SCF iteration only D (N^2 numbers) goes up and
   J, K (2 N^2 numbers) come down; the N^4 tensor never moves again.
3. **Three precision modes** (`precision=`):
   * `fp64` - everything in double precision (not available on Apple MPS: Metal has no FP64);
   * `fp32` - G, D, J, K, matmul and eigh in single precision;
   * `mixed` - an *incremental Fock build*: the first J/K is FP64 (the "anchor"), afterwards only the change
     dD = D_n - D_(n-1) goes through FP32 J/K on the device and is added to J, K in FP64 on the host.
     The FP32 rounding error is proportional to |dD|, which shrinks as the SCF converges.  Whenever rms(dD)
     has fallen by `anchor_factor` (default 1e-3) relative to the first increment after the last anchor, a
     fresh FP64 J/K is taken.  `anchor_factor=0` keeps the single initial anchor.  On a device without FP64
     (MPS) the anchors run on the CPU with NumPy, and matmul/eigh stay FP64 on the host.
4. **A CUDA ERI kernel** (instructor demo, `eri="cuda"`; CuPy `RawModule`, compiled at run time by NVRTC).
   It is the "sym8 + reuse" algorithm of the C++/Rust stages split into three kernels:
   * `pair_kernel` - one thread per *primitive* pair: p, P and the Hermite coefficients E_tuv;
   * `schwarz_kernel` - one thread per function pair: Q_ij = sqrt|(ij|ij)|;
   * `quartet_kernel` - one thread per *unique contracted quartet* (ij|kl), kl <= ij: loop over the
     primitive quartets, Boys function (series + downward recursion, or erf + upward recursion for T >= 30),
     McMurchie-Davidson R recursion, contraction with E, then 8 symmetric writes into the N^4 tensor.
   FP64 or FP32 is a compile-time switch (`-DREAL=double|float`); every literal is cast with `R_()` so the
   FP32 build never silently promotes to FP64.  Pair order `sorted` (default) groups pairs by angular
   momentum and contraction length so that the 32 threads of a warp run the same loop bounds.

## 2. Why it is faster (and when it is not)

**J/K is a memory-bandwidth problem.**  Each element of G is read once for J and once for K and used in
one multiply-add.  On a CPU core this runs at ~15 GB/s; an A100's HBM2 is rated at ~1.5 TB/s and an L40S's
GDDR6 at ~0.86 TB/s (vendor specifications, not measured here).  Measured: benzene J/K (G = 866 MB) takes 1.31 ms in FP64 on the A100, i.e.
2 x 866 MB / 1.31 ms = 1.32 TB/s effective - about 85 % of the rated bandwidth, and 82x faster than one
Xeon 6338 core (108 ms, 16 GB/s).  FP32 halves the bytes and roughly halves the time (0.73 ms): for a
bandwidth-bound kernel the gain from FP32 is "half the bytes", not "more FLOPs".

**The ERI is a compute problem with ~10^7-10^8 independent tasks.**  Benzene has 13.6 million unique
quartets (the same count as the C++ sym8 code; 178,195 screened by Schwarz), naphthalene 80.7 million.
That is enough independent work to fill 108 SMs x 2048 threads.  The A100 computes the benzene ERI tensor
in 0.117 s (device total, FP64) - 407x the single-core C++ time (47.7 s, Pete Gold 6130F).  The
comparison is one GPU against **one** CPU core; with ideal 32-core scaling the C++ code would need
~1.5 s (extrapolated, not measured - the multi-core CPU results of stage 4 were not available when this
was written), i.e. the GPU is still ~13x ahead of a whole CPU node.

**What limits the ERI kernel.**  CuPy reports 100 registers and 3,536 bytes of *local memory* per thread
for the FP64 quartet kernel: the McMurchie-Davidson R table (up to 9 x 48 numbers for a (dd|dd) quartet)
cannot live in registers and spills to L1/L2-cached local memory.  The 100 registers allow only 4 blocks
of 128 threads per SM: theoretical occupancy 25 % (FP32 build: 72 registers, 44 %).  Ordering the pair
list by angular-momentum class makes the kernel 2.5x faster than the natural i >= j order (warp
divergence: in natural order neighbouring threads mix s, p and d quartets with different loop lengths and
the warp executes all branches).  Block size 64 vs 128 makes no difference; 256 is slower.

**Consumer and workstation GPUs have little FP64 hardware.**  On the A100 the FP64 ERI kernel is only
2.2x slower than FP32 (data-centre GPUs have FP64 units at half the FP32 rate).  On the L40S
(Ada Lovelace, a graphics/AI card) FP64 is 11x slower than FP32 for benzene (0.350 vs 0.032 s quartet
kernel), and in FP32 the L40S is faster than the A100.  Laptop GPUs (GeForce, Apple) behave like the L40S
or worse - which is why the student path needs a precision strategy.

**Small problems lose on a GPU.**  Every call has a fixed cost of tens of microseconds (kernel launch,
host<->device copy, synchronisation).  GPU J/K costs ~0.17 ms whatever the size up to c2h4; the CPU is
faster below ~20 basis functions.  GPU `eigh` (cuSOLVER) is slower than LAPACK on the CPU for every
molecule here (N <= 166) and only breaks even at naphthalene.  Nsight Systems on the H2O SCF
(`results/gpu/raw/nsys_h2o_*.txt`) shows 340 tiny `cudaMemcpyAsync` calls (~3 kB each) - latency, not
bandwidth.  It also shows a load-balance effect: the Schwarz kernel has only 190 threads for H2O (one per
pair) and takes as long as the 18,145-thread quartet kernel.

**Transfers.**  Copying the benzene ERI tensor to the device over PCIe 4.0 takes 0.10 s from ordinary
(pageable) memory (~9 GB/s) and 0.040 s from pinned memory (~22 GB/s).  This is as long as *computing* the
tensor on the A100 (0.117 s): once the ERI kernel is on the GPU, the PCIe bus, not the arithmetic, would be
the bottleneck of the CPU-ERI path.

**Amdahl again.**  With ERI and J/K on the A100 the benzene SCF takes 0.95 s in total - compared with
60.5 s for the fastest single-core CPU SCF (Rust sym8 unchecked, Pete 6226R) - but half of that time is
now the one-electron integrals (S, T, V: 0.49 s, NumPy on the CPU), which nobody moved to the GPU.

## 3. Precision: how wrong is FP32, and the mixed-precision fix

FP32 carries ~7 significant digits (unit round-off 6e-8).  A total energy of -230 Eh therefore cannot be
better than ~1e-5 Eh, and the SCF "noise floor" prevents the usual convergence test (dE < 1e-10 Eh) from
ever being met: FP32 runs use dE < 1e-7, rms dD < 1e-5 and still often stop at the 100-iteration limit
(marked NC).  Errors scatter in sign and magnitude between devices and libraries because the summation
order differs - FP32 error is noise, not a systematic offset.

**A100, CUDA ERI + CuPy J/K** (dE = E - E_PySCF in Eh, iterations in brackets, NC = not converged):

| molecule | nbf | fp64: dE Eh (iter) | fp32: dE Eh (iter) | fp32-jk-only: dE Eh (iter) | mixed: dE Eh (iter) | mixed-no-reanchor: dE Eh (iter) |
|---|---|---|---|---|---|---|
| h2 | 4 | -3.4e-14 (5) | +5.2e-07 (6) | -8.3e-08 (5) | -3.4e-14 (6) | +1.3e-08 (5) |
| h2o | 19 | +1.4e-13 (12) | -7.0e-05 (60) | -2.7e-06 (11) | +1.6e-13 (13) | -3.7e-06 (13) |
| nh3 | 21 | +7.1e-14 (12) | +2.0e-05 (100, NC) | -5.2e-06 (17) | +8.5e-14 (12) | +9.3e-07 (42) |
| ch4 | 23 | +1.4e-14 (11) | +2.0e-05 (100, NC) | +1.3e-06 (19) | +0.0e+00 (12) | +1.9e-06 (10) |
| c2h4 | 38 | +2.3e-13 (11) | +1.6e-05 (27) | +2.5e-06 (17) | +2.0e-13 (12) | -1.3e-05 (11) |
| benzene | 102 | +2.4e-12 (12) | +6.1e-05 (100, NC) | +3.0e-05 (50) | +2.1e-12 (13) | -4.1e-05 (100, NC) |
| naphthalene | 166 | +5.2e-12 (15) | -7.4e-05 (100, NC) | +2.9e-05 (100, NC) | +4.2e-12 (16) | +3.3e-05 (100, NC) |

**L40S, same code:**

| molecule | nbf | fp64: dE Eh (iter) | fp32: dE Eh (iter) | fp32-jk-only: dE Eh (iter) | mixed: dE Eh (iter) | mixed-no-reanchor: dE Eh (iter) |
|---|---|---|---|---|---|---|
| h2 | 4 | -3.3e-14 (5) | +9.9e-07 (19) | -6.1e-08 (5) | -3.4e-14 (6) | +1.1e-08 (5) |
| h2o | 19 | +2.1e-13 (12) | -8.8e-05 (100, NC) | -9.9e-07 (29) | +2.3e-13 (13) | -4.0e-06 (13) |
| nh3 | 21 | +7.1e-14 (12) | +4.0e-06 (100, NC) | -4.5e-06 (16) | +8.5e-14 (12) | -2.5e-06 (14) |
| ch4 | 23 | +1.4e-14 (11) | +1.2e-05 (10) | -6.3e-07 (38) | +0.0e+00 (11) | -1.3e-07 (10) |
| c2h4 | 38 | +2.6e-13 (11) | +2.6e-05 (100, NC) | +4.3e-06 (23) | +2.6e-13 (12) | -1.1e-05 (17) |
| benzene | 102 | +2.6e-12 (12) | +5.6e-05 (100, NC) | +3.4e-05 (63) | +2.2e-12 (13) | -7.0e-05 (100, NC) |
| naphthalene | 166 | +4.9e-12 (15) | -2.6e-05 (100, NC) | +2.9e-05 (100, NC) | +4.5e-12 (16) | +3.7e-05 (25) |

**CPU FP32 emulation** (`target="numpy"`, the FP64 tensor from the A100 run; this is what a student
without an NVIDIA GPU measures):

| molecule | nbf | fp64: dE Eh (iter) | fp32: dE Eh (iter) | fp32-jk-only: dE Eh (iter) | mixed: dE Eh (iter) | mixed-no-reanchor: dE Eh (iter) |
|---|---|---|---|---|---|---|
| h2 | 4 | -3.4e-14 (5) | -5.2e-08 (8) | +4.1e-09 (6) | -3.5e-14 (6) | +1.1e-08 (5) |
| h2o | 19 | +5.7e-14 (12) | -1.4e-06 (17) | +4.0e-07 (19) | +5.7e-14 (13) | +3.6e-06 (12) |
| nh3 | 21 | +9.9e-14 (12) | -5.9e-06 (66) | +8.7e-07 (24) | +7.1e-14 (12) | +3.2e-06 (13) |
| ch4 | 23 | -7.1e-14 (11) | -9.8e-06 (18) | -3.5e-06 (26) | -7.1e-14 (12) | +7.6e-06 (17) |
| c2h4 | 38 | +2.8e-13 (11) | -2.4e-06 (100, NC) | +4.0e-06 (29) | +3.4e-13 (13) | -1.5e-05 (14) |
| benzene | 102 | +2.8e-12 (12) | +4.7e-05 (44) | +7.7e-06 (56) | +2.5e-12 (13) | -4.0e-05 (100, NC) |
| naphthalene | 166 |  |  |  |  |  |

**Local laptop CPU** (Windows, torch-cpu and NumPy targets, H2O; torch-cpu runs the same code path as
Apple MPS, but Metal's FP32 kernels sum in a different order, so MPS numbers will differ in detail -
**the MPS target itself was not run: no Apple hardware was available**):

| target | mode | molecule | dE vs ref (Eh) | iterations | FP64/FP32 J/K |
|---|---|---|---|---|---|
| torch-cpu | fp64 | h2o | +8.5e-14 | 12 | 13/0 |
| torch-cpu | fp32 | h2o | -2.2e-05 | 35 | 0/36 |
| torch-cpu | fp32-jk-only | h2o | -3.5e-07 | 14 | 0/15 |
| torch-cpu | mixed | h2o | +8.5e-14 | 13 | 3/11 |
| torch-cpu | mixed-no-reanchor | h2o | +2.9e-07 | 12 | 1/12 |
| numpy | fp64 | h2o | +1.3e-13 | 12 | 13/0 |
| numpy | fp32 | h2o | +4.6e-06 | 37 | 0/38 |
| numpy | fp32-jk-only | h2o | +5.0e-06 | 69 | 0/70 |
| numpy | mixed | h2o | +1.4e-13 | 13 | 3/11 |
| numpy | mixed-no-reanchor | h2o | +3.7e-07 | 12 | 1/12 |

Summary of what was achieved:

* `fp64`: within 1e-8 Eh of PySCF for every molecule and device (largest |dE| 5.2e-12 Eh, naphthalene).
* `fp32` (CUDA-ERI runs on the two GPUs): |dE| from 5.2e-07 Eh (H2) to 8.8e-05 Eh (CPU emulation goes down to 5e-8 Eh for H2); 9 of the 14 FP32 SCF runs with the CUDA ERI (7 molecules x 2 GPUs)
  never reach dE < 1e-7 within 100 iterations.
* `fp32-jk-only` (FP64 matmul/eigh): 4e-9 to 3.4e-5 Eh - most of the damage comes from J/K, but FP32
  orthogonalisation/diagonalisation adds to it.
* `mixed` with re-anchoring: within 1e-8 Eh everywhere (largest |dE| 4.5e-12 Eh), with 3 FP64 and 10-14
  FP32 J/K builds per SCF.  The FP32 increments are exact enough because by the time they matter, dD is
  small.
* `mixed-no-reanchor`: 1e-8 to 7e-5 Eh (benzene/naphthalene partly not converged) - the first increments (dD ~ D) are as inaccurate as plain FP32
  and their error is never removed.

On the A100 the mixed scheme is not faster (FP64 is cheap there).  It pays off where FP64 is slow or
missing: consumer GPUs and Apple Silicon - there all but ~3 Fock builds run at FP32 speed.

## 4. Measured numbers (Sooner, single GPU)

Hardware: NVIDIA A100-PCIE-40GB (sm_80, 108 SMs, node c851) and NVIDIA L40S (sm_89, 142 SMs, node c1033),
both hosted by Intel Xeon Gold 6338 CPUs; CuPy 14.2.0, CUDA runtime 12.9 (driver reports CUDA 13.3).
"CPU" columns measured on the same node with one thread (`OMP_NUM_THREADS=1`); the "CPU C++ sym8" column
is the Phase-1 Pete single-core number (Gold 6130F).  GPU times are the best of 2-3 runs after a warm-up
(kernel compilation excluded; it takes ~1.3 s the first time and is cached on disk).

**ERI tensor (device total = basis upload + pair + Schwarz + quartet kernels):**

| molecule | nbf | unique quartets computed | CPU C++ sym8, 1 core (s) | A100 FP64 (s) | A100 FP32 (s) | L40S FP64 (s) | L40S FP32 (s) | A100 FP64 vs 1 CPU core | FP64 max abs err vs PySCF | FP32 max rel err vs FP64 |
|---|---|---|---|---|---|---|---|---|---|---|
| h2 | 4 | 55 |  | 0.00214 | 0.00144 | 0.00244 | 0.00084 |  | 9.5e-15 | 1.4e-07 |
| h2o | 19 | 18145 | 0.0539 | 0.0148 | 0.0114 | 0.0124 | 0.0038 | 4x | 2.2e-14 | 3.0e-07 |
| nh3 | 21 | 26796 | 0.0795 | 0.00829 | 0.00642 | 0.0125 | 0.00384 | 10x | 2.1e-14 | 8.8e-07 |
| ch4 | 23 | 38226 | 0.115 | 0.00796 | 0.0064 | 0.0128 | 0.00385 | 14x | (no ref) | 3.1e-07 |
| c2h4 | 38 | 274911 | 0.943 | 0.0167 | 0.00986 | 0.0411 | 0.00563 | 56x | (no ref) | 3.1e-07 |
| benzene | 102 | 13621436 | 47.7 | 0.117 | 0.055 | 0.369 | 0.0383 | 407x | (no ref) | 3.1e-07 |
| naphthalene | 166 | 80658247 | 214 | 0.581 | 0.287 | 1.96 | 0.231 | 368x | (no ref) | 3.1e-07 |

**Phases of the ERI computation:**

| GPU | molecule | precision | upload basis + alloc G (s) | pair kernel (s) | Schwarz kernel (s) | quartet kernel (s) | device total (s) | host pair tables (s) |
|---|---|---|---|---|---|---|---|---|
| A100 | naphthalene | fp64 | 0.0081 | 0.00006 | 0.0081 | 0.5648 | 0.5810 | 0.0045 |
| A100 | naphthalene | fp32 | 0.0050 | 0.00004 | 0.0051 | 0.2766 | 0.2867 | 0.0044 |
| L40S | naphthalene | fp64 | 0.0101 | 0.00009 | 0.0174 | 1.9306 | 1.9581 | 0.0046 |
| L40S | naphthalene | fp32 | 0.0087 | 0.00005 | 0.0027 | 0.2192 | 0.2306 | 0.0046 |
| A100 | benzene | fp64 | 0.0029 | 0.00003 | 0.0079 | 0.1064 | 0.1172 | 0.0019 |
| A100 | benzene | fp32 | 0.0018 | 0.00002 | 0.0050 | 0.0481 | 0.0550 | 0.0016 |
| L40S | benzene | fp64 | 0.0024 | 0.00004 | 0.0157 | 0.3504 | 0.3686 | 0.0021 |
| L40S | benzene | fp32 | 0.0040 | 0.00002 | 0.0027 | 0.0317 | 0.0383 | 0.0020 |

**Pair order, block size, registers and occupancy** (CuPy `RawKernel.attributes` and
`occupancyMaxActiveBlocksPerMultiprocessor`; Nsight Compute could not be used, see below):

| GPU | molecule | precision | pair order | block | quartet kernel (s) | registers/thread | local mem/thread (B) | blocks/SM | theoretical occupancy |
|---|---|---|---|---|---|---|---|---|---|
| A100 | c2h4 | fp64 | sorted | 128 | 0.0097 | 100 | 3536 | 4 | 0.250 |
| A100 | c2h4 | fp32 | sorted | 128 | 0.0051 | 72 | 1776 | 7 | 0.438 |
| A100 | c2h4 | fp64 | natural | 128 | 0.0089 | 100 | 3536 | 4 | 0.250 |
| A100 | c2h4 | fp64 | sorted | 64 | 0.0092 | 100 | 3536 | 8 | 0.250 |
| A100 | c2h4 | fp64 | sorted | 256 | 0.0096 | 100 | 3536 | 2 | 0.250 |
| A100 | c2h4 | fp32 | natural | 128 | 0.0057 | 72 | 1776 | 7 | 0.438 |
| L40S | c2h4 | fp64 | sorted | 128 | 0.0286 | 96 | 3536 | 5 | 0.417 |
| L40S | c2h4 | fp32 | sorted | 128 | 0.0026 | 64 | 1776 | 8 | 0.667 |
| L40S | c2h4 | fp64 | natural | 128 | 0.0292 | 96 | 3536 | 5 | 0.417 |
| L40S | c2h4 | fp64 | sorted | 64 | 0.0275 | 96 | 3536 | 10 | 0.417 |
| L40S | c2h4 | fp64 | sorted | 256 | 0.0348 | 96 | 3536 | 2 | 0.333 |
| L40S | c2h4 | fp32 | natural | 128 | 0.0033 | 64 | 1776 | 8 | 0.667 |
| A100 | benzene | fp64 | sorted | 128 | 0.1064 | 100 | 3536 | 4 | 0.250 |
| A100 | benzene | fp32 | sorted | 128 | 0.0481 | 72 | 1776 | 7 | 0.438 |
| A100 | benzene | fp64 | natural | 128 | 0.2635 | 100 | 3536 | 4 | 0.250 |
| A100 | benzene | fp64 | sorted | 64 | 0.1055 | 100 | 3536 | 8 | 0.250 |
| A100 | benzene | fp64 | sorted | 256 | 0.1424 | 100 | 3536 | 2 | 0.250 |
| A100 | benzene | fp32 | natural | 128 | 0.1330 | 72 | 1776 | 7 | 0.438 |
| L40S | benzene | fp64 | sorted | 128 | 0.3504 | 96 | 3536 | 5 | 0.417 |
| L40S | benzene | fp32 | sorted | 128 | 0.0317 | 64 | 1776 | 8 | 0.667 |
| L40S | benzene | fp64 | natural | 128 | 0.7186 | 96 | 3536 | 5 | 0.417 |
| L40S | benzene | fp64 | sorted | 64 | 0.3504 | 96 | 3536 | 10 | 0.417 |
| L40S | benzene | fp64 | sorted | 256 | 0.3516 | 96 | 3536 | 2 | 0.333 |
| L40S | benzene | fp32 | natural | 128 | 0.0782 | 64 | 1776 | 8 | 0.667 |
| A100 | naphthalene | fp64 | sorted | 128 | 0.5648 | 100 | 3536 | 4 | 0.250 |
| A100 | naphthalene | fp32 | sorted | 128 | 0.2766 | 72 | 1776 | 7 | 0.438 |
| A100 | naphthalene | fp64 | natural | 128 | 1.4832 | 100 | 3536 | 4 | 0.250 |
| L40S | naphthalene | fp64 | sorted | 128 | 1.9306 | 96 | 3536 | 5 | 0.417 |
| L40S | naphthalene | fp32 | sorted | 128 | 0.2192 | 64 | 1776 | 8 | 0.667 |
| L40S | naphthalene | fp64 | natural | 128 | 3.6053 | 96 | 3536 | 5 | 0.417 |

**Host <-> device copies of the full ERI tensor:**

| GPU | molecule | precision | G size (MB) | host->device pageable (s) | pageable GB/s | pinned GB/s | device->host pageable GB/s |
|---|---|---|---|---|---|---|---|
| A100 | naphthalene | fp64 | 6074.7 | 0.652 | 9.32 | 24.5 | 4.48 |
| A100 | naphthalene | fp32 | 3037.3 | 0.324 | 9.38 | 24.6 | 4.4 |
| L40S | naphthalene | fp64 | 6074.7 | 0.715 | 8.5 | 24.4 | 3.9 |
| L40S | naphthalene | fp32 | 3037.3 | 0.355 | 8.55 | 24.5 | 3.93 |
| A100 | h2o | fp64 | 1.0 | 0.000144 | 7.23 | 19.9 | 3.32 |
| A100 | h2o | fp32 | 0.5 | 8.44e-05 | 6.17 | 16.5 | 6.75 |
| A100 | c2h4 | fp64 | 16.7 | 0.00198 | 8.44 | 24.2 | 10.4 |
| A100 | c2h4 | fp32 | 8.3 | 0.000928 | 8.99 | 23.9 | 10.2 |
| A100 | benzene | fp64 | 865.9 | 0.102 | 8.49 | 21.8 | 4.44 |
| A100 | benzene | fp32 | 433.0 | 0.0462 | 9.37 | 24.6 | 4.39 |
| L40S | h2o | fp64 | 1.0 | 0.000151 | 6.93 | 19.6 | 2.86 |
| L40S | h2o | fp32 | 0.5 | 8.81e-05 | 5.92 | 16.4 | 8.11 |
| L40S | c2h4 | fp64 | 16.7 | 0.00211 | 7.91 | 24.1 | 10.5 |
| L40S | c2h4 | fp32 | 8.3 | 0.000967 | 8.62 | 23.8 | 15.1 |
| L40S | benzene | fp64 | 865.9 | 0.0978 | 8.85 | 24.5 | 4.03 |
| L40S | benzene | fp32 | 433.0 | 0.0488 | 8.87 | 24.5 | 4.02 |

**J/K, eigh, matmul per call (random symmetric D, best of 5-20):**

| GPU | molecule | nbf | CPU J/K 1 core (s) | GPU J/K FP64 (s) | GPU J/K FP32 (s) | GPU J/K einsum FP64 (s) | CPU eigh (s) | GPU eigh FP64 (s) | CPU matmul (s) | GPU matmul FP64 (s) | NxN round trip (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A100 | h2 | 4 | 7.06e-06 | 0.000165 | 0.000166 | 0.000206 | 8.46e-06 | 0.000105 | 1.27e-06 | 2.03e-05 | 3.27e-05 |
| A100 | h2o | 19 | 8.65e-05 | 0.000168 | 0.000167 | 0.000207 | 3.95e-05 | 0.000258 | 1.79e-06 | 1.88e-05 | 3.57e-05 |
| A100 | nh3 | 21 | 0.000153 | 0.000168 | 0.000168 | 0.000206 | 4.82e-05 | 0.00027 | 1.8e-06 | 1.87e-05 | 3.39e-05 |
| A100 | ch4 | 23 | 0.000192 | 0.000168 | 0.000167 | 0.000207 | 5.65e-05 | 0.000276 | 2.05e-06 | 1.88e-05 | 3.38e-05 |
| A100 | c2h4 | 38 | 0.00115 | 0.000167 | 0.000174 | 0.00022 | 0.000136 | 0.000705 | 3.45e-06 | 1.95e-05 | 3.71e-05 |
| A100 | benzene | 102 | 0.108 | 0.00131 | 0.000728 | 0.0049 | 0.000836 | 0.00193 | 4.56e-05 | 2.11e-05 | 5.17e-05 |
| A100 | naphthalene | 166 | 0.771 | 0.00875 | 0.00475 | 0.0425 | 0.00216 | 0.0021 | 0.000182 | 2.19e-05 | 7.58e-05 |
| L40S | h2 | 4 | 8.18e-06 | 0.000204 | 0.000198 | 0.000255 | 9.92e-06 | 0.000124 | 1.52e-06 | 2.42e-05 | 3.73e-05 |
| L40S | h2o | 19 | 0.000106 | 0.000202 | 0.000209 | 0.000252 | 4.72e-05 | 0.000394 | 2.17e-06 | 2.73e-05 | 4.09e-05 |
| L40S | nh3 | 21 | 0.000153 | 0.000204 | 0.000212 | 0.000251 | 5.69e-05 | 0.000424 | 2.19e-06 | 2.73e-05 | 3.9e-05 |
| L40S | ch4 | 23 | 0.000221 | 0.000205 | 0.000204 | 0.000251 | 6.72e-05 | 0.000437 | 2.4e-06 | 2.73e-05 | 3.94e-05 |
| L40S | c2h4 | 38 | 0.00128 | 0.000204 | 0.000202 | 0.00028 | 0.000176 | 0.00101 | 4.52e-06 | 3.11e-05 | 4.22e-05 |
| L40S | benzene | 102 | 0.117 | 0.00252 | 0.00119 | 0.0121 | 0.00108 | 0.00333 | 5.9e-05 | 4.66e-05 | 5.86e-05 |
| L40S | naphthalene | 166 | 0.866 | 0.0167 | 0.00847 | 0.095 | 0.00279 | 0.00424 | 0.000236 | 6.19e-05 | 8.81e-05 |

**SCF wall time and breakdown (CUDA ERI path):**

| GPU | molecule | mode | SCF wall (s) | S,T,V CPU (s) | ERI (s) | J/K total (s) | J/K calls | diag (s) | matmul (s) | FP64/FP32 J/K builds |
|---|---|---|---|---|---|---|---|---|---|---|
| A100 | naphthalene | fp64 | 3.32 | 1.25 | 1.84 | 0.144 | 16 | 0.0408 | 0.0218 | 16/0 |
| A100 | naphthalene | fp32 | 3.74 | 1.25 | 1.55 | 0.502 | 101 | 0.194 | 0.134 | 0/101 |
| A100 | naphthalene | mixed | 3.41 | 1.25 | 1.97 | 0.0988 | 17 | 0.0478 | 0.0226 | 3/14 |
| L40S | naphthalene | fp64 | 5.31 | 1.52 | 3.4 | 0.272 | 16 | 0.0772 | 0.0275 | 16/0 |
| L40S | naphthalene | fp32 | 4.49 | 1.51 | 1.66 | 0.881 | 101 | 0.144 | 0.157 | 0/101 |
| L40S | naphthalene | mixed | 5.39 | 1.51 | 3.56 | 0.177 | 17 | 0.0936 | 0.0284 | 3/14 |
| A100 | c2h4 | fp64 | 0.133 | 0.0811 | 0.024 | 0.00388 | 12 | 0.0109 | 0.009 | 12/0 |
| A100 | c2h4 | fp32 | 0.161 | 0.0818 | 0.0142 | 0.00921 | 28 | 0.0262 | 0.0217 | 0/28 |
| A100 | c2h4 | mixed | 0.15 | 0.0817 | 0.0318 | 0.00474 | 13 | 0.0174 | 0.0101 | 3/10 |
| A100 | benzene | fp64 | 0.947 | 0.487 | 0.393 | 0.0189 | 13 | 0.0287 | 0.0127 | 13/0 |
| A100 | benzene | fp32 | 1.13 | 0.485 | 0.235 | 0.088 | 101 | 0.172 | 0.0984 | 0/101 |
| A100 | benzene | mixed | 0.849 | 0.484 | 0.298 | 0.0149 | 14 | 0.0311 | 0.0137 | 3/11 |
| L40S | c2h4 | fp64 | 0.184 | 0.0982 | 0.0513 | 0.00448 | 12 | 0.015 | 0.0104 | 12/0 |
| L40S | c2h4 | fp32 | 0.332 | 0.0986 | 0.0118 | 0.0371 | 101 | 0.0581 | 0.0929 | 0/101 |
| L40S | c2h4 | mixed | 0.182 | 0.0984 | 0.0469 | 0.00522 | 13 | 0.0159 | 0.0112 | 3/10 |
| L40S | benzene | fp64 | 1.27 | 0.59 | 0.577 | 0.0347 | 13 | 0.0492 | 0.0154 | 13/0 |
| L40S | benzene | fp32 | 1.27 | 0.59 | 0.242 | 0.133 | 101 | 0.127 | 0.115 | 0/101 |
| L40S | benzene | mixed | 1.27 | 0.591 | 0.574 | 0.0241 | 14 | 0.054 | 0.0165 | 3/11 |

**Student path on the A100 node** (ERI on the CPU with NumPy - the C++/Rust extensions were not built on
Sooner - then copied to the GPU):

| molecule | nbf | mode | dE vs ref (Eh) | iterations | SCF wall (s) | CPU ERI (s) | upload G (s) | J/K total (s) | ERI backend |
|---|---|---|---|---|---|---|---|---|---|
| h2o | 19 | fp64 | +1.7e-13 | 12 | 0.212 | 0.162 | 0.000249 | 0.00403 | numpy |
| h2o | 19 | fp32 | -7.3e-05 | 76 | 0.309 | 0.16 | 0.000217 | 0.0239 | numpy |
| h2o | 19 | mixed | +1.6e-13 | 13 | 0.207 | 0.161 | 0.000195 | 0.00419 | numpy |
| nh3 | 21 | fp64 | +7.1e-14 | 12 | 0.358 | 0.305 | 0.000661 | 0.00402 | numpy |
| nh3 | 21 | fp32 | +4.5e-05 | 61 | 0.439 | 0.306 | 0.00026 | 0.0193 | numpy |
| nh3 | 21 | mixed | +5.7e-14 | 12 | 0.357 | 0.305 | 0.000243 | 0.00394 | numpy |
| ch4 | 23 | fp64 | +1.4e-14 | 11 | 0.592 | 0.529 | 0.000689 | 0.0039 | numpy |
| ch4 | 23 | fp32 | +2.7e-05 | 52 | 0.668 | 0.53 | 0.000367 | 0.0168 | numpy |
| ch4 | 23 | mixed | -1.4e-14 | 12 | 0.594 | 0.53 | 0.000334 | 0.00418 | numpy |
| c2h4 | 38 | fp64 | +2.6e-13 | 11 | 2.44 | 2.32 | 0.0029 | 0.00393 | numpy |
| c2h4 | 38 | fp32 | +1.1e-05 | 100 | 2.67 | 2.32 | 0.00288 | 0.0338 | numpy |
| c2h4 | 38 | mixed | +2.0e-13 | 12 | 2.46 | 2.33 | 0.00332 | 0.0082 | numpy |

## 5. Commands for students

Install the package as in the earlier stages (`python -m pip install -e ".[dev]"`), then:

```bash
# which GPU library is usable here?
python tools/doctor.py

# macOS on Apple Silicon (M1-M4): PyTorch with the Metal (MPS) backend, FP32 only
python -m pip install torch
python ext/cuda/precision_study.py h2o --target mps
python ext/cuda/precision_study.py c2h4 --target mps --csv my_mps_errors.csv

# Linux (or WSL) laptop with an NVIDIA GPU: CuPy
python -m pip install cupy-cuda12x
python -m hfperf h2o --backend gpu --opt target=cupy
python ext/cuda/precision_study.py c2h4 --target cupy

# any laptop without a usable GPU: the same code on the CPU (FP32 is emulated exactly by NumPy float32)
python ext/cuda/precision_study.py h2o --target numpy
python ext/cuda/precision_study.py h2o --target torch-cpu      # if torch is installed

# tests (GPU tests are skipped automatically when the device is missing)
python -m pytest -q tests/test_gpu.py
```

In Python (choose your own SCF thresholds for FP32):

```python
from hfperf import load_molecule, rhf
from hfperf.backends.gpu import GpuBackend
be = GpuBackend(target="auto", precision="mixed")      # or "fp32" / "fp64"
res = rhf(load_molecule("c2h4"), be)
print(res.energy, be.gpu_stats)                        # FP64 vs FP32 builds, transfer/kernel times
res32 = rhf(load_molecule("c2h4"), GpuBackend(precision="fp32"), e_tol=1e-7, d_tol=1e-5)
```

`python -m hfperf ... --opt precision=fp32` also works but uses the default thresholds (1e-10 / 1e-8),
which FP32 never reaches, so it runs to 100 iterations and prints "NOT converged".

Apple-specific notes: `torch.linalg.eigh` is not implemented on MPS in many PyTorch versions; the backend
then runs eigh on the CPU and records `gpu_stats["eigh_on"] == "cpu-fallback"`.  Requesting
`precision="fp64"` on MPS raises a `ValueError`.

## 6. Instructor demo on Sooner

```bash
# once (venv on /scratch because the /home quota is small; scratch is purged, re-run if needed)
bash demo/sooner/setup_env.sh
# from the repo root on Sooner
sbatch demo/sooner/gpu_bench.sbatch                                    # A100 (--constraint=public_a100)
sbatch --constraint=public_l40s_quad -t 01:30:00 demo/sooner/gpu_bench.sbatch   # L40S comparison
python ext/cuda/collect.py                                             # raw JSON -> results/gpu/*.csv
```

The job runs the full test suite, `ext/cuda/bench_gpu.py` for h2 ... benzene and naphthalene, and the
profilers if they exist.  Jobs used for the numbers above: 36366284 (A100) and 36366287 (L40S).

## 7. What was not done or could not be measured

* **Apple MPS** was not run (no Apple hardware).  The torch code path was tested with `torch-cpu` on Windows.
* **The local RTX 5070 laptop GPU** was not used: the agent's sandbox on that PC cannot see the NVIDIA driver
  (`nvcuda.dll` missing, CuPy reports driver version 0), so the "small local CuPy check" was skipped.
* **Nsight Compute** failed on both Sooner GPUs with `ERR_NVGPUCTRPERM` (no permission to read the GPU
  performance counters).  Registers, local memory and theoretical occupancy therefore come from the CUDA
  driver API through CuPy, not from measured achieved occupancy.  Nsight Systems worked
  (`results/gpu/raw/nsys_h2o_*.txt`).
* The CPU-ERI student path on Sooner used the NumPy ERI (C++/Rust extensions were not built there).
* The multi-core CPU comparison is an extrapolation (ideal 32x scaling of the single-core C++ time); replace
  it with the stage-4 measured numbers when available.
