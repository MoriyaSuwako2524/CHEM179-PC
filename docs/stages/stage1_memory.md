# Stage 1 — Memory tricks and algorithmic tricks (NumPy backend)

Stage 0 (`backends/naive`) spends almost all of its time in the Python interpreter.
Stage 1 makes two separate points:

1. **Memory hierarchy.** For matrix multiplication, loop order and cache blocking matter a lot
   once the code is compiled (C). In pure Python the interpreter cost per operation hides them.
   Then BLAS (`numpy @`) shows what a carefully blocked, vectorized kernel can reach.
2. **Do less work.** For the two-electron integrals the large gains come from algorithms,
   not from memory tricks: permutational symmetry, shell-pair data reuse, screening, a faster
   Boys function, and moving the innermost loops from Python into NumPy array operations.

All numbers below were measured on one core of a Pete compute node
(`Intel Xeon Gold 6226R @ 2.90GHz`, 32 KB L1d, 1 MB L2/core, 22 MB L3), with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, Python 3.10, NumPy 2.2.6 (OpenBLAS), and GCC 13.2.
Raw data are in `results/memory/`.

## What changed with respect to stage 0

| kernel | stage 0 (`naive`) | stage 1 (`numpy`) |
|---|---|---|
| `matmul`, `transpose` | triple Python loop | `A @ B` (BLAS dgemm), `A.T` (a view, no copy) |
| `eigh` | cyclic Jacobi in Python | `numpy.linalg.eigh` (LAPACK `dsyevd`) |
| `solve` (DIIS) | Gaussian elimination in Python | LAPACK LU (`scipy.linalg.lu_factor`), same singularity test → `ValueError` |
| `jk` | 4 nested Python loops | J: one `gemv` on `G.reshape(N², N²)`; K: batched `matmul` over contiguous (j,l) blocks of G, so G is not copied |
| S, T, V | per function pair, recursive E | per shell pair, E tables vectorized over primitive pairs |
| ERI | all N⁴ contracted integrals, recursive E and R | ladder R0…R5 below; the default is R5 |

`NumpyBackend()` uses the fastest correct combination:
`NumpyBackend(eri="vector", screening=1e-12, boys="table", onee="vector", jk="blas", linalg="lapack")`.
Each trick is a constructor switch, so a single change can be timed by itself:

```python
from hfperf import load_molecule, rhf
from hfperf.backends.numpy_opt import NumpyBackend
be = NumpyBackend(eri="shell", screening=0, boys="series")   # rung R2
rhf(load_molecule("h2o"), be)
print(be.stats)        # shell quartets, contracted / primitive integrals evaluated, Schwarz skips
```

## Part A — matrix multiplication and the memory hierarchy

`benchmarks/matmul/` contains:
* `py_ladder.py`: Python-level variants;
* `matmul.c`: a small plain-C library loaded with `ctypes` (ijk, ikj, transposed-B, blocked with tile size `t`, STREAM triad, FMA peak probe);
* `c_ladder.py`: the driver for the C library.

### A1. Python: the interpreter hides cache effects (GFLOP/s; 1 GFLOP/s = 10⁹ floating-point operations per second, 2N³ per product)

| variant | N=64 | N=128 | N=192 | N=256 |
|---|---|---|---|---|
| ijk, NumPy element indexing `A[i,k]` | 0.00575 | 0.0078 | 0.00739 | 0.0073 |
| ijk, nested lists | 0.0223 | 0.0293 | 0.0264 | 0.0258 |
| ikj, nested lists | 0.0214 | 0.027 | 0.0275 | 0.0277 |
| transpose B first, lists | 0.0257 | 0.0331 | 0.031 | 0.031 |
| blocked (tile 32), lists | 0.0195 | 0.0243 | 0.0249 | 0.0248 |
| row-vectorized `A[i,:] @ B` | 3.21 | 7.84 | 11 | 12.1 |
| column-vectorized `A @ B[:,j]` (strided) | 3.23 | 8.99 | 13.1 | 10.2 |
| `A @ B` (BLAS, 1 thread) | 47.1 | 66.1 | 69.8 | 72.8 |

In Python every `a*b` and `s += …` executes tens of bytecodes, with type checks and object
allocation, which costs about 30–80 ns. A cache miss costs about 5–100 ns. At N = 256 all five pure-Python orders land
between 0.025 and 0.031 GFLOP/s, so the loop order changes the time by
less than 25%.
Element indexing on NumPy arrays is 3.5× *slower* than lists, because each
`A[i,k]` builds a new Python float object. Moving only the innermost loop into compiled code (`A[i,:] @ B`)
already gives ~391× and full BLAS ~2,353× over the best pure-Python loop (N = 256).

### A2. C: now loop order and blocking matter (GFLOP/s)

`O2` = `gcc -O2`; `O3n` = `gcc -O3 -march=native` (AVX-512 available but GCC prefers 256-bit vectors on Skylake-SP);
`O3n512` = same plus `-mprefer-vector-width=512`.

| variant | N=128 | N=256 | N=512 | N=1000 | N=1024 | N=2048 |
|---|---|---|---|---|---|---|
| BLAS (`numpy @`, OpenBLAS, 1 thread) | 22.8 | 54 | 79.9 | 85.5 | 66.5 | 95.3 |
| O2 ijk | 0.753 | 1.54 | 0.66 | 1.59 | 0.684 | 0.289 |
| O2 ikj | 1.19 | 2.64 | 3.1 | 3.5 | 3.61 | 2.66 |
| O2 tB | 0.887 | 1.58 | 1.91 | 1.86 | 1.87 | 1.7 |
| O2 blocked16 | 1.42 | 2.6 | 3.29 | 3.24 | 3.31 | 3.28 |
| O2 blocked32 | 1.6 | 2.75 | 3.48 | 3.41 | 3.5 | 3.44 |
| O2 blocked64 | 1.63 | 2.8 | 3.48 | 3.51 | 3.55 | 3.47 |
| O2 blocked128 | 1.71 | 2.59 | 3.16 | 3.35 | 3.34 | 2.54 |
| O3n ijk | 1.22 | 1.56 | 0.661 | 1.61 | 0.685 | 0.301 |
| O3n ikj | 5.35 | 6.96 | 5.56 | 6.2 | 6.36 | 4.11 |
| O3n tB | 1.49 | 1.74 | 1.98 | 1.91 | 1.85 | 1.73 |
| O3n blocked16 | 3.98 | 4.79 | 5.05 | 6.46 | 5.13 | 5.06 |
| O3n blocked32 | 4.79 | 5.49 | 6.1 | 9.66 | 5.55 | 6.25 |
| O3n blocked64 | 5.37 | 6.48 | 7.39 | 8.8 | 7.16 | 7.51 |
| O3n blocked128 | 5.97 | 7.02 | 7.94 | 9.11 | 9 | 5.65 |
| O3n512 ijk | 1.43 | 1.62 | 0.66 | 1.62 | 0.685 | 0.305 |
| O3n512 ikj | 6.61 | 9.49 | 6.44 | 6.53 | 6.52 | 4.62 |
| O3n512 tB | 1.72 | 2.16 | 1.92 | 1.82 | 1.84 | 1.73 |
| O3n512 blocked16 | 5.59 | 6.86 | 5.55 | 8.68 | 6.02 | 5.89 |
| O3n512 blocked32 | 6.67 | 7.44 | 6.31 | 10.5 | 7.04 | 6.94 |
| O3n512 blocked64 | 6.7 | 8.24 | 7.83 | 7.14 | 8.93 | 9.08 |
| O3n512 blocked128 | 7.08 | 8.93 | 8.45 | 7.32 | 9.94 | 6.03 |

How to read this table:
* **ijk** reads B down a column. Each multiply-add touches a new 64-byte cache line and uses 8 bytes of it.
  When N is a power of two (512, 1024, 2048), every element of a column maps to the same few cache sets
  (associativity conflicts), and the rate drops to ~0.3 GFLOP/s at N = 2048, versus
  1.6 at N = 1000. The `s +=` reduction is also a serial dependency chain,
  so the compiler cannot vectorize it without `-ffast-math`. Compiler flags do not help ijk.
* **ikj** walks rows of B and C with stride 1. Every byte loaded is used, and the inner loop
  `c[j] += a*b[j]` vectorizes (SIMD). This gives 9× over ijk at N = 1024 (O3n).
* **transposed B** also gives stride-1 access. The inner loop is still a dot-product reduction, which is not vectorized
  without `-ffast-math`, so it stays near 2 GFLOP/s.
* **blocked** keeps three t×t tiles (3·8·t² bytes; 24 KB at t = 32, 98 KB at t = 64, 393 KB at t = 128)
  in L1/L2 and reuses each loaded element t times. This holds the rate up at N = 2048, where plain ikj falls off
  (4.1 → 7.5 GFLOP/s with t = 64, O3n;
  4.6 → 9.1 with O3n512).
* **BLAS** adds register blocking, packing of panels into contiguous buffers, and hand-written AVX-512 FMA micro-kernels.
  It reaches 95 GFLOP/s at N = 2048, about 10× our best C.
  The dip at N = 1024 for BLAS (67 vs 85 at N = 1000) is again a power-of-two effect.

### A3. Machine limits for a roofline (one core)

| quantity | value | how |
|---|---|---|
| theoretical peak at base clock | 92.8 GFLOP/s | 2.9 GHz × 8 doubles (AVX-512) × 2 (FMA = mul+add) × 2 FMA units |
| measured FMA peak (O3n / O3n512) | 117.3 / 116.8 GFLOP/s | `peak_fma`: 12 independent 512-bit FMA chains, best of 3 × 0.4 s |
| implied clock under AVX-512 load | 3.66 GHz | measured peak / 32 FLOP per cycle (turbo is above the 2.9 GHz base) |
| measured peak, `-O2` (SSE2 only, no FMA; the 64-byte vector type is split into 128-bit pieces and probably spilled to the stack) | 6.4 GFLOP/s | same source code, no `-march=native` |
| STREAM triad bandwidth | 13.7 GB/s | `a[i] = b[i] + s*c[i]`, 3 × 160 MB arrays, 24 bytes per element |
| ridge point (peak / bandwidth) | 8.6 FLOP/byte | a kernel needs more FLOPs per byte than this to be compute-bound |

Matrix multiply does 2N³ FLOPs on 3·8·N² bytes, so its arithmetic intensity grows like N/12. Above N ≈ 100 it can be compute-bound,
but only if the data really is reused from cache. Without that reuse (ijk), each multiply-add streams at least 8 bytes from memory.
At 0.25 FLOP/byte the memory roof is about 3.4 GFLOP/s, which matches the ijk numbers.

Measured peak and STREAM come from `results/memory/machine_v3.json` (node n192, the same node and CPU model as the ladder).
An earlier short probe in `machine.json` (40 ms per run) gave an inconsistent 79 GFLOP/s for O3n because the clock was still ramping;
that probe was superseded. `machine_v2.json` is the same longer probe on a Gold 6130F node (2.1 GHz base), kept for comparison.

## Part B — algorithmic tricks for the ERIs and J/K

### The ERI ladder (cumulative; every rung uses the stage-1 one-electron integrals, J/K and LAPACK)

| rung | what is added | switch |
|---|---|---|
| R0 | all N⁴ contracted integrals, stage-0 integral code | `eri="full"` |
| R1 | (a) 8-fold symmetry (ij\|kl) = (ji\|kl) = (ij\|lk) = (kl\|ij) = …: only i≥j, k≥l, ij≥kl | `eri="sym8"` |
| R2 | (b) loop over **shell quartets**; per shell pair the product exponents p, centres P and the Hermite E tables (times contraction coefficients) are built **once**. One Boys/R evaluation per primitive quartet serves all Cartesian components (81 for (pp\|pp)). The E·R·E contraction is two small matrix products. Python loop over primitive pairs | `eri="shell", screening=0, boys="series"` |
| R3 | (c) Schwarz screening: skip the quartet if max√(ab\|ab) · max√(cd\|cd) < 10⁻¹² | `screening=1e-12` |
| R4 | (d) Boys function from a table (ΔT = 0.1) + 6-term Taylor for the top order only + downward recursion for the rest | `boys="table"` |
| R5 | (e) primitive-pair loops become NumPy array operations: Boys, the R recursion (one array op per level) and E·R·E (two `tensordot`/BLAS calls) act on all P×Q primitive quartets of a shell quartet at once | `eri="vector"` (default) |

Measured ERI time and total SCF time (seconds, one core):

| molecule | N | ERI R0 | ERI R1 | ERI R2 | ERI R3 | ERI R4 | ERI R5 | SCF R5 | ERI speed-up R0→R5 |
|---|---|---|---|---|---|---|---|---|---|
| h2o | 19 | 150 | 21.6 | 1.02 | 0.813 | 0.798 | 0.242 | 0.333 | 621× |
| nh3 | 21 | 206 | 28.9 | 1.52 | 1.52 | 1.07 | 0.454 | 0.568 | 453× |
| ch4 | 23 | 268 | 37.9 | 2.75 | 2.63 | 1.73 | 0.825 | 1.03 | 325× |
| c2h4 | 38 | 3,353 | 441 | 13 | 13 | 8.18 | 3.47 | 3.86 | 968× |
| benzene | 102 | not run | not run | not run | not run | not run | 92.6 | 96.5 | — |

All 24 small-molecule entries were measured; none are extrapolated. R0 for C₂H₄ (2,085,136 integrals) took 56 min, just inside the 1 h express limit. R0 for h2o/nh3/ch4 ran on node n192; all other ERI runs ran on node n183. Both nodes have the same CPU model. Per integral, the stage-0 code costs 1.2–1.6 ms on these nodes (about 2× slower than the ~0.5 ms measured on the Windows development PC).
Benzene was run with R5 only; it fits easily in the 1 h express limit. Energies agree with PySCF (`tests/data/reference_energies.json`) to
|ΔE| ≤ 2.2e-12 Eh in every run (contract: 10⁻⁸ Eh).

Work actually done (from `NumpyBackend.stats`):

| molecule | N | contracted integrals R0 (= N⁴) | R1 (unique) | R2–R5 contracted (in evaluated shell quartets) | shell quartets evaluated | primitive quartets | Schwarz-skipped shell quartets |
|---|---|---|---|---|---|---|---|
| h2o | 19 | 130,321 | 18,145 | 23,557 | 1,540 | 47,793 | 0 |
| nh3 | 21 | 194,481 | 26,796 | 33,111 | 3,081 | 85,551 | 0 |
| ch4 | 23 | 279,841 | 38,226 | 45,528 | 5,565 | 142,720 | 0 |
| c2h4 | 38 | 2,085,136 | 274,911 | 310,998 | 22,155 | 650,854 | 0 |
| benzene | 102 | 108,243,216 | 13,799,631 | 14,369,139 | 663,739 | 18,700,593 | 28,403 |

The shell-quartet count of contracted integrals is a little above the R1 count. When two shells in a quartet are the same
(for example (aa|bb)), the whole block is computed although some of its elements are symmetry copies. This redundancy is a few percent.

Why each rung helps:
* **R0→R1 (7.0× for H₂O):** 1/8 of the work. The symmetry copies are memory writes, not integral evaluations.
* **R1→R2 (21× H₂O, 34× C₂H₄):** the largest gain on the ladder. Stage 0 recomputes the same
  E coefficients and Boys values recursively for every Cartesian component and every primitive. Shell-pair data
  are computed once and reused across the whole (nab × ncd) block (memoization / data reuse).
* **R2→R3 (Schwarz):** for these small, compact molecules no quartet falls below 10⁻¹². R3 ≈ R2; the differences of up to ~20% on sub-second runs are run-to-run noise.
  The first skips appear for benzene (28,403 of 692,142 shell quartets, 4%).
  Screening only pays off for larger, spatially extended molecules, where the number of significant integrals grows ~N² instead of N⁴.
* **R3→R4 (Boys table, 1.6× for C₂H₄):** one Taylor series for the highest order plus a cheap downward recursion
  replaces one 20–80-term power series per order n.
* **R4→R5 (vectorization, 2.4× for C₂H₄):** the Python loop over primitive pairs is gone.
  The remaining cost is ~139 µs of Python/NumPy call overhead *per shell quartet* (benzene: 93 s / 663,739 quartets).
  Most shell quartets hold only a handful of primitive quartets, so NumPy arrays of length 1–36 cannot amortize their per-call overhead.
  Removing that overhead requires compiled code, which is stage 2 (C++/Rust).

### J/K build and diagonalization: loops vs BLAS/LAPACK (one call, converged density / Fock matrix)

| molecule | N | J/K loop (s) | J/K einsum (s) | J/K BLAS (s) | loop/BLAS | Jacobi eigh (s) | LAPACK eigh (s) | Jacobi/LAPACK |
|---|---|---|---|---|---|---|---|---|
| h2o | 19 | 0.0952 | 0.000255 | 0.00012 | 793× | 0.0406 | 5.97e-05 | 681× |
| nh3 | 21 | 0.115 | 0.000287 | 0.000177 | 648× | 0.0535 | 7.1e-05 | 753× |
| ch4 | 23 | 0.165 | 0.000384 | 0.00025 | 659× | 0.0736 | 6.72e-05 | 1,095× |
| c2h4 | 38 | 1.22 | 0.00268 | 0.00173 | 705× | 0.295 | 0.000198 | 1,491× |
| benzene | 102 | 62.3 | 0.15 | 0.136 | 457× | 8.23 | 0.000818 | 10,065× |

J and K read the whole N⁴ tensor once per SCF iteration (benzene: 102⁴ × 8 B = 0.87 GB). They do 2 FLOPs per 8-byte element, i.e. 0.25 FLOP/byte,
far below the ridge point of 8.6 FLOP/byte, so this step is **memory-bandwidth bound**. The BLAS version needs 0.14 s for two passes over 0.87 GB
(≈12.7 GB/s, 93% of the STREAM triad bandwidth, measured on a different node of the same model). This is about as fast as this kernel can run on one core; the next speed-ups must reduce memory traffic (for example, never store G). The Python loop is interpreter-bound and 457× slower.
`einsum` without `optimize=` is within 1.1–2.2× of the BLAS route. Jacobi diagonalization is O(N³) per sweep in Python;
LAPACK `dsyevd` uses Householder tridiagonalization plus divide-and-conquer in compiled, blocked code. At N = 102 the gap is ~10,065×.

## Commands (laptop: macOS or Linux)

```bash
cd chem179-hf-perf
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_numpy_opt.py          # ~5 s: every ERI rung vs PySCF references

# single-thread runs (otherwise BLAS uses all cores)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m hfperf h2o --backend numpy                 # default = rung R5, ~0.3 s
python -m hfperf.backends.numpy_opt.bench --molecules h2o nh3 --rungs R1 R2 R3 R4 R5 --out my_eri.json
python -m hfperf.backends.numpy_opt.bench --molecules h2o c2h4 --kernels --out my_kernels.json
# R0 (all N^4 integrals, stage-0 code) takes 1–3 min for H2O on a laptop (150 s on Pete); add --rungs R0 if you are patient

# matmul ladder
cd benchmarks/matmul
python py_ladder.py --sizes 64 128 192 --out my_py.csv          # ~1 min
make                                       # Linux (gcc).  macOS: make CC=clang NATIVE=-mcpu=native EXT=dylib
python c_ladder.py --builds O2 O3native --sizes 128 256 512 1000 1024 --out my_c.csv
python c_ladder.py --builds O2 O3native --machine --out my_machine.json   # STREAM triad + FMA peak
```

Notes for laptops:
* Apple Silicon has no AVX-512. `peak_fma` uses GCC/Clang vector extensions, so it still compiles (as 4 × 128-bit NEON ops per vector);
  the peak FLOP/cycle formula changes (M-series P-cores: 4 × 128-bit FMA pipes → 16 DP FLOP/cycle).
  `-mprefer-vector-width=512` is an x86 GCC flag, so skip the `O3native512` build there (`make libmm_O2.dylib libmm_O3native.dylib EXT=dylib ...`).
* macOS NumPy wheels use OpenBLAS (or Accelerate in recent builds). Set `VECLIB_MAXIMUM_THREADS=1` as well.
* Windows: use WSL.

## Files

* `src/hfperf/backends/numpy_opt/`: `backend.py` (switches), `eri.py` (ladder), `shells.py` (shell / shell-pair data),
  `hermite.py` (vectorized E tables and R recursion), `boys.py` (table + Taylor + downward recursion), `onee.py` (S, T, V), `bench.py` (driver).
* `tests/test_numpy_opt.py`: 25 tests (h2/h2o), all pass locally (Windows, Python 3.12) and on Pete (Linux, Python 3.10).
* `benchmarks/matmul/`: `matmul.c`, `Makefile`, `py_ladder.py`, `c_ladder.py`, `common.py`.
* `results/memory/`: `py_ladder.csv`, `c_ladder.csv` (tidy: variant, N, seconds, gflops, cpu), `machine*.json`, `eri_*.json` (raw),
  `eri_ladder.csv` and `jk_eigh.csv` (tidy), `raw_lscpu_*.txt`.
