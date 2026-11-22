# Stage 2a — C++ kernels via pybind11

**What changes:** only two methods of the backend, `eri(funcs)` and `jk(G, D)`, move from Python to C++17
(`ext/cpp/hfcpp.cpp`, exposed with pybind11). `CppBackend` subclasses `NaiveBackend`, so the one-electron
integrals, matrix multiplies and Jacobi diagonalization are still the stage-0 Python loops. Any change in
run time is therefore caused by these two kernels alone.

| file | role |
|---|---|
| `ext/cpp/hfcpp.cpp` | the kernels + Python bindings (436 code lines; see LOC below) |
| `ext/cpp/build.py` | one-command build (g++ / Apple clang), puts `_hfcpp*.so` into `src/hfperf/backends/cpp/` |
| `src/hfperf/backends/cpp/backend.py` | `CppBackend` (and a `CppOmpBackend` stub for the threading stage) |
| `ext/cpp/bench.py`, `ext/cpp/collect.py` | benchmark driver and CSV aggregation |
| `tests/test_cpp.py` | correctness checks (skip automatically if the module is not built) |

## Three ERI variants: separating *language* from *algorithm*

| variant | what it does | how to select |
|---|---|---|
| `naive` | **the same algorithm as stage 0**, line by line: one contracted quartet at a time, recursive Hermite `E` and `R`, series Boys function, all N⁴ quartets | `CppBackend(eri_algorithm="naive")` |
| `sym8-noreuse` | same per-quartet kernel, but only unique quartets `(ij|kl)` with i≥j, k≥l, ij≥kl (≈N⁴/8) are computed and copied to the 8 equivalent positions; quartets with Schwarz bound `Q_ij·Q_kl < 1e-12` are skipped | `CppBackend(reuse=False)` |
| `sym8` (default) | as above + **reuse**: for every basis-function pair (ij) the primitive-pair data (p, P, and the products E_t·E_u·E_v·c_a·c_b) are computed once and stored; the R table is built iteratively (no recursion, no recomputation) from a single Boys evaluation F_L(T) followed by downward recursion | `CppBackend()` / `--backend cpp` |

J/K comes in three loop orders over the stored N⁴ tensor:
`jk_naive` (the Python loop order), `jk_twopass` (one pass for J, one for K) and `jk` (default; one fused pass in which every
row `G[i,k,j,:]` is loaded once and used for both `J[i,k]` and `K[i,j]`).

## Why it is faster

1. **Interpreter overhead disappears (naive → C++ naive, ~48×).** In Python every `a*b` is a dynamic type check, a
   heap-allocated float object and a reference-count update; a recursive call costs ~100 ns of bookkeeping. In C++ the
   same multiply is one machine instruction on registers, and a function call is a few cycles. The *algorithm* has not
   changed: the same 130,321 quartets, the same recursion tree.
2. **Doing less work (symmetry, ~7×).** (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij) = … — for H₂O 18,145 unique quartets instead of
   130,321 (ratio 7.2). Schwarz screening removes quartets whose upper bound is below 1e-12; it does nothing for the small
   molecules and removes 1.3 % (benzene) / 16 % (naphthalene) of unique quartets here.
3. **Not recomputing (reuse, ~10×).** The recursive `hermite_E` is re-evaluated for every (t,u,v,τ,ν,φ) combination and the
   recursive `hermite_R` re-evaluates the Boys function at every leaf of its tree. Tabulating them once per pair / per
   primitive quartet removes this redundancy. This is an *algorithmic* change that is just as available in Python — the
   language only decides how expensive the redundancy is.
4. **J/K is limited by memory traffic, not arithmetic.** Each element of G (8 bytes) is used for one multiply-add. Reading
   0.87 GB (benzene) at the ~5–6 GB/s one core can stream takes ~0.16 s — and that is exactly the measured time of the
   fused kernel. The two-pass version reads G twice and is 1.7× slower. Our first "cache-friendly" rewrite (the two-pass
   one) was *slower* than the original loop order, because the Python order was already unit-stride in its inner loop; the
   number of passes over memory mattered more than the loop order. (A measured counter-example worth showing.)

### Compiler flags (the teaching point)

`-O0` → `-O2` makes `eri_naive` only 1.45× faster and `-O3 -march=native` adds nothing. The naive kernel spends its time in
`exp`/`pow` library calls, deep recursion and a chain of dependent divisions in the Boys series — none of which the
optimizer can remove or vectorize. The table-driven `sym8` kernel has simple loops over arrays, so the optimizer has more to
work with (2.5× from `-O0` to `-O2`). `-march=native` gives no measurable gain on either kernel, because neither contains
loops the compiler vectorizes. **Flags cannot rescue an algorithm; algorithm changes dominate.** We avoid `-ffast-math`: it
allows the compiler to reorder sums, which breaks the bit-reproducibility of the 1e-10 accuracy contract checks.

### Amdahl's law

With `sym8`, the ERI and J/K kernels are no longer the bottleneck: 71–88 % of the remaining SCF time is in the
Python one-electron integrals, Jacobi diagonalization and triple-loop matmul that were deliberately *not* ported
(benzene: diag 97 s, matmul 25 s, S/T/V 18 s vs ERI 48 s and J/K 2 s). Further speed-ups must come from those kernels
(the NumPy/LAPACK stage) — the total time cannot drop below the time of the parts you did not touch.

## Commands (macOS and Linux; Windows users: inside WSL)

```bash
# one-time: a C++17 compiler and pybind11
#   macOS:  xcode-select --install          Ubuntu/WSL:  sudo apt install g++
python -m pip install pybind11

python ext/cpp/build.py              # -> src/hfperf/backends/cpp/_hfcpp*.so  (-O3 + native CPU tuning)
python -m pytest -q tests/test_cpp.py
python -m hfperf h2o --backend cpp   # full RHF; the timing table shows S,T,V / ERI / JK / diag / matmul

# separate language from algorithm
python ext/cpp/bench.py scf h2o --variant naive      # stage-0 algorithm, compiled
python ext/cpp/bench.py eri h2o --variant sym8-noreuse
python ext/cpp/bench.py scf h2o --variant sym8

# compiler flags: builds _hfcpp_O0/_O2/_O3/_native and times eri on H2O
python ext/cpp/build.py --all-profiles --verbose
python ext/cpp/bench.py flags h2o

# memory traffic in J/K
python ext/cpp/bench.py jk h2o c2h4 benzene
python ext/cpp/bench.py eri-slab benzene --i0 0 --i1 26   # part of the naive benzene tensor (~25 min)
python ext/cpp/collect.py            # results/cpp/*.csv
python ext/cpp/loc.py                # lines of code per kernel
```

Build details: one compiler call, `c++ -O3 -march=native -std=c++17 -shared -fPIC -fvisibility=hidden` plus the
pybind11 and Python include paths; on macOS `-undefined dynamic_lookup` is added (Python symbols are resolved at import
time), and on Apple silicon `-mcpu=native` is used if the compiler rejects `-march=native` (the script probes). No OpenMP
yet. Native Windows/MSVC is not supported (use WSL). **Tested on Linux (GCC 13.2, Pete); the macOS path has not been run
yet.**

Memory: the full tensor needs 8·N⁴ bytes — 0.87 GB for benzene (102 bf), 6.07 GB for naphthalene (166 bf). On an 8 GB
laptop naphthalene will swap or fail; stop at benzene.

## Measured numbers

Single core (`-c 1`, `OMP/OPENBLAS/MKL_NUM_THREADS=1`), Intel Xeon Gold 6130F @ 2.10 GHz (OSU Pete), GCC 13.2.0,
Python 3.10.14, NumPy 2.2.6. Default build `-O3 -march=native`. All energies from the full SCF; "|ΔE|" is versus the
PySCF reference in `tests/data/reference_energies.json`. Raw JSON: `results/cpp/raw/`, tables: `results/cpp/{scf,eri,flags,jk}.csv`.

### ERI time (s)

| molecule | N | Python naive (stage 0) | C++ naive | C++ sym8-noreuse | C++ sym8 | sym8 quartets computed / screened |
|---|---|---|---|---|---|---|
| H₂O | 19 | 193.5 | 4.03 | 0.571 | 0.054 | 18,145 / 0 |
| NH₃ | 21 | – | 5.54 | 0.763 | 0.080 | 26,796 / 0 |
| CH₄ | 23 | – | 7.15 | 0.998 | 0.115 | 38,226 / 0 |
| C₂H₄ | 38 | – | 83.5 | 11.1 | 0.943 | 274,911 / 0 |
| benzene | 102 | – | 5,338 ‡ | 766 | 47.7 | 13,621,436 / 178,195 |
| naphthalene | 166 | – | not run (est. 7–9 h from 31–40 µs/quartet × 166⁴) | not run | 214 | 80,658,247 / 15,412,344 |

‡ Benzene C++ naive: the full N⁴ tensor (108,243,216 quartets) was computed as four row-slabs `G[i0:i1]` in four separate
single-core jobs (Pete's express partition has a 1 h limit and the batch queue did not start the job); 5,338 s is the **sum of
the measured slab times** (1648 + 1418 + 1310 + 962 s, all on Xeon Gold 6130F), not an extrapolation. Every slab agrees with
the `sym8` tensor to 1.0e-12. A single-process benzene SCF with `eri_algorithm="naive"` was **not run**; the "≈ 5,484 s" SCF
total is a composite (naive ERI time + the measured non-ERI time of the sym8 run) and its energy is not separately
measured. Raw data: `results/cpp/raw/slab_benzene_naive_*.json`, `results/cpp/eri_slabs.csv`.
On benzene, sym8 is 112× faster than the C++ naive port and 7.0× of that comes from symmetry alone.

H₂O ladder: Python → C++ (same algorithm) **48×**; + symmetry/screening **7.1×**; + reuse **10.6×**; total **3,590×**.
Python costs 1.48 ms per contracted quartet, the C++ port of the same code 31 µs.

### Total SCF time (s) and accuracy

| molecule | N | variant | wall | ERI | J/K per call | S,T,V + diag + matmul (Python) | iterations | \|ΔE\| (Eh) |
|---|---|---|---|---|---|---|---|---|
| H₂O | 19 | Python naive | 195.7 | 193.5 | 0.100 | 0.79 | 12 | 5.7e-14 |
| H₂O | 19 | C++ naive | 4.93 | 4.03 | 0.00020 | 0.78 | 12 | 8.5e-14 |
| H₂O | 19 | C++ sym8 | 0.961 | 0.054 | 0.00018 | 0.79 | 12 | 1.3e-13 |
| NH₃ | 21 | C++ naive | 7.26 | 5.54 | 0.00026 | 1.58 | 12 | 7.1e-14 |
| NH₃ | 21 | C++ sym8 | 1.73 | 0.080 | 0.00026 | 1.52 | 12 | 1.4e-14 |
| CH₄ | 23 | C++ naive | 9.19 | 7.15 | 0.00037 | 1.85 | 11 | 2.8e-14 |
| CH₄ | 23 | C++ sym8 | 2.17 | 0.115 | 0.00039 | 1.88 | 11 | 9.9e-14 |
| C₂H₄ | 38 | C++ naive | 90.2 | 83.5 | 0.0027 | 6.28 | 11 | 3.4e-13 |
| C₂H₄ | 38 | C++ sym8 | 8.00 | 0.943 | 0.0029 | 6.63 | 11 | 3.1e-13 |
| benzene | 102 | C++ naive | ≈ 5,484 (composite ‡) | 5,338 ‡ | – | – | – | – |
| benzene | 102 | C++ sym8 | 193.3 | 47.7 | 0.160 | 140.5 | 12 | 3.8e-12 |
| naphthalene | 166 | C++ sym8 | 843.8 | 214.0 | 1.14 | 602.0 | 15 | 5.0e-12 |

All energies agree with PySCF to < 1e-8 Eh (largest deviation 5.0e-12 Eh). ERI tensors agree with PySCF elementwise to
2.2e-14 (H₂O, NH₃, all variants).

### J/K contraction, one call (s)

| molecule | N | size of G | Python loops | C++ Python-order | C++ two-pass | C++ fused |
|---|---|---|---|---|---|---|
| H₂O | 19 | 1.0 MB | 0.0994 | 0.000213 | 0.000292 | 0.000152 |
| C₂H₄ | 38 | 16.7 MB | – | 0.00360 | 0.00451 | 0.00263 |
| benzene | 102 | 0.87 GB | – | 0.198 | 0.277 | 0.159 |
| naphthalene | 166 | 6.07 GB | – | 1.42 | 1.81 | 1.12 |

### Compiler flags, H₂O ERI (best of 3, s)

| flags | C++ naive | C++ sym8 |
|---|---|---|
| `-O0` | 6.10 | 0.124 |
| `-O2` | 4.20 | 0.0492 |
| `-O3` | 3.91 | 0.0518 |
| `-O3 -march=native` | 4.09 | 0.0564 |

Differences of ≲ 5–10 % between `-O2`, `-O3` and `-O3 -march=native` are within node-to-node noise (an earlier run on a
Xeon Gold 6226R gave 3.12 / 2.99 / 3.04 s, see `results/cpp/raw/supplementary/`).

### Lines of code (non-blank, non-comment; `python ext/cpp/loc.py`)

| kernel | C++ | Python stage 0 |
|---|---|---|
| naive ERI (Boys, E, R, primitive, contraction, tensor) | 93 | integrals.py 156 (incl. one-electron) + boys.py 23 |
| sym8 + Schwarz + pair reuse | 175 | – |
| J/K (three loop orders) | 57 | fock.py 18 |
| whole file incl. bindings | 436 | |

## Notes and caveats

* Timings are single runs (SCF) or best-of-3 (flags, J/K); expect ±5–10 % between nodes of the same type.
* Python-stage-0 numbers were only measured for H₂O (195.7 s total); larger molecules take hours in pure Python.
* Screening threshold 1e-12 with exact zeros for skipped quartets; energies remain within 5e-12 Eh of PySCF.
* The tensor is still stored in full (N⁴) so that `jk(G, D)` keeps the stage-0 interface; a direct-SCF or
  unique-quartet J/K would cut memory 8× and is left for later stages.
