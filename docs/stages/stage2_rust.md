# Stage 2b: the same kernel in Rust (PyO3 + maturin)

**Files:** `ext/rust/` (crate, `Cargo.toml`, `pyproject.toml`, `bench_rust.py`, `loc.py`),
`src/hfperf/backends/rust/` (Python side), `tests/test_rust.py`, raw data in `results/rust/`.

Stage 2a moved the integral kernel into C++. This stage writes **the same algorithms in Rust**.
That gives us one more data point for "compiled code is fast", and one new question:
*what does a memory-safe language cost?*

## 1. What changed compared with the previous stage

| piece | stage 0 (naive Python) | stage 2b (`--backend rust`) |
|---|---|---|
| ERI `(ij\|kl)` | recursive McMurchie–Davidson in Python | the same recursion, compiled (`eri.rs`), plus symmetry/reuse variants |
| J, K | 4 nested Python loops | `fock.rs`, three flavours (safe iterator, checked index, unchecked index) |
| S, T, V | Python | the same algorithm compiled (`one_e.rs`) |
| matmul, eigh | Python triple loop, Jacobi | NumPy/LAPACK (the stage-1 choice). `RustBackend(linalg="naive", onee="naive")` restores stage 0 ("strict" mode, as in CppBackend) |
| SCF driver | unchanged | unchanged (`scf.py` never knows which language did the work) |

The Python side is a thin wrapper. `RustBackend(NaiveBackend)` packs the basis with
`pack_basis(funcs)`, which gives flat NumPy arrays, calls into Rust, and gets NumPy arrays back.
The Rust code holds no Python objects. It borrows the NumPy buffers as `&[f64]` slices and
releases the GIL (`py.allow_threads`) while it computes.

Kernel variants (same interface; give an instance to `rhf(mol, RustUncheckedBackend())`):

| benchmark label | Python class | Rust entry point | what it does |
|---|---|---|---|
| `naive` | `RustNaiveBackend` | `eri_naive` | line-by-line translation of stage 0: every one of the N⁴ quartets, recursive E and R, series Boys |
| `sym8_noreuse` | `RustSym8Backend` | `eri_sym8(reuse=False)` | the identical quartet routine, but only for unique quartets (i≥j, k≥l, ij≥kl), with Schwarz screening (1e-12) |
| `sym8_reuse` (**default**, `--backend rust`) | `RustBackend` | `eri_sym8(reuse=True, checked=True)` | Hermite E coefficients tabulated once per primitive pair. Boys F₀..F_L from one series plus downward recursion. R table filled iteratively. Bounds-checked indexing |
| `sym8_reuse_unchecked` | `RustUncheckedBackend` | `eri_sym8(..., checked=False)`, `jk(mode="unchecked")` | the same kernel compiled with `get_unchecked` (unsafe) in the hot loops |

`eri_sym8(reuse=…)` deliberately mirrors `CppBackend(eri_algorithm="sym8", reuse=…)` from stage 2a.

## 2. Measured numbers

All timings are **single core** (`#SBATCH -c 1`, `OMP/OPENBLAS/MKL_NUM_THREADS=1`) on one OSU
Pete node (n183, Intel Xeon Gold 6226R @ 2.90 GHz). The node is shared with other users' jobs, so
treat differences below ~7% as noise. The build is portable (`maturin develop --release`, generic
x86-64, no `-C target-cpu=native`). rustc 1.100.0, maturin 1.16.0, PyO3 0.22, numpy crate 0.22, Python 3.10,
NumPy 2.2.6. The raw data (CSV, JSON and `lscpu` for every job) is in `results/rust/raw/`. The
merged tables are `results/rust/summary_*.csv`, `jk_micro.csv` and `safety_bounds_checks.csv`.

### 2.1 ERI time (seconds)

| molecule | N_bf | naive | sym8_noreuse | sym8_reuse (default) | sym8_reuse_unchecked |
|---|---|---|---|---|---|
| h2o | 19 | 1.919 | 0.334 | 0.134 | 0.080 |
| nh3 | 21 | 2.480 | 0.334 | 0.148 | 0.092 |
| ch4 | 23 | 3.457 | 0.461 | 0.200 | 0.127 |
| c2h4 | 38 | 42.33 | 5.538 | 1.721 | 1.080 |
| benzene | 102 | 3073.5 | 380.5 | 92.18 | 56.73 |
| naphthalene | 166 | — | 2143.4 | 496.8 | 305.1 |

Naive naphthalene was **not run**. At the measured ~28 µs per quartet it would take about
5.5 h (7.6×10⁸ quartets), which is more than the 1 h `express` limit.

### 2.2 Total SCF wall time (seconds; includes ERI)

| molecule | N_bf | SCF iters | naive | sym8_noreuse | sym8_reuse | sym8_reuse_unchecked | sym8_reuse **strict** |
|---|---|---|---|---|---|---|---|
| h2o | 19 | 12 | 1.988 | 0.411 | 0.207 | 0.154 | 1.041 |
| nh3 | 21 | 12 | 2.557 | 0.410 | 0.222 | 0.166 | 1.399 |
| ch4 | 23 | 11 | 3.557 | 0.558 | 0.298 | 0.226 | 1.780 |
| c2h4 | 38 | 11 | 42.56 | 5.762 | 1.946 | 1.311 | 6.883 |
| benzene | 102 | 12 | 3077.4 | 384.2 | 95.88 | 60.52 | 204.3 |
| naphthalene | 166 | 15 | — | 2166.2 | 519.0 | 327.5 | — |

"Strict" means only ERI and J/K are in Rust; S, T, V, matmul and the Jacobi diagonaliser stay in
Python, as in CppBackend. For benzene that adds ~110 s on top of the same 91 s of ERI. Every
energy matches the PySCF reference: the largest |E − E_ref| over all runs in the tables is
1.5e-11 Eh (contract: 1e-8). Every run converged.

### 2.3 Same algorithm, different language (vs. stage 0 on the same CPU model)

Stage-0 numbers are from `results/naive/naive_baseline.csv`. That file was measured by the
naive-stage track, also on a Xeon Gold 6226R (single core), but on a different Pete node.

| molecule | Python ERI (s) | Rust `eri_naive` (s) | speed-up | Python J/K per SCF (s) | Rust J/K per SCF (s) | speed-up |
|---|---|---|---|---|---|---|
| h2o | 150.4 | 1.92 | 78x | 1.015 | 0.0019 | 524x |
| nh3 | 207.6 | 2.48 | 84x | 1.502 | 0.0026 | 580x |
| ch4 | 270.8 | 3.46 | 78x | 1.920 | 0.0041 | 467x |
| c2h4 | 3237.7 | 42.33 | 76x | 14.614 | 0.0237 | 616x |

### 2.4 Cost per computed quartet (µs)

| molecule | naive | sym8_noreuse | sym8_reuse | sym8_reuse_unchecked |
|---|---|---|---|---|
| h2o | 14.7 | 18.4 | 7.4 | 4.4 |
| nh3 | 12.8 | 12.5 | 5.5 | 3.4 |
| ch4 | 12.4 | 12.1 | 5.2 | 3.3 |
| c2h4 | 20.3 | 20.1 | 6.3 | 3.9 |
| benzene | 28.4 | 27.9 | 6.8 | 4.2 |
| naphthalene | — | 26.6 | 6.2 | 3.8 |

Schwarz screening (threshold 1e-12) skips these unique quartets:

| molecule | computed | screened | screened share |
|---|---|---|---|
| h2o | 18,145 | 0 | 0.0% |
| nh3 | 26,796 | 0 | 0.0% |
| ch4 | 38,226 | 0 | 0.0% |
| c2h4 | 274,911 | 0 | 0.0% |
| benzene | 13,621,436 | 178,195 | 1.3% |
| naphthalene | 80,658,247 | 15,412,344 | 16.0% |

### 2.5 Safety experiment: bounds checks on/off

ERI time with the reuse kernel, `checked=True` vs. `checked=False`:

| molecule | build | checked (s) | unchecked (s) | checked / unchecked |
|---|---|---|---|---|
| h2o | portable | 0.134 | 0.080 | 1.68 |
| h2o | native | 0.133 | 0.086 | 1.54 |
| nh3 | portable | 0.148 | 0.092 | 1.62 |
| ch4 | portable | 0.200 | 0.127 | 1.57 |
| c2h4 | portable | 1.721 | 1.080 | 1.59 |
| c2h4 | native | 1.719 | 1.163 | 1.48 |
| benzene | portable | 92.18 | 56.73 | 1.62 |
| benzene | native | 91.73 | 61.00 | 1.50 |
| naphthalene | portable | 496.8 | 305.1 | 1.63 |

J/K at the converged density (best of 3, milliseconds). `iter` is the idiomatic safe version:

| molecule | N_bf | iter (safe) | checked index | unchecked index |
|---|---|---|---|---|
| h2o | 19 | 0.14 | 0.19 | 0.18 |
| nh3 | 21 | 0.17 | 0.23 | 0.22 |
| ch4 | 23 | 0.25 | 0.33 | 0.31 |
| c2h4 | 38 | 1.93 | 2.40 | 2.36 |
| benzene | 102 | 152.77 | 180.52 | 162.06 |
| naphthalene | 166 | 986.01 | 1069.65 | 1010.94 |

### 2.6 Portable vs. `-C target-cpu=native` build

| molecule | variant | portable (s) | native (s) | portable / native |
|---|---|---|---|---|
| h2o | sym8_noreuse | 0.334 | 0.363 | 0.92 |
| h2o | sym8_reuse | 0.134 | 0.133 | 1.01 |
| h2o | sym8_reuse_unchecked | 0.080 | 0.086 | 0.93 |
| c2h4 | sym8_noreuse | 5.538 | 5.864 | 0.94 |
| c2h4 | sym8_reuse | 1.721 | 1.719 | 1.00 |
| c2h4 | sym8_reuse_unchecked | 1.080 | 1.163 | 0.93 |
| benzene | sym8_noreuse | 380.5 | 405.2 | 0.94 |
| benzene | sym8_reuse | 92.18 | 91.73 | 1.00 |
| benzene | sym8_reuse_unchecked | 56.73 | 61.00 | 0.93 |

A native-CPU build (AVX-512 is available on this Xeon) gives **no gain**: the ratios are 0.92–1.01,
which is within the noise. This is unlike much numerical code. The ERI kernel is dominated by
`exp`, `sqrt`, divisions and short, data-dependent loops of 1–27 elements, so the compiler finds
nothing to vectorise. Note that the C++ track builds with `-march=native` by default; this table is
why that difference should not matter for this kernel.

Build cost: a clean `maturin build --release` (all dependencies plus the crate, fat LTO) took
**24 s wall on 4 cores** (`results/rust/raw/build_portable.log`). Rebuilds after editing one `.rs`
file take roughly 10–15 s. The wheel is tagged `manylinux2014`, so it runs on glibc ≥ 2.17.

## 3. Why it is faster: where the time went

1. **Interpreter overhead (≈ 80× for the identical algorithm).** In Python, every `a*b`, every
   function call in the E/R recursion and every loop iteration is an operation on boxed objects.
   Python looks up the type, dispatches, and allocates a new float object. The compiled
   `eri_naive` does *exactly the same arithmetic*, but each step is one or two machine
   instructions on values held in registers. Nothing else changed, and that alone takes H₂O from
   150 s to 1.9 s.
2. **Do less work: 8-fold permutational symmetry (≈ 8×).** (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij) = …
   so only ~N⁴/8 quartets are unique. The per-quartet cost does not change (28.4 vs 27.9 µs for
   benzene), so the speed-up is just the ratio of quartet counts: 1.08×10⁸ / 1.36×10⁷ ≈ 8.0.
   Schwarz screening removes another 1.3% (benzene) to 16% (naphthalene). It matters more as
   molecules get bigger, because distant charge distributions do not overlap.
3. **Do not recompute (≈ 4×).** The recursive kernel recomputes the same E^{ij}_t coefficients
   for every (t,u,v,τ,ν,φ) term and calls the Boys series at every leaf of the R recursion. The
   reuse kernel builds E once per *primitive pair* and keeps it for every quartet that contains the
   pair. It gets F₀..F_L from one series evaluation, and fills R iteratively into a small table.
   This is an algorithm change, not a language change, and it would help Python just as much.
4. **Bounds checks (≈ 1.5–1.7× in the ERI kernel, ≈ 1.1× in J/K).** See section 4.

What did *not* help: `target-cpu=native` (section 2.6). The J/K build is now limited by memory
bandwidth. It streams the 0.85 GB benzene tensor in 0.15 s (~5.5 GB/s on one core), about the
same as NumPy `einsum` on this machine (0.150 s in `results/memory/kernels_benzene.json`). This
backend's ERI time grows as ~N⁴ (benzene → naphthalene: ×5.4 for 1.63× more basis functions; the
pure N⁴ prediction is ×7.0, and screening makes up the difference). So the ERI kernel stays the
bottleneck and is the target for threads (stage 4).

## 4. Memory safety vs. speed: what a chemist should take away

In C, C++ or Fortran, `R[i]` with an `i` past the end of the array silently reads whatever happens
to be in memory there. The program does not crash, and you get a slightly wrong integral. That is
the worst possible failure in a quantum-chemistry code: an energy that is wrong in the 6th digit
with no error message. Rust makes `R[i]` check `i < R.len()` and **stop the program with a clear
message** if it is not. Safe Rust has no way to read out of bounds.

The experiment (`ext/rust/src/idx.rs`): every array read in the hot loops of `eri_tab.rs` and
`fock.rs` goes through `ld::<CHECKED>(slice, i)`. With `CHECKED = true` that is `slice[i]`, which is
checked. With `CHECKED = false` it is `unsafe { *slice.get_unchecked(i) }`, which is unchecked,
like C. `CHECKED` is a compile-time constant, so the compiler builds two copies of the kernel from
**one** source. The test suite runs both and requires identical results to 1e-10.

Results:

* **ERI reuse kernel: checked is ~1.6× slower** (1.57–1.68 for the portable build across all molecules; 1.48–1.54 for the native-CPU build). The innermost loop
  multiplies tiny tables (≤ 27 E coefficients × a few hundred R values) with indices like
  `((t+τ)*s + u+ν)*s + v+φ`. The compiler cannot prove these are in range, so every load gets a
  compare-and-branch. Worse, a possible panic is a side effect, and that stops the compiler from
  reordering, unrolling or merging the loads.
* **J/K: almost free.** Writing it idiomatically (slice once with `&g[base..base+n*n]`, then
  `zip` two iterators) costs *one* check per row, and that version is the **fastest of the three**:
  153 ms for benzene, vs. 181 ms checked-index and 162 ms unchecked-index. The safe iterator form
  tells the compiler the lengths up front, which is better than switching checks off.
* Lesson: write the hot loop so that the compiler can see the bounds (slices, iterators,
  fixed-size arrays). Reach for `unsafe` only where profiling shows checks still cost something,
  keep it to a few lines (here, 2 lines in `idx.rs`), and test the checked build of the same code.
  For a production integral code, a 1.6× gain in one kernel is often worth an audited `unsafe`
  block. For a code you are still developing, the checked build is the one that catches your
  off-by-one in the d-function indexing on day one.
* What Rust checks that C++ does not, beyond indices: no use-after-free or dangling references
  (the borrow checker). No data races, which matters for stage 4: Rayon will refuse to compile a
  parallel loop where two threads write the same `J[i][j]`. No uninitialised memory. Integer
  overflow is checked in debug builds.

## 5. Code size (non-blank, non-comment lines; `python ext/rust/loc.py`)

| part | Rust lines |
|---|---|
| ERI + J/K kernel (`basis`, `boys`, `hermite`, `eri`, `eri_tab`, `fock`, `idx`) | 482 |
| one-electron S, T, V (`one_e.rs`, not in the C++ track) | 82 |
| PyO3 bindings (`lib.rs`) | 117 |
| **total** | **681** |
| Python wrapper (`backends/rust/backend.py`) | ~90 |

For comparison, `ext/cpp/hfcpp.cpp` (ERI + J/K + pybind11 bindings, no S/T/V) had 436
non-blank, non-comment lines when this was written. That file belongs to the C++ track and may
still change, so re-count with `python ext/cpp/loc.py`. The Rust version is longer mainly
because of type annotations on every function signature, explicit `as f64` / `as usize`
conversions (Rust never converts numbers implicitly), and the checked/unchecked machinery.

## 6. Prepared for Rayon (stage 4)

`eri_contracted(bs, i, j, k, l)` and `quartet(bra, ket, …)` are pure functions: they read the basis
and the pair tables and write nothing. All writes happen in the outer loops (`scatter8`), and the
only per-thread state is the `Work` scratch struct (Boys, power and R tables). Stage 4 will turn
the outer `i` loop into `(0..n).into_par_iter()`, with one `Work` per thread, and collect the unique
integrals before scattering. In `fock.rs`, each (i, j) element of J and K is independent. The
registry name `rust-rayon` already points at `RustRayonBackend`, which for now raises
`NotImplementedError`.

## 7. Run it yourself (macOS or Linux; Windows → use WSL)

```bash
# 1. Rust toolchain (user-local, no sudo).  macOS: run `xcode-select --install` first (linker).
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
rustc --version

# 2. Build tool, inside the same virtualenv/conda env you use for hfperf
python -m pip install maturin
python -m pip install -e ".[dev]"            # from the repo root, if not done yet

# 3. Build and install the extension into the active environment (~30 s the first time)
cd ext/rust
maturin develop --release                   # needs an activated venv or conda env
cd ../..
#    alternative without an activated env:  python -m pip install ./ext/rust

# 4. Check it
python -m pytest -q tests/test_rust.py      # 25 tests, a few seconds
python -m hfperf h2o --backend rust         # default kernel; compare with --backend naive

# 5. Benchmark the variants (single core)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python ext/rust/bench_rust.py --variants naive sym8_noreuse sym8_reuse sym8_reuse_unchecked \
       --molecules h2o nh3 ch4 c2h4 --jk-micro --out results/rust/my_laptop
python ext/rust/loc.py
```

Notes:
* If `maturin develop` says "Couldn't find a virtualenv or conda environment", activate one, or use
  the `pip install ./ext/rust` line.
* After editing a `.rs` file, rerun `maturin develop --release`. Python imports the new module
  only in a **new** interpreter.
* Without `--release` you get an unoptimised debug build that is much slower (not measured here).
  Always benchmark release builds.
* Try it (exercise; not run for this document): in `fock.rs`, function `jk_index`, change
  `for kl in 0..n * n` to `for kl in 0..n * n + 1`, rebuild, then call
  `hfperf_rust.jk(G, D, mode="checked")` and `mode="unchecked"` on small arrays. The checked build
  should stop with Rust's `index out of bounds` message, which PyO3 raises in Python as
  `pyo3_runtime.PanicException`. The unchecked build reads past the end of `D`: it returns a wrong
  J, or crashes. Undo the change afterwards. A subtler point: an off-by-one inside one record
  of a big flat table (e.g. `0..=bx` in `eri_tab.rs::quartet`) lands *inside* the slice, in the next
  record, so no bounds check catches it. Only the tests against PySCF do. Bounds checks protect
  memory, not physics.
* `-C target-cpu=native` (`RUSTFLAGS="-C target-cpu=native" maturin develop --release`) made no
  difference on our test machine (section 2.6). Feel free to try it on your laptop.

## 8. What was not done or is not comparable

* Naive-kernel naphthalene was not run (estimated at ~5.5 h; see 2.1).
* No C++ timings are quoted here. When this was written there was no `results/cpp/` data on the
  same node. When it exists, compare like with like: ERI and J/K columns directly; SCF totals only
  against the **strict** column, because by default RustBackend also moves S, T, V and the linear
  algebra out of Python.
* The stage-0 comparison in 2.3 uses another track's measurement on the same CPU model but a
  different node.
* One early run of the naive kernel landed on a Xeon Gold 6130F node (`results/rust/raw/naive_small.csv`).
  It is kept for transparency but not used in the tables. It was ~20–28% slower than n183
  (2.1 vs 2.9 GHz).
