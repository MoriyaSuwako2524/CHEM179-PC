//! Stage 4: shared-memory threads with Rayon.
//!
//! Same work unit as the C++/OpenMP version: one unique pair row ij (all kl <= ij).
//!
//!  eri_sym8_par   the tabulated sym8 kernel (eri_tab.rs) over rows in parallel.
//!                 Safe Rust will not let two threads write into one `Vec` through
//!                 shared references, and a unique quartet writes 8 scattered tensor
//!                 elements.  So the work is split in two safe phases:
//!                   1. compute: each row ij fills its OWN slice of a packed triangle
//!                      (the Vec is cut into disjoint `&mut` row slices with split_at_mut);
//!                   2. gather: each (i, j) slab G[i, j, :, :] reads its values from the
//!                      packed triangle (every output element written by exactly one task).
//!                 mode = "steal" (Rayon work stealing, the default) or "block"
//!                 (one contiguous block of rows per thread = OpenMP schedule(static)).
//!  jk_par         J/K from the stored tensor, rows i in parallel.
//!  DirectCore     integral-direct J/K (no tensor): thread-private accumulation with
//!                 fold/reduce ("fold"), or shared atomics ("atomic", f64 add via a
//!                 compare-and-swap loop on AtomicU64 - Rust has no atomic f64).
use crate::basis::Basis;
use crate::eri_tab::{build_pairs, pair_index, quartet, Pair, Work};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Default)]
pub struct ParInfo {
    pub computed: u64,
    pub screened: u64,
    pub busy: Vec<f64>, // seconds of row work done by each worker thread
    pub setup_s: f64,
    pub loop_s: f64,
    pub gather_s: f64,  // eri: unpacking the triangle into the N^4 tensor
    pub reduce_s: f64,  // direct/fold: combining the thread-private J/K
}

/// Run `f` on a pool of `threads` workers (0 = Rayon's global pool, sized by
/// RAYON_NUM_THREADS or the number of logical CPUs).
pub fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    if threads == 0 {
        f()
    } else {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool").install(f)
    }
}

/// Per-thread busy time (nanoseconds), indexed by the Rayon worker index.
struct Busy(Vec<AtomicU64>);

impl Busy {
    fn new() -> Self { Busy((0..rayon::current_num_threads()).map(|_| AtomicU64::new(0)).collect()) }
    fn add(&self, t0: Instant) {
        let i = rayon::current_thread_index().unwrap_or(0) % self.0.len();
        self.0[i].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    fn secs(&self) -> Vec<f64> { self.0.iter().map(|a| a.load(Ordering::Relaxed) as f64 * 1e-9).collect() }
}

fn schwarz(pairs: &[Pair], data: &[f64], lw: usize) -> Vec<f64> {
    pairs
        .par_iter()
        .map_init(|| Work::new(lw), |w, pr| quartet::<true>(pr, pr, data, w).abs().sqrt())
        .collect()
}

/// All kl <= ij of one row into `row` (screened entries stay 0); returns (computed, screened).
#[inline]
fn row_kernel(ij: usize, row: &mut [f64], pairs: &[Pair], data: &[f64], qs: &[f64], screen: f64,
              w: &mut Work) -> (u64, u64) {
    let (mut c, mut s) = (0u64, 0u64);
    for kl in 0..=ij {
        if qs[ij] * qs[kl] < screen {
            s += 1;
            continue;
        }
        row[kl] = quartet::<true>(&pairs[ij], &pairs[kl], data, w);
        c += 1;
    }
    (c, s)
}

/// Cut `buf` into consecutive mutable slices of the given lengths (disjoint borrows).
fn split_rows<'a>(mut buf: &'a mut [f64], lens: impl Iterator<Item = usize>) -> Vec<&'a mut [f64]> {
    let mut out = Vec::new();
    for len in lens {
        let (a, b) = std::mem::take(&mut buf).split_at_mut(len);
        out.push(a);
        buf = b;
    }
    out
}

fn add2(a: (u64, u64), b: (u64, u64)) -> (u64, u64) { (a.0 + b.0, a.1 + b.1) }

/// Packed unique quartets for every pair row, computed in parallel.
fn packed_rows(pairs: &[Pair], data: &[f64], qs: &[f64], screen: f64, lw: usize, block: bool,
               busy: &Busy) -> (Vec<f64>, (u64, u64)) {
    let npair = pairs.len();
    let mut packed = vec![0.0; npair * (npair + 1) / 2];
    let counts = {
        let mut rows = split_rows(&mut packed, (0..npair).map(|ij| ij + 1));
        if block {
            let nt = rayon::current_num_threads().max(1);
            let chunk = (npair + nt - 1) / nt;
            rows.par_chunks_mut(chunk.max(1))
                .enumerate()
                .map(|(b, blk)| {
                    let mut w = Work::new(lw);
                    let t0 = Instant::now();
                    let mut cs = (0, 0);
                    for (o, r) in blk.iter_mut().enumerate() {
                        cs = add2(cs, row_kernel(b * chunk + o, r, pairs, data, qs, screen, &mut w));
                    }
                    busy.add(t0);
                    cs
                })
                .reduce(|| (0, 0), add2)
        } else {
            rows.into_par_iter()
                .enumerate()
                .map_init(|| Work::new(lw), |w, (ij, r)| {
                    let t0 = Instant::now();
                    let cs = row_kernel(ij, r, pairs, data, qs, screen, w);
                    busy.add(t0);
                    cs
                })
                .reduce(|| (0, 0), add2)
        }
    };
    (packed, counts)
}

pub fn eri_sym8_par(bs: &Basis, screen: f64, block: bool) -> (Vec<f64>, ParInfo) {
    let t0 = Instant::now();
    let n = bs.nbf();
    let lw = 4 * bs.max_l();
    let (pairs, data) = build_pairs(bs);
    let qs = schwarz(&pairs, &data, lw);
    let setup_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let busy = Busy::new();
    let (packed, (computed, screened)) = packed_rows(&pairs, &data, &qs, screen, lw, block, &busy);
    let loop_s = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let mut g = vec![0.0; n * n * n * n];
    let tri = |a: usize, b: usize| if a >= b { pair_index(a, b) } else { pair_index(b, a) };
    g.par_chunks_mut(n * n).enumerate().for_each(|(ab, slab)| {
        let p1 = tri(ab / n, ab % n);
        for c in 0..n {
            for d in 0..n {
                let p2 = tri(c, d);
                let (hi, lo) = if p1 >= p2 { (p1, p2) } else { (p2, p1) };
                slab[c * n + d] = packed[hi * (hi + 1) / 2 + lo];
            }
        }
    });
    let gather_s = t2.elapsed().as_secs_f64();
    (g, ParInfo { computed, screened, busy: busy.secs(), setup_s, loop_s, gather_s, reduce_s: 0.0 })
}

/// J/K from the stored tensor; rows i in parallel (row i covers j <= i, so rows
/// differ in length - work stealing balances them).
pub fn jk_par(g: &[f64], d: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).fold(0.0, |s, (x, y)| s + x * y);
    let rows: Vec<(Vec<f64>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut jr = vec![0.0; i + 1];
            let mut kr = vec![0.0; i + 1];
            for j in 0..=i {
                let base = (i * n + j) * n * n;
                jr[j] = dot(&g[base..base + n * n], d);
                kr[j] = (0..n)
                    .map(|k| {
                        let row = ((i * n + k) * n + j) * n;
                        dot(&g[row..row + n], &d[k * n..k * n + n])
                    })
                    .sum();
            }
            (jr, kr)
        })
        .collect();
    let mut jm = vec![0.0; n * n];
    let mut km = vec![0.0; n * n];
    for (i, (jr, kr)) in rows.into_iter().enumerate() {
        for j in 0..=i {
            jm[i * n + j] = jr[j];
            jm[j * n + i] = jr[j];
            km[i * n + j] = kr[j];
            km[j * n + i] = kr[j];
        }
    }
    (jm, km)
}

fn atomic_add(a: &AtomicU64, x: f64) {
    let mut cur = a.load(Ordering::Relaxed);
    loop {
        let new = (f64::from_bits(cur) + x).to_bits();
        match a.compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(v) => cur = v,
        }
    }
}

/// Integral-direct J/K builder (owns copies of the pair tables).
pub struct DirectCore {
    pub n: usize,
    lw: usize,
    pairs: Vec<Pair>,
    data: Vec<f64>,
    qs: Vec<f64>,
    pi: Vec<usize>,
    pj: Vec<usize>,
    screen: f64,
    pub rows: Vec<usize>,
    row_off: Vec<usize>,
    pub store: bool,
    pub cache: Option<Vec<f64>>,
    cache_counts: (u64, u64),
    pub last: ParInfo,
}

impl DirectCore {
    pub fn new(bs: &Basis, screen: f64, store: bool, rows: Option<Vec<usize>>) -> Result<Self, String> {
        let t0 = Instant::now();
        let n = bs.nbf();
        let lw = 4 * bs.max_l();
        let (pairs, data) = build_pairs(bs);
        let qs = schwarz(&pairs, &data, lw);
        let npair = pairs.len();
        let (mut pi, mut pj) = (Vec::with_capacity(npair), Vec::with_capacity(npair));
        for i in 0..n {
            for j in 0..=i {
                pi.push(i);
                pj.push(j);
            }
        }
        let rows = match rows {
            Some(r) => r, // exactly these rows (an MPI rank's share; may be empty)
            None => (0..npair).collect(),
        };
        if rows.iter().any(|&r| r >= npair) {
            return Err(format!("row index out of range (npair = {npair})"));
        }
        let mut row_off = vec![0usize; rows.len() + 1];
        for (r, &ij) in rows.iter().enumerate() {
            row_off[r + 1] = row_off[r] + ij + 1;
        }
        let last = ParInfo { setup_s: t0.elapsed().as_secs_f64(), ..Default::default() };
        Ok(DirectCore { n, lw, pairs, data, qs, pi, pj, screen, rows, row_off, store, cache: None,
                        cache_counts: (0, 0), last })
    }

    pub fn npair(&self) -> usize { self.pairs.len() }
    pub fn n_quartets(&self) -> usize { *self.row_off.last().unwrap() }

    /// The six (target, value) updates of one unique quartet; targets < n*n go to
    /// Jt, the rest (offset by n*n) to Kt.  See the C++ DirectJK for the formula.
    #[inline(always)]
    fn updates(&self, ij: usize, kl: usize, v: f64, d: &[f64]) -> [(usize, f64); 6] {
        let n = self.n;
        let (i, j, k, l) = (self.pi[ij], self.pj[ij], self.pi[kl], self.pj[kl]);
        let deg = (if i == j { 1.0 } else { 2.0 }) * (if k == l { 1.0 } else { 2.0 })
            * (if ij == kl { 1.0 } else { 2.0 });
        let w = v * deg;
        let nn = n * n;
        [(i * n + j, d[k * n + l] * w), (k * n + l, d[i * n + j] * w),
         (nn + i * n + k, d[j * n + l] * w), (nn + j * n + l, d[i * n + k] * w),
         (nn + i * n + l, d[j * n + k] * w), (nn + j * n + k, d[i * n + l] * w)]
    }

    /// Visit every non-screened quartet of row number r: f(kl, value).
    #[inline(always)]
    fn visit_row(&self, r: usize, w: &mut Work, mut f: impl FnMut(usize, f64)) -> (u64, u64) {
        let ij = self.rows[r];
        if let Some(c) = &self.cache {
            let row = &c[self.row_off[r]..self.row_off[r + 1]];
            for (kl, &v) in row.iter().enumerate() {
                if v != 0.0 { f(kl, v) }
            }
            return (0, 0);
        }
        let (mut cn, mut sn) = (0u64, 0u64);
        for kl in 0..=ij {
            if self.qs[ij] * self.qs[kl] < self.screen {
                sn += 1;
                continue;
            }
            cn += 1;
            f(kl, quartet::<true>(&self.pairs[ij], &self.pairs[kl], &self.data, w));
        }
        (cn, sn)
    }

    fn fill_cache(&mut self, busy: &Busy) -> (u64, u64) {
        let mut cache = vec![0.0; self.n_quartets()];
        let counts = {
            let lens = self.rows.iter().map(|&ij| ij + 1);
            let slices = split_rows(&mut cache, lens);
            let me = &*self;
            slices
                .into_par_iter()
                .enumerate()
                .map_init(|| Work::new(me.lw), |w, (r, row)| {
                    let t0 = Instant::now();
                    let cs = row_kernel(me.rows[r], row, &me.pairs, &me.data, &me.qs, me.screen, w);
                    busy.add(t0);
                    cs
                })
                .reduce(|| (0, 0), add2)
        };
        self.cache = Some(cache);
        self.cache_counts = counts;
        counts
    }

    /// J and K for the symmetric density d; atomic = shared AtomicU64 accumulators.
    pub fn build(&mut self, d: &[f64], atomic: bool) -> (Vec<f64>, Vec<f64>) {
        let n = self.n;
        let nn = n * n;
        let t0 = Instant::now();
        let busy = Busy::new();
        let mut counts = (0u64, 0u64);
        if self.store && self.cache.is_none() {
            counts = self.fill_cache(&busy);  // first build of a semi-direct run: compute + keep
        }
        let me = &*self;
        let nrow = me.rows.len();
        let mut reduce_s = 0.0;
        let acc: Vec<f64> = if atomic {
            let shared: Vec<AtomicU64> = (0..2 * nn).map(|_| AtomicU64::new(0)).collect();
            let cs = (0..nrow)
                .into_par_iter()
                .map_init(|| Work::new(me.lw), |w, r| {
                    let tb = Instant::now();
                    let ij = me.rows[r];
                    let cs = me.visit_row(r, w, |kl, v| {
                        for (t, x) in me.updates(ij, kl, v, d) {
                            atomic_add(&shared[t], x);
                        }
                    });
                    busy.add(tb);
                    cs
                })
                .reduce(|| (0, 0), add2);
            counts = add2(counts, cs);
            shared.into_iter().map(|a| f64::from_bits(a.into_inner())).collect()
        } else {
            let (acc, cs, red) = (0..nrow)
                .into_par_iter()
                .fold(|| (vec![0.0; 2 * nn], (0u64, 0u64), Work::new(me.lw)), |(mut a, c, mut w), r| {
                    let tb = Instant::now();
                    let ij = me.rows[r];
                    let cs = me.visit_row(r, &mut w, |kl, v| {
                        for (t, x) in me.updates(ij, kl, v, d) {
                            a[t] += x;
                        }
                    });
                    busy.add(tb);
                    (a, add2(c, cs), w)
                })
                .map(|(a, c, _)| (a, c, 0.0f64))
                .reduce(|| (vec![0.0; 2 * nn], (0, 0), 0.0), |(mut a, c1, r1), (b, c2, r2)| {
                    let tr = Instant::now();
                    for (x, y) in a.iter_mut().zip(&b) {
                        *x += y;
                    }
                    (a, add2(c1, c2), r1 + r2 + tr.elapsed().as_secs_f64())
                });
            counts = add2(counts, cs);
            reduce_s = red;
            acc
        };
        if counts == (0, 0) {
            counts = self.cache_counts;
        }
        let (jt, kt) = acc.split_at(nn);
        let mut jm = vec![0.0; nn];
        let mut km = vec![0.0; nn];
        for a in 0..n {
            for b in 0..n {
                jm[a * n + b] = 0.25 * (jt[a * n + b] + jt[b * n + a]);
                km[a * n + b] = 0.125 * (kt[a * n + b] + kt[b * n + a]);
            }
        }
        self.last = ParInfo { computed: counts.0, screened: counts.1, busy: busy.secs(),
                              setup_s: self.last.setup_s, loop_s: t0.elapsed().as_secs_f64(),
                              gather_s: 0.0, reduce_s };
        (jm, km)
    }
}
