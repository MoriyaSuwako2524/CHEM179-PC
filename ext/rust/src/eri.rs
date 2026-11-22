//! Two-electron integrals with the *naive* McMurchie-Davidson algorithm.
//!
//! `eri_naive` is a line-by-line translation of backends/naive: per basis
//! function quartet, per primitive quartet, recursive E and R, series Boys,
//! all N^4 elements.  `eri_sym8` calls the identical quartet routine but only
//! for the ~N^4/8 unique quartets and skips those that the Schwarz inequality
//! |(ij|kl)| <= sqrt((ij|ij)) sqrt((kl|kl)) proves negligible.
//!
//! Parallel-ready structure: `eri_contracted(bs, i, j, k, l)` is a pure
//! function of its arguments; only the outer loops write to memory.  The later
//! Rayon stage parallelises the outer `i` loop without touching the physics.
use crate::basis::Basis;
use crate::hermite::{dist, hermite_e, hermite_r, product_center};
use std::f64::consts::PI;

#[allow(clippy::too_many_arguments)]
pub fn eri_prim(a: f64, l1: [i32; 3], ca: [f64; 3], b: f64, l2: [i32; 3], cb: [f64; 3],
                c: f64, l3: [i32; 3], cc: [f64; 3], d: f64, l4: [i32; 3], cd: [f64; 3]) -> f64 {
    let p = a + b;
    let q = c + d;
    let alpha = p * q / (p + q);
    let pp = product_center(a, ca, b, cb);
    let qq = product_center(c, cc, d, cd);
    let rpq = dist(pp, qq);
    let mut val = 0.0;
    for t in 0..=l1[0] + l2[0] {
        for u in 0..=l1[1] + l2[1] {
            for v in 0..=l1[2] + l2[2] {
                for tau in 0..=l3[0] + l4[0] {
                    for nu in 0..=l3[1] + l4[1] {
                        for phi in 0..=l3[2] + l4[2] {
                            let sign = if (tau + nu + phi) % 2 == 0 { 1.0 } else { -1.0 };
                            val += hermite_e(l1[0], l2[0], t, ca[0] - cb[0], a, b)
                                * hermite_e(l1[1], l2[1], u, ca[1] - cb[1], a, b)
                                * hermite_e(l1[2], l2[2], v, ca[2] - cb[2], a, b)
                                * hermite_e(l3[0], l4[0], tau, cc[0] - cd[0], c, d)
                                * hermite_e(l3[1], l4[1], nu, cc[1] - cd[1], c, d)
                                * hermite_e(l3[2], l4[2], phi, cc[2] - cd[2], c, d)
                                * sign
                                * hermite_r(t + tau, u + nu, v + phi, 0, alpha,
                                            pp[0] - qq[0], pp[1] - qq[1], pp[2] - qq[2], rpq);
                        }
                    }
                }
            }
        }
    }
    2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt()) * val
}

/// (ij|kl) over contracted basis functions.
pub fn eri_contracted(bs: &Basis, i: usize, j: usize, k: usize, l: usize) -> f64 {
    let (l1, l2, l3, l4) = (bs.lmn(i), bs.lmn(j), bs.lmn(k), bs.lmn(l));
    let (c1, c2, c3, c4) = (bs.center(i), bs.center(j), bs.center(k), bs.center(l));
    let mut s = 0.0;
    for p1 in bs.prims(i) {
        for p2 in bs.prims(j) {
            for p3 in bs.prims(k) {
                for p4 in bs.prims(l) {
                    s += bs.coefs[p1] * bs.coefs[p2] * bs.coefs[p3] * bs.coefs[p4]
                        * eri_prim(bs.exps[p1], l1, c1, bs.exps[p2], l2, c2,
                                   bs.exps[p3], l3, c3, bs.exps[p4], l4, c4);
                }
            }
        }
    }
    s
}

/// All N^4 integrals, one at a time, no symmetry (same as the naive stage).
pub fn eri_naive(bs: &Basis) -> Vec<f64> {
    let n = bs.nbf();
    let mut g = vec![0.0; n * n * n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    g[((i * n + j) * n + k) * n + l] = eri_contracted(bs, i, j, k, l);
                }
            }
        }
    }
    g
}

/// Write one unique integral into all 8 symmetry-equivalent positions.
#[inline]
pub fn scatter8(g: &mut [f64], n: usize, i: usize, j: usize, k: usize, l: usize, v: f64) {
    let idx = |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
    g[idx(i, j, k, l)] = v;
    g[idx(j, i, k, l)] = v;
    g[idx(i, j, l, k)] = v;
    g[idx(j, i, l, k)] = v;
    g[idx(k, l, i, j)] = v;
    g[idx(l, k, i, j)] = v;
    g[idx(k, l, j, i)] = v;
    g[idx(l, k, j, i)] = v;
}

/// Unique quartets: i >= j, k >= l, (ij) >= (kl).  Returns (G, computed, screened).
pub fn eri_sym8(bs: &Basis, screen: f64) -> (Vec<f64>, u64, u64) {
    let n = bs.nbf();
    // Schwarz factors Q_ij = sqrt|(ij|ij)|
    let mut qs = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = eri_contracted(bs, i, j, i, j).abs().sqrt();
            qs[i * n + j] = v;
            qs[j * n + i] = v;
        }
    }
    let mut g = vec![0.0; n * n * n * n];
    let (mut computed, mut screened) = (0u64, 0u64);
    for i in 0..n {
        for j in 0..=i {
            for k in 0..=i {
                let lmax = if k == i { j } else { k };
                for l in 0..=lmax {
                    if qs[i * n + j] * qs[k * n + l] < screen {
                        screened += 1;
                        continue;
                    }
                    computed += 1;
                    scatter8(&mut g, n, i, j, k, l, eri_contracted(bs, i, j, k, l));
                }
            }
        }
    }
    (g, computed, screened)
}
