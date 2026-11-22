//! eri_sym8_tab: the same 8-fold symmetry + Schwarz screening as `eri_sym8`,
//! but the inner work is reorganised so that nothing is computed twice:
//!   * the Hermite expansion coefficients E^{ab}_{tuv} (times the contraction
//!     coefficients) are tabulated ONCE per primitive pair and reused for
//!     every quartet that contains that pair;
//!   * F_0..F_L (Boys) come from one series evaluation + downward recursion;
//!   * R^0_{tuv} is filled iteratively into a table instead of re-derived by
//!     recursion for every (t,u,v).
//! All table reads go through `idx::ld::<CHECKED>` -> bounds-check experiment.
use crate::basis::Basis;
use crate::boys::boys_array;
use crate::eri::scatter8;
use crate::hermite::{hermite_e, product_center};
use crate::idx::{ld, st};
use std::f64::consts::PI;

/// Hermite tables for every primitive pair of one basis-function pair (i, j).
/// Record layout in `data`: [p, Px, Py, Pz, E[0..ncoef]] for each primitive pair,
/// E index = (t*dy + u)*dz + v with t <= lx_i+lx_j etc.
pub(crate) struct Pair {
    pub(crate) start: usize,
    pub(crate) nprim: usize,
    pub(crate) dims: [usize; 3],
    pub(crate) l: usize, // lx+ly+lz summed over both functions
}

impl Pair {
    fn ncoef(&self) -> usize { self.dims[0] * self.dims[1] * self.dims[2] }
    fn rec(&self) -> usize { 4 + self.ncoef() }
}

pub(crate) fn build_pairs(bs: &Basis) -> (Vec<Pair>, Vec<f64>) {
    let n = bs.nbf();
    let mut pairs = Vec::with_capacity(n * (n + 1) / 2);
    let mut data = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            let (li, lj, ci, cj) = (bs.lmn(i), bs.lmn(j), bs.center(i), bs.center(j));
            let dims = [(li[0] + lj[0] + 1) as usize, (li[1] + lj[1] + 1) as usize,
                        (li[2] + lj[2] + 1) as usize];
            let start = data.len();
            let mut nprim = 0;
            for pa in bs.prims(i) {
                for pb in bs.prims(j) {
                    let (a, b) = (bs.exps[pa], bs.exps[pb]);
                    let cab = bs.coefs[pa] * bs.coefs[pb];
                    let pc = product_center(a, ci, b, cj);
                    data.extend_from_slice(&[a + b, pc[0], pc[1], pc[2]]);
                    for t in 0..dims[0] {
                        let ex = hermite_e(li[0], lj[0], t as i32, ci[0] - cj[0], a, b);
                        for u in 0..dims[1] {
                            let ey = hermite_e(li[1], lj[1], u as i32, ci[1] - cj[1], a, b);
                            for v in 0..dims[2] {
                                let ez = hermite_e(li[2], lj[2], v as i32, ci[2] - cj[2], a, b);
                                data.push(cab * ex * ey * ez);
                            }
                        }
                    }
                    nprim += 1;
                }
            }
            pairs.push(Pair { start, nprim, dims, l: dims.iter().sum::<usize>() - 3 });
        }
    }
    (pairs, data)
}

#[inline(always)]
pub(crate) fn pair_index(i: usize, j: usize) -> usize { i * (i + 1) / 2 + j }

/// Scratch space reused for every primitive quartet (one per thread later).
pub(crate) struct Work {
    s: usize,        // stride of the R table = max total L + 1
    f: Vec<f64>,     // Boys F_0..F_L
    pw: Vec<f64>,    // (-2 alpha)^n
    r: Vec<f64>,     // R^n_{tuv}, index ((n*s + t)*s + u)*s + v
}

impl Work {
    pub(crate) fn new(lmax_total: usize) -> Self {
        let s = lmax_total + 1;
        Work { s, f: vec![0.0; s], pw: vec![0.0; s], r: vec![0.0; s * s * s * s] }
    }
}

/// Fill R^n_{tuv} for all t+u+v <= l - n (iteratively, from high n down to 0).
#[inline(always)]
fn r_table<const C: bool>(w: &mut Work, l: usize, alpha: f64, x: f64, y: f64, z: f64) {
    let s = w.s;
    let id = |n: usize, t: usize, u: usize, v: usize| ((n * s + t) * s + u) * s + v;
    let mut p = 1.0;
    for n in 0..=l {
        w.pw[n] = p;
        p *= -2.0 * alpha;
    }
    for n in (0..=l).rev() {
        let rem = l - n;
        for t in 0..=rem {
            for u in 0..=rem - t {
                for v in 0..=rem - t - u {
                    let val = if t > 0 {
                        let mut a = x * ld::<C>(&w.r, id(n + 1, t - 1, u, v));
                        if t > 1 { a += (t - 1) as f64 * ld::<C>(&w.r, id(n + 1, t - 2, u, v)); }
                        a
                    } else if u > 0 {
                        let mut a = y * ld::<C>(&w.r, id(n + 1, 0, u - 1, v));
                        if u > 1 { a += (u - 1) as f64 * ld::<C>(&w.r, id(n + 1, 0, u - 2, v)); }
                        a
                    } else if v > 0 {
                        let mut a = z * ld::<C>(&w.r, id(n + 1, 0, 0, v - 1));
                        if v > 1 { a += (v - 1) as f64 * ld::<C>(&w.r, id(n + 1, 0, 0, v - 2)); }
                        a
                    } else {
                        ld::<C>(&w.pw, n) * ld::<C>(&w.f, n)
                    };
                    st::<C>(&mut w.r, id(n, t, u, v), val);
                }
            }
        }
    }
}

/// (ij|kl) from two pair tables.
#[inline(always)]
pub(crate) fn quartet<const C: bool>(bra: &Pair, ket: &Pair, data: &[f64], w: &mut Work) -> f64 {
    let s = w.s;
    let l = bra.l + ket.l;
    let [bx, by, bz] = bra.dims;
    let [kx, ky, kz] = ket.dims;
    let mut total = 0.0;
    for ab in 0..bra.nprim {
        let o1 = bra.start + ab * bra.rec();
        let p = ld::<C>(data, o1);
        let (px, py, pz) = (ld::<C>(data, o1 + 1), ld::<C>(data, o1 + 2), ld::<C>(data, o1 + 3));
        for cd in 0..ket.nprim {
            let o2 = ket.start + cd * ket.rec();
            let q = ld::<C>(data, o2);
            let (x, y, z) = (px - ld::<C>(data, o2 + 1), py - ld::<C>(data, o2 + 2),
                             pz - ld::<C>(data, o2 + 3));
            let alpha = p * q / (p + q);
            boys_array(l, alpha * (x * x + y * y + z * z), &mut w.f);
            r_table::<C>(w, l, alpha, x, y, z);
            let mut val = 0.0;
            for t in 0..bx {
                for u in 0..by {
                    for v in 0..bz {
                        let eab = ld::<C>(data, o1 + 4 + (t * by + u) * bz + v);
                        let mut inner = 0.0;
                        for tau in 0..kx {
                            for nu in 0..ky {
                                for phi in 0..kz {
                                    let ecd = ld::<C>(data, o2 + 4 + (tau * ky + nu) * kz + phi);
                                    let r = ld::<C>(&w.r, ((t + tau) * s + u + nu) * s + v + phi);
                                    if (tau + nu + phi) & 1 == 0 { inner += ecd * r } else { inner -= ecd * r }
                                }
                            }
                        }
                        val += eab * inner;
                    }
                }
            }
            total += 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt()) * val;
        }
    }
    total
}

pub fn eri_sym8_tab<const C: bool>(bs: &Basis, screen: f64) -> (Vec<f64>, u64, u64) {
    let n = bs.nbf();
    let (pairs, data) = build_pairs(bs);
    let mut w = Work::new(4 * bs.max_l());
    let mut qs = vec![0.0; pairs.len()];
    for (ij, pr) in pairs.iter().enumerate() {
        qs[ij] = quartet::<C>(pr, pr, &data, &mut w).abs().sqrt();
    }
    let mut g = vec![0.0; n * n * n * n];
    let (mut computed, mut screened) = (0u64, 0u64);
    for i in 0..n {
        for j in 0..=i {
            let ij = pair_index(i, j);
            for k in 0..=i {
                let lmax = if k == i { j } else { k };
                for l in 0..=lmax {
                    let kl = pair_index(k, l);
                    if qs[ij] * qs[kl] < screen {
                        screened += 1;
                        continue;
                    }
                    computed += 1;
                    let v = quartet::<C>(&pairs[ij], &pairs[kl], &data, &mut w);
                    scatter8(&mut g, n, i, j, k, l, v);
                }
            }
        }
    }
    (g, computed, screened)
}
