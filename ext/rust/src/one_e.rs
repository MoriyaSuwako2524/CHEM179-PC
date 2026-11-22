//! One-electron integrals S, T, V: the naive algorithm, compiled.
use crate::basis::Basis;
use crate::hermite::{dist, hermite_e, hermite_r, product_center};
use std::f64::consts::PI;

fn overlap_prim(a: f64, l1: [i32; 3], ca: [f64; 3], b: f64, l2: [i32; 3], cb: [f64; 3]) -> f64 {
    let s1 = hermite_e(l1[0], l2[0], 0, ca[0] - cb[0], a, b);
    let s2 = hermite_e(l1[1], l2[1], 0, ca[1] - cb[1], a, b);
    let s3 = hermite_e(l1[2], l2[2], 0, ca[2] - cb[2], a, b);
    s1 * s2 * s3 * (PI / (a + b)).powf(1.5)
}

fn kinetic_prim(a: f64, l1: [i32; 3], ca: [f64; 3], b: f64, l2: [i32; 3], cb: [f64; 3]) -> f64 {
    let [l, m, n] = l2;
    let s = |d: [i32; 3]| overlap_prim(a, l1, ca, b, d, cb);
    let term0 = b * (2 * (l + m + n) + 3) as f64 * s(l2);
    let term1 = -2.0 * b * b * (s([l + 2, m, n]) + s([l, m + 2, n]) + s([l, m, n + 2]));
    // s() with a negative power is multiplied by l(l-1) = 0, but hermite_e must
    // still not recurse forever: guard it explicitly.
    let sm = |d: [i32; 3], f: i32| if f == 0 { 0.0 } else { f as f64 * s(d) };
    let term2 = -0.5 * (sm([l - 2, m, n], l * (l - 1)) + sm([l, m - 2, n], m * (m - 1))
        + sm([l, m, n - 2], n * (n - 1)));
    term0 + term1 + term2
}

fn nuclear_prim(a: f64, l1: [i32; 3], ca: [f64; 3], b: f64, l2: [i32; 3], cb: [f64; 3], c: [f64; 3]) -> f64 {
    let p = a + b;
    let pp = product_center(a, ca, b, cb);
    let rpc = dist(pp, c);
    let mut val = 0.0;
    for t in 0..=l1[0] + l2[0] {
        for u in 0..=l1[1] + l2[1] {
            for v in 0..=l1[2] + l2[2] {
                val += hermite_e(l1[0], l2[0], t, ca[0] - cb[0], a, b)
                    * hermite_e(l1[1], l2[1], u, ca[1] - cb[1], a, b)
                    * hermite_e(l1[2], l2[2], v, ca[2] - cb[2], a, b)
                    * hermite_r(t, u, v, 0, p, pp[0] - c[0], pp[1] - c[1], pp[2] - c[2], rpc);
            }
        }
    }
    2.0 * PI / p * val
}

fn contract2<F: Fn(f64, [i32; 3], [f64; 3], f64, [i32; 3], [f64; 3]) -> f64>(
    bs: &Basis, i: usize, j: usize, prim: F) -> f64 {
    let (li, lj, ci, cj) = (bs.lmn(i), bs.lmn(j), bs.center(i), bs.center(j));
    let mut s = 0.0;
    for pa in bs.prims(i) {
        for pb in bs.prims(j) {
            s += bs.coefs[pa] * bs.coefs[pb] * prim(bs.exps[pa], li, ci, bs.exps[pb], lj, cj);
        }
    }
    s
}

pub fn overlap(bs: &Basis) -> Vec<f64> {
    let n = bs.nbf();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = contract2(bs, i, j, overlap_prim);
        }
    }
    out
}

pub fn kinetic(bs: &Basis) -> Vec<f64> {
    let n = bs.nbf();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = contract2(bs, i, j, kinetic_prim);
        }
    }
    out
}

/// charges[a], coords[3a..3a+3] in Bohr.
pub fn nuclear(bs: &Basis, charges: &[f64], coords: &[f64]) -> Vec<f64> {
    let n = bs.nbf();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for (a, &z) in charges.iter().enumerate() {
                let c = [coords[3 * a], coords[3 * a + 1], coords[3 * a + 2]];
                out[i * n + j] += -z * contract2(bs, i, j, |a1, l1, c1, b1, l2, c2| {
                    nuclear_prim(a1, l1, c1, b1, l2, c2, c)
                });
            }
        }
    }
    out
}
