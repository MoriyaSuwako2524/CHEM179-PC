//! J_ij = sum_kl (ij|kl) D_kl      K_ij = sum_kl (ik|jl) D_kl
//! from the full N^4 tensor (row-major), using J = J^T and K = K^T.
//!
//! Three flavours of the *same* loop for the safety experiment:
//!   jk_index::<true>   explicit index arithmetic, every access bounds-checked
//!   jk_index::<false>  explicit index arithmetic, `get_unchecked` (unsafe)
//!   jk_iter            idiomatic safe Rust: slice once, then zip iterators;
//!                      one check per slice, none inside the loop.
use crate::idx::ld;

pub fn jk_index<const C: bool>(g: &[f64], d: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut jm = vec![0.0; n * n];
    let mut km = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let base = (i * n + j) * n * n;
            let mut sj = 0.0;
            for kl in 0..n * n {
                sj += ld::<C>(g, base + kl) * ld::<C>(d, kl);
            }
            let mut sk = 0.0;
            for k in 0..n {
                let row = ((i * n + k) * n + j) * n;
                for l in 0..n {
                    sk += ld::<C>(g, row + l) * ld::<C>(d, k * n + l);
                }
            }
            jm[i * n + j] = sj;
            jm[j * n + i] = sj;
            km[i * n + j] = sk;
            km[j * n + i] = sk;
        }
    }
    (jm, km)
}

pub fn jk_iter(g: &[f64], d: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut jm = vec![0.0; n * n];
    let mut km = vec![0.0; n * n];
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).fold(0.0, |s, (x, y)| s + x * y);
    for i in 0..n {
        for j in 0..=i {
            let base = (i * n + j) * n * n;
            let sj = dot(&g[base..base + n * n], d);
            let sk: f64 = (0..n)
                .map(|k| {
                    let row = ((i * n + k) * n + j) * n;
                    dot(&g[row..row + n], &d[k * n..k * n + n])
                })
                .sum();
            jm[i * n + j] = sj;
            jm[j * n + i] = sj;
            km[i * n + j] = sk;
            km[j * n + i] = sk;
        }
    }
    (jm, km)
}
