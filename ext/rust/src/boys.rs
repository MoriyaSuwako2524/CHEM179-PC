//! Boys function F_n(T) = int_0^1 t^{2n} exp(-T t^2) dt.
use std::f64::consts::PI;

/// Same algorithm as backends/naive/boys.py: power series for T < 30,
/// erf + upward recursion above.
pub fn boys(n: i32, t: f64) -> f64 {
    if t < 1e-12 {
        return 1.0 / (2 * n + 1) as f64;
    }
    if t < 30.0 {
        let mut term = 1.0 / (2 * n + 1) as f64;
        let mut total = term;
        let mut k = 0;
        while term > 1e-17 * total {
            k += 1;
            term *= 2.0 * t / (2 * n + 2 * k + 1) as f64;
            total += term;
        }
        return (-t).exp() * total;
    }
    let mut f = 0.5 * (PI / t).sqrt() * libm::erf(t.sqrt());
    let e = (-t).exp();
    for m in 0..n {
        f = ((2 * m + 1) as f64 * f - e) / (2.0 * t);
    }
    f
}

/// F_0 .. F_nmax at once (used by the table kernel): one series evaluation for
/// F_nmax, then the stable downward recursion F_n = (2T F_{n+1} + e^{-T}) / (2n+1).
pub fn boys_array(nmax: usize, t: f64, out: &mut [f64]) {
    if t < 1e-12 {
        for n in 0..=nmax {
            out[n] = 1.0 / (2 * n + 1) as f64;
        }
    } else if t < 30.0 {
        out[nmax] = boys(nmax as i32, t);
        let e = (-t).exp();
        for n in (0..nmax).rev() {
            out[n] = (2.0 * t * out[n + 1] + e) / (2 * n + 1) as f64;
        }
    } else {
        let e = (-t).exp();
        out[0] = 0.5 * (PI / t).sqrt() * libm::erf(t.sqrt());
        for m in 0..nmax {
            out[m + 1] = ((2 * m + 1) as f64 * out[m] - e) / (2.0 * t);
        }
    }
}
