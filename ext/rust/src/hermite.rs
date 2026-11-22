//! McMurchie-Davidson building blocks, a line-by-line translation of
//! backends/naive/integrals.py (recursive, nothing cached).
use crate::boys::boys;

/// E^{ij}_t for one Cartesian direction; qx = A_x - B_x.
pub fn hermite_e(i: i32, j: i32, t: i32, qx: f64, a: f64, b: f64) -> f64 {
    let p = a + b;
    let q = a * b / p;
    if t < 0 || t > i + j {
        return 0.0;
    }
    if i == 0 && j == 0 && t == 0 {
        return (-q * qx * qx).exp();
    }
    if j == 0 {
        return 1.0 / (2.0 * p) * hermite_e(i - 1, j, t - 1, qx, a, b)
            - q * qx / a * hermite_e(i - 1, j, t, qx, a, b)
            + (t + 1) as f64 * hermite_e(i - 1, j, t + 1, qx, a, b);
    }
    1.0 / (2.0 * p) * hermite_e(i, j - 1, t - 1, qx, a, b)
        + q * qx / b * hermite_e(i, j - 1, t, qx, a, b)
        + (t + 1) as f64 * hermite_e(i, j - 1, t + 1, qx, a, b)
}

/// Hermite Coulomb integral R^n_{tuv} (recursive).
#[allow(clippy::too_many_arguments)]
pub fn hermite_r(t: i32, u: i32, v: i32, n: i32, p: f64, x: f64, y: f64, z: f64, rpc: f64) -> f64 {
    let tt = p * rpc * rpc;
    if t == 0 && u == 0 && v == 0 {
        return (-2.0 * p).powi(n) * boys(n, tt);
    }
    if t == 0 && u == 0 {
        let mut val = 0.0;
        if v > 1 {
            val += (v - 1) as f64 * hermite_r(t, u, v - 2, n + 1, p, x, y, z, rpc);
        }
        return val + z * hermite_r(t, u, v - 1, n + 1, p, x, y, z, rpc);
    }
    if t == 0 {
        let mut val = 0.0;
        if u > 1 {
            val += (u - 1) as f64 * hermite_r(t, u - 2, v, n + 1, p, x, y, z, rpc);
        }
        return val + y * hermite_r(t, u - 1, v, n + 1, p, x, y, z, rpc);
    }
    let mut val = 0.0;
    if t > 1 {
        val += (t - 1) as f64 * hermite_r(t - 2, u, v, n + 1, p, x, y, z, rpc);
    }
    val + x * hermite_r(t - 1, u, v, n + 1, p, x, y, z, rpc)
}

#[inline]
pub fn product_center(a: f64, ca: [f64; 3], b: f64, cb: [f64; 3]) -> [f64; 3] {
    let p = a + b;
    [(a * ca[0] + b * cb[0]) / p, (a * ca[1] + b * cb[1]) / p, (a * ca[2] + b * cb[2]) / p]
}

#[inline]
pub fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
