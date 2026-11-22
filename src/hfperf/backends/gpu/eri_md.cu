// Stage 5 (instructor demo): two-electron integrals on an NVIDIA GPU.
//
// Same mathematics as the C++/Rust "sym8 + reuse" kernels (McMurchie-Davidson,
// Hermite E tables per primitive pair, iterative R recursion, Boys function),
// reorganised for a GPU:
//
//   pair_kernel     one thread per PRIMITIVE pair (a on function i, b on j, i >= j):
//                   p = a + b, P, and E_tuv = c_a c_b E^x_t E^y_u E^z_v.
//   schwarz_kernel  one thread per function pair ij:  Q_ij = sqrt|(ij|ij)|.
//   quartet_kernel  one thread per UNIQUE contracted quartet (ij|kl), kl <= ij
//                   in a (possibly reordered) pair list; the thread loops over
//                   all primitive quartets and writes the 8 symmetry copies of
//                   the result into the full N^4 tensor.
//
// REAL is set at compile time: -DREAL=double (FP64) or -DREAL=float (FP32,
// together with -DREAL_IS_FLOAT=1).  Every literal is wrapped in R_() so that
// an FP32 build never silently promotes to FP64 arithmetic (which is 2x-64x
// slower on consumer GPUs).
//
// Compiled at run time by CuPy (NVRTC) from hfperf/backends/gpu/cuda_eri.py.
#ifndef REAL
#define REAL double
#endif
#ifndef REAL_IS_FLOAT
#define REAL_IS_FLOAT 0
#endif
#define R_(x) ((REAL)(x))

#define LMAX_F 2              // highest l of one basis function (d)
#define LTOT_MAX 8            // highest total L of a quartet: (dd|dd)
#define RSIZE (9 * 48)        // (L+1) * max (Lx+1)(Ly+1)(Lz+1) with Lx+Ly+Lz <= 8

typedef long long i64;

// ---------------------------------------------------------------- Boys
// F_0..F_L(T): series for F_L then downward recursion (T < 30), or the
// erf closed form for F_0 and upward recursion (T >= 30) - as in hfcpp.cpp.
__device__ void boys_all(int L, REAL T, REAL* F) {
    const REAL e = exp(-T);
    if (T < R_(1e-12)) {
        for (int n = 0; n <= L; ++n) F[n] = R_(1.0) / R_(2 * n + 1);
    } else if (T < R_(30.0)) {
        REAL term = R_(1.0) / R_(2 * L + 1), total = term;
#if REAL_IS_FLOAT
        const REAL eps = R_(1e-8);
#else
        const REAL eps = R_(1e-17);
#endif
        for (int k = 1; k < 200 && term > eps * total; ++k) {
            term *= R_(2.0) * T / R_(2 * L + 2 * k + 1);
            total += term;
        }
        F[L] = e * total;
        for (int n = L; n > 0; --n) F[n - 1] = (R_(2.0) * T * F[n] + e) / R_(2 * n - 1);
    } else {
        F[0] = R_(0.5) * sqrt(R_(3.14159265358979323846) / T) * erf(sqrt(T));
        for (int n = 0; n < L; ++n) F[n + 1] = (R_(2 * n + 1) * F[n] - e) / (R_(2.0) * T);
    }
}

// ---------------------------------------------------------------- Hermite E
// E^{ij}_t, t = 0..li+lj, bottom-up (same recursion as hermite_E_row in hfcpp.cpp).
__device__ void hermite_E_row(int li, int lj, REAL Qx, REAL a, REAL b, REAL* out) {
    REAL T[LMAX_F + 1][LMAX_F + 1][2 * LMAX_F + 2];
    for (int i = 0; i <= LMAX_F; ++i)
        for (int j = 0; j <= LMAX_F; ++j)
            for (int t = 0; t < 2 * LMAX_F + 2; ++t) T[i][j][t] = R_(0.0);
    const REAL p = a + b, q = a * b / p, h = R_(1.0) / (R_(2.0) * p);
    T[0][0][0] = exp(-q * Qx * Qx);
    for (int i = 0; i <= li; ++i)
        for (int j = 0; j <= lj; ++j) {
            if (i == 0 && j == 0) continue;
            for (int t = 0; t <= i + j; ++t) {
                if (j == 0)
                    T[i][0][t] = (t > 0 ? h * T[i - 1][0][t - 1] : R_(0.0)) - q * Qx / a * T[i - 1][0][t]
                                 + R_(t + 1) * T[i - 1][0][t + 1];
                else
                    T[i][j][t] = (t > 0 ? h * T[i][j - 1][t - 1] : R_(0.0)) + q * Qx / b * T[i][j - 1][t]
                                 + R_(t + 1) * T[i][j - 1][t + 1];
            }
        }
    for (int t = 0; t <= li + lj; ++t) out[t] = T[li][lj][t];
}

extern "C" {

// One thread per primitive pair pp.  pp belongs to function pair pp_pair[pp]
// = (i, j) and combines primitive pa of i with primitive pb of j.
__global__ void pair_kernel(const i64 npp, const int* __restrict__ pp_pair, const int* __restrict__ pp_a,
                            const int* __restrict__ pp_b, const i64* __restrict__ pp_hoff,
                            const int* __restrict__ pair_i, const int* __restrict__ pair_j,
                            const REAL* __restrict__ centers, const int* __restrict__ lmn,
                            const REAL* __restrict__ exps, const REAL* __restrict__ coefs,
                            REAL* __restrict__ pp_p, REAL* __restrict__ pp_P, REAL* __restrict__ E) {
    const i64 pp = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (pp >= npp) return;
    const int ij = pp_pair[pp], i = pair_i[ij], j = pair_j[ij];
    const int pa = pp_a[pp], pb = pp_b[pp];
    const REAL a = exps[pa], b = exps[pb], p = a + b, cc = coefs[pa] * coefs[pb];
    pp_p[pp] = p;
    REAL Ex[2 * LMAX_F + 1], Ey[2 * LMAX_F + 1], Ez[2 * LMAX_F + 1];
    for (int k = 0; k < 3; ++k) {
        const REAL A = centers[3 * i + k], B = centers[3 * j + k];
        pp_P[3 * pp + k] = (a * A + b * B) / p;
        hermite_E_row(lmn[3 * i + k], lmn[3 * j + k], A - B, a, b, k == 0 ? Ex : (k == 1 ? Ey : Ez));
    }
    const int Lx = lmn[3 * i] + lmn[3 * j], Ly = lmn[3 * i + 1] + lmn[3 * j + 1], Lz = lmn[3 * i + 2] + lmn[3 * j + 2];
    REAL* out = E + pp_hoff[pp];
    int h = 0;
    for (int t = 0; t <= Lx; ++t)
        for (int u = 0; u <= Ly; ++u)
            for (int v = 0; v <= Lz; ++v, ++h) out[h] = cc * Ex[t] * Ey[u] * Ez[v];
}

}  // extern "C"

// Contracted (bra|ket) from the precomputed primitive-pair data.
__device__ REAL quartet(int bra, int ket, const int* __restrict__ pair_L, const int* __restrict__ pair_ppoff,
                        const REAL* __restrict__ pp_p, const REAL* __restrict__ pp_P,
                        const i64* __restrict__ pp_hoff, const REAL* __restrict__ E) {
    const int bLx = pair_L[3 * bra], bLy = pair_L[3 * bra + 1], bLz = pair_L[3 * bra + 2];
    const int kLx = pair_L[3 * ket], kLy = pair_L[3 * ket + 1], kLz = pair_L[3 * ket + 2];
    const int Lx = bLx + kLx, Ly = bLy + kLy, Lz = bLz + kLz, L = Lx + Ly + Lz;
    const int su = Lz + 1, st = (Ly + 1) * su, sn = (Lx + 1) * st;  // R strides (v stride 1)
    REAL R[RSIZE];               // too large for registers -> "local memory" (cached in L1/L2)
    REAL F[LTOT_MAX + 1];
    const REAL pref0 = R_(34.986836655249725);  // 2 pi^(5/2)
    REAL total = R_(0.0);
    for (int ab = pair_ppoff[bra]; ab < pair_ppoff[bra + 1]; ++ab) {
        const REAL p = pp_p[ab];
        const REAL Px = pp_P[3 * ab], Py = pp_P[3 * ab + 1], Pz = pp_P[3 * ab + 2];
        const REAL* Eb = E + pp_hoff[ab];
        for (int cd = pair_ppoff[ket]; cd < pair_ppoff[ket + 1]; ++cd) {
            const REAL q = pp_p[cd];
            const REAL* Ek = E + pp_hoff[cd];
            const REAL alpha = p * q / (p + q);
            const REAL X = Px - pp_P[3 * cd], Y = Py - pp_P[3 * cd + 1], Z = Pz - pp_P[3 * cd + 2];
            boys_all(L, alpha * (X * X + Y * Y + Z * Z), F);
            REAL m2a = R_(1.0);
            for (int n = 0; n <= L; ++n) {  // R^n_000 = (-2 alpha)^n F_n
                R[n * sn] = m2a * F[n];
                m2a *= R_(-2.0) * alpha;
            }
            for (int n = L - 1; n >= 0; --n) {
                const int rem = L - n;
                REAL* Rn = R + n * sn;
                const REAL* R1 = R + (n + 1) * sn;
                for (int t = 0; t <= min(Lx, rem); ++t)
                    for (int u = 0; u <= min(Ly, rem - t); ++u)
                        for (int v = 0; v <= min(Lz, rem - t - u); ++v) {
                            const int id = t * st + u * su + v;
                            REAL val;
                            if (t > 0)
                                val = (t > 1 ? R_(t - 1) * R1[id - 2 * st] : R_(0.0)) + X * R1[id - st];
                            else if (u > 0)
                                val = (u > 1 ? R_(u - 1) * R1[id - 2 * su] : R_(0.0)) + Y * R1[id - su];
                            else if (v > 0)
                                val = (v > 1 ? R_(v - 1) * R1[id - 2] : R_(0.0)) + Z * R1[id - 1];
                            else
                                continue;
                            Rn[id] = val;
                        }
            }
            REAL val = R_(0.0);
            int hb = 0;
            for (int t = 0; t <= bLx; ++t)
                for (int u = 0; u <= bLy; ++u)
                    for (int v = 0; v <= bLz; ++v, ++hb) {
                        REAL inner = R_(0.0);
                        int hk = 0;
                        for (int tau = 0; tau <= kLx; ++tau)
                            for (int nu = 0; nu <= kLy; ++nu)
                                for (int phi = 0; phi <= kLz; ++phi, ++hk) {
                                    const REAL e = ((tau + nu + phi) & 1) ? -Ek[hk] : Ek[hk];
                                    inner += e * R[(t + tau) * st + (u + nu) * su + (v + phi)];
                                }
                        val += Eb[hb] * inner;
                    }
            total += pref0 / (p * q * sqrt(p + q)) * val;
        }
    }
    return total;
}

extern "C" {

__global__ void schwarz_kernel(const int npair, const int* __restrict__ pair_L, const int* __restrict__ pair_ppoff,
                               const REAL* __restrict__ pp_p, const REAL* __restrict__ pp_P,
                               const i64* __restrict__ pp_hoff, const REAL* __restrict__ E,
                               double* __restrict__ Q) {
    const int ij = blockIdx.x * blockDim.x + threadIdx.x;
    if (ij >= npair) return;
    Q[ij] = sqrt(fabs((double)quartet(ij, ij, pair_L, pair_ppoff, pp_p, pp_P, pp_hoff, E)));
}

// One thread per unique quartet.  Thread t -> (a, b) with b <= a on the
// triangle of the (reordered) pair list; bra = order[a], ket = order[b].
__global__ void quartet_kernel(const int n, const i64 nquart, const int* __restrict__ order,
                               const int* __restrict__ pair_i, const int* __restrict__ pair_j,
                               const int* __restrict__ pair_L, const int* __restrict__ pair_ppoff,
                               const REAL* __restrict__ pp_p, const REAL* __restrict__ pp_P,
                               const i64* __restrict__ pp_hoff, const REAL* __restrict__ E,
                               const double* __restrict__ Q, const double tol, REAL* __restrict__ G) {
    const i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nquart) return;
    i64 a = (i64)((sqrt(8.0 * (double)t + 1.0) - 1.0) * 0.5);  // row of the triangle
    while (a * (a + 1) / 2 > t) --a;
    while ((a + 1) * (a + 2) / 2 <= t) ++a;
    const i64 b = t - a * (a + 1) / 2;
    const int bra = order[a], ket = order[b];
    if (Q[bra] * Q[ket] < tol) return;  // Schwarz: |(ij|kl)| <= Q_ij Q_kl; G is pre-zeroed
    const REAL v = quartet(bra, ket, pair_L, pair_ppoff, pp_p, pp_P, pp_hoff, E);
    const i64 i = pair_i[bra], j = pair_j[bra], k = pair_i[ket], l = pair_j[ket], N = n;
#define AT(p, q, r, s) G[(((p) * N + (q)) * N + (r)) * N + (s)]
    AT(i, j, k, l) = v; AT(j, i, k, l) = v; AT(i, j, l, k) = v; AT(j, i, l, k) = v;
    AT(k, l, i, j) = v; AT(l, k, i, j) = v; AT(k, l, j, i) = v; AT(l, k, j, i) = v;
#undef AT
}

}  // extern "C"
