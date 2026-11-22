// Stage 2a: two-electron integrals and the J/K build in C++17, exposed to
// Python with pybind11.  Everything else in the SCF stays in Python, so any
// speed-up measured with CppBackend comes from these kernels alone.
//
// Entry points (all take the flat arrays produced by hfperf.basis.pack_basis):
//   eri_naive(...)          the SAME algorithm as backends/naive/integrals.py:
//                           one contracted function quartet at a time, recursive
//                           McMurchie-Davidson E and R, series Boys, no symmetry.
//   eri_sym8(..., tol, reuse)
//                           8-fold permutational symmetry + Schwarz screening.
//                           reuse=false: per-quartet kernel identical to eri_naive
//                                        (isolates the effect of symmetry+screening)
//                           reuse=true : precomputed primitive-pair data (Hermite E
//                                        tables, P, p), iterative R table and one
//                                        Boys evaluation per primitive quartet.
//   jk(G, D), jk_naive(G, D), jk_twopass(G, D)
//                           J_ij = sum_kl (ij|kl) D_kl,  K_ij = sum_kl (ik|jl) D_kl.
//                           jk_naive  : the Python loop order (two read streams of G)
//                           jk_twopass: one unit-stride pass for J, another for K
//                           jk        : one fused pass, each G element read once
//
// Stage 4 (threads) adds, in the "par" kernel section below:
//   eri_sym8_par(..., threads, schedule, chunk)   OpenMP over unique pair rows ij
//                           of eri_sym8(reuse=true); schedule = static|dynamic|guided
//   jk_par(G, D, threads)   OpenMP over rows i of the fused J/K pass
//   DirectJK                direct-SCF Fock build (no N^4 tensor): J/K accumulated
//                           from unique quartets with thread-private matrices +
//                           reduction ("private") or `omp atomic` updates ("atomic");
//                           optional row subset (for MPI ranks) and value cache.
// The same file compiles without -fopenmp (pragmas ignored -> serial results).
//
// OpenMP readiness: every outer loop marked "OMP:" writes to disjoint output
// elements and keeps its scratch space local to the iteration, so a later stage
// can add `#pragma omp parallel for` there without touching the math.
//
// LOC markers: lines between "BEGIN KERNEL <name>" and "END KERNEL <name>" are
// counted by ext/cpp/loc.py.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef HFCPP_MODNAME
#define HFCPP_MODNAME _hfcpp
#endif

namespace py = pybind11;

namespace hf {

constexpr double PI = 3.14159265358979323846;

// Flat structure-of-arrays basis (a copy of pack_basis output).
struct Basis {
    int n = 0;
    std::vector<double> center;  // n*3
    std::vector<int> lmn;        // n*3
    std::vector<int> off;        // n+1, primitives of f are [off[f], off[f+1])
    std::vector<double> exps, coefs;
};

// ===================================================== BEGIN KERNEL naive
// Line-by-line port of backends/naive (boys.py + integrals.py).

double boys(int n, double T) {
    if (T < 1e-12) return 1.0 / (2 * n + 1);
    if (T < 30.0) {
        double term = 1.0 / (2 * n + 1), total = term;
        int k = 0;
        while (term > 1e-17 * total) {
            ++k;
            term *= 2.0 * T / (2 * n + 2 * k + 1);
            total += term;
        }
        return std::exp(-T) * total;
    }
    double f = 0.5 * std::sqrt(PI / T) * std::erf(std::sqrt(T));
    const double e = std::exp(-T);
    for (int m = 0; m < n; ++m) f = ((2 * m + 1) * f - e) / (2 * T);
    return f;
}

double hermite_E(int i, int j, int t, double Qx, double a, double b) {
    const double p = a + b, q = a * b / p;
    if (t < 0 || t > i + j) return 0.0;
    if (i == 0 && j == 0 && t == 0) return std::exp(-q * Qx * Qx);
    if (j == 0)
        return 1.0 / (2 * p) * hermite_E(i - 1, j, t - 1, Qx, a, b)
               - q * Qx / a * hermite_E(i - 1, j, t, Qx, a, b)
               + (t + 1) * hermite_E(i - 1, j, t + 1, Qx, a, b);
    return 1.0 / (2 * p) * hermite_E(i, j - 1, t - 1, Qx, a, b)
           + q * Qx / b * hermite_E(i, j - 1, t, Qx, a, b)
           + (t + 1) * hermite_E(i, j - 1, t + 1, Qx, a, b);
}

double hermite_R(int t, int u, int v, int n, double p, double X, double Y, double Z, double RPC) {
    const double T = p * RPC * RPC;
    if (t == 0 && u == 0 && v == 0) return std::pow(-2 * p, n) * boys(n, T);
    double val = 0.0;
    if (t == 0 && u == 0) {
        if (v > 1) val += (v - 1) * hermite_R(t, u, v - 2, n + 1, p, X, Y, Z, RPC);
        return val + Z * hermite_R(t, u, v - 1, n + 1, p, X, Y, Z, RPC);
    }
    if (t == 0) {
        if (u > 1) val += (u - 1) * hermite_R(t, u - 2, v, n + 1, p, X, Y, Z, RPC);
        return val + Y * hermite_R(t, u - 1, v, n + 1, p, X, Y, Z, RPC);
    }
    if (t > 1) val += (t - 1) * hermite_R(t - 2, u, v, n + 1, p, X, Y, Z, RPC);
    return val + X * hermite_R(t - 1, u, v, n + 1, p, X, Y, Z, RPC);
}

// (ab|cd) over primitive Cartesian Gaussians.
double eri_prim(double a, const int* l1, const double* A, double b, const int* l2, const double* B,
                double c, const int* l3, const double* C, double d, const int* l4, const double* D) {
    const double p = a + b, q = c + d, alpha = p * q / (p + q);
    double P[3], Q[3];
    for (int k = 0; k < 3; ++k) {
        P[k] = (a * A[k] + b * B[k]) / p;
        Q[k] = (c * C[k] + d * D[k]) / q;
    }
    const double RPQ = std::sqrt((P[0] - Q[0]) * (P[0] - Q[0]) + (P[1] - Q[1]) * (P[1] - Q[1])
                                 + (P[2] - Q[2]) * (P[2] - Q[2]));
    double val = 0.0;
    for (int t = 0; t <= l1[0] + l2[0]; ++t)
        for (int u = 0; u <= l1[1] + l2[1]; ++u)
            for (int v = 0; v <= l1[2] + l2[2]; ++v)
                for (int tau = 0; tau <= l3[0] + l4[0]; ++tau)
                    for (int nu = 0; nu <= l3[1] + l4[1]; ++nu)
                        for (int phi = 0; phi <= l3[2] + l4[2]; ++phi)
                            val += hermite_E(l1[0], l2[0], t, A[0] - B[0], a, b)
                                   * hermite_E(l1[1], l2[1], u, A[1] - B[1], a, b)
                                   * hermite_E(l1[2], l2[2], v, A[2] - B[2], a, b)
                                   * hermite_E(l3[0], l4[0], tau, C[0] - D[0], c, d)
                                   * hermite_E(l3[1], l4[1], nu, C[1] - D[1], c, d)
                                   * hermite_E(l3[2], l4[2], phi, C[2] - D[2], c, d)
                                   * (((tau + nu + phi) % 2) ? -1.0 : 1.0)
                                   * hermite_R(t + tau, u + nu, v + phi, 0, alpha,
                                               P[0] - Q[0], P[1] - Q[1], P[2] - Q[2], RPQ);
    return 2 * std::pow(PI, 2.5) / (p * q * std::sqrt(p + q)) * val;
}

double eri_contracted(const Basis& B, int i, int j, int k, int l) {
    double s = 0.0;
    for (int pa = B.off[i]; pa < B.off[i + 1]; ++pa)
        for (int pb = B.off[j]; pb < B.off[j + 1]; ++pb)
            for (int pc = B.off[k]; pc < B.off[k + 1]; ++pc)
                for (int pd = B.off[l]; pd < B.off[l + 1]; ++pd)
                    s += B.coefs[pa] * B.coefs[pb] * B.coefs[pc] * B.coefs[pd]
                         * eri_prim(B.exps[pa], &B.lmn[3 * i], &B.center[3 * i],
                                    B.exps[pb], &B.lmn[3 * j], &B.center[3 * j],
                                    B.exps[pc], &B.lmn[3 * k], &B.center[3 * k],
                                    B.exps[pd], &B.lmn[3 * l], &B.center[3 * l]);
    return s;
}

// All N^4 integrals, one at a time, no symmetry (same as eri_tensor in Python).
// Rows i in [i0, i1) only, so a long run can be split into pieces (default: all).
void eri_naive(const Basis& B, double* G, int i0, int i1) {
    const std::int64_t n = B.n;
    // OMP: iterations over i write disjoint slabs G[i,:,:,:]
    for (int i = i0; i < i1; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    G[((i * n + j) * n + k) * n + l] = eri_contracted(B, i, j, k, l);
}
// ===================================================== END KERNEL naive

// ===================================================== BEGIN KERNEL sym8
// Reusable data for one basis-function pair (i, j): for every primitive pair
// (a, b) we store p = a + b, the product center P and the Hermite expansion
// coefficients E_tuv = c_a c_b E^{ij}_t(x) E^{ij}_u(y) E^{ij}_v(z).  These are
// computed once per pair instead of once per quartet.
constexpr int LMAX = 4;  // up to g functions per index; 6-31G* needs 2

struct Pair {
    int L[3] = {0, 0, 0};      // l_i + l_j along x, y, z
    int nherm = 1;             // (Lx+1)(Ly+1)(Lz+1)
    std::vector<double> p, P;  // nprim pairs; P is 3*nprim pairs
    std::vector<double> E;     // nprim pairs * nherm  (bra use)
    std::vector<double> Es;    // same with (-1)^(t+u+v)  (ket use)
};

// E^{ij}_t for t = 0..li+lj, built bottom-up (no recursion, no recomputation).
static void hermite_E_row(int li, int lj, double Qx, double a, double b, double* out) {
    double T[LMAX + 1][LMAX + 1][2 * LMAX + 2] = {};
    const double p = a + b, q = a * b / p, h = 1.0 / (2 * p);
    T[0][0][0] = std::exp(-q * Qx * Qx);
    for (int i = 0; i <= li; ++i)
        for (int j = 0; j <= lj; ++j) {
            if (i == 0 && j == 0) continue;
            for (int t = 0; t <= i + j; ++t) {
                if (j == 0)
                    T[i][0][t] = (t > 0 ? h * T[i - 1][0][t - 1] : 0.0) - q * Qx / a * T[i - 1][0][t]
                                 + (t + 1) * T[i - 1][0][t + 1];
                else
                    T[i][j][t] = (t > 0 ? h * T[i][j - 1][t - 1] : 0.0) + q * Qx / b * T[i][j - 1][t]
                                 + (t + 1) * T[i][j - 1][t + 1];
            }
        }
    for (int t = 0; t <= li + lj; ++t) out[t] = T[li][lj][t];
}

static Pair make_pair(const Basis& B, int i, int j) {
    Pair pr;
    const int* li = &B.lmn[3 * i];
    const int* lj = &B.lmn[3 * j];
    const double* A = &B.center[3 * i];
    const double* C = &B.center[3 * j];
    for (int k = 0; k < 3; ++k) pr.L[k] = li[k] + lj[k];
    pr.nherm = (pr.L[0] + 1) * (pr.L[1] + 1) * (pr.L[2] + 1);
    double Ex[2 * LMAX + 1], Ey[2 * LMAX + 1], Ez[2 * LMAX + 1];
    for (int pa = B.off[i]; pa < B.off[i + 1]; ++pa)
        for (int pb = B.off[j]; pb < B.off[j + 1]; ++pb) {
            const double a = B.exps[pa], b = B.exps[pb], p = a + b, cc = B.coefs[pa] * B.coefs[pb];
            pr.p.push_back(p);
            for (int k = 0; k < 3; ++k) pr.P.push_back((a * A[k] + b * C[k]) / p);
            hermite_E_row(li[0], lj[0], A[0] - C[0], a, b, Ex);
            hermite_E_row(li[1], lj[1], A[1] - C[1], a, b, Ey);
            hermite_E_row(li[2], lj[2], A[2] - C[2], a, b, Ez);
            for (int t = 0; t <= pr.L[0]; ++t)
                for (int u = 0; u <= pr.L[1]; ++u)
                    for (int v = 0; v <= pr.L[2]; ++v) {
                        const double e = cc * Ex[t] * Ey[u] * Ez[v];
                        pr.E.push_back(e);
                        pr.Es.push_back(((t + u + v) % 2) ? -e : e);
                    }
        }
    return pr;
}

// F_0..F_L(T) with one series (or closed form) evaluation + recursion.
static void boys_all(int L, double T, double* F) {
    const double e = std::exp(-T);
    if (T < 1e-12) {
        for (int n = 0; n <= L; ++n) F[n] = 1.0 / (2 * n + 1);
    } else if (T < 30.0) {
        F[L] = boys(L, T);
        for (int n = L; n > 0; --n) F[n - 1] = (2 * T * F[n] + e) / (2 * n - 1);  // downward: stable
    } else {
        F[0] = 0.5 * std::sqrt(PI / T) * std::erf(std::sqrt(T));
        for (int n = 0; n < L; ++n) F[n + 1] = ((2 * n + 1) * F[n] - e) / (2 * T);  // upward: fine at large T
    }
}

// Scratch space for one quartet; one per thread when OpenMP is added.
struct Scratch {
    std::vector<double> R;  // (L+1) * (Lx+1)(Ly+1)(Lz+1)
    double F[4 * LMAX + 1];
};

// Contracted (ij|kl) from two precomputed pairs.
static double quartet(const Pair& bra, const Pair& ket, Scratch& s) {
    const int Lx = bra.L[0] + ket.L[0], Ly = bra.L[1] + ket.L[1], Lz = bra.L[2] + ket.L[2];
    const int L = Lx + Ly + Lz;
    const int sv = 1, su = Lz + 1, st = (Ly + 1) * (Lz + 1), sn = (Lx + 1) * st;  // R strides
    s.R.resize(static_cast<size_t>(L + 1) * sn);
    double* R = s.R.data();
    const int npb = static_cast<int>(bra.p.size()), npk = static_cast<int>(ket.p.size());
    const double pref0 = 2 * std::pow(PI, 2.5);
    double total = 0.0;
    for (int ab = 0; ab < npb; ++ab) {
        const double p = bra.p[ab];
        const double* P = &bra.P[3 * ab];
        const double* Eb = &bra.E[static_cast<size_t>(ab) * bra.nherm];
        for (int cd = 0; cd < npk; ++cd) {
            const double q = ket.p[cd];
            const double* Q = &ket.P[3 * cd];
            const double* Ek = &ket.Es[static_cast<size_t>(cd) * ket.nherm];
            const double alpha = p * q / (p + q);
            const double X = P[0] - Q[0], Y = P[1] - Q[1], Z = P[2] - Q[2];
            boys_all(L, alpha * (X * X + Y * Y + Z * Z), s.F);
            // R^n_{tuv}, filled from n = L down to n = 0 (iterative MD recursion)
            double m2a = 1.0;
            for (int n = 0; n <= L; ++n) {  // R^n_000 = (-2 alpha)^n F_n
                R[n * sn] = m2a * s.F[n];
                m2a *= -2 * alpha;
            }
            for (int n = L - 1; n >= 0; --n) {
                const int rem = L - n;
                double* Rn = R + n * sn;
                const double* R1 = R + (n + 1) * sn;
                for (int t = 0; t <= std::min(Lx, rem); ++t)
                    for (int u = 0; u <= std::min(Ly, rem - t); ++u)
                        for (int v = 0; v <= std::min(Lz, rem - t - u); ++v) {
                            const int id = t * st + u * su + v * sv;
                            double val;
                            if (t > 0)
                                val = (t > 1 ? (t - 1) * R1[id - 2 * st] : 0.0) + X * R1[id - st];
                            else if (u > 0)
                                val = (u > 1 ? (u - 1) * R1[id - 2 * su] : 0.0) + Y * R1[id - su];
                            else if (v > 0)
                                val = (v > 1 ? (v - 1) * R1[id - 2 * sv] : 0.0) + Z * R1[id - sv];
                            else
                                continue;  // R^n_000 already set
                            Rn[id] = val;
                        }
            }
            // contract with the Hermite coefficients of bra and ket
            double val = 0.0;
            int hb = 0;
            for (int t = 0; t <= bra.L[0]; ++t)
                for (int u = 0; u <= bra.L[1]; ++u)
                    for (int v = 0; v <= bra.L[2]; ++v, ++hb) {
                        double inner = 0.0;
                        int hk = 0;
                        for (int tau = 0; tau <= ket.L[0]; ++tau)
                            for (int nu = 0; nu <= ket.L[1]; ++nu)
                                for (int phi = 0; phi <= ket.L[2]; ++phi, ++hk)
                                    inner += Ek[hk] * R[(t + tau) * st + (u + nu) * su + (v + phi)];
                        val += Eb[hb] * inner;
                    }
            total += pref0 / (p * q * std::sqrt(p + q)) * val;
        }
    }
    return total;
}

struct Sym8Stats {
    std::int64_t computed = 0, screened = 0;
};

// Unique quartets ij >= kl (i >= j, k >= l), Schwarz bound |(ij|kl)| <= Q_ij Q_kl.
Sym8Stats eri_sym8(const Basis& B, double tol, bool reuse, double* G) {
    const std::int64_t n = B.n;
    const std::int64_t npair = n * (n + 1) / 2;
    std::vector<int> pi(npair), pj(npair);
    for (int i = 0, ij = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j, ++ij) {
            pi[ij] = i;
            pj[ij] = j;
        }
    std::vector<Pair> pairs;
    if (reuse) {
        pairs.reserve(npair);
        for (std::int64_t ij = 0; ij < npair; ++ij) pairs.push_back(make_pair(B, pi[ij], pj[ij]));
    }
    Scratch s;
    auto value = [&](std::int64_t ij, std::int64_t kl) {
        return reuse ? quartet(pairs[ij], pairs[kl], s) : eri_contracted(B, pi[ij], pj[ij], pi[kl], pj[kl]);
    };
    std::vector<double> Q(npair);
    for (std::int64_t ij = 0; ij < npair; ++ij) Q[ij] = std::sqrt(std::fabs(value(ij, ij)));

    Sym8Stats st;
    auto at = [&](std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d) -> double& {
        return G[((a * n + b) * n + c) * n + d];
    };
    // OMP: dynamic schedule over ij (row lengths grow with ij); each (ij,kl)
    // writes its own 8 elements; Scratch and stats become thread-private.
    for (std::int64_t ij = 0; ij < npair; ++ij) {
        const int i = pi[ij], j = pj[ij];
        for (std::int64_t kl = 0; kl <= ij; ++kl) {
            if (Q[ij] * Q[kl] < tol) {
                ++st.screened;
                continue;
            }
            const int k = pi[kl], l = pj[kl];
            const double v = value(ij, kl);
            ++st.computed;
            at(i, j, k, l) = at(j, i, k, l) = at(i, j, l, k) = at(j, i, l, k) = v;
            at(k, l, i, j) = at(l, k, i, j) = at(k, l, j, i) = at(l, k, j, i) = v;
        }
    }
    return st;
}
// ===================================================== END KERNEL sym8

// ===================================================== BEGIN KERNEL jk
// Same loop order as backends/naive/fock.py: K reads G[i,k,j,l] with stride n^2 in k.
void jk_naive(std::int64_t n, const double* G, const double* D, double* J, double* K) {
    for (std::int64_t i = 0; i < n; ++i)  // OMP: rows i are independent
        for (std::int64_t j = 0; j < n; ++j) {
            double sj = 0.0, sk = 0.0;
            for (std::int64_t k = 0; k < n; ++k)
                for (std::int64_t l = 0; l < n; ++l) {
                    sj += G[((i * n + j) * n + k) * n + l] * D[k * n + l];
                    sk += G[((i * n + k) * n + j) * n + l] * D[k * n + l];
                }
            J[i * n + j] = sj;
            K[i * n + j] = sk;
        }
}

// Fused single pass (default): the J/K build is limited by memory bandwidth
// (one multiply-add per 8-byte load), so the number of passes over the N^4
// tensor sets the time.  Read each row G[i,k,j,:] once and use it twice:
//   J_ik += sum_l G[i,k,j,l] D[j,l]      K_ij += sum_l G[i,k,j,l] D[k,l]
void jk(std::int64_t n, const double* G, const double* D, double* J, double* K) {
    for (std::int64_t i = 0; i < n; ++i) {  // OMP: rows i are independent
        double* Ji = J + i * n;
        double* Ki = K + i * n;
        for (std::int64_t j = 0; j < n; ++j) Ji[j] = Ki[j] = 0.0;
        for (std::int64_t k = 0; k < n; ++k) {
            const double* dk = D + k * n;
            double sj = 0.0;
            for (std::int64_t j = 0; j < n; ++j) {
                const double* g = G + ((i * n + k) * n + j) * n;
                const double* dj = D + j * n;
                double a = 0.0, b = 0.0;
                for (std::int64_t l = 0; l < n; ++l) {
                    a += g[l] * dj[l];
                    b += g[l] * dk[l];
                }
                sj += a;
                Ki[j] += b;
            }
            Ji[k] = sj;
        }
    }
}

// Two passes (J, then K), each unit-stride: reads G twice -> about 2x the
// memory traffic of the fused version.  Kept as a measured counter-example.
void jk_twopass(std::int64_t n, const double* G, const double* D, double* J, double* K) {
    const std::int64_t n2 = n * n;
    for (std::int64_t i = 0; i < n; ++i) {      // OMP: rows i are independent
        for (std::int64_t j = 0; j < n; ++j) {  // J_ij = <G[i,j,:,:], D>  (a length-n^2 dot product)
            const double* g = G + (i * n + j) * n2;
            double s = 0.0;
            for (std::int64_t kl = 0; kl < n2; ++kl) s += g[kl] * D[kl];
            J[i * n + j] = s;
        }
        double* Ki = K + i * n;
        for (std::int64_t j = 0; j < n; ++j) Ki[j] = 0.0;
        for (std::int64_t k = 0; k < n; ++k)  // K_ij += sum_l G[i,k,j,l] D[k,l]
            for (std::int64_t j = 0; j < n; ++j) {
                const double* g = G + ((i * n + k) * n + j) * n;
                const double* d = D + k * n;
                double s = 0.0;
                for (std::int64_t l = 0; l < n; ++l) s += g[l] * d[l];
                Ki[j] += s;
            }
    }
}
// ===================================================== END KERNEL jk

// ===================================================== BEGIN KERNEL par
// Stage 4: shared-memory threads with OpenMP.
//
// Work unit = one unique pair row ij (all kl <= ij).  Row ij holds ij+1 quartets,
// so rows get longer as ij grows: a plain block partition ("static") hands the
// last thread ~2x the average work, while "dynamic"/"guided" (or static with a
// small chunk = round robin) balance it.  Per-thread busy times are returned so
// the imbalance can be measured, not just inferred.

inline double wtime() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

inline int resolve_threads(int nt) {
#ifdef _OPENMP
    return nt > 0 ? nt : omp_get_max_threads();
#else
    (void)nt;
    return 1;
#endif
}

// Map "static" | "dynamic" | "guided" (+ chunk; 0 = OpenMP default) onto schedule(runtime).
inline void set_schedule(const std::string& kind, int chunk) {
#ifdef _OPENMP
    omp_sched_t k;
    if (kind == "static") k = omp_sched_static;
    else if (kind == "dynamic") k = omp_sched_dynamic;
    else if (kind == "guided") k = omp_sched_guided;
    else throw std::invalid_argument("schedule must be static, dynamic or guided");
    omp_set_schedule(k, chunk);
#else
    if (kind != "static" && kind != "dynamic" && kind != "guided")
        throw std::invalid_argument("schedule must be static, dynamic or guided");
    (void)chunk;
#endif
}

inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Pair data + Schwarz factors for all n(n+1)/2 pairs, built in parallel.
struct PairSet {
    std::int64_t n = 0, npair = 0;
    std::vector<int> pi, pj;
    std::vector<Pair> pairs;
    std::vector<double> Q;
};

static PairSet make_pairset(const Basis& B, int nt) {
    PairSet P;
    P.n = B.n;
    P.npair = P.n * (P.n + 1) / 2;
    P.pi.resize(P.npair);
    P.pj.resize(P.npair);
    for (int i = 0, ij = 0; i < B.n; ++i)
        for (int j = 0; j <= i; ++j, ++ij) {
            P.pi[ij] = i;
            P.pj[ij] = j;
        }
    P.pairs.resize(P.npair);
    P.Q.resize(P.npair);
    const std::int64_t npair = P.npair;
#pragma omp parallel num_threads(nt)
    {
        Scratch s;
#pragma omp for schedule(dynamic, 16)
        for (std::int64_t ij = 0; ij < npair; ++ij) {
            P.pairs[ij] = make_pair(B, P.pi[ij], P.pj[ij]);
            P.Q[ij] = std::sqrt(std::fabs(quartet(P.pairs[ij], P.pairs[ij], s)));
        }
    }
    return P;
}

struct ParStats {
    std::int64_t computed = 0, screened = 0;
    std::vector<double> busy;  // per-thread seconds spent in the quartet loop
    double setup_s = 0.0, loop_s = 0.0, reduce_s = 0.0;
};

// eri_sym8(reuse=true) with the ij loop shared among threads.  Each unique
// quartet writes its own 8 tensor elements, so no two iterations write the same
// address: no locks, no atomics, no reduction needed.
static ParStats eri_sym8_par(const Basis& B, double tol, int nthreads, const std::string& sched, int chunk,
                             double* G) {
    const int nt = resolve_threads(nthreads);
    set_schedule(sched, chunk);
    ParStats st;
    double t0 = wtime();
    const PairSet P = make_pairset(B, nt);
    st.setup_s = wtime() - t0;
    const std::int64_t n = P.n, npair = P.npair;
    st.busy.assign(nt, 0.0);
    std::int64_t computed = 0, screened = 0;
    t0 = wtime();
#pragma omp parallel num_threads(nt) reduction(+ : computed, screened)
    {
        Scratch s;
        const double tb = wtime();
#pragma omp for schedule(runtime) nowait
        for (std::int64_t ij = 0; ij < npair; ++ij) {
            const std::int64_t i = P.pi[ij], j = P.pj[ij];
            for (std::int64_t kl = 0; kl <= ij; ++kl) {
                if (P.Q[ij] * P.Q[kl] < tol) {
                    ++screened;
                    continue;
                }
                const std::int64_t k = P.pi[kl], l = P.pj[kl];
                const double v = quartet(P.pairs[ij], P.pairs[kl], s);
                ++computed;
                G[((i * n + j) * n + k) * n + l] = G[((j * n + i) * n + k) * n + l] = v;
                G[((i * n + j) * n + l) * n + k] = G[((j * n + i) * n + l) * n + k] = v;
                G[((k * n + l) * n + i) * n + j] = G[((l * n + k) * n + i) * n + j] = v;
                G[((k * n + l) * n + j) * n + i] = G[((l * n + k) * n + j) * n + i] = v;
            }
        }
        st.busy[thread_id()] = wtime() - tb;  // time until THIS thread ran out of work
    }
    st.loop_s = wtime() - t0;
    st.computed = computed;
    st.screened = screened;
    return st;
}

// The fused J/K pass of `jk`, rows i shared among threads (rows are equal work
// and write disjoint rows J[i,:], K[i,:]).  This loop is memory-bandwidth bound,
// so it stops scaling once the cores saturate the memory channels.
static void jk_par(std::int64_t n, const double* G, const double* D, double* J, double* K, int nthreads) {
    const int nt = resolve_threads(nthreads);
#pragma omp parallel for schedule(static) num_threads(nt)
    for (std::int64_t i = 0; i < n; ++i) {
        double* Ji = J + i * n;
        double* Ki = K + i * n;
        for (std::int64_t j = 0; j < n; ++j) Ji[j] = Ki[j] = 0.0;
        for (std::int64_t k = 0; k < n; ++k) {
            const double* dk = D + k * n;
            double sj = 0.0;
            for (std::int64_t j = 0; j < n; ++j) {
                const double* g = G + ((i * n + k) * n + j) * n;
                const double* dj = D + j * n;
                double a = 0.0, b = 0.0;
                for (std::int64_t l = 0; l < n; ++l) {
                    a += g[l] * dj[l];
                    b += g[l] * dk[l];
                }
                sj += a;
                Ki[j] += b;
            }
            Ji[k] = sj;
        }
    }
}

// Direct-SCF J/K: no N^4 tensor; every build loops over the unique quartets of
// the rows it owns and scatters each value into J and K.  With D symmetric:
//   v' = (ij|kl) * deg,  deg = (i==j ? 1 : 2) (k==l ? 1 : 2) (ij==kl ? 1 : 2)
//   Jt_ij += D_kl v'   Jt_kl += D_ij v'
//   Kt_ik += D_jl v'   Kt_jl += D_ik v'   Kt_il += D_jk v'   Kt_jk += D_il v'
//   J = (Jt + Jt^T) / 4,  K = (Kt + Kt^T) / 8
// Different quartets update the SAME J/K elements -> a data race unless each
// thread accumulates privately (then reduces) or every update is atomic.
struct DirectJK {
    Basis B;
    PairSet P;
    double tol;
    std::vector<std::int64_t> rows;       // pair rows ij handled by this object (an MPI rank's share)
    bool store;                           // keep the computed values for later builds ("semi-direct")
    bool have_cache = false;
    std::vector<std::int64_t> row_off;    // cache offsets per row: sum of (ij+1) for earlier rows
    std::vector<double> cache;
    ParStats last;

    // all_rows = true: own every pair row (rows_ ignored); false: exactly rows_ (may be empty)
    DirectJK(Basis b, double tol_, bool store_, std::vector<std::int64_t> rows_, bool all_rows, int nthreads)
        : B(std::move(b)), tol(tol_), rows(std::move(rows_)), store(store_) {
        const double t0 = wtime();
        P = make_pairset(B, resolve_threads(nthreads));
        if (all_rows) {
            rows.resize(P.npair);
            for (std::int64_t r = 0; r < P.npair; ++r) rows[r] = r;
        }
        for (auto r : rows)
            if (r < 0 || r >= P.npair) throw std::invalid_argument("row index out of range");
        row_off.assign(rows.size() + 1, 0);
        for (size_t r = 0; r < rows.size(); ++r) row_off[r + 1] = row_off[r] + rows[r] + 1;
        last.setup_s = wtime() - t0;
    }

    std::int64_t n_quartets() const { return row_off.back(); }

    // accumulate into Jt, Kt (n x n, zeroed by caller); ACC(ptr, value) does the add
    template <class Acc>
    inline void scatter(std::int64_t ij, std::int64_t kl, double v, const double* D, double* Jt, double* Kt,
                        Acc acc) const {
        const std::int64_t n = P.n;
        const std::int64_t i = P.pi[ij], j = P.pj[ij], k = P.pi[kl], l = P.pj[kl];
        const double deg = (i == j ? 1.0 : 2.0) * (k == l ? 1.0 : 2.0) * (ij == kl ? 1.0 : 2.0);
        const double w = v * deg;
        acc(&Jt[i * n + j], D[k * n + l] * w);
        acc(&Jt[k * n + l], D[i * n + j] * w);
        acc(&Kt[i * n + k], D[j * n + l] * w);
        acc(&Kt[j * n + l], D[i * n + k] * w);
        acc(&Kt[i * n + l], D[j * n + k] * w);
        acc(&Kt[j * n + k], D[i * n + l] * w);
    }

    // one pass over the owned rows; Acc chooses plain (+=) or atomic adds
    template <class Acc>
    void sweep(const double* D, double* Jt, double* Kt, Scratch& s, bool fill, std::int64_t& computed,
               std::int64_t& screened, Acc acc) {
        const std::int64_t nrow = static_cast<std::int64_t>(rows.size());
#pragma omp for schedule(runtime) nowait
        for (std::int64_t r = 0; r < nrow; ++r) {
            const std::int64_t ij = rows[r];
            double* c = store ? cache.data() + row_off[r] : nullptr;
            for (std::int64_t kl = 0; kl <= ij; ++kl) {
                double v;
                if (have_cache) {
                    v = c[kl];
                    if (v == 0.0) continue;
                } else {
                    if (P.Q[ij] * P.Q[kl] < tol) {
                        ++screened;
                        if (fill) c[kl] = 0.0;
                        continue;
                    }
                    v = quartet(P.pairs[ij], P.pairs[kl], s);
                    ++computed;
                    if (fill) c[kl] = v;
                }
                scatter(ij, kl, v, D, Jt, Kt, acc);
            }
        }
    }

    // J, K for density D (n x n, symmetric).  mode = "private" | "atomic".
    void build(const double* D, double* J, double* K, int nthreads, const std::string& mode,
               const std::string& sched, int chunk) {
        const int nt = resolve_threads(nthreads);
        set_schedule(sched, chunk);
        if (mode != "private" && mode != "atomic") throw std::invalid_argument("mode must be private or atomic");
        const std::int64_t n = P.n, nn = n * n;
        const bool fill = store && !have_cache;
        if (fill) cache.assign(n_quartets(), 0.0);
        std::vector<double> Jt(nn, 0.0), Kt(nn, 0.0);
        ParStats st;
        st.setup_s = last.setup_s;
        st.busy.assign(nt, 0.0);
        std::int64_t computed = 0, screened = 0;
        double reduce_s = 0.0;
        const double t0 = wtime();
        if (mode == "private") {
#pragma omp parallel num_threads(nt) reduction(+ : computed, screened)
            {
                Scratch s;
                std::vector<double> Jp(nn, 0.0), Kp(nn, 0.0);  // this thread's partial J, K
                const double tb = wtime();
                sweep(D, Jp.data(), Kp.data(), s, fill, computed, screened,
                      [](double* p, double x) { *p += x; });
                st.busy[thread_id()] = wtime() - tb;
                const double tr = wtime();
#pragma omp critical(hfcpp_reduce)
                {
                    for (std::int64_t a = 0; a < nn; ++a) {
                        Jt[a] += Jp[a];
                        Kt[a] += Kp[a];
                    }
                    reduce_s = std::max(reduce_s, wtime() - tr);  // includes waiting for the lock
                }
            }
        } else {
            double* jt = Jt.data();
            double* kt = Kt.data();
#pragma omp parallel num_threads(nt) reduction(+ : computed, screened)
            {
                Scratch s;
                const double tb = wtime();
                sweep(D, jt, kt, s, fill, computed, screened, [](double* p, double x) {
#pragma omp atomic
                    *p += x;
                });
                st.busy[thread_id()] = wtime() - tb;
            }
        }
        st.loop_s = wtime() - t0;
        st.reduce_s = reduce_s;
        if (fill) have_cache = true;
        if (computed + screened > 0) {
            st.computed = computed;
            st.screened = screened;
        } else {  // cached build: report what the cache holds
            st.computed = last.computed;
            st.screened = last.screened;
        }
        last = st;
        for (std::int64_t a = 0; a < n; ++a)
            for (std::int64_t b = 0; b < n; ++b) {
                J[a * n + b] = 0.25 * (Jt[a * n + b] + Jt[b * n + a]);
                K[a * n + b] = 0.125 * (Kt[a * n + b] + Kt[b * n + a]);
            }
    }
};
// ===================================================== END KERNEL par

}  // namespace hf

// ----------------------------------------------------------- Python bindings
using DArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<int, py::array::c_style | py::array::forcecast>;

static hf::Basis make_basis(const DArr& centers, const IArr& lmn, const IArr& offsets, const DArr& exps,
                            const DArr& coefs) {
    hf::Basis B;
    B.n = static_cast<int>(offsets.size()) - 1;
    if (B.n < 0 || centers.size() != 3 * B.n || lmn.size() != 3 * B.n)
        throw std::invalid_argument("inconsistent basis arrays (expected centers/lmn of shape (nbf, 3))");
    B.center.assign(centers.data(), centers.data() + centers.size());
    B.lmn.assign(lmn.data(), lmn.data() + lmn.size());
    B.off.assign(offsets.data(), offsets.data() + offsets.size());
    B.exps.assign(exps.data(), exps.data() + exps.size());
    B.coefs.assign(coefs.data(), coefs.data() + coefs.size());
    if (B.off.back() != static_cast<int>(B.exps.size()) || B.exps.size() != B.coefs.size())
        throw std::invalid_argument("offsets[-1] must equal len(exps) == len(coefs)");
    for (int v : B.lmn)
        if (v < 0 || v > hf::LMAX) throw std::invalid_argument("angular momentum component > LMAX");
    return B;
}

static DArr new_tensor(py::ssize_t n) {
    DArr G({n, n, n, n});
    std::fill(G.mutable_data(), G.mutable_data() + G.size(), 0.0);
    return G;
}

template <class F>
static py::tuple jk_wrap(const DArr& G, const DArr& D, F kernel) {
    if (D.ndim() != 2 || D.shape(0) != D.shape(1)) throw std::invalid_argument("D must be square");
    const py::ssize_t n = D.shape(0);
    if (G.ndim() != 4 || G.shape(0) != n || G.shape(1) != n || G.shape(2) != n || G.shape(3) != n)
        throw std::invalid_argument("G must have shape (n, n, n, n) matching D");
    DArr J({n, n}), K({n, n});
    const double* g = G.data();
    const double* d = D.data();
    double* jp = J.mutable_data();
    double* kp = K.mutable_data();
    {
        py::gil_scoped_release nogil;
        kernel(n, g, d, jp, kp);
    }
    return py::make_tuple(J, K);
}

PYBIND11_MODULE(HFCPP_MODNAME, m) {
    m.doc() = "Chem179 stage 2a: C++ two-electron integrals and J/K";
    m.def("boys", &hf::boys, py::arg("n"), py::arg("T"));
    m.def(
        "eri_quartet",
        [](DArr c, IArr l, IArr o, DArr e, DArr k, int i, int j, int kk, int ll) {
            return hf::eri_contracted(make_basis(c, l, o, e, k), i, j, kk, ll);
        },
        "One contracted (ij|kl) with the naive kernel");
    m.def(
        "eri_naive",
        [](DArr c, IArr l, IArr o, DArr e, DArr k, int i_begin, int i_end) {
            hf::Basis B = make_basis(c, l, o, e, k);
            if (i_end < 0 || i_end > B.n) i_end = B.n;
            if (i_begin < 0 || i_begin > i_end) throw std::invalid_argument("need 0 <= i_begin <= i_end");
            DArr G = new_tensor(B.n);
            double* g = G.mutable_data();
            {
                py::gil_scoped_release nogil;
                hf::eri_naive(B, g, i_begin, i_end);
            }
            return G;
        },
        py::arg("centers"), py::arg("lmn"), py::arg("offsets"), py::arg("exps"), py::arg("coefs"),
        py::arg("i_begin") = 0, py::arg("i_end") = -1,
        "Full (ij|kl) tensor, same algorithm as the naive Python backend "
        "(optionally only rows G[i_begin:i_end], the rest left zero)");
    m.def(
        "eri_sym8",
        [](DArr c, IArr l, IArr o, DArr e, DArr k, double tol, bool reuse) {
            hf::Basis B = make_basis(c, l, o, e, k);
            DArr G = new_tensor(B.n);
            double* g = G.mutable_data();
            hf::Sym8Stats st;
            {
                py::gil_scoped_release nogil;
                st = hf::eri_sym8(B, tol, reuse, g);
            }
            return py::make_tuple(G, st.computed, st.screened);
        },
        py::arg("centers"), py::arg("lmn"), py::arg("offsets"), py::arg("exps"), py::arg("coefs"),
        py::arg("tol") = 1e-12, py::arg("reuse") = true,
        "Full tensor from unique quartets; returns (G, n_computed, n_screened)");
    m.def("jk", [](DArr G, DArr D) { return jk_wrap(G, D, hf::jk); }, py::arg("G"), py::arg("D"));
    m.def("jk_naive", [](DArr G, DArr D) { return jk_wrap(G, D, hf::jk_naive); }, py::arg("G"),
          py::arg("D"));
    m.def("jk_twopass", [](DArr G, DArr D) { return jk_wrap(G, D, hf::jk_twopass); }, py::arg("G"),
          py::arg("D"));
    // ---------------------------------------------------------------- stage 4 (threads)
    m.def("openmp_info", []() {
        py::dict d;
#ifdef _OPENMP
        d["enabled"] = true;
        d["version"] = static_cast<int>(_OPENMP);
        d["max_threads"] = omp_get_max_threads();
        d["num_procs"] = omp_get_num_procs();
#else
        d["enabled"] = false;
        d["version"] = 0;
        d["max_threads"] = 1;
        d["num_procs"] = 1;
#endif
        return d;
    });
    m.def(
        "eri_sym8_par",
        [](DArr c, IArr l, IArr o, DArr e, DArr k, double tol, int threads, std::string schedule, int chunk) {
            hf::Basis B = make_basis(c, l, o, e, k);
            DArr G = new_tensor(B.n);
            double* g = G.mutable_data();
            hf::ParStats st;
            {
                py::gil_scoped_release nogil;
                st = hf::eri_sym8_par(B, tol, threads, schedule, chunk, g);
            }
            py::dict info;
            info["computed"] = st.computed;
            info["screened"] = st.screened;
            info["busy"] = st.busy;
            info["setup_s"] = st.setup_s;
            info["loop_s"] = st.loop_s;
            return py::make_tuple(G, info);
        },
        py::arg("centers"), py::arg("lmn"), py::arg("offsets"), py::arg("exps"), py::arg("coefs"),
        py::arg("tol") = 1e-12, py::arg("threads") = 0, py::arg("schedule") = "dynamic", py::arg("chunk") = 1,
        "eri_sym8(reuse=True) with OpenMP over pair rows; returns (G, info dict with per-thread busy times)");
    m.def(
        "jk_par",
        [](DArr G, DArr D, int threads) {
            return jk_wrap(G, D, [threads](std::int64_t n, const double* g, const double* d, double* j, double* k) {
                hf::jk_par(n, g, d, j, k, threads);
            });
        },
        py::arg("G"), py::arg("D"), py::arg("threads") = 0);
    py::class_<hf::DirectJK>(m, "DirectJK", py::module_local(),  // _hfcpp and _hfcpp_omp each own one
                             
                             "Direct (integral-direct) J/K builder: no N^4 tensor is stored.\n"
                             "rows: pair rows ij this object owns (default: all) - used by the MPI driver.\n"
                             "store=True keeps the computed values after the first build (semi-direct).")
        .def(py::init([](DArr c, IArr l, IArr o, DArr e, DArr k, double tol, bool store,
                         std::optional<std::vector<std::int64_t>> rows, int threads) {
                 hf::Basis B = make_basis(c, l, o, e, k);
                 const bool all = !rows.has_value();
                 std::vector<std::int64_t> r = all ? std::vector<std::int64_t>{} : *rows;
                 py::gil_scoped_release nogil;
                 return new hf::DirectJK(std::move(B), tol, store, std::move(r), all, threads);
             }),
             py::arg("centers"), py::arg("lmn"), py::arg("offsets"), py::arg("exps"), py::arg("coefs"),
             py::arg("tol") = 1e-12, py::arg("store") = false, py::arg("rows") = py::none(),
             py::arg("threads") = 0)
        .def(
            "build",
            [](hf::DirectJK& self, DArr D, int threads, std::string mode, std::string schedule, int chunk) {
                const py::ssize_t n = static_cast<py::ssize_t>(self.P.n);
                if (D.ndim() != 2 || D.shape(0) != n || D.shape(1) != n)
                    throw std::invalid_argument("D must be (nbf, nbf)");
                DArr J({n, n}), K({n, n});
                const double* d = D.data();
                double* jp = J.mutable_data();
                double* kp = K.mutable_data();
                {
                    py::gil_scoped_release nogil;
                    self.build(d, jp, kp, threads, mode, schedule, chunk);
                }
                return py::make_tuple(J, K);
            },
            py::arg("D"), py::arg("threads") = 0, py::arg("mode") = "private", py::arg("schedule") = "dynamic",
            py::arg("chunk") = 1)
        .def_property_readonly("nbf", [](const hf::DirectJK& s) { return s.P.n; })
        .def_property_readonly("npair", [](const hf::DirectJK& s) { return s.P.npair; })
        .def_property_readonly("n_rows", [](const hf::DirectJK& s) { return s.rows.size(); })
        .def_property_readonly("n_quartets", &hf::DirectJK::n_quartets)
        .def_property_readonly("cached", [](const hf::DirectJK& s) { return s.have_cache; })
        .def_property_readonly("last_stats", [](const hf::DirectJK& s) {
            py::dict info;
            info["computed"] = s.last.computed;
            info["screened"] = s.last.screened;
            info["busy"] = s.last.busy;
            info["setup_s"] = s.last.setup_s;
            info["loop_s"] = s.last.loop_s;
            info["reduce_s"] = s.last.reduce_s;
            return info;
        });
}
