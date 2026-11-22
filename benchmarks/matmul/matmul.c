/* matmul.c - a tiny matrix-multiply library for the memory-hierarchy ladder.
 *
 * All matrices are n x n, row-major (C order), double precision; C = A B.
 * Build (see Makefile):  gcc -O2 -fPIC -shared matmul.c -o libmm_O2.so
 *                        gcc -O3 -march=native -fPIC -shared matmul.c -o libmm_O3native.so
 * Loaded from Python with ctypes (no pybind11 needed).
 */
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* textbook order: the inner loop walks B down a column -> stride n*8 bytes,
 * a new cache line for every multiply-add once a column no longer fits in cache */
void mm_ijk(int n, const double *A, const double *B, double *C)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
}

/* swapped loops: the inner loop walks rows of B and C with stride 1
 * -> every byte of a cache line is used, and the compiler can vectorize */
void mm_ikj(int n, const double *A, const double *B, double *C)
{
    memset(C, 0, sizeof(double) * (size_t)n * n);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            const double a = A[(size_t)i * n + k];
            const double *b = B + (size_t)k * n;
            double *c = C + (size_t)i * n;
            for (int j = 0; j < n; j++)
                c[j] += a * b[j];
        }
}

/* copy B^T first (O(n^2) extra work and memory), then an ijk loop whose
 * inner loop reads both operands with stride 1 (a dot product of two rows) */
void mm_tB(int n, const double *A, const double *B, double *C)
{
    double *Bt = (double *)malloc(sizeof(double) * (size_t)n * n);
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            Bt[(size_t)j * n + k] = B[(size_t)k * n + j];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            const double *a = A + (size_t)i * n, *b = Bt + (size_t)j * n;
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += a[k] * b[k];
            C[(size_t)i * n + j] = s;
        }
    free(Bt);
}

/* cache blocking: work on tile x tile sub-blocks so that the three blocks being
 * combined (3 * tile^2 * 8 bytes) stay in L1/L2 and are reused tile times */
void mm_blocked(int n, const double *A, const double *B, double *C, int tile)
{
    memset(C, 0, sizeof(double) * (size_t)n * n);
    for (int ii = 0; ii < n; ii += tile)
        for (int kk = 0; kk < n; kk += tile)
            for (int jj = 0; jj < n; jj += tile) {
                const int ie = ii + tile < n ? ii + tile : n;
                const int ke = kk + tile < n ? kk + tile : n;
                const int je = jj + tile < n ? jj + tile : n;
                for (int i = ii; i < ie; i++)
                    for (int k = kk; k < ke; k++) {
                        const double a = A[(size_t)i * n + k];
                        const double *b = B + (size_t)k * n;
                        double *c = C + (size_t)i * n;
                        for (int j = jj; j < je; j++)
                            c[j] += a * b[j];
                    }
            }
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* STREAM triad a = b + s*c; returns the best time of `reps` sweeps (seconds).
 * Bytes moved per sweep (STREAM convention): 3 * 8 * n  (2 reads + 1 write). */
double stream_triad(long n, double *a, const double *b, const double *c, double s, int reps)
{
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        for (long i = 0; i < n; i++)
            a[i] = b[i] + s * c[i];
        double t = now() - t0;
        if (t < best) best = t;
    }
    return best;
}

/* Peak-FLOP probe: NACC independent vector FMA chains (no memory traffic).
 * With -march=native on an AVX-512 core each v8d is one zmm register.
 * Returns seconds; FLOPs = iters * NACC * 8 lanes * 2 (multiply + add). */
typedef double v8d __attribute__((vector_size(64)));
#define NACC 12
double peak_fma(long iters, double *sink)
{
    v8d acc[NACC], x, y;
    for (int l = 0; l < 8; l++) { x[l] = 1.0 + 1e-9 * l; y[l] = 1e-12; }
    for (int a = 0; a < NACC; a++) acc[a] = x * (double)(a + 1);
    double t0 = now();
    for (long it = 0; it < iters; it++) {
        for (int a = 0; a < NACC; a++)
            acc[a] = acc[a] * x + y;
    }
    double t = now() - t0;
    double s = 0.0;
    for (int a = 0; a < NACC; a++)
        for (int l = 0; l < 8; l++) s += acc[a][l];
    *sink = s;
    return t;
}
