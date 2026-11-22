"""The Boys function F_n(T) = int_0^1 t^{2n} exp(-T t^2) dt.

Stage 0 evaluates it with a convergent power series for small T and the
closed form + downward-free upward recursion for large T.  It is called
once per (primitive quartet, n) - one of the hot spots you will profile.
"""
import math


def boys(n: int, T: float) -> float:
    if T < 1e-12:
        return 1.0 / (2 * n + 1)
    if T < 30.0:
        # F_n(T) = exp(-T) * sum_k (2T)^k / ((2n+1)(2n+3)...(2n+2k+1))
        term = 1.0 / (2 * n + 1)
        total = term
        k = 0
        while term > 1e-17 * total:
            k += 1
            term *= 2.0 * T / (2 * n + 2 * k + 1)
            total += term
        return math.exp(-T) * total
    # large T: F_0 = sqrt(pi/T)/2 erf(sqrt T), then upward recursion (stable here)
    f = 0.5 * math.sqrt(math.pi / T) * math.erf(math.sqrt(T))
    e = math.exp(-T)
    for m in range(n):
        f = ((2 * m + 1) * f - e) / (2 * T)
    return f
