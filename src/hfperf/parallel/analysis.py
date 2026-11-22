"""Strong-scaling arithmetic.

speedup     S(p) = T(1) / T(p)
efficiency  E(p) = S(p) / p
Amdahl      T(p) = T1 * (s + (1 - s) / p), s = serial fraction.  T(p) = a + b/p is linear
            in (a, b), so the fit is an ordinary least-squares problem (we weight by 1/T(p)
            so that the many-thread points, which are small numbers, count as much as T(1)).
Karp-Flatt  e(p) = (1/S - 1/p) / (1 - 1/p): the serial fraction that would explain each
            measured point on its own.  If e grows with p, the loss is not a fixed serial
            part but overhead that grows with p (imbalance, memory bandwidth, sync).
"""
from __future__ import annotations

import numpy as np


def speedup_table(threads, times):
    p = np.asarray(threads, float)
    t = np.asarray(times, float)
    t1 = t[p == 1][0] if np.any(p == 1) else t[0] * p[0]
    s = t1 / t
    return [{"p": int(pi), "time_s": float(ti), "speedup": float(si), "efficiency": float(si / pi)}
            for pi, ti, si in zip(p, t, s)]


def amdahl_fit(threads, times) -> dict:
    """Least-squares fit of T(p) = a + b/p; returns serial fraction s, T1 and S_max = 1/s."""
    p = np.asarray(threads, float)
    t = np.asarray(times, float)
    A = np.stack([np.ones_like(p), 1.0 / p], axis=1) / t[:, None]
    (a, b), *_ = np.linalg.lstsq(A, np.ones_like(t), rcond=None)
    a, b = max(a, 0.0), max(b, 0.0)
    s = a / (a + b) if a + b > 0 else float("nan")
    pred = a + b / p
    return {"serial_fraction": float(s), "T1_fit": float(a + b), "max_speedup": float(1 / s) if s > 0 else float("inf"),
            "rms_rel_resid": float(np.sqrt(np.mean(((pred - t) / t) ** 2)))}


def karp_flatt(threads, times):
    out = []
    for r in speedup_table(threads, times):
        p, s = r["p"], r["speedup"]
        out.append(float("nan") if p == 1 else (1 / s - 1 / p) / (1 - 1 / p))
    return out


def load_imbalance(busy) -> float:
    """max/mean of per-thread busy times (1.0 = perfect)."""
    b = np.asarray([x for x in busy if x > 0], float)
    return float(b.max() / b.mean()) if b.size else float("nan")
