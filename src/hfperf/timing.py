"""Kernel timer registry.

SCF code wraps each kernel call in ``with timer("name"):``.  For every call we
record
  * wall time   (time.perf_counter)
  * CPU time    (time.process_time, summed over all threads of the process)
  * optionally the peak Python-heap growth inside the call (tracemalloc)

CPU/wall is the *effective parallelism*: ~1 for single-threaded code, ~T
for a kernel that keeps T cores busy, < 1 when the process waits (I/O, GPU).
"""
from __future__ import annotations

import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager

TIMINGS: dict[str, list[float]] = defaultdict(list)    # wall seconds per call
CPU_TIMES: dict[str, list[float]] = defaultdict(list)  # CPU seconds per call
MEM_PEAK: dict[str, list[int]] = defaultdict(list)     # bytes, only when tracking memory
_TRACK_MEMORY = False


def track_memory(on: bool = True) -> None:
    """Record per-kernel Python-heap peaks (tracemalloc; slows Python code ~2-4x)."""
    global _TRACK_MEMORY
    _TRACK_MEMORY = on
    if on and not tracemalloc.is_tracing():
        tracemalloc.start()
    if not on and tracemalloc.is_tracing():
        tracemalloc.stop()


@contextmanager
def timer(name: str):
    if _TRACK_MEMORY:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
    c0 = time.process_time()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        TIMINGS[name].append(time.perf_counter() - t0)
        CPU_TIMES[name].append(time.process_time() - c0)
        if _TRACK_MEMORY:
            MEM_PEAK[name].append(max(0, tracemalloc.get_traced_memory()[1] - base))


def reset() -> None:
    TIMINGS.clear()
    CPU_TIMES.clear()
    MEM_PEAK.clear()


def summary() -> dict[str, dict[str, float]]:
    out = {}
    for k, v in TIMINGS.items():
        rec = {"calls": len(v), "total_s": sum(v), "cpu_s": sum(CPU_TIMES.get(k, []))}
        if MEM_PEAK.get(k):
            rec["peak_mem_bytes"] = max(MEM_PEAK[k])
        out[k] = rec
    return out


def report() -> str:
    rows = sorted(summary().items(), key=lambda kv: -kv[1]["total_s"])
    tot = sum(r["total_s"] for _, r in rows) or 1.0
    mem = any("peak_mem_bytes" in r for _, r in rows)
    head = f"{'kernel':<14}{'calls':>7}{'wall s':>11}{'share':>8}{'cpu/wall':>10}" + (f"{'peak MB':>10}" if mem else "")
    lines = [head]
    for k, r in rows:
        # CPU clocks are coarse (~15 ms on Windows): only show the ratio for kernels >= 50 ms
        ratio = f"{r['cpu_s'] / r['total_s']:.2f}" if r["total_s"] >= 0.05 else "-"
        line = f"{k:<14}{r['calls']:>7}{r['total_s']:>11.4f}{100 * r['total_s'] / tot:>7.1f}%{ratio:>10}"
        if mem:
            line += f"{r.get('peak_mem_bytes', 0) / 2**20:>10.1f}"
        lines.append(line)
    return "\n".join(lines)
